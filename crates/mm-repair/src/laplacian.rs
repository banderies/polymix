//! Discrete Laplacian weights and the implicit fairing systems built from them.
//!
//! Implicit systems use the symmetric form `L = W − D` (`W` = edge weights, `D` = row sums):
//!
//! * order 1 (membrane / harmonic): `−L x = 0` on the free vertices;
//! * order 2 (thin plate / bilaplacian): `L M⁻¹ L x = 0` on the free vertices, with `M` the
//!   vertex mass (degree for uniform weights — which makes this exactly the
//!   "umbrella-of-umbrella" operator `U² x = 0` scaled by the degree — and the barycentric
//!   area for cotangent weights, the usual FEM discretisation of Δ²).
//!
//! Both are symmetric positive definite once every free component touches a fixed vertex,
//! so they are solved with preconditioned conjugate gradients ([`crate::sparse`]): Jacobi
//! for small systems, smoothed-aggregation AMG (near-null space: constants and, for order
//! 2, the coordinate functions) from 1500 unknowns on. Values of fixed vertices move to
//! the right-hand side (Dirichlet constraints). Order 2 rows reach two rings out, so the
//! fixed set effectively consists of the two rings of vertices surrounding the free region.

use crate::sparse::{solve_pcg_preconditioned, solve_pcg_with_callback, Amg, AmgParams, CsrMatrix, Element, PcgParams, PcgStats};
use crate::util::LocalIndex;
use crate::{RepairError, Result};
use glam::DVec3;
use mm_core::mesh::Mesh;
use mm_core::INVALID;
use rayon::prelude::*;
use smallvec::SmallVec;

/// Edge weighting scheme of the discrete Laplacian.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LaplacianWeights {
    /// All neighbours weigh 1 (umbrella operator). Robust on any input, but the result
    /// depends on the connectivity, and explicit smoothing also relaxes tangentially.
    Uniform,
    /// Cotangent weights `(cot α + cot β) / 2` (the linear FEM Laplacian). Purely geometric,
    /// little tangential drift; exact for linear functions on planar meshes.
    #[default]
    Cotangent,
    /// Floater's mean-value weights `(tan(γ₁/2) + tan(γ₂/2)) / |e|`; always positive.
    /// Symmetrised for implicit systems.
    MeanValue,
}

/// Largest |cot| used: triangles with a larger cotangent (an angle below ~1.1° or above
/// ~178.9°) have all three cotangents scaled down by the same factor, which bounds the
/// weights while keeping the element stiffness matrix positive semi-definite.
const COT_MAX: f64 = 50.0;

/// Cotangent of the angle between `a` and `b` (clamped; for flip tests and similar).
#[inline]
pub(crate) fn cot(a: DVec3, b: DVec3) -> f64 {
    let c = a.cross(b).length();
    let d = a.dot(b);
    if c <= 1e-300 {
        return if d >= 0.0 { COT_MAX } else { -COT_MAX };
    }
    (d / c).clamp(-COT_MAX, COT_MAX)
}

/// `tan(γ/2)` for the angle γ between `a` and `b`.
#[inline]
fn tan_half(a: DVec3, b: DVec3) -> f64 {
    let c = a.cross(b).length();
    let la = a.length();
    let lb = b.length();
    if c <= 1e-300 {
        return if a.dot(b) >= 0.0 { 0.0 } else { 1e6 };
    }
    ((la * lb - a.dot(b)) / c).max(0.0)
}

/// Half the (scaled) cotangent of the angle at `po` in triangle (`pa`, `pb`, `po`), i.e. the
/// triangle's FEM contribution to the weight of edge (a, b). Zero for degenerate triangles.
#[inline]
fn cot_contribution(pa: DVec3, pb: DVec3, po: DVec3) -> f64 {
    let (ea, eb, ab) = (pa - po, pb - po, pb - pa);
    let a2 = ea.cross(eb).length();
    let l2 = ea.length_squared().max(eb.length_squared()).max(ab.length_squared());
    // Also rejects NaN.
    if a2.is_nan() || a2 <= 1e-14 * l2 {
        return 0.0;
    }
    let c_o = ea.dot(eb) / a2;
    let c_a = (-ea).dot(ab) / a2;
    let c_b = (-eb).dot(-ab) / a2;
    let s = c_o.abs().max(c_a.abs()).max(c_b.abs());
    let f = if s > COT_MAX { COT_MAX / s } else { 1.0 };
    0.5 * c_o * f
}

/// Cotangent (linear FEM) weight of edge `e`, with positions from `pos`. May be negative
/// next to obtuse angles; the resulting Laplacian is still positive semi-definite.
pub(crate) fn cot_weight<P: Fn(u32) -> DVec3>(mesh: &Mesh, e: u32, pos: &P) -> f64 {
    let [a, b] = mesh.edge_vertices(e);
    let (pa, pb) = (pos(a), pos(b));
    let mut w = 0.0;
    for o in mesh.edge_opposite_vertices(e) {
        if o != INVALID {
            w += cot_contribution(pa, pb, pos(o));
        }
    }
    w
}

/// Mean-value weight of edge `e` as seen from its endpoint `i`.
pub(crate) fn mean_value_weight<P: Fn(u32) -> DVec3>(mesh: &Mesh, e: u32, i: u32, pos: &P) -> f64 {
    let j = mesh.edge_other_vertex(e, i);
    let pi = pos(i);
    let a = pos(j) - pi;
    let len = a.length();
    if len <= 1e-300 {
        return 1e-12;
    }
    let mut w = 0.0;
    for o in mesh.edge_opposite_vertices(e) {
        if o != INVALID {
            w += tan_half(a, pos(o) - pi);
        }
    }
    let w = w / len;
    if w.is_finite() {
        w.max(1e-12)
    } else {
        1e-12
    }
}

/// Weight of edge `e` from vertex `i`'s point of view (non-symmetric for mean value).
#[inline]
/// Explicit smoothing uses only non-negative weights so each step stays a convex
/// combination of the neighbours.
pub(crate) fn directed_weight(mesh: &Mesh, e: u32, i: u32, kind: LaplacianWeights) -> f64 {
    let pos = |v: u32| mesh.vertex(v);
    match kind {
        LaplacianWeights::Uniform => 1.0,
        LaplacianWeights::Cotangent => cot_weight(mesh, e, &pos).max(0.0),
        LaplacianWeights::MeanValue => mean_value_weight(mesh, e, i, &pos),
    }
}

/// Symmetric weight of edge `e` for implicit systems.
#[inline]
pub(crate) fn symmetric_weight<P: Fn(u32) -> DVec3>(mesh: &Mesh, e: u32, kind: LaplacianWeights, pos: &P) -> f64 {
    match kind {
        LaplacianWeights::Uniform => 1.0,
        LaplacianWeights::Cotangent => cot_weight(mesh, e, pos),
        LaplacianWeights::MeanValue => {
            let [a, b] = mesh.edge_vertices(e);
            0.5 * (mean_value_weight(mesh, e, a, pos) + mean_value_weight(mesh, e, b, pos))
        }
    }
}

/// One Laplacian row: neighbours with symmetric weights, the row sum and the vertex mass.
struct LapRow {
    nbrs: SmallVec<[(u32, f64); 8]>,
    diag: f64,
    mass: f64,
}

fn lap_row<P: Fn(u32) -> DVec3>(mesh: &Mesh, v: u32, kind: LaplacianWeights, pos: &P) -> LapRow {
    let mut nbrs: SmallVec<[(u32, f64); 8]> = SmallVec::new();
    let mut diag = 0.0;
    let mut len2 = 0.0;
    for &e in mesh.vertex_edges(v) {
        let w = symmetric_weight(mesh, e, kind, pos);
        let o = mesh.edge_other_vertex(e, v);
        len2 += pos(o).distance_squared(pos(v));
        nbrs.push((o, w));
        diag += w;
    }
    let mass = match kind {
        LaplacianWeights::Cotangent => {
            let area: f64 = mesh
                .vertex_triangles(v)
                .iter()
                .map(|&t| {
                    let [a, b, c] = mesh.triangle(t).map(pos);
                    0.5 * (b - a).cross(c - a).length()
                })
                .sum::<f64>()
                / 3.0;
            let floor = 1e-6 * len2 / nbrs.len().max(1) as f64;
            area.max(floor).max(1e-300)
        }
        _ => diag.max(1e-300),
    };
    LapRow { nbrs, diag, mass }
}

/// Assembled Dirichlet fairing problem over a set of free vertices.
pub(crate) struct FairSystem {
    pub free: Vec<u32>,
    pub matrix: CsrMatrix,
    pub rhs: Vec<DVec3>,
}

/// Drop free vertices whose connected component (through free vertices) has no fixed
/// neighbour: their system would be singular.
fn constrained_free_set(mesh: &Mesh, free: &[u32]) -> Vec<u32> {
    let mut idx = LocalIndex::new(mesh.max_vid(), free.len());
    let mut list: Vec<u32> = Vec::with_capacity(free.len());
    for &v in free {
        if mesh.is_vertex(v) && idx.get(v).is_none() && mesh.vertex_valence(v) > 0 {
            idx.insert(v, list.len() as u32);
            list.push(v);
        }
    }
    let mut comp = vec![u32::MAX; list.len()];
    let mut keep = vec![false; list.len()];
    let mut stack = Vec::new();
    let mut members = Vec::new();
    for s in 0..list.len() {
        if comp[s] != u32::MAX {
            continue;
        }
        comp[s] = s as u32;
        stack.push(s);
        members.clear();
        let mut constrained = false;
        while let Some(i) = stack.pop() {
            members.push(i);
            for nb in mesh.vertex_neighbors(list[i]) {
                match idx.get(nb) {
                    Some(j) => {
                        if comp[j as usize] == u32::MAX {
                            comp[j as usize] = s as u32;
                            stack.push(j as usize);
                        }
                    }
                    None => constrained = true,
                }
            }
        }
        if constrained {
            for &i in &members {
                keep[i] = true;
            }
        }
    }
    list.into_iter().zip(keep).filter(|(_, k)| *k).map(|(v, _)| v).collect()
}

/// Build the order-1 or order-2 system for `free` (other vertices are fixed).
pub(crate) fn build_fair_system(mesh: &Mesh, free: &[u32], order: u32, kind: LaplacianWeights) -> FairSystem {
    build_fair_system_with(mesh, free, order, kind, &|v| mesh.vertex(v))
}

/// [`build_fair_system`] with the weights (and masses) computed from positions `wpos`
/// instead of the mesh's (the right-hand side always uses the real positions).
pub(crate) fn build_fair_system_with<P: Fn(u32) -> DVec3 + Sync>(
    mesh: &Mesh,
    free: &[u32],
    order: u32,
    kind: LaplacianWeights,
    wpos: &P,
) -> FairSystem {
    let free = constrained_free_set(mesh, free);
    let nf = free.len();
    let mut fidx = LocalIndex::new(mesh.max_vid(), nf);
    for (i, &v) in free.iter().enumerate() {
        fidx.insert(v, i as u32);
    }
    let par = nf >= 2048;

    if order <= 1 {
        let build = |&v: &u32| -> (Vec<(u32, f64)>, DVec3) {
            let row = lap_row(mesh, v, kind, wpos);
            let mut ent = Vec::with_capacity(row.nbrs.len() + 1);
            let mut rhs = DVec3::ZERO;
            ent.push((fidx.get(v).unwrap(), row.diag));
            for &(k, w) in &row.nbrs {
                match fidx.get(k) {
                    Some(l) => ent.push((l, -w)),
                    None => rhs += mesh.vertex(k) * w,
                }
            }
            (ent, rhs)
        };
        let rows: Vec<(Vec<(u32, f64)>, DVec3)> =
            if par { free.par_iter().map(build).collect() } else { free.iter().map(build).collect() };
        let (ent, rhs): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
        return FairSystem { matrix: CsrMatrix::from_rows(nf, nf, ent), rhs, free };
    }

    // Order 2: Laplacian rows are needed for the free vertices and their one ring.
    let mut ridx = LocalIndex::new(mesh.max_vid(), nf * 2);
    let mut row_verts: Vec<u32> = Vec::with_capacity(nf * 2);
    for &v in &free {
        if ridx.get(v).is_none() {
            ridx.insert(v, row_verts.len() as u32);
            row_verts.push(v);
        }
        for nb in mesh.vertex_neighbors(v) {
            if ridx.get(nb).is_none() {
                ridx.insert(nb, row_verts.len() as u32);
                row_verts.push(nb);
            }
        }
    }
    let lap: Vec<LapRow> = if par {
        row_verts.par_iter().map(|&v| lap_row(mesh, v, kind, wpos)).collect()
    } else {
        row_verts.iter().map(|&v| lap_row(mesh, v, kind, wpos)).collect()
    };
    let build = |&i: &u32| -> (Vec<(u32, f64)>, DVec3) {
        let mut ent: Vec<(u32, f64)> = Vec::with_capacity(64);
        let mut rhs = DVec3::ZERO;
        let li = &lap[ridx.get(i).unwrap() as usize];
        let mut add = |k: u32, val: f64, ent: &mut Vec<(u32, f64)>| match fidx.get(k) {
            Some(l) => ent.push((l, val)),
            None => rhs -= mesh.vertex(k) * val,
        };
        // Σ_j L_ij M_j⁻¹ L_jk over j ∈ {i} ∪ N(i).
        let js = std::iter::once((i, -li.diag)).chain(li.nbrs.iter().copied());
        for (j, lij) in js {
            let lj = &lap[ridx.get(j).unwrap() as usize];
            let c = lij / lj.mass;
            add(j, -c * lj.diag, &mut ent);
            for &(k, wjk) in &lj.nbrs {
                add(k, c * wjk, &mut ent);
            }
        }
        (ent, rhs)
    };
    let rows: Vec<(Vec<(u32, f64)>, DVec3)> =
        if par { free.par_iter().map(build).collect() } else { free.iter().map(build).collect() };
    let (ent, rhs): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    FairSystem { matrix: CsrMatrix::from_rows(nf, nf, ent), rhs, free }
}

/// Systems at least this large are preconditioned with smoothed-aggregation AMG.
const AMG_MIN_UNKNOWNS: usize = 1500;

/// Low-energy vectors of the fairing operator on `free`: constants (order 1), plus the
/// coordinate functions, whose thin-plate energy vanishes (order 2).
fn near_null_space(mesh: &Mesh, free: &[u32], order: u32) -> Vec<Vec<f64>> {
    let n = free.len();
    let mut out = vec![vec![1.0; n]];
    if order >= 2 && n > 0 {
        let c = free.iter().map(|&v| mesh.vertex(v)).sum::<DVec3>() / n as f64;
        let r = free.iter().map(|&v| mesh.vertex(v).distance(c)).fold(0.0, f64::max).max(1e-300);
        for axis in 0..3 {
            out.push(free.iter().map(|&v| (mesh.vertex(v) - c)[axis] / r).collect());
        }
    }
    out
}

/// PCG with AMG for large systems, Jacobi otherwise.
#[allow(clippy::too_many_arguments)]
fn solve_fair<T: Element>(
    mesh: &Mesh,
    free: &[u32],
    order: u32,
    a: &CsrMatrix,
    b: &[T],
    x: &mut [T],
    pcg: &PcgParams,
    callback: Option<&dyn Fn(usize, f64) -> bool>,
) -> PcgStats {
    if free.len() >= AMG_MIN_UNKNOWNS {
        if let Some(amg) = Amg::new(a, &near_null_space(mesh, free, order), &AmgParams::default()) {
            return solve_pcg_preconditioned(a, b, x, pcg, &amg.preconditioner(1), callback);
        }
    }
    solve_pcg_with_callback(a, b, x, pcg, callback)
}

/// Solve the fairing system for `free` and write the result into the mesh.
/// Returns the solver statistics; positions are only written if the solution is finite.
pub(crate) fn fair_vertices(
    mesh: &mut Mesh,
    free: &[u32],
    order: u32,
    kind: LaplacianWeights,
    pcg: &PcgParams,
    callback: Option<&dyn Fn(usize, f64) -> bool>,
) -> Result<PcgStats> {
    if order == 0 || order > 2 {
        return Err(RepairError::InvalidParameter(format!("fairing order must be 1 or 2, got {order}")));
    }
    let sys = build_fair_system(mesh, free, order, kind);
    if sys.free.is_empty() {
        return Ok(PcgStats { converged: true, ..Default::default() });
    }
    let mut x: Vec<DVec3> = sys.free.iter().map(|&v| mesh.vertex(v)).collect();
    let stats = solve_fair(mesh, &sys.free, order, &sys.matrix, &sys.rhs, &mut x, pcg, callback);
    if stats.breakdown {
        return Err(RepairError::Solver("fairing matrix is not positive definite".into()));
    }
    if x.iter().any(|p| !p.is_finite()) {
        return Err(RepairError::Solver("non-finite solution".into()));
    }
    let pos = mesh.positions_mut();
    for (&v, p) in sys.free.iter().zip(x) {
        pos[v as usize] = p;
    }
    mesh.bump_shape_version();
    Ok(stats)
}

/// Direction along which [`fair_vertices_along`] moves vertices.
#[derive(Clone, Copy, Debug)]
pub(crate) enum FairDirection {
    /// Area-weighted vertex normals of the current surface.
    VertexNormals,
    /// One fixed unit direction for all vertices (e.g. a plane normal).
    Fixed(DVec3),
}

/// Fair `free` while only allowing displacements along a per-vertex direction `nᵢ`.
///
/// Minimises the same quadratic energy as [`fair_vertices`] over `xᵢ = x⁰ᵢ + sᵢ nᵢ`; the
/// reduced system `Kᵢₖ = Aᵢₖ (nᵢ·nₖ)` is again symmetric positive definite. Keeping the
/// tangential vertex distribution fixed avoids the drift (and the resulting distortion)
/// that a full 3D solve shows when the free region is meshed more finely than its
/// surroundings.
pub(crate) fn fair_vertices_along(
    mesh: &mut Mesh,
    free: &[u32],
    order: u32,
    kind: LaplacianWeights,
    dir: FairDirection,
    pcg: &PcgParams,
) -> Result<PcgStats> {
    if order == 0 || order > 2 {
        return Err(RepairError::InvalidParameter(format!("fairing order must be 1 or 2, got {order}")));
    }
    let sys = match dir {
        // A fixed direction is a height field over the plane ⟂ to it: compute the weights
        // in that plane (the parametrisation domain), not on the tilted surface.
        FairDirection::Fixed(n) => {
            let n = n.normalize_or_zero();
            build_fair_system_with(mesh, free, order, kind, &|v| {
                let p = mesh.vertex(v);
                p - n * p.dot(n)
            })
        }
        FairDirection::VertexNormals => build_fair_system(mesh, free, order, kind),
    };
    let nf = sys.free.len();
    if nf == 0 {
        return Ok(PcgStats { converged: true, ..Default::default() });
    }
    let normals: Vec<DVec3> = match dir {
        FairDirection::VertexNormals => sys.free.iter().map(|&v| mesh.vertex_normal(v)).collect(),
        FairDirection::Fixed(n) => vec![n.normalize_or_zero(); nf],
    };
    let x0: Vec<DVec3> = sys.free.iter().map(|&v| mesh.vertex(v)).collect();
    let mut ax = vec![DVec3::ZERO; nf];
    sys.matrix.mul_vec(&x0, &mut ax);
    let rhs: Vec<f64> = (0..nf).map(|i| normals[i].dot(sys.rhs[i] - ax[i])).collect();
    let rows: Vec<Vec<(u32, f64)>> = (0..nf)
        .map(|i| {
            let (cols, vals) = sys.matrix.row(i);
            cols.iter().zip(vals).map(|(&k, &a)| (k, a * normals[i].dot(normals[k as usize]))).collect()
        })
        .collect();
    let k = CsrMatrix::from_rows(nf, nf, rows);
    let mut s = vec![0.0f64; nf];
    let stats = solve_fair(mesh, &sys.free, order, &k, &rhs, &mut s, pcg, None);
    if stats.breakdown {
        return Err(RepairError::Solver("fairing matrix is not positive definite".into()));
    }
    if s.iter().any(|x| !x.is_finite()) {
        return Err(RepairError::Solver("non-finite solution".into()));
    }
    let pos = mesh.positions_mut();
    for i in 0..nf {
        pos[sys.free[i] as usize] = x0[i] + normals[i] * s[i];
    }
    mesh.bump_shape_version();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn order2_system_is_symmetric() {
        let m = primitives::icosphere(1.0, 2);
        let free: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).z > 0.6).collect();
        for kind in [LaplacianWeights::Uniform, LaplacianWeights::Cotangent, LaplacianWeights::MeanValue] {
            for order in [1, 2] {
                let s = build_fair_system(&m, &free, order, kind);
                assert_eq!(s.free.len(), free.len());
                assert!(s.matrix.is_symmetric(1e-9), "{kind:?} order {order}");
            }
        }
    }

    #[test]
    fn planar_cot_laplacian_reproduces_linear_functions() {
        // Interior vertices of a jittered planar grid are fixed points of the cot system.
        let mut m = primitives::plane_grid(2.0, 2.0, 8, 8);
        let ids: Vec<u32> = m.vertex_ids().collect();
        for (k, &v) in ids.iter().enumerate() {
            if !m.is_boundary_vertex(v) {
                let p = m.vertex(v);
                let j = 0.05 * ((k as f64 * 1.7).sin());
                m.set_vertex(v, p + DVec3::new(j, -j * 0.5, 0.0));
            }
        }
        let free: Vec<u32> = m.vertex_ids().filter(|&v| !m.is_boundary_vertex(v)).collect();
        let before: Vec<DVec3> = free.iter().map(|&v| m.vertex(v)).collect();
        let st = fair_vertices(&mut m, &free, 1, LaplacianWeights::Cotangent, &PcgParams::default(), None).unwrap();
        assert!(st.converged);
        for (&v, p) in free.iter().zip(before) {
            assert!(m.vertex(v).distance(p) < 1e-6);
        }
    }

    #[test]
    fn unconstrained_components_are_skipped() {
        let mut m = primitives::icosphere(1.0, 1);
        let all: Vec<u32> = m.vertex_ids().collect();
        let st = fair_vertices(&mut m, &all, 2, LaplacianWeights::Uniform, &PcgParams::default(), None).unwrap();
        assert_eq!(st.iterations, 0);
        m.check_validity().unwrap();
    }
}
