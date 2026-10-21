//! Discrete Laplacian weights and the implicit fairing systems built from them.
//!
//! Implicit systems use the symmetric form `L = W − D` (`W` = edge weights, `D` = row sums):
//!
//! * order 1 (membrane / harmonic): `−L x = 0` on the free vertices;
//! * order 2 (thin plate / bilaplacian): `L M⁻¹ L x = 0` on the free vertices, with `M` the
//!   vertex mass (degree for uniform weights — which makes this exactly the
//!   "umbrella-of-umbrella" operator `U² x = 0` scaled by the degree — and the barycentric
//!   area for cotangent weights, the usual FEM discretisation of Δ²);
//! * order 3 (triharmonic, minimum curvature variation): `−L M⁻¹ L M⁻¹ L x = 0`, which
//!   blends C² with three rings of fixed surroundings.
//!
//! Both are symmetric positive definite once every free component touches a fixed vertex,
//! so they are solved with preconditioned conjugate gradients ([`crate::sparse`]): Jacobi
//! for small systems, smoothed-aggregation AMG (near-null space: constants and, for order
//! 2, the coordinate functions) from 1500 unknowns on. Values of fixed vertices move to
//! the right-hand side (Dirichlet constraints). Order 2 rows reach two rings out, so the
//! fixed set effectively consists of the two rings of vertices surrounding the free region.

use crate::sparse::{solve_pcg_preconditioned, solve_pcg_with_callback, Amg, AmgParams, CsrMatrix, Element, Jacobi, PcgParams, PcgStats};
use crate::util::LocalIndex;
use crate::{RepairError, Result};
use glam::DVec3;
use mm_core::hash::FastMap;
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

/// An order 1 or 2 fairing matrix over the constrained free set, and, per free row, the fixed
/// vertices its right-hand side gathers with their coefficients, in the order the assembly
/// visits them (so the right-hand side for any fixed values sums exactly as a direct assembly
/// would).
struct FairAssembly {
    free: Vec<u32>,
    matrix: CsrMatrix,
    /// Row `i` gathers `terms[term_ptr[i]..term_ptr[i + 1]]`.
    term_ptr: Vec<usize>,
    terms: Vec<(u32, f64)>,
    /// Order 2 rows subtract their terms (`rhs -= x_k c`); order 1 rows add them.
    subtract: bool,
}

impl FairAssembly {
    /// The right-hand side for fixed vertices at `vpos`.
    fn rhs<V: Fn(u32) -> DVec3 + Sync>(&self, vpos: &V) -> Vec<DVec3> {
        let row = |i: usize| -> DVec3 {
            let mut rhs = DVec3::ZERO;
            for &(k, c) in &self.terms[self.term_ptr[i]..self.term_ptr[i + 1]] {
                if self.subtract {
                    rhs -= vpos(k) * c;
                } else {
                    rhs += vpos(k) * c;
                }
            }
            rhs
        };
        // Only rows next to the fixed vertices have terms: usually too few to be worth a
        // parallel pass.
        let n = self.free.len();
        if self.terms.len() >= 1 << 16 {
            (0..n).into_par_iter().map(row).collect()
        } else {
            (0..n).map(row).collect()
        }
    }
}

/// One free row of a fairing system: its matrix entries, and the fixed vertices its
/// right-hand side gathers with their coefficients.
type Row = (Vec<(u32, f64)>, Vec<(u32, f64)>);

/// Matrix rows and right-hand-side terms per free row, flattened into a [`FairAssembly`].
fn flatten_rows(free: Vec<u32>, rows: Vec<Row>, subtract: bool) -> FairAssembly {
    let nf = free.len();
    let mut term_ptr = Vec::with_capacity(nf + 1);
    term_ptr.push(0);
    let mut terms = Vec::with_capacity(rows.iter().map(|r| r.1.len()).sum());
    let mut ent = Vec::with_capacity(nf);
    for (e, t) in rows {
        terms.extend(t);
        term_ptr.push(terms.len());
        ent.push(e);
    }
    FairAssembly { matrix: CsrMatrix::from_rows(nf, nf, ent), free, term_ptr, terms, subtract }
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

/// Build the order-1, 2 or 3 system for `free` (other vertices are fixed).
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
    build_fair_system_at(mesh, free, order, kind, wpos, &|v| mesh.vertex(v))
}

/// [`build_fair_system_with`] with the fixed vertices' values (the right-hand side) taken from
/// `vpos` instead of the mesh.
fn build_fair_system_at<W: Fn(u32) -> DVec3 + Sync, V: Fn(u32) -> DVec3 + Sync>(
    mesh: &Mesh,
    free: &[u32],
    order: u32,
    kind: LaplacianWeights,
    wpos: &W,
    vpos: &V,
) -> FairSystem {
    if order >= 3 {
        let free = constrained_free_set(mesh, free);
        let mut fidx = LocalIndex::new(mesh.max_vid(), free.len());
        for (i, &v) in free.iter().enumerate() {
            fidx.insert(v, i as u32);
        }
        return build_order3_system(mesh, free, fidx, kind, wpos, vpos);
    }
    let a = assemble(mesh, free, order, kind, wpos);
    let rhs = a.rhs(vpos);
    FairSystem { free: a.free, matrix: a.matrix, rhs }
}

/// The order 1 (or less) or 2 matrix for `free` (other vertices fixed), with the right-hand
/// side's terms left unevaluated.
fn assemble<W: Fn(u32) -> DVec3 + Sync>(mesh: &Mesh, free: &[u32], order: u32, kind: LaplacianWeights, wpos: &W) -> FairAssembly {
    let free = constrained_free_set(mesh, free);
    let nf = free.len();
    let mut fidx = LocalIndex::new(mesh.max_vid(), nf);
    for (i, &v) in free.iter().enumerate() {
        fidx.insert(v, i as u32);
    }
    let par = nf >= 2048;

    if order <= 1 {
        let build = |&v: &u32| -> Row {
            let row = lap_row(mesh, v, kind, wpos);
            let mut ent = Vec::with_capacity(row.nbrs.len() + 1);
            let mut terms = Vec::new();
            ent.push((fidx.get(v).unwrap(), row.diag));
            for &(k, w) in &row.nbrs {
                match fidx.get(k) {
                    Some(l) => ent.push((l, -w)),
                    None => terms.push((k, w)),
                }
            }
            (ent, terms)
        };
        let rows: Vec<Row> = if par { free.par_iter().map(build).collect() } else { free.iter().map(build).collect() };
        return flatten_rows(free, rows, false);
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
    let build = |&i: &u32| -> Row {
        let mut ent: Vec<(u32, f64)> = Vec::with_capacity(64);
        let mut terms = Vec::new();
        let li = &lap[ridx.get(i).unwrap() as usize];
        let mut add = |k: u32, val: f64, ent: &mut Vec<(u32, f64)>| match fidx.get(k) {
            Some(l) => ent.push((l, val)),
            None => terms.push((k, val)),
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
        (ent, terms)
    };
    let rows: Vec<Row> = if par { free.par_iter().map(build).collect() } else { free.iter().map(build).collect() };
    flatten_rows(free, rows, true)
}

/// Order 3: rows `−(L M⁻¹ L M⁻¹ L)ᵢ` for the free vertices, from Laplacian rows of the free
/// vertices and their two rings.
fn build_order3_system<P: Fn(u32) -> DVec3 + Sync, V: Fn(u32) -> DVec3 + Sync>(
    mesh: &Mesh,
    free: Vec<u32>,
    fidx: LocalIndex,
    kind: LaplacianWeights,
    wpos: &P,
    vpos: &V,
) -> FairSystem {
    let nf = free.len();
    let mut ridx = LocalIndex::new(mesh.max_vid(), nf * 3);
    let mut row_verts: Vec<u32> = Vec::with_capacity(nf * 3);
    let mut ring: Vec<u32> = free.clone();
    for &v in &ring {
        ridx.insert(v, row_verts.len() as u32);
        row_verts.push(v);
    }
    for _ in 0..2 {
        let mut next = Vec::new();
        for &v in &ring {
            for nb in mesh.vertex_neighbors(v) {
                if ridx.get(nb).is_none() {
                    ridx.insert(nb, row_verts.len() as u32);
                    row_verts.push(nb);
                    next.push(nb);
                }
            }
        }
        ring = next;
    }
    let lap: Vec<LapRow> = if nf >= 2048 {
        row_verts.par_iter().map(|&v| lap_row(mesh, v, kind, wpos)).collect()
    } else {
        row_verts.iter().map(|&v| lap_row(mesh, v, kind, wpos)).collect()
    };
    let row_of = |v: u32| &lap[ridx.get(v).unwrap() as usize];
    // Row v of L as (column, value), diagonal first.
    let l_row = |v: u32| std::iter::once((v, -row_of(v).diag)).chain(row_of(v).nbrs.iter().copied());
    let build = |&i: &u32| -> (Vec<(u32, f64)>, DVec3) {
        // r = e_iᵀ L M⁻¹ L M⁻¹ L, accumulated sparsely.
        let mut acc: FastMap<u32, f64> = FastMap::with_capacity_and_hasher(64, Default::default());
        for (j, lij) in l_row(i) {
            let cj = lij / row_of(j).mass;
            for (l, ljl) in l_row(j) {
                let cl = cj * ljl / row_of(l).mass;
                for (k, llk) in l_row(l) {
                    *acc.entry(k).or_insert(0.0) += cl * llk;
                }
            }
        }
        // The right-hand side is summed in vertex order, so the result does not depend on the
        // map's layout (it used to vary from run to run with the random hash seed).
        let mut terms: Vec<(u32, f64)> = acc.into_iter().collect();
        terms.sort_unstable_by_key(|t| t.0);
        let mut ent = Vec::with_capacity(terms.len());
        let mut rhs = DVec3::ZERO;
        for (k, val) in terms {
            // Negated: L is negative semi-definite, so −L M⁻¹ L M⁻¹ L is positive.
            match fidx.get(k) {
                Some(l) => ent.push((l, -val)),
                None => rhs += vpos(k) * val,
            }
        }
        (ent, rhs)
    };
    let rows: Vec<(Vec<(u32, f64)>, DVec3)> = if nf >= 2048 { free.par_iter().map(build).collect() } else { free.iter().map(build).collect() };
    let (ent, rhs): (Vec<_>, Vec<_>) = rows.into_iter().unzip();
    FairSystem { matrix: CsrMatrix::from_rows(nf, nf, ent), rhs, free }
}

/// Systems at least this large are preconditioned with smoothed-aggregation AMG.
const AMG_MIN_UNKNOWNS: usize = 1500;

/// Low-energy vectors of the fairing operator on `free` (at positions `vpos`): constants
/// (order 1), plus the coordinate functions, whose thin-plate energy vanishes (order 2).
fn near_null_space<V: Fn(u32) -> DVec3>(vpos: &V, free: &[u32], order: u32) -> Vec<Vec<f64>> {
    let n = free.len();
    let mut out = vec![vec![1.0; n]];
    if order >= 2 && n > 0 {
        let c = free.iter().map(|&v| vpos(v)).sum::<DVec3>() / n as f64;
        let r = free.iter().map(|&v| vpos(v).distance(c)).fold(0.0, f64::max).max(1e-300);
        for axis in 0..3 {
            out.push(free.iter().map(|&v| (vpos(v) - c)[axis] / r).collect());
        }
    }
    out
}

/// The AMG hierarchy [`solve_fair`] uses for a system on `free` (at positions `vpos`), if the
/// system is large enough to want one and the matrix coarsens.
fn fair_amg<V: Fn(u32) -> DVec3>(vpos: &V, free: &[u32], order: u32, a: &CsrMatrix) -> Option<Amg> {
    if free.len() >= AMG_MIN_UNKNOWNS {
        Amg::new(a, &near_null_space(vpos, free, order), &AmgParams::default())
    } else {
        None
    }
}

/// PCG with AMG for large systems, Jacobi otherwise.
#[allow(clippy::too_many_arguments)]
fn solve_fair<T: Element, V: Fn(u32) -> DVec3>(
    vpos: &V,
    free: &[u32],
    order: u32,
    a: &CsrMatrix,
    b: &[T],
    x: &mut [T],
    pcg: &PcgParams,
    callback: Option<&dyn Fn(usize, f64) -> bool>,
) -> PcgStats {
    if let Some(amg) = fair_amg(vpos, free, order, a) {
        return solve_pcg_preconditioned(a, b, x, pcg, &amg.preconditioner(1), callback);
    }
    solve_pcg_with_callback(a, b, x, pcg, callback)
}

/// The checks [`fair_vertices`] makes on a solve before using it.
fn check_solution(stats: &PcgStats, x: &[DVec3]) -> Result<()> {
    if stats.breakdown {
        return Err(RepairError::Solver("fairing matrix is not positive definite".into()));
    }
    if x.iter().any(|p| !p.is_finite()) {
        return Err(RepairError::Solver("non-finite solution".into()));
    }
    Ok(())
}

fn check_order(order: u32) -> Result<()> {
    if order == 0 || order > 3 {
        return Err(RepairError::InvalidParameter(format!("fairing order must be 1, 2 or 3, got {order}")));
    }
    Ok(())
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
    let m: &Mesh = mesh;
    let sol = fair_positions_with(m, free, order, kind, &|v| m.vertex(v), pcg, callback)?;
    if sol.free.is_empty() {
        return Ok(sol.stats);
    }
    let pos = mesh.positions_mut();
    for (&v, p) in sol.free.iter().zip(sol.positions) {
        pos[v as usize] = p;
    }
    mesh.bump_shape_version();
    Ok(sol.stats)
}

/// Result of [`fair_positions`].
#[derive(Clone, Debug, Default)]
pub struct FairSolution {
    /// The vertices solved for: the free vertices given, in order, less those whose connected
    /// free region has no fixed neighbour.
    pub free: Vec<u32>,
    /// Their new positions.
    pub positions: Vec<DVec3>,
    /// Solver statistics.
    pub stats: PcgStats,
}

/// Fair `free` (order 1, 2 or 3, every other vertex fixed) on positions `pos` instead of the
/// mesh's own, which gives only the connectivity: what [`crate::fair_region`] would compute
/// (with its default solver settings replaced by `pcg`) on a copy of the mesh whose vertex `v`
/// sits at `pos(v)`, without making the copy or writing anything. Every position the solve
/// reads (weights, masses, the fixed values, the starting guess and the near-null space of
/// the multigrid preconditioner) comes from `pos`.
pub fn fair_positions<P: Fn(u32) -> DVec3 + Sync>(mesh: &Mesh, free: &[u32], order: u32, kind: LaplacianWeights, pos: &P, pcg: &PcgParams) -> Result<FairSolution> {
    fair_positions_with(mesh, free, order, kind, pos, pcg, None)
}

/// [`fair_positions`] with a solver callback (see [`crate::sparse::solve_pcg_with_callback`]).
fn fair_positions_with<P: Fn(u32) -> DVec3 + Sync>(
    mesh: &Mesh,
    free: &[u32],
    order: u32,
    kind: LaplacianWeights,
    pos: &P,
    pcg: &PcgParams,
    callback: Option<&dyn Fn(usize, f64) -> bool>,
) -> Result<FairSolution> {
    check_order(order)?;
    let sys = build_fair_system_at(mesh, free, order, kind, pos, pos);
    if sys.free.is_empty() {
        return Ok(FairSolution { stats: PcgStats { converged: true, ..Default::default() }, ..Default::default() });
    }
    let mut x: Vec<DVec3> = sys.free.iter().map(|&v| pos(v)).collect();
    let stats = solve_fair(pos, &sys.free, order, &sys.matrix, &sys.rhs, &mut x, pcg, callback);
    check_solution(&stats, &x)?;
    Ok(FairSolution { free: sys.free, positions: x, stats })
}

/// An order 1 or 2 fairing system assembled once (matrix, preconditioner) and solved for
/// many sets of fixed values, e.g. a displacement field re-solved on every move of a drag:
/// each [`FairSolver::solve`] only gathers the right-hand side and runs the solver.
///
/// The weights and masses come from the positions given to [`FairSolver::new`]. A solve
/// for values `values` gives exactly what [`fair_positions`] gives for positions `values`
/// whenever the system does not depend on them (uniform weights) and `values` agrees with the
/// construction positions on the free vertices, which set the multigrid's near-null space.
pub struct FairSolver {
    assembly: FairAssembly,
    order: u32,
    amg: Option<Amg>,
    jacobi: Jacobi,
}

impl FairSolver {
    /// Assemble the system for `free` (other vertices fixed) on positions `pos`. Orders 1
    /// and 2 only.
    pub fn new<P: Fn(u32) -> DVec3 + Sync>(mesh: &Mesh, free: &[u32], order: u32, kind: LaplacianWeights, pos: &P) -> Result<Self> {
        if order == 0 || order > 2 {
            return Err(RepairError::InvalidParameter(format!("a cached fairing system must be of order 1 or 2, got {order}")));
        }
        let assembly = assemble(mesh, free, order, kind, pos);
        let amg = fair_amg(pos, &assembly.free, order, &assembly.matrix);
        let jacobi = Jacobi::new(&assembly.matrix);
        Ok(Self { assembly, order, amg, jacobi })
    }

    /// The vertices solved for (see [`FairSolution::free`]).
    pub fn free(&self) -> &[u32] {
        &self.assembly.free
    }

    /// Fairing order.
    pub fn order(&self) -> u32 {
        self.order
    }

    /// New positions of [`FairSolver::free`], with every other vertex at `values` (which
    /// also gives the free vertices' starting guess).
    pub fn solve<P: Fn(u32) -> DVec3 + Sync>(&self, values: &P, pcg: &PcgParams) -> Result<(Vec<DVec3>, PcgStats)> {
        let free = &self.assembly.free;
        if free.is_empty() {
            return Ok((Vec::new(), PcgStats { converged: true, ..Default::default() }));
        }
        let rhs = self.assembly.rhs(values);
        let mut x: Vec<DVec3> = free.iter().map(|&v| values(v)).collect();
        let a = &self.assembly.matrix;
        let stats = match &self.amg {
            Some(amg) => solve_pcg_preconditioned(a, &rhs, &mut x, pcg, &amg.preconditioner(1), None),
            None => solve_pcg_preconditioned(a, &rhs, &mut x, pcg, &self.jacobi, None),
        };
        check_solution(&stats, &x)?;
        Ok((x, stats))
    }
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
    let stats = solve_fair(&|v| mesh.vertex(v), &sys.free, order, &k, &rhs, &mut s, pcg, None);
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
            for order in [1, 2, 3] {
                let s = build_fair_system(&m, &free, order, kind);
                assert_eq!(s.free.len(), free.len());
                assert!(s.matrix.is_symmetric(1e-9), "{kind:?} order {order}");
            }
        }
    }

    #[test]
    fn order3_reproduces_a_quartic_height_field() {
        // Uniform Δ³ of x⁴ vanishes on a regular grid (Δ² of it does not): order 3 restores the
        // field from a disturbed start, order 2 cannot.
        let field = |p: DVec3| 0.02 * p.x.powi(4);
        let mut m = primitives::plane_grid(16.0, 16.0, 16, 16);
        let ids: Vec<u32> = m.vertex_ids().collect();
        for &v in &ids {
            let p = m.vertex(v);
            m.set_vertex(v, DVec3::new(p.x, p.y, field(p)));
        }
        let free: Vec<u32> = ids.iter().copied().filter(|&v| m.vertex(v).x.abs() < 3.5 && m.vertex(v).y.abs() < 3.5).collect();
        let err = |order: u32| {
            let mut m = m.clone();
            for &v in &free {
                let p = m.vertex(v);
                m.set_vertex(v, p + DVec3::new(0.0, 0.0, 0.3));
            }
            fair_vertices(&mut m, &free, order, LaplacianWeights::Uniform, &PcgParams { tolerance: 1e-12, max_iterations: 20_000 }, None).unwrap();
            free.iter().map(|&v| (m.vertex(v).z - field(m.vertex(v))).abs()).fold(0.0, f64::max)
        };
        assert!(err(3) < 1e-6, "order 3 restores x⁴ ({})", err(3));
        assert!(err(2) > 1e-3, "order 2 does not ({})", err(2));
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

    fn bits(p: DVec3) -> [u64; 3] {
        p.to_array().map(f64::to_bits)
    }

    /// Positions to fair from: the mesh's own, bumped.
    fn bumped(m: &Mesh) -> Vec<DVec3> {
        (0..m.max_vid()).map(|v| m.vertex(v as u32) * (1.0 + 0.05 * (v as f64 * 0.37).sin())).collect()
    }

    #[test]
    fn fair_positions_match_fairing_a_copy() {
        // Below and above the AMG threshold (1500 unknowns).
        for (m, zcut) in [(primitives::icosphere(1.0, 3), 0.4), (primitives::icosphere(1.0, 5), 0.2)] {
            let free: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).z > zcut).collect();
            let p = bumped(&m);
            let at = |v: u32| p[v as usize];
            let mut copy = m.clone();
            for v in copy.vertex_ids().collect::<Vec<_>>() {
                copy.set_vertex(v, p[v as usize]);
            }
            for kind in [LaplacianWeights::Uniform, LaplacianWeights::Cotangent, LaplacianWeights::MeanValue] {
                for order in [1, 2, 3] {
                    let pcg = PcgParams { tolerance: 1e-9, max_iterations: 5000 };
                    let mut c = copy.clone();
                    let st = fair_vertices(&mut c, &free, order, kind, &pcg, None).unwrap();
                    let sol = fair_positions(&m, &free, order, kind, &at, &pcg).unwrap();
                    assert_eq!(sol.free, free, "every free vertex has a fixed neighbour");
                    assert_eq!(sol.stats.iterations, st.iterations, "{kind:?} order {order}");
                    for (&v, x) in sol.free.iter().zip(&sol.positions) {
                        assert_eq!(bits(c.vertex(v)), bits(*x), "{kind:?} order {order}");
                    }
                }
            }
        }
    }

    #[test]
    fn a_cached_system_solves_like_a_fresh_one() {
        // A displacement field (zero but on a few fixed vertices) re-solved for several moves,
        // as Stretchy soft transforms and warps do.
        for (m, zcut) in [(primitives::icosphere(1.0, 3), 0.4), (primitives::icosphere(1.0, 5), 0.2)] {
            let free: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).z > zcut).collect();
            let ring: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).z <= zcut && m.vertex(v).z > zcut - 0.15).collect();
            for order in [1, 2] {
                let solver = FairSolver::new(&m, &free, order, LaplacianWeights::Uniform, &|_| DVec3::ZERO).unwrap();
                assert_eq!(solver.free(), &free[..]);
                for k in 0..3 {
                    let lift = DVec3::new(0.1 * k as f64, -0.05, 0.3 + 0.1 * k as f64);
                    let values = |v: u32| if ring.binary_search(&v).is_ok() { lift * (1.0 + m.vertex(v).x) } else { DVec3::ZERO };
                    let pcg = PcgParams { tolerance: 1e-7, max_iterations: 10_000 };
                    let (x, st) = solver.solve(&values, &pcg).unwrap();
                    let fresh = fair_positions(&m, &free, order, LaplacianWeights::Uniform, &values, &pcg).unwrap();
                    assert_eq!(st.iterations, fresh.stats.iterations);
                    assert!(x.iter().zip(&fresh.positions).all(|(a, b)| bits(*a) == bits(*b)), "order {order}, move {k}");
                    assert!(x.iter().any(|p| p.length() > 1e-3), "the field is not trivial");
                }
            }
        }
        assert!(FairSolver::new(&primitives::icosphere(1.0, 1), &[0], 3, LaplacianWeights::Uniform, &|_| DVec3::ZERO).is_err());
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
