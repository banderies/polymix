//! Sparse linear algebra: compressed-sparse-row matrices and preconditioned conjugate
//! gradients for symmetric positive-definite systems, with a diagonal ([`Jacobi`]) or a
//! smoothed-aggregation algebraic multigrid ([`Amg`]) preconditioner. Jacobi needs O(n)
//! iterations on thin-plate (bilaplacian) systems; AMG keeps it at tens of iterations.
//!
//! The solver is generic over the vector [`Element`], so the three coordinate systems of a
//! mesh-fairing problem can be solved in a single pass with `DVec3` elements: each
//! component runs an independent CG recurrence with its own step sizes, but they share the
//! matrix traversal.

use glam::DVec3;
use rayon::prelude::*;
use std::ops::{Add, AddAssign, Mul, Sub, SubAssign};

/// Row count above which matrix/vector kernels run in parallel.
const PAR_THRESHOLD: usize = 4096;

/// Vector element usable by [`solve_pcg`]: `f64` or `DVec3` (component-wise semantics).
pub trait Element:
    Copy
    + Send
    + Sync
    + Default
    + PartialEq
    + std::fmt::Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<f64, Output = Self>
    + AddAssign
    + SubAssign
{
    /// Additive identity.
    fn zero() -> Self;
    /// All components equal to `v`.
    fn splat(v: f64) -> Self;
    /// Component-wise product.
    fn cmul(self, o: Self) -> Self;
    /// Component-wise quotient; components whose denominator is zero or non-finite give 0.
    fn cdiv(self, o: Self) -> Self;
    /// Component-wise square root of non-negative components (negative → 0).
    fn csqrt(self) -> Self;
    /// Component-wise `if self > 0 { self } else { alt }`.
    fn or_if_zero(self, alt: Self) -> Self;
    /// True if every component of `self` is `<=` the matching component of `o`.
    fn all_le(self, o: Self) -> bool;
    /// True if any component is negative.
    fn any_negative(self) -> bool;
    /// True if every component is finite.
    fn is_finite_elem(self) -> bool;
    /// Largest component.
    fn max_component(self) -> f64;
}

impl Element for f64 {
    #[inline]
    fn zero() -> Self {
        0.0
    }
    #[inline]
    fn splat(v: f64) -> Self {
        v
    }
    #[inline]
    fn cmul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn cdiv(self, o: Self) -> Self {
        let r = self / o;
        if o == 0.0 || !r.is_finite() {
            0.0
        } else {
            r
        }
    }
    #[inline]
    fn csqrt(self) -> Self {
        self.max(0.0).sqrt()
    }
    #[inline]
    fn or_if_zero(self, alt: Self) -> Self {
        if self > 0.0 {
            self
        } else {
            alt
        }
    }
    #[inline]
    fn all_le(self, o: Self) -> bool {
        self <= o
    }
    #[inline]
    fn any_negative(self) -> bool {
        self < 0.0
    }
    #[inline]
    fn is_finite_elem(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn max_component(self) -> f64 {
        self
    }
}

impl Element for DVec3 {
    #[inline]
    fn zero() -> Self {
        DVec3::ZERO
    }
    #[inline]
    fn splat(v: f64) -> Self {
        DVec3::splat(v)
    }
    #[inline]
    fn cmul(self, o: Self) -> Self {
        self * o
    }
    #[inline]
    fn cdiv(self, o: Self) -> Self {
        DVec3::new(self.x.cdiv(o.x), self.y.cdiv(o.y), self.z.cdiv(o.z))
    }
    #[inline]
    fn csqrt(self) -> Self {
        DVec3::new(self.x.csqrt(), self.y.csqrt(), self.z.csqrt())
    }
    #[inline]
    fn or_if_zero(self, alt: Self) -> Self {
        DVec3::new(self.x.or_if_zero(alt.x), self.y.or_if_zero(alt.y), self.z.or_if_zero(alt.z))
    }
    #[inline]
    fn all_le(self, o: Self) -> bool {
        self.cmple(o).all()
    }
    #[inline]
    fn any_negative(self) -> bool {
        self.cmplt(DVec3::ZERO).any()
    }
    #[inline]
    fn is_finite_elem(self) -> bool {
        self.is_finite()
    }
    #[inline]
    fn max_component(self) -> f64 {
        self.max_element()
    }
}

/// Sparse matrix in compressed-sparse-row form. Column indices within a row are sorted
/// and unique.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CsrMatrix {
    nrows: usize,
    ncols: usize,
    row_ptr: Vec<usize>,
    col_idx: Vec<u32>,
    values: Vec<f64>,
}

/// Coordinate-format accumulator; duplicate entries are summed by [`TripletMatrix::to_csr`].
#[derive(Clone, Debug, Default)]
pub struct TripletMatrix {
    /// Row count.
    pub nrows: usize,
    /// Column count.
    pub ncols: usize,
    /// (row, column, value) entries.
    pub entries: Vec<(u32, u32, f64)>,
}

impl TripletMatrix {
    /// Empty `nrows × ncols` matrix.
    pub fn new(nrows: usize, ncols: usize) -> Self {
        Self { nrows, ncols, entries: Vec::new() }
    }
    /// Add `v` to entry (r, c).
    #[inline]
    pub fn push(&mut self, r: usize, c: usize, v: f64) {
        debug_assert!(r < self.nrows && c < self.ncols);
        self.entries.push((r as u32, c as u32, v));
    }
    /// Convert to CSR, summing duplicates.
    pub fn to_csr(&self) -> CsrMatrix {
        CsrMatrix::from_triplets(self.nrows, self.ncols, &self.entries)
    }
}

impl CsrMatrix {
    /// Build from (row, col, value) triplets, summing duplicates.
    pub fn from_triplets(nrows: usize, ncols: usize, entries: &[(u32, u32, f64)]) -> Self {
        let mut counts = vec![0usize; nrows + 1];
        for &(r, c, _) in entries {
            assert!((r as usize) < nrows && (c as usize) < ncols, "triplet out of range");
            counts[r as usize + 1] += 1;
        }
        for i in 0..nrows {
            counts[i + 1] += counts[i];
        }
        let mut fill = counts.clone();
        let mut cols = vec![0u32; entries.len()];
        let mut vals = vec![0f64; entries.len()];
        for &(r, c, v) in entries {
            let k = &mut fill[r as usize];
            cols[*k] = c;
            vals[*k] = v;
            *k += 1;
        }
        let mut rows: Vec<Vec<(u32, f64)>> = Vec::with_capacity(nrows);
        for r in 0..nrows {
            rows.push((counts[r]..counts[r + 1]).map(|k| (cols[k], vals[k])).collect());
        }
        Self::from_rows(nrows, ncols, rows)
    }

    /// Build from per-row entry lists (any order, duplicates summed).
    pub fn from_rows(nrows: usize, ncols: usize, mut rows: Vec<Vec<(u32, f64)>>) -> Self {
        assert_eq!(rows.len(), nrows);
        let merge = |row: &mut Vec<(u32, f64)>| {
            row.sort_unstable_by_key(|e| e.0);
            let mut w = 0usize;
            for r in 0..row.len() {
                if w > 0 && row[w - 1].0 == row[r].0 {
                    row[w - 1].1 += row[r].1;
                } else {
                    row[w] = row[r];
                    w += 1;
                }
            }
            row.truncate(w);
        };
        if nrows >= PAR_THRESHOLD {
            rows.par_iter_mut().for_each(merge);
        } else {
            rows.iter_mut().for_each(merge);
        }
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        row_ptr.push(0);
        let nnz: usize = rows.iter().map(|r| r.len()).sum();
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for row in rows {
            for (c, v) in row {
                assert!((c as usize) < ncols, "column out of range");
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    /// `n × n` identity.
    pub fn identity(n: usize) -> Self {
        Self { nrows: n, ncols: n, row_ptr: (0..=n).collect(), col_idx: (0..n as u32).collect(), values: vec![1.0; n] }
    }

    /// Row count.
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    /// Column count.
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    /// Number of stored entries.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    /// Column indices and values of row `r`.
    pub fn row(&self, r: usize) -> (&[u32], &[f64]) {
        let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
        (&self.col_idx[s..e], &self.values[s..e])
    }
    /// Entry (r, c), zero if not stored.
    pub fn get(&self, r: usize, c: usize) -> f64 {
        let (cols, vals) = self.row(r);
        match cols.binary_search(&(c as u32)) {
            Ok(k) => vals[k],
            Err(_) => 0.0,
        }
    }
    /// Diagonal entries.
    pub fn diagonal(&self) -> Vec<f64> {
        (0..self.nrows.min(self.ncols)).map(|r| self.get(r, r)).collect()
    }

    /// `y = A x`.
    pub fn mul_vec<T: Element>(&self, x: &[T], y: &mut [T]) {
        assert_eq!(x.len(), self.ncols);
        assert_eq!(y.len(), self.nrows);
        let f = |(r, yr): (usize, &mut T)| {
            let (s, e) = (self.row_ptr[r], self.row_ptr[r + 1]);
            let mut acc = T::zero();
            for k in s..e {
                acc += x[self.col_idx[k] as usize] * self.values[k];
            }
            *yr = acc;
        };
        if self.nrows >= PAR_THRESHOLD {
            y.par_iter_mut().enumerate().for_each(f);
        } else {
            y.iter_mut().enumerate().for_each(f);
        }
    }

    /// Transposed matrix.
    pub fn transpose(&self) -> Self {
        let mut counts = vec![0usize; self.ncols + 1];
        for &c in &self.col_idx {
            counts[c as usize + 1] += 1;
        }
        for i in 0..self.ncols {
            counts[i + 1] += counts[i];
        }
        let row_ptr = counts.clone();
        let mut fill = counts;
        let mut col_idx = vec![0u32; self.nnz()];
        let mut values = vec![0f64; self.nnz()];
        for r in 0..self.nrows {
            let (cols, vals) = self.row(r);
            for (&c, &v) in cols.iter().zip(vals) {
                let k = &mut fill[c as usize];
                col_idx[*k] = r as u32;
                values[*k] = v;
                *k += 1;
            }
        }
        Self { nrows: self.ncols, ncols: self.nrows, row_ptr, col_idx, values }
    }

    /// Sparse product `A * B` (row-wise Gustavson with a dense accumulator per thread).
    pub fn mul_mat(&self, b: &CsrMatrix) -> CsrMatrix {
        assert_eq!(self.ncols, b.nrows);
        let nc = b.ncols;
        let row_of = |acc: &mut (Vec<f64>, Vec<u32>, Vec<bool>), r: usize| -> Vec<(u32, f64)> {
            let (dense, touched, mark) = acc;
            let (ca, va) = self.row(r);
            for (&k, &a) in ca.iter().zip(va) {
                let (cb, vb) = b.row(k as usize);
                for (&c, &bv) in cb.iter().zip(vb) {
                    if !mark[c as usize] {
                        mark[c as usize] = true;
                        touched.push(c);
                    }
                    dense[c as usize] += a * bv;
                }
            }
            touched.sort_unstable();
            let out: Vec<(u32, f64)> = touched.iter().map(|&c| (c, dense[c as usize])).collect();
            for &c in touched.iter() {
                dense[c as usize] = 0.0;
                mark[c as usize] = false;
            }
            touched.clear();
            out
        };
        let init = || (vec![0.0f64; nc], Vec::<u32>::new(), vec![false; nc]);
        let rows: Vec<Vec<(u32, f64)>> = if self.nrows >= PAR_THRESHOLD {
            (0..self.nrows).into_par_iter().map_init(init, row_of).collect()
        } else {
            let mut acc = init();
            (0..self.nrows).map(|r| row_of(&mut acc, r)).collect()
        };
        Self::from_sorted_rows(self.nrows, nc, rows)
    }

    /// Build from rows whose entries are already sorted by column and unique.
    fn from_sorted_rows(nrows: usize, ncols: usize, rows: Vec<Vec<(u32, f64)>>) -> Self {
        let nnz: usize = rows.iter().map(|r| r.len()).sum();
        let mut row_ptr = Vec::with_capacity(nrows + 1);
        row_ptr.push(0);
        let mut col_idx = Vec::with_capacity(nnz);
        let mut values = Vec::with_capacity(nnz);
        for row in rows {
            for (c, v) in row {
                col_idx.push(c);
                values.push(v);
            }
            row_ptr.push(col_idx.len());
        }
        Self { nrows, ncols, row_ptr, col_idx, values }
    }

    /// Multiply row `r` by `s[r]`.
    pub fn scale_rows(&mut self, s: &[f64]) {
        assert_eq!(s.len(), self.nrows);
        for (r, &f) in s.iter().enumerate() {
            for v in &mut self.values[self.row_ptr[r]..self.row_ptr[r + 1]] {
                *v *= f;
            }
        }
    }

    /// True if `|A - Aᵀ|` is below `tol` (relative to the largest entry).
    pub fn is_symmetric(&self, tol: f64) -> bool {
        if self.nrows != self.ncols {
            return false;
        }
        let maxv = self.values.iter().fold(0.0f64, |m, v| m.max(v.abs())).max(1e-300);
        for r in 0..self.nrows {
            let (cols, vals) = self.row(r);
            for (&c, &v) in cols.iter().zip(vals) {
                if (self.get(c as usize, r) - v).abs() > tol * maxv {
                    return false;
                }
            }
        }
        true
    }
}

/// Parameters of [`solve_pcg`].
#[derive(Clone, Copy, Debug)]
pub struct PcgParams {
    /// Stop when `‖r‖ ≤ tolerance · ‖b‖` (per component; `‖r₀‖` is used when `b = 0`).
    pub tolerance: f64,
    /// Iteration limit.
    pub max_iterations: usize,
}

impl Default for PcgParams {
    fn default() -> Self {
        Self { tolerance: 1e-8, max_iterations: 5000 }
    }
}

/// Outcome of [`solve_pcg`].
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct PcgStats {
    /// Iterations performed.
    pub iterations: usize,
    /// Largest per-component relative residual at exit.
    pub relative_residual: f64,
    /// The tolerance was reached.
    pub converged: bool,
    /// The callback requested a stop (the current iterate is still returned in `x`).
    pub cancelled: bool,
    /// The matrix was found not to be positive definite (`pᵀAp ≤ 0`).
    pub breakdown: bool,
}

#[inline]
fn dot<T: Element>(a: &[T], b: &[T]) -> T {
    if a.len() >= PAR_THRESHOLD {
        a.par_iter().zip(b.par_iter()).map(|(x, y)| x.cmul(*y)).reduce(T::zero, |p, q| p + q)
    } else {
        let mut s = T::zero();
        for (x, y) in a.iter().zip(b) {
            s += x.cmul(*y);
        }
        s
    }
}

/// A symmetric positive-definite preconditioner: `z ≈ A⁻¹ r`.
pub trait Preconditioner<T: Element>: Sync {
    /// Apply the preconditioner to `r`, writing the result into `z`.
    fn apply(&self, r: &[T], z: &mut [T]);
}

/// Diagonal (Jacobi) preconditioner.
#[derive(Clone, Debug)]
pub struct Jacobi {
    inv_diag: Vec<f64>,
}

impl Jacobi {
    /// Inverse of the diagonal of `a` (entries that are not positive and finite use 1).
    pub fn new(a: &CsrMatrix) -> Self {
        Self { inv_diag: a.diagonal().iter().map(|&d| if d > 0.0 && d.is_finite() { 1.0 / d } else { 1.0 }).collect() }
    }
}

impl<T: Element> Preconditioner<T> for Jacobi {
    fn apply(&self, r: &[T], z: &mut [T]) {
        if r.len() >= PAR_THRESHOLD {
            z.par_iter_mut().zip(r.par_iter().zip(self.inv_diag.par_iter())).for_each(|(zi, (ri, d))| *zi = *ri * *d);
        } else {
            for ((zi, ri), d) in z.iter_mut().zip(r).zip(&self.inv_diag) {
                *zi = *ri * *d;
            }
        }
    }
}

/// Solve `A x = b` for symmetric positive-definite `A` by Jacobi-preconditioned conjugate
/// gradients, starting from the initial guess in `x`.
pub fn solve_pcg<T: Element>(a: &CsrMatrix, b: &[T], x: &mut [T], params: &PcgParams) -> PcgStats {
    solve_pcg_with_callback(a, b, x, params, None)
}

/// [`solve_pcg`] with a callback `(iteration, relative_residual) -> continue?` invoked on the
/// calling thread every few iterations.
pub fn solve_pcg_with_callback<T: Element>(
    a: &CsrMatrix,
    b: &[T],
    x: &mut [T],
    params: &PcgParams,
    callback: Option<&dyn Fn(usize, f64) -> bool>,
) -> PcgStats {
    solve_pcg_preconditioned(a, b, x, params, &Jacobi::new(a), callback)
}

/// Preconditioned conjugate gradients with any symmetric positive-definite preconditioner
/// (e.g. [`Jacobi`] or [`Amg`]), starting from the initial guess in `x`.
pub fn solve_pcg_preconditioned<T: Element>(
    a: &CsrMatrix,
    b: &[T],
    x: &mut [T],
    params: &PcgParams,
    precond: &dyn Preconditioner<T>,
    callback: Option<&dyn Fn(usize, f64) -> bool>,
) -> PcgStats {
    let n = a.nrows();
    assert_eq!(a.ncols(), n, "PCG needs a square matrix");
    assert_eq!(b.len(), n);
    assert_eq!(x.len(), n);
    let mut stats = PcgStats::default();
    if n == 0 {
        stats.converged = true;
        return stats;
    }
    let par = n >= PAR_THRESHOLD;
    let mut r = vec![T::zero(); n];
    a.mul_vec(x, &mut r);
    for (ri, bi) in r.iter_mut().zip(b) {
        *ri = *bi - *ri;
    }
    let mut z = vec![T::zero(); n];
    precond.apply(&r, &mut z);
    let mut p = z.clone();
    let mut rz = dot(&r, &z);
    let bnorm = dot(b, b).csqrt();
    let mut rnorm = dot(&r, &r).csqrt();
    let reference = bnorm.or_if_zero(rnorm);
    let target = reference * params.tolerance;
    let mut q = vec![T::zero(); n];
    loop {
        stats.relative_residual = rnorm.cdiv(reference).max_component();
        if rnorm.all_le(target) {
            stats.converged = true;
            break;
        }
        if stats.iterations >= params.max_iterations {
            break;
        }
        if let Some(cb) = callback {
            if stats.iterations % 16 == 0 && !cb(stats.iterations, stats.relative_residual) {
                stats.cancelled = true;
                break;
            }
        }
        a.mul_vec(&p, &mut q);
        let pq = dot(&p, &q);
        if pq.any_negative() || !pq.is_finite_elem() {
            stats.breakdown = true;
            break;
        }
        let alpha = rz.cdiv(pq);
        let update = |((xi, ri), (pi, qi)): ((&mut T, &mut T), (&T, &T))| -> T {
            *xi += alpha.cmul(*pi);
            *ri -= alpha.cmul(*qi);
            ri.cmul(*ri)
        };
        let rr = if par {
            x.par_iter_mut().zip(r.par_iter_mut()).zip(p.par_iter().zip(q.par_iter())).map(update).reduce(T::zero, |u, v| u + v)
        } else {
            x.iter_mut().zip(r.iter_mut()).zip(p.iter().zip(q.iter())).map(update).fold(T::zero(), |u, v| u + v)
        };
        precond.apply(&r, &mut z);
        let rz_new = dot(&r, &z);
        let beta = rz_new.cdiv(rz);
        if par {
            p.par_iter_mut().zip(z.par_iter()).for_each(|(pi, zi)| *pi = *zi + beta.cmul(*pi));
        } else {
            for (pi, zi) in p.iter_mut().zip(z.iter()) {
                *pi = *zi + beta.cmul(*pi);
            }
        }
        rz = rz_new;
        rnorm = rr.csqrt();
        stats.iterations += 1;
    }
    stats
}

// ---------------------------------------------------------------------------------------
// Smoothed-aggregation algebraic multigrid
// ---------------------------------------------------------------------------------------

/// Parameters of [`Amg`].
#[derive(Clone, Copy, Debug)]
pub struct AmgParams {
    /// Off-diagonal `a_ij` is a strong connection if `|a_ij| ≥ θ √(a_ii a_jj)`.
    pub strength_threshold: f64,
    /// Levels are added until the coarsest has at most this many unknowns (solved densely).
    pub max_coarse: usize,
    /// Maximum number of levels.
    pub max_levels: usize,
    /// Damped-Jacobi pre- and post-smoothing sweeps per level.
    pub smoothing_sweeps: usize,
}

impl Default for AmgParams {
    fn default() -> Self {
        Self { strength_threshold: 0.02, max_coarse: 400, max_levels: 12, smoothing_sweeps: 2 }
    }
}

struct AmgLevel {
    a: CsrMatrix,
    p: CsrMatrix,
    r: CsrMatrix,
    /// Smoother weight times the inverse diagonal.
    w_inv_diag: Vec<f64>,
}

/// Smoothed-aggregation algebraic multigrid (Vaněk, Mandel & Brezina 1996) used as a
/// symmetric V-cycle preconditioner for [`solve_pcg_preconditioned`].
///
/// The caller supplies the near-null space (vectors of low energy, e.g. constants for
/// Laplace problems, constants and the coordinate functions for thin-plate problems); the
/// tentative prolongator interpolates them exactly on each aggregate, and is then smoothed
/// with one damped-Jacobi step. Coarse operators are Galerkin products `Pᵀ A P`.
pub struct Amg {
    levels: Vec<AmgLevel>,
    coarse: DenseCholesky,
}

impl Amg {
    /// Build the hierarchy. `near_null` holds `k` vectors of length `a.nrows()`. Returns
    /// `None` if the matrix does not coarsen (the caller should fall back to [`Jacobi`]).
    pub fn new(a: &CsrMatrix, near_null: &[Vec<f64>], params: &AmgParams) -> Option<Self> {
        let n = a.nrows();
        let k = near_null.len().max(1);
        // Row-major n × k near-null basis.
        let mut b: Vec<f64> = vec![1.0; n * k];
        if !near_null.is_empty() {
            for (c, v) in near_null.iter().enumerate() {
                assert_eq!(v.len(), n);
                for i in 0..n {
                    b[i * k + c] = v[i];
                }
            }
        }
        let mut levels = Vec::new();
        let mut cur = a.clone();
        while cur.nrows() > params.max_coarse && levels.len() + 1 < params.max_levels {
            let inv_diag: Vec<f64> = cur.diagonal().iter().map(|&d| if d > 0.0 && d.is_finite() { 1.0 / d } else { 0.0 }).collect();
            let lambda = estimate_lambda_max(&cur, &inv_diag);
            let omega = 4.0 / (3.0 * lambda);
            let (agg, n_agg) = aggregate(&cur, params.strength_threshold);
            if n_agg == 0 {
                break;
            }
            let (ptent, bc) = tentative_prolongator(&agg, n_agg, &b, k);
            if ptent.ncols() == 0 || ptent.ncols() * 10 > cur.nrows() * 9 {
                break; // not coarsening
            }
            // P = (I − ω D⁻¹ A) P_tent
            let ap = cur.mul_mat(&ptent);
            let rows: Vec<Vec<(u32, f64)>> = (0..cur.nrows())
                .map(|i| {
                    let (tc, tv) = ptent.row(i);
                    let (ac, av) = ap.row(i);
                    let s = omega * inv_diag[i];
                    tc.iter().zip(tv).map(|(&c, &v)| (c, v)).chain(ac.iter().zip(av).map(|(&c, &v)| (c, -s * v))).collect()
                })
                .collect();
            let p = CsrMatrix::from_rows(cur.nrows(), ptent.ncols(), rows);
            let r = p.transpose();
            let ac = r.mul_mat(&cur.mul_mat(&p));
            let w_inv_diag = inv_diag.iter().map(|d| d * omega).collect();
            levels.push(AmgLevel { a: cur, p, r, w_inv_diag });
            cur = ac;
            b = bc;
        }
        if cur.nrows() > 4 * params.max_coarse.max(1000) {
            return None;
        }
        let coarse = DenseCholesky::new(&cur)?;
        Some(Self { levels, coarse })
    }

    /// Number of levels including the dense coarsest one.
    pub fn num_levels(&self) -> usize {
        self.levels.len() + 1
    }

    /// (rows, stored entries) of each sparse level's operator, then of the coarse level.
    pub fn level_sizes(&self) -> Vec<(usize, usize)> {
        let mut v: Vec<(usize, usize)> = self.levels.iter().map(|l| (l.a.nrows(), l.a.nnz())).collect();
        v.push((self.coarse.n, self.coarse.n * self.coarse.n));
        v
    }

    fn vcycle<T: Element>(&self, l: usize, b: &[T], x: &mut [T], sweeps: usize) {
        if l == self.levels.len() {
            self.coarse.solve(b, x);
            return;
        }
        let lv = &self.levels[l];
        let n = b.len();
        let mut tmp = vec![T::zero(); n];
        let smooth = |x: &mut [T], tmp: &mut [T]| {
            lv.a.mul_vec(x, tmp);
            for i in 0..n {
                x[i] += (b[i] - tmp[i]) * lv.w_inv_diag[i];
            }
        };
        for (xi, (bi, w)) in x.iter_mut().zip(b.iter().zip(&lv.w_inv_diag)) {
            *xi = *bi * *w;
        }
        for _ in 1..sweeps {
            smooth(x, &mut tmp);
        }
        lv.a.mul_vec(x, &mut tmp);
        for i in 0..n {
            tmp[i] = b[i] - tmp[i];
        }
        let nc = lv.p.ncols();
        let mut bc = vec![T::zero(); nc];
        lv.r.mul_vec(&tmp, &mut bc);
        let mut xc = vec![T::zero(); nc];
        self.vcycle(l + 1, &bc, &mut xc, sweeps);
        lv.p.mul_vec(&xc, &mut tmp);
        for i in 0..n {
            x[i] += tmp[i];
        }
        for _ in 0..sweeps {
            smooth(x, &mut tmp);
        }
    }
}

/// One V(ν,ν) cycle with damped-Jacobi smoothing (symmetric, so valid for CG).
pub struct AmgPreconditioner<'a> {
    amg: &'a Amg,
    sweeps: usize,
}

impl Amg {
    /// The V-cycle preconditioner with `sweeps` smoothing sweeps (≥ 1).
    pub fn preconditioner(&self, sweeps: usize) -> AmgPreconditioner<'_> {
        AmgPreconditioner { amg: self, sweeps: sweeps.max(1) }
    }
}

impl<T: Element> Preconditioner<T> for AmgPreconditioner<'_> {
    fn apply(&self, r: &[T], z: &mut [T]) {
        self.amg.vcycle(0, r, z, self.sweeps);
    }
}

/// Largest eigenvalue of `D⁻¹A` by power iteration (slightly overestimated).
fn estimate_lambda_max(a: &CsrMatrix, inv_diag: &[f64]) -> f64 {
    let n = a.nrows();
    let mut v: Vec<f64> = (0..n).map(|i| 1.0 + ((i as f64) * 0.618_033_988_7).fract()).collect();
    let mut w = vec![0.0; n];
    let mut lambda = 1.0;
    for _ in 0..15 {
        let nv = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if nv == 0.0 {
            break;
        }
        a.mul_vec(&v, &mut w);
        for i in 0..n {
            w[i] *= inv_diag[i];
        }
        lambda = w.iter().map(|x| x * x).sum::<f64>().sqrt() / nv;
        std::mem::swap(&mut v, &mut w);
    }
    (lambda * 1.1).max(1e-12)
}

/// Greedy aggregation over the strong-connection graph (standard three-phase SA scheme).
fn aggregate(a: &CsrMatrix, theta: f64) -> (Vec<u32>, usize) {
    const NONE: u32 = u32::MAX;
    let n = a.nrows();
    let diag = a.diagonal();
    let strong = |i: usize| -> Vec<u32> {
        let (cols, vals) = a.row(i);
        cols.iter()
            .zip(vals)
            .filter(|(&j, &v)| j as usize != i && v.abs() >= theta * (diag[i] * diag[j as usize]).abs().sqrt())
            .map(|(&j, _)| j)
            .collect()
    };
    let nbrs: Vec<Vec<u32>> = if n >= PAR_THRESHOLD { (0..n).into_par_iter().map(strong).collect() } else { (0..n).map(strong).collect() };
    let mut agg = vec![NONE; n];
    let mut na = 0u32;
    for i in 0..n {
        if agg[i] != NONE || nbrs[i].iter().any(|&j| agg[j as usize] != NONE) {
            continue;
        }
        agg[i] = na;
        for &j in &nbrs[i] {
            agg[j as usize] = na;
        }
        na += 1;
    }
    let phase1 = agg.clone();
    for i in 0..n {
        if agg[i] == NONE {
            if let Some(&j) = nbrs[i].iter().find(|&&j| phase1[j as usize] != NONE) {
                agg[i] = phase1[j as usize];
            }
        }
    }
    for i in 0..n {
        if agg[i] == NONE {
            agg[i] = na;
            for &j in &nbrs[i] {
                if agg[j as usize] == NONE {
                    agg[j as usize] = na;
                }
            }
            na += 1;
        }
    }
    (agg, na as usize)
}

/// Tentative prolongator: on each aggregate, an orthonormal basis (modified Gram-Schmidt,
/// rank-revealing) of the near-null vectors restricted to it. Returns `P_tent` and the
/// coarse near-null basis (row-major, `k` columns).
fn tentative_prolongator(agg: &[u32], n_agg: usize, b: &[f64], k: usize) -> (CsrMatrix, Vec<f64>) {
    let n = agg.len();
    let mut members: Vec<Vec<u32>> = vec![Vec::new(); n_agg];
    for (i, &g) in agg.iter().enumerate() {
        members[g as usize].push(i as u32);
    }
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut bc: Vec<f64> = Vec::new();
    let mut ncoarse = 0u32;
    for m in &members {
        if m.is_empty() {
            continue;
        }
        let len = m.len();
        // Columns of B restricted to the aggregate.
        let mut q: Vec<Vec<f64>> = (0..k).map(|c| m.iter().map(|&i| b[i as usize * k + c]).collect()).collect();
        let mut kept: Vec<usize> = Vec::new();
        let mut rmat = vec![vec![0.0; k]; k];
        for c in 0..k {
            let orig = q[c].iter().map(|x| x * x).sum::<f64>().sqrt();
            for &pk in &kept {
                // Kept columns always precede the current one.
                let (lo, hi) = q.split_at_mut(c);
                let (qp, qc) = (&lo[pk], &mut hi[0]);
                let d: f64 = qp.iter().zip(qc.iter()).map(|(a, b)| a * b).sum();
                rmat[pk][c] = d;
                for (x, p) in qc.iter_mut().zip(qp) {
                    *x -= d * p;
                }
            }
            let nrm = q[c].iter().map(|x| x * x).sum::<f64>().sqrt();
            if nrm > 1e-8 * orig.max(1e-300) && nrm > 0.0 && kept.len() < len {
                for x in q[c].iter_mut() {
                    *x /= nrm;
                }
                rmat[c][c] = nrm;
                kept.push(c);
            }
        }
        for (slot, &c) in kept.iter().enumerate() {
            let col = ncoarse + slot as u32;
            for (t, &i) in m.iter().enumerate() {
                rows[i as usize].push((col, q[c][t]));
            }
            bc.extend((0..k).map(|cc| if cc >= c { rmat[c][cc] } else { 0.0 }));
        }
        ncoarse += kept.len() as u32;
    }
    (CsrMatrix::from_rows(n, ncoarse as usize, rows), bc)
}

/// Dense Cholesky factorisation for the coarsest AMG level.
struct DenseCholesky {
    n: usize,
    l: Vec<f64>,
}

impl DenseCholesky {
    fn new(a: &CsrMatrix) -> Option<Self> {
        let n = a.nrows();
        let mut l = vec![0.0; n * n];
        for i in 0..n {
            let (cols, vals) = a.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                l[i * n + c as usize] = v;
            }
        }
        let maxd = (0..n).map(|i| l[i * n + i].abs()).fold(0.0, f64::max).max(1e-300);
        for j in 0..n {
            let mut d = l[j * n + j];
            for k in 0..j {
                d -= l[j * n + k] * l[j * n + k];
            }
            // Regularise (near-)singular pivots instead of failing.
            let d = if d > 1e-12 * maxd { d } else { 1e-12 * maxd };
            let djj = d.sqrt();
            l[j * n + j] = djj;
            for i in j + 1..n {
                let mut s = l[i * n + j];
                for k in 0..j {
                    s -= l[i * n + k] * l[j * n + k];
                }
                l[i * n + j] = s / djj;
            }
        }
        if l.iter().any(|x| !x.is_finite()) {
            return None;
        }
        Some(Self { n, l })
    }

    #[allow(clippy::needless_range_loop)]
    fn solve<T: Element>(&self, b: &[T], x: &mut [T]) {
        let n = self.n;
        for i in 0..n {
            let mut s = b[i];
            for k in 0..i {
                s -= x[k] * self.l[i * n + k];
            }
            x[i] = s * (1.0 / self.l[i * n + i]);
        }
        for i in (0..n).rev() {
            let mut s = x[i];
            for k in i + 1..n {
                s -= x[k] * self.l[k * n + i];
            }
            x[i] = s * (1.0 / self.l[i * n + i]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 1D Poisson matrix (tridiagonal 2, -1).
    fn poisson1d(n: usize) -> CsrMatrix {
        let mut t = TripletMatrix::new(n, n);
        for i in 0..n {
            t.push(i, i, 2.0);
            if i > 0 {
                t.push(i, i - 1, -1.0);
            }
            if i + 1 < n {
                t.push(i, i + 1, -1.0);
            }
        }
        t.to_csr()
    }

    #[test]
    fn triplets_sum_duplicates() {
        let m = CsrMatrix::from_triplets(2, 3, &[(0, 2, 1.0), (0, 0, 2.0), (0, 2, 3.0), (1, 1, -1.0)]);
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.get(0, 2), 4.0);
        assert_eq!(m.get(0, 0), 2.0);
        assert_eq!(m.get(1, 1), -1.0);
        assert_eq!(m.get(1, 0), 0.0);
        let t = m.transpose();
        assert_eq!(t.get(2, 0), 4.0);
        assert_eq!(t.nrows(), 3);
    }

    #[test]
    fn mat_mat_and_vec() {
        let a = poisson1d(5);
        let i = CsrMatrix::identity(5);
        assert_eq!(a.mul_mat(&i), a);
        let a2 = a.mul_mat(&a);
        assert!(a2.is_symmetric(1e-12));
        assert_eq!(a2.get(2, 2), 6.0);
        assert_eq!(a2.get(2, 0), 1.0);
        let x = vec![1.0; 5];
        let mut y = vec![0.0; 5];
        a.mul_vec(&x, &mut y);
        assert_eq!(y, vec![1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn pcg_solves_poisson() {
        for n in [10usize, 300, 5000] {
            let a = poisson1d(n);
            let xs: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin()).collect();
            let mut b = vec![0.0; n];
            a.mul_vec(&xs, &mut b);
            let mut x = vec![0.0; n];
            let st = solve_pcg(&a, &b, &mut x, &PcgParams { tolerance: 1e-12, max_iterations: 20 * n });
            assert!(st.converged, "{st:?}");
            let err = x.iter().zip(&xs).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-6, "n={n} err={err}");
        }
    }

    #[test]
    fn pcg_vec3_independent_components() {
        let n = 200;
        let a = poisson1d(n);
        let xs: Vec<DVec3> = (0..n).map(|i| DVec3::new(i as f64, (i as f64).sqrt(), 0.0)).collect();
        let mut b = vec![DVec3::ZERO; n];
        a.mul_vec(&xs, &mut b);
        let mut x = vec![DVec3::ZERO; n];
        let st = solve_pcg(&a, &b, &mut x, &PcgParams { tolerance: 1e-12, max_iterations: 10_000 });
        assert!(st.converged);
        for (p, q) in x.iter().zip(&xs) {
            assert!(p.distance(*q) < 1e-6);
        }
    }

    /// Dirichlet 5-point Laplacian on an n×n grid.
    fn poisson2d(n: usize) -> CsrMatrix {
        let idx = |i: usize, j: usize| i * n + j;
        let mut t = TripletMatrix::new(n * n, n * n);
        for i in 0..n {
            for j in 0..n {
                t.push(idx(i, j), idx(i, j), 4.0);
                if i > 0 {
                    t.push(idx(i, j), idx(i - 1, j), -1.0);
                }
                if i + 1 < n {
                    t.push(idx(i, j), idx(i + 1, j), -1.0);
                }
                if j > 0 {
                    t.push(idx(i, j), idx(i, j - 1), -1.0);
                }
                if j + 1 < n {
                    t.push(idx(i, j), idx(i, j + 1), -1.0);
                }
            }
        }
        t.to_csr()
    }

    #[test]
    fn amg_preconditions_laplace_and_bilaplace() {
        let n = 60;
        let l = poisson2d(n);
        let bl = l.mul_mat(&l);
        let coords: Vec<Vec<f64>> = vec![
            vec![1.0; n * n],
            (0..n * n).map(|k| (k / n) as f64 / n as f64).collect(),
            (0..n * n).map(|k| (k % n) as f64 / n as f64).collect(),
        ];
        for (a, nn, max_its) in [(&l, &coords[..1], 40), (&bl, &coords[..], 150)] {
            let amg = Amg::new(a, nn, &AmgParams::default()).expect("hierarchy");
            assert!(amg.num_levels() >= 2);
            let xs: Vec<f64> = (0..n * n).map(|k| ((k as f64) * 0.013).sin()).collect();
            let mut b = vec![0.0; n * n];
            a.mul_vec(&xs, &mut b);
            let mut x = vec![0.0; n * n];
            let p = PcgParams { tolerance: 1e-10, max_iterations: 10_000 };
            let st = solve_pcg_preconditioned(a, &b, &mut x, &p, &amg.preconditioner(1), None);
            let mut xj = vec![0.0; n * n];
            let sj = solve_pcg(a, &b, &mut xj, &p);
            assert!(st.converged && st.iterations < max_its, "AMG {st:?}, Jacobi {sj:?}");
            assert!(st.iterations * 5 < sj.iterations);
            let err = x.iter().zip(&xs).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(err < 1e-5, "{err}");
            // Vector elements use the same hierarchy.
            let b3: Vec<DVec3> = b.iter().map(|&v| DVec3::new(v, -v, 2.0 * v)).collect();
            let mut x3 = vec![DVec3::ZERO; n * n];
            let s3 = solve_pcg_preconditioned(a, &b3, &mut x3, &p, &amg.preconditioner(1), None);
            assert!(s3.converged);
            assert!((x3[17].y + xs[17]).abs() < 1e-5);
        }
    }

    #[test]
    fn pcg_callback_cancels() {
        let a = poisson1d(1000);
        let b = vec![1.0; 1000];
        let mut x = vec![0.0; 1000];
        let st = solve_pcg_with_callback(&a, &b, &mut x, &PcgParams::default(), Some(&|it, _| it < 32));
        assert!(st.cancelled && !st.converged);
    }
}
