// SPDX-License-Identifier: AGPL-3.0-only
//! Sparse linear algebra for systems too large for the dense Cholesky in `linalg`: a matrix
//! in compressed rows (CSR), assembled from triplets with their duplicates summed, and the
//! conjugate gradient method for symmetric positive-definite systems, preconditioned by the
//! diagonal (Jacobi). Ported from ScanRuler (`src/core/fit/sparse.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! Dense Cholesky costs n³/3: four hundred unknowns is a few milliseconds, ten thousand is
//! minutes and 800 MB of matrix. A least-squares fit over a mesh or a large net touches each
//! unknown with a few dozen others, so the matrix is mostly zeros, and conjugate gradient
//! needs only products with it — each costs the non-zeros, not n². Its convergence goes with
//! the square root of the condition number; scaling by the diagonal takes out what comes
//! from unknowns weighed differently (a heavy row beside light ones), which is most of it in
//! a fit. What is left is the problem's own: a Laplacian over a mesh k vertices across takes
//! about k iterations.
//!
//! The solver takes any operator that can multiply a vector — a CSR matrix is one, and a
//! system kept in pieces need not be assembled to be solved. A system that is not
//! symmetric takes BiCGSTAB instead, at twice the products an iteration.

/// A square matrix in compressed rows: row i's entries are `cols`/`vals` in
/// `row_ptr[i]..row_ptr[i + 1]`, their columns ascending and each once.
#[derive(Debug, Clone, PartialEq)]
pub struct CsrMatrix {
    pub n: usize,
    pub row_ptr: Vec<usize>,
    pub cols: Vec<usize>,
    pub vals: Vec<f64>,
}

/// A symmetric positive-definite operator: its size, its product with a vector (written
/// into `out`, which is never `x`), and its diagonal for the preconditioner where it is
/// known.
pub trait LinearOperator {
    fn size(&self) -> usize;
    fn apply(&self, x: &[f64], out: &mut [f64]);
    fn diagonal(&self) -> Option<Vec<f64>> {
        None
    }
}

/// A CSR matrix as an operator, its diagonal read off it.
impl LinearOperator for CsrMatrix {
    fn size(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        csr_multiply(self, x, out);
    }
    fn diagonal(&self) -> Option<Vec<f64>> {
        Some(csr_diagonal(self))
    }
}

/// An operator given as a product function and, where known, its diagonal.
pub struct FnOperator<F: Fn(&[f64], &mut [f64])> {
    pub n: usize,
    pub apply: F,
    pub diagonal: Option<Vec<f64>>,
}

impl<F: Fn(&[f64], &mut [f64])> LinearOperator for FnOperator<F> {
    fn size(&self) -> usize {
        self.n
    }
    fn apply(&self, x: &[f64], out: &mut [f64]) {
        (self.apply)(x, out);
    }
    fn diagonal(&self) -> Option<Vec<f64>> {
        self.diagonal.clone()
    }
}

/// Entries gathered in any order, a position given more than once summed — what a fit's
/// terms add to its normal equations, point by point.
#[derive(Debug, Clone, Default)]
pub struct Triplets {
    pub n: usize,
    rows: Vec<usize>,
    cols_at: Vec<usize>,
    vals_at: Vec<f64>,
}

impl Triplets {
    pub fn new(n: usize, capacity: usize) -> Self {
        let c = capacity.max(16);
        Triplets { n, rows: Vec::with_capacity(c), cols_at: Vec::with_capacity(c), vals_at: Vec::with_capacity(c) }
    }

    /// How many entries have been added (a zero is not one).
    pub fn size(&self) -> usize {
        self.rows.len()
    }

    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        if v == 0.0 {
            return;
        }
        self.rows.push(i);
        self.cols_at.push(j);
        self.vals_at.push(v);
    }

    /// The entries `v · w[a] · w[b]` for every pair of `idx`: the term v (Σ w[a] x[idx[a]])²
    /// adds to a system's matrix.
    pub fn add_outer(&mut self, idx: &[usize], w: &[f64], v: f64) {
        for a in 0..idx.len() {
            let va = v * w[a];
            if va == 0.0 {
                continue;
            }
            for b in 0..idx.len() {
                self.add(idx[a], idx[b], va * w[b]);
            }
        }
    }

    pub fn to_csr(&self) -> CsrMatrix {
        csr_from_triplets(self.n, &self.rows, &self.cols_at, &self.vals_at, self.rows.len())
    }
}

/// A CSR matrix of the first `count` triplets, duplicates summed. Two passes by row, then a
/// marker per column within each row: linear in the entries, with no sort but the columns'
/// order within a row.
pub fn csr_from_triplets(n: usize, rows: &[usize], cols: &[usize], vals: &[f64], count: usize) -> CsrMatrix {
    let count = count.min(rows.len()).min(cols.len()).min(vals.len());
    let mut start = vec![0usize; n + 1];
    for &r in &rows[..count] {
        start[r + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut by_row_col = vec![0usize; count];
    let mut by_row_val = vec![0.0f64; count];
    let mut cursor = start[..n].to_vec();
    for k in 0..count {
        let at = cursor[rows[k]];
        cursor[rows[k]] += 1;
        by_row_col[at] = cols[k];
        by_row_val[at] = vals[k];
    }
    // Duplicates summed into the column's first place in its row.
    let mut marker: Vec<Option<usize>> = vec![None; n];
    let mut row_ptr = vec![0usize; n + 1];
    let mut out_col = vec![0usize; count];
    let mut out_val = vec![0.0f64; count];
    let mut w = 0usize;
    for i in 0..n {
        let first = w;
        for k in start[i]..start[i + 1] {
            let c = by_row_col[k];
            match marker[c] {
                Some(m) if m >= first => out_val[m] += by_row_val[k],
                _ => {
                    marker[c] = Some(w);
                    out_col[w] = c;
                    out_val[w] = by_row_val[k];
                    w += 1;
                }
            }
        }
        sort_row(&mut out_col, &mut out_val, first, w);
        row_ptr[i + 1] = w;
    }
    out_col.truncate(w);
    out_val.truncate(w);
    CsrMatrix { n, row_ptr, cols: out_col, vals: out_val }
}

/// A row's entries in column order — insertion sort, a row being short.
fn sort_row(cols: &mut [usize], vals: &mut [f64], from: usize, to: usize) {
    for i in from + 1..to {
        let c = cols[i];
        let v = vals[i];
        let mut j = i;
        while j > from && cols[j - 1] > c {
            cols[j] = cols[j - 1];
            vals[j] = vals[j - 1];
            j -= 1;
        }
        cols[j] = c;
        vals[j] = v;
    }
}

/// out = A x.
pub fn csr_multiply(a: &CsrMatrix, x: &[f64], out: &mut [f64]) {
    for i in 0..a.n {
        let mut s = 0.0;
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            s += a.vals[k] * x[a.cols[k]];
        }
        out[i] = s;
    }
}

/// The matrix's diagonal.
pub fn csr_diagonal(a: &CsrMatrix) -> Vec<f64> {
    let mut d = vec![0.0; a.n];
    for (i, di) in d.iter_mut().enumerate() {
        for k in a.row_ptr[i]..a.row_ptr[i + 1] {
            if a.cols[k] == i {
                *di = a.vals[k];
            }
        }
    }
    d
}

/// How a conjugate gradient solve runs.
#[derive(Debug, Clone)]
pub struct CgOptions {
    /// Where to start; zero when absent. A solution of a nearby system — the last round of an
    /// iteration — saves most of the work.
    pub x0: Option<Vec<f64>>,
    /// Stop once the residual is this share of the right-hand side's norm.
    pub tolerance: f64,
    /// At most this many iterations; `max(100, 10 n)` when absent.
    pub max_iterations: Option<usize>,
    /// Scale by the diagonal (Jacobi); on unless asked off.
    pub precondition: bool,
}

impl Default for CgOptions {
    fn default() -> Self {
        CgOptions { x0: None, tolerance: 1e-10, max_iterations: None, precondition: true }
    }
}

/// What a solve came to.
#[derive(Debug, Clone, PartialEq)]
pub struct CgResult {
    pub x: Vec<f64>,
    pub iterations: usize,
    /// The residual's norm over the right-hand side's, at the end.
    pub residual: f64,
    pub converged: bool,
}

/// Solve A x = b for a symmetric positive-definite A by preconditioned conjugate gradient. A
/// zero right-hand side gives zero at once.
pub fn conjugate_gradient(op: &dyn LinearOperator, b: &[f64], opts: &CgOptions) -> CgResult {
    let n = op.size();
    let tolerance = opts.tolerance;
    let max_iterations = opts.max_iterations.unwrap_or(100.max(10 * n));
    let mut x = match &opts.x0 {
        Some(x0) => x0.clone(),
        None => vec![0.0; n],
    };
    x.resize(n, 0.0);
    let mut r = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut q = vec![0.0; n];
    // The preconditioner: 1 / diagonal, where the diagonal is positive.
    let mut inv = vec![1.0; n];
    if opts.precondition {
        if let Some(diagonal) = op.diagonal() {
            for i in 0..n.min(diagonal.len()) {
                let d = diagonal[i];
                inv[i] = if d > 0.0 { 1.0 / d } else { 1.0 };
            }
        }
    }
    let b_norm = b.iter().take(n).map(|v| v * v).sum::<f64>().sqrt();
    if !(b_norm > 0.0) {
        return CgResult { x: vec![0.0; n], iterations: 0, residual: 0.0, converged: true };
    }

    op.apply(&x, &mut q);
    let mut rr = 0.0;
    for i in 0..n {
        r[i] = b[i] - q[i];
        rr += r[i] * r[i];
    }
    let mut rz = 0.0;
    for i in 0..n {
        z[i] = inv[i] * r[i];
        p[i] = z[i];
        rz += r[i] * z[i];
    }
    let goal = tolerance * b_norm;
    let mut iterations = 0;
    while rr.sqrt() > goal && iterations < max_iterations {
        op.apply(&p, &mut q);
        let mut pq = 0.0;
        for i in 0..n {
            pq += p[i] * q[i];
        }
        // Not positive along p: the operator is not definite, or the round-off has caught up
        // with the solution. What there is stands.
        if !(pq > 0.0) {
            break;
        }
        let alpha = rz / pq;
        rr = 0.0;
        for i in 0..n {
            x[i] += alpha * p[i];
            r[i] -= alpha * q[i];
            rr += r[i] * r[i];
        }
        let mut rz_next = 0.0;
        for i in 0..n {
            z[i] = inv[i] * r[i];
            rz_next += r[i] * z[i];
        }
        let beta = rz_next / rz;
        rz = rz_next;
        for i in 0..n {
            p[i] = z[i] + beta * p[i];
        }
        iterations += 1;
    }
    let residual = rr.sqrt() / b_norm;
    CgResult { x, iterations, residual, converged: residual <= tolerance }
}

/// Solve A x = b for a square A that need not be symmetric — a system of weighted means whose
/// weights differ either way along an edge — by BiCGSTAB (van der Vorst), preconditioned by
/// the diagonal. It wants an A whose spectrum keeps clear of zero, as a diagonally dominant
/// one does. Two products with A an iteration.
pub fn biconjugate_gradient_stabilized(op: &dyn LinearOperator, b: &[f64], opts: &CgOptions) -> CgResult {
    let n = op.size();
    let tolerance = opts.tolerance;
    let max_iterations = opts.max_iterations.unwrap_or(100.max(10 * n));
    let mut x = match &opts.x0 {
        Some(x0) => x0.clone(),
        None => vec![0.0; n],
    };
    x.resize(n, 0.0);
    let mut inv = vec![1.0; n];
    if opts.precondition {
        if let Some(diagonal) = op.diagonal() {
            for i in 0..n.min(diagonal.len()) {
                inv[i] = if diagonal[i] != 0.0 { 1.0 / diagonal[i] } else { 1.0 };
            }
        }
    }
    let b_norm = b.iter().take(n).map(|v| v * v).sum::<f64>().sqrt();
    if !(b_norm > 0.0) {
        return CgResult { x: vec![0.0; n], iterations: 0, residual: 0.0, converged: true };
    }
    let mut r = vec![0.0; n];
    let mut r0 = vec![0.0; n];
    let mut p = vec![0.0; n];
    let mut v = vec![0.0; n];
    let mut s = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut z = vec![0.0; n];
    op.apply(&x, &mut t);
    let mut rr = 0.0;
    for i in 0..n {
        r[i] = b[i] - t[i];
        r0[i] = r[i];
        rr += r[i] * r[i];
    }
    let goal = tolerance * b_norm;
    let mut rho = 1.0;
    let mut alpha = 1.0;
    let mut omega = 1.0;
    let mut iterations = 0;
    while rr.sqrt() > goal && iterations < max_iterations {
        let mut rho_next = 0.0;
        for i in 0..n {
            rho_next += r0[i] * r[i];
        }
        if rho_next == 0.0 {
            break;
        }
        let beta = (rho_next / rho) * (alpha / omega);
        rho = rho_next;
        for i in 0..n {
            p[i] = r[i] + beta * (p[i] - omega * v[i]);
            y[i] = inv[i] * p[i];
        }
        op.apply(&y, &mut v);
        let mut r0v = 0.0;
        for i in 0..n {
            r0v += r0[i] * v[i];
        }
        if r0v == 0.0 {
            break;
        }
        alpha = rho / r0v;
        let mut ss = 0.0;
        for i in 0..n {
            s[i] = r[i] - alpha * v[i];
            ss += s[i] * s[i];
        }
        if ss.sqrt() <= goal {
            for i in 0..n {
                x[i] += alpha * y[i];
            }
            rr = ss;
            iterations += 1;
            break;
        }
        for i in 0..n {
            z[i] = inv[i] * s[i];
        }
        op.apply(&z, &mut t);
        let mut tt = 0.0;
        let mut ts = 0.0;
        for i in 0..n {
            tt += t[i] * t[i];
            ts += t[i] * s[i];
        }
        omega = if tt > 0.0 { ts / tt } else { 0.0 };
        rr = 0.0;
        for i in 0..n {
            x[i] += alpha * y[i] + omega * z[i];
            r[i] = s[i] - omega * t[i];
            rr += r[i] * r[i];
        }
        iterations += 1;
        if omega == 0.0 {
            break;
        }
    }
    let residual = rr.sqrt() / b_norm;
    CgResult { x, iterations, residual, converged: residual <= tolerance }
}

// Ported from ScanRuler's tests/sparse.test.ts: the sparse solver checked against dense
// Cholesky on the 2D Laplacian — the five-point stencil over a grid with its edge held —
// and on a system whose rows are weighed very differently, where the diagonal's scaling
// earns its keep.
#[cfg(test)]
mod tests {
    use super::*;
    use crate::linalg::cholesky_solve;

    /// The five-point Laplacian over a k × k grid of unknowns, the cells past its edge held
    /// at zero, plus `shift` on the diagonal.
    fn laplacian(k: usize, shift: f64) -> Triplets {
        let mut t = Triplets::new(k * k, 16);
        let at = |i: usize, j: usize| i * k + j;
        for i in 0..k {
            for j in 0..k {
                t.add(at(i, j), at(i, j), 4.0 + shift);
                if i > 0 {
                    t.add(at(i, j), at(i - 1, j), -1.0);
                }
                if i + 1 < k {
                    t.add(at(i, j), at(i + 1, j), -1.0);
                }
                if j > 0 {
                    t.add(at(i, j), at(i, j - 1), -1.0);
                }
                if j + 1 < k {
                    t.add(at(i, j), at(i, j + 1), -1.0);
                }
            }
        }
        t
    }

    fn dense(n: usize, csr: &CsrMatrix) -> Vec<f64> {
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            for p in csr.row_ptr[i]..csr.row_ptr[i + 1] {
                a[i * n + csr.cols[p]] = csr.vals[p];
            }
        }
        a
    }

    #[test]
    fn sums_duplicates_and_keeps_each_row_in_column_order() {
        let csr = csr_from_triplets(3, &[0, 2, 0, 1, 0, 2], &[2, 0, 0, 1, 2, 0], &[1.0, 5.0, 3.0, 4.0, 2.0, 1.0], 6);
        assert_eq!(csr.row_ptr, vec![0, 2, 3, 4]);
        assert_eq!(csr.cols, vec![0, 2, 1, 0]);
        assert_eq!(csr.vals, vec![3.0, 3.0, 4.0, 6.0]);
        assert_eq!(csr_diagonal(&csr), vec![3.0, 4.0, 0.0]);
        let mut out = vec![0.0; 3];
        csr_multiply(&csr, &[1.0, 2.0, 3.0], &mut out);
        assert_eq!(out, vec![12.0, 8.0, 6.0]);
    }

    #[test]
    fn solves_a_2d_laplacian_as_dense_cholesky_does() {
        let k = 20;
        let n = k * k;
        let csr = laplacian(k, 0.0).to_csr();
        assert_eq!(csr.vals.len(), 5 * n - 4 * k);
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.37).sin() + (i % 7) as f64 * 0.1).collect();
        let exact = cholesky_solve(n, &dense(n, &csr), &b, 1e-12).unwrap();
        let cg = conjugate_gradient(&csr, &b, &CgOptions { tolerance: 1e-12, ..Default::default() });
        assert!(cg.converged);
        // About k iterations per factor of e — a few times k in all.
        assert!(cg.iterations < 6 * k);
        let worst = (0..n).map(|i| (cg.x[i] - exact[i]).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-9);
    }

    #[test]
    fn takes_a_start_and_takes_the_system_as_an_operator() {
        let k = 12;
        let n = k * k;
        let csr = laplacian(k, 0.5).to_csr();
        let b = vec![1.0; n];
        let cold = conjugate_gradient(&csr, &b, &CgOptions { tolerance: 1e-10, ..Default::default() });
        let warm = conjugate_gradient(&csr, &b, &CgOptions { tolerance: 1e-10, x0: Some(cold.x.clone()), ..Default::default() });
        assert!(warm.iterations <= 1);
        let diagonal = csr_diagonal(&csr);
        let op = FnOperator { n, apply: |x: &[f64], out: &mut [f64]| csr_multiply(&csr, x, out), diagonal: Some(diagonal) };
        let as_operator = conjugate_gradient(&op, &b, &CgOptions { tolerance: 1e-10, ..Default::default() });
        for i in 0..n {
            // toBeCloseTo(…, 9): within half of 1e-9.
            assert!((as_operator.x[i] - cold.x[i]).abs() < 5e-10);
        }
    }

    #[test]
    fn scales_away_rows_weighed_a_millionfold_apart() {
        // The Laplacian with every other row and column scaled by 1000: the same system in
        // other units, which plain conjugate gradient crawls through and the diagonal's
        // scaling does not.
        let k = 16;
        let n = k * k;
        let base = laplacian(k, 0.1).to_csr();
        let s = |i: usize| if i.is_multiple_of(2) { 1000.0 } else { 1.0 };
        let mut t = Triplets::new(n, 16);
        for i in 0..n {
            for p in base.row_ptr[i]..base.row_ptr[i + 1] {
                t.add(i, base.cols[p], base.vals[p] * s(i) * s(base.cols[p]));
            }
        }
        let csr = t.to_csr();
        let b: Vec<f64> = (0..n).map(|i| s(i) * (i as f64).cos()).collect();
        let exact = cholesky_solve(n, &dense(n, &csr), &b, 1e-12).unwrap();
        let jacobi = conjugate_gradient(&csr, &b, &CgOptions { tolerance: 1e-12, ..Default::default() });
        let plain = conjugate_gradient(&csr, &b, &CgOptions { tolerance: 1e-12, precondition: false, max_iterations: Some(jacobi.iterations * 3), ..Default::default() });
        assert!(jacobi.converged);
        assert!(!plain.converged);
        for i in 0..n {
            assert!((jacobi.x[i] - exact[i]).abs() < 1e-8 * exact[i].abs().max(1.0));
        }
    }

    #[test]
    fn answers_zero_for_a_zero_right_hand_side() {
        let csr = laplacian(4, 0.0).to_csr();
        let r = conjugate_gradient(&csr, &[0.0; 16], &CgOptions::default());
        assert_eq!(r.iterations, 0);
        assert!(r.x.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn bicgstab_solves_an_unsymmetric_diagonally_dominant_system() {
        // Not in ScanRuler's tests: a check that the BiCGSTAB port solves what it claims to.
        let k = 10;
        let n = k * k;
        let mut t = laplacian(k, 1.0);
        for i in 0..n - 1 {
            t.add(i, i + 1, -0.3);
        }
        let csr = t.to_csr();
        let b: Vec<f64> = (0..n).map(|i| (i as f64 * 0.21).cos()).collect();
        let r = biconjugate_gradient_stabilized(&csr, &b, &CgOptions { tolerance: 1e-12, ..Default::default() });
        assert!(r.converged);
        let mut ax = vec![0.0; n];
        csr_multiply(&csr, &r.x, &mut ax);
        for i in 0..n {
            assert!((ax[i] - b[i]).abs() < 1e-9);
        }
    }
}
