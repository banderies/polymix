// SPDX-License-Identifier: AGPL-3.0-only
//! Small dense linear algebra for the fits and the sketch solver: Cholesky, Gaussian
//! elimination (in `fit2d`) and the symmetric Jacobi eigen-decomposition. Ported from
//! ScanRuler (`src/core/fit/linalg.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;

pub use crate::fit2d::solve_linear;

/// Solve `a x = b` for a symmetric positive definite row-major `n×n` matrix by Cholesky.
/// `None` when a pivot falls under `tiny` (the matrix is not definite to working precision).
pub fn cholesky_solve(n: usize, a: &[f64], b: &[f64], tiny: f64) -> Option<Vec<f64>> {
    let mut l = vec![0.0; n * n];
    for j in 0..n {
        let rj = j * n;
        let mut d = a[rj + j];
        for k in 0..j {
            d -= l[rj + k] * l[rj + k];
        }
        if !(d > tiny) {
            return None;
        }
        let ljj = d.sqrt();
        l[rj + j] = ljj;
        for i in j + 1..n {
            let ri = i * n;
            let mut s = a[ri + j];
            for k in 0..j {
                s -= l[ri + k] * l[rj + k];
            }
            l[ri + j] = s / ljj;
        }
    }
    let mut y = vec![0.0; n];
    for i in 0..n {
        let ri = i * n;
        let mut s = b[i];
        for k in 0..i {
            s -= l[ri + k] * y[k];
        }
        y[i] = s / l[ri + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in i + 1..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

/// The eigenvalues (ascending) and unit eigenvectors of a symmetric row-major `n×n` matrix,
/// by cyclic Jacobi rotations.
pub fn symmetric_eigen_n(n: usize, input: &[f64]) -> (Vec<f64>, Vec<Vec<f64>>) {
    let mut a = input[..n * n].to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
    }
    for _sweep in 0..64 {
        let mut off = 0.0;
        let mut diag = 0.0;
        for p in 0..n {
            diag += a[p * n + p].abs();
            for q in p + 1..n {
                off += a[p * n + q].abs();
            }
        }
        if off == 0.0 || off <= 1e-14 * diag {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                let apq = a[p * n + q];
                if apq == 0.0 {
                    continue;
                }
                let theta = (a[q * n + q] - a[p * n + p]) / (2.0 * apq);
                let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
                let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for k in 0..n {
                    let (akp, akq) = (a[k * n + p], a[k * n + q]);
                    a[k * n + p] = c * akp - s * akq;
                    a[k * n + q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p * n + k], a[q * n + k]);
                    a[p * n + k] = c * apk - s * aqk;
                    a[q * n + k] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let (vkp, vkq) = (v[k * n + p], v[k * n + q]);
                    v[k * n + p] = c * vkp - s * vkq;
                    v[k * n + q] = s * vkp + c * vkq;
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| a[i * n + i].total_cmp(&a[j * n + j]));
    let values = order.iter().map(|&i| a[i * n + i]).collect();
    let vectors = order
        .iter()
        .map(|&i| {
            let col: Vec<f64> = (0..n).map(|k| v[k * n + i]).collect();
            let len = col.iter().map(|x| x * x).sum::<f64>().sqrt();
            if len > 1e-20 {
                col.iter().map(|x| x / len).collect()
            } else {
                col
            }
        })
        .collect();
    (values, vectors)
}

fn normalize_or_z(v: DVec3) -> DVec3 {
    let l = v.length();
    if l > 1e-300 {
        v / l
    } else {
        DVec3::Z
    }
}

/// Two unit vectors square to `n` and to each other: a deterministic in-plane basis.
pub fn ortho_basis(n: DVec3) -> (DVec3, DVec3) {
    let a = n.abs();
    let helper = if a.x <= a.y && a.x <= a.z {
        DVec3::X
    } else if a.y <= a.z {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let u = normalize_or_z(n.cross(helper));
    let v = normalize_or_z(n.cross(u));
    (u, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cholesky_and_eigen_agree() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let x = cholesky_solve(3, &a, &[1.0, 2.0, 3.0], 1e-12).unwrap();
        let back: Vec<f64> = (0..3).map(|i| (0..3).map(|j| a[i * 3 + j] * x[j]).sum()).collect();
        assert!(back.iter().zip([1.0, 2.0, 3.0]).all(|(b, w)| (b - w).abs() < 1e-12));
        let (values, vectors) = symmetric_eigen_n(3, &a);
        for (lambda, vec) in values.iter().zip(&vectors) {
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i * 3 + j] * vec[j]).sum();
                assert!((av - lambda * vec[i]).abs() < 1e-10);
            }
        }
        assert!(values[0] <= values[1] && values[1] <= values[2]);
    }
}
