//! Small dense linear algebra: symmetric 3x3 eigen-decomposition (cyclic Jacobi).

use glam::{DMat3, DVec3};

/// Eigen-decomposition of a symmetric 3x3 matrix (row-major). Returns eigenvalues in
/// ascending order with matching unit eigenvectors.
pub fn sym_eigen3(m: [[f64; 3]; 3]) -> ([f64; 3], [DVec3; 3]) {
    let mut a = m;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let scale: f64 = a.iter().flatten().map(|x| x * x).sum::<f64>().max(1e-300);
    for _sweep in 0..64 {
        let off = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
        if off <= 1e-30 * scale {
            break;
        }
        for &(p, q) in &[(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p][q];
            if apq.abs() <= 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            // a <- Jᵀ a J with J = I except J[p][p]=J[q][q]=c, J[p][q]=s, J[q][p]=-s.
            let mut j = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
            j[p][p] = c;
            j[q][q] = c;
            j[p][q] = s;
            j[q][p] = -s;
            a = mul(transpose(j), mul(a, j));
            v = mul(v, j);
            a[p][q] = 0.0;
            a[q][p] = 0.0;
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&x, &y| a[x][x].partial_cmp(&a[y][y]).unwrap_or(std::cmp::Ordering::Equal));
    let vals = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let col = |k: usize| DVec3::new(v[0][k], v[1][k], v[2][k]).normalize_or_zero();
    (vals, [col(idx[0]), col(idx[1]), col(idx[2])])
}

fn mul(a: [[f64; 3]; 3], b: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
        }
    }
    r
}

fn transpose(a: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let mut r = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..3 {
            r[i][j] = a[j][i];
        }
    }
    r
}

/// Accumulate `w * x xᵀ` into a symmetric matrix.
#[inline]
pub fn add_outer(m: &mut [[f64; 3]; 3], x: DVec3, w: f64) {
    let a = [x.x, x.y, x.z];
    for i in 0..3 {
        for j in 0..3 {
            m[i][j] += w * a[i] * a[j];
        }
    }
}

/// Closest rotation to `s` (orthogonal, det +1), via the eigen-decomposition of `sᵀs`.
/// For `s = Σ bᵢ aᵢᵀ` it is the rotation that best maps the `aᵢ` onto the `bᵢ`.
pub fn closest_rotation(s: DMat3) -> DMat3 {
    let sts = s.transpose() * s;
    let a = [0, 1, 2].map(|i| [0, 1, 2].map(|j| sts.col(j)[i]));
    let (vals, vecs) = sym_eigen3(a);
    // s = U Σ Vᵀ with V = eigenvectors of sᵀs (ascending), U columns = s v / σ.
    let v = DMat3::from_cols(vecs[2], vecs[1], vecs[0]);
    let sig = [vals[2], vals[1], vals[0]].map(|x| x.max(0.0).sqrt());
    let mut u0 = s * v.col(0);
    u0 = if sig[0] > 1e-12 { u0 / sig[0] } else { DVec3::X };
    let mut u1 = s * v.col(1);
    u1 = if sig[1] > 1e-12 * sig[0].max(1e-300) { u1 / sig[1] } else { crate::geom::orthonormal_basis(u0).0 };
    u1 = (u1 - u0 * u0.dot(u1)).normalize_or_zero();
    if u1 == DVec3::ZERO {
        u1 = crate::geom::orthonormal_basis(u0).0;
    }
    let mut u2 = u0.cross(u1);
    // Keep det(U Vᵀ) = +1.
    if v.determinant() < 0.0 {
        u2 = -u2;
    }
    let u = DMat3::from_cols(u0, u1, u2);
    // R maps original edges onto deformed ones: R = U Vᵀ for s = Σ e' eᵀ.
    u * v.transpose()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen_matches_definition() {
        let m = [[4.0, 1.0, 0.5], [1.0, 3.0, 0.2], [0.5, 0.2, 1.0]];
        let (vals, vecs) = sym_eigen3(m);
        assert!(vals[0] <= vals[1] && vals[1] <= vals[2]);
        for k in 0..3 {
            let x = vecs[k];
            let mx = DVec3::new(
                m[0][0] * x.x + m[0][1] * x.y + m[0][2] * x.z,
                m[1][0] * x.x + m[1][1] * x.y + m[1][2] * x.z,
                m[2][0] * x.x + m[2][1] * x.y + m[2][2] * x.z,
            );
            assert!((mx - x * vals[k]).length() < 1e-9, "eigenpair {k} wrong");
        }
    }

    #[test]
    fn rotation_fit_recovers_rotations() {
        let r0 = DMat3::from_quat(glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 2.0));
        let pts = [DVec3::new(1.0, 0.2, 0.0), DVec3::new(-0.3, 1.0, 0.5), DVec3::new(0.1, -0.4, 1.0), DVec3::new(0.7, 0.7, -0.2)];
        let s = pts.iter().fold(DMat3::ZERO, |acc, &a| {
            let b = r0 * a;
            acc + DMat3::from_cols(b * a.x, b * a.y, b * a.z)
        });
        let r = closest_rotation(s);
        assert!((r - r0).to_cols_array().iter().all(|x| x.abs() < 1e-9));
        // Planar input (rank 2) still gives a proper rotation.
        let flat = DMat3::from_cols(DVec3::X, DVec3::Y, DVec3::ZERO);
        assert!((closest_rotation(flat).determinant() - 1.0).abs() < 1e-9);
    }
}
