// SPDX-License-Identifier: AGPL-3.0-only
//! The principal frame of a point cloud: its centroid and the axes it scatters along, most
//! first. Ported from ScanRuler (`src/core/deviation/prealign.ts` `principalFrame`, and
//! `symmetricEigen3`, `normalizeOrZ` and `orthoBasis` from `src/core/fit/linalg.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Plane and cylinder fitting both reduce to "which direction does this cloud of points or
//! normals scatter least along", which is exactly the smallest eigenvector of a 3×3
//! covariance; the fitted surfaces take their base plane from the same decomposition.
//!
//! Also the JavaScript number semantics the surface modules keep: `js_cmp` (a comparator's
//! order) and `js_hypot` (V8's `Math.hypot`, bit for bit, which ScanRuler reads every length
//! with).

use glam::DVec3;

/// The eigen decomposition of a symmetric 3×3 matrix: the eigenvalues ascending, and the
/// matching unit eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigen3 {
    /// Eigenvalues in ascending order.
    pub values: [f64; 3],
    /// Matching unit eigenvectors.
    pub vectors: [DVec3; 3],
}

/// JavaScript's `Math.hypot` as V8 computes it, to the last bit: every value over the
/// largest, their squares summed with Kahan's compensation, the root times the largest.
/// A plain `sqrt(x² + y² + z²)` — or libm's `hypot` — differs from it in the last bit for
/// about a third of all inputs, and ScanRuler's distances, normals and frames are all read
/// with `Math.hypot`. Infinity where any value is infinite, else NaN where any is NaN.
pub fn js_hypot(values: &[f64]) -> f64 {
    let mut max = 0.0f64;
    let mut nan = false;
    for &x in values {
        if x.is_nan() {
            nan = true;
        } else if x.abs() > max {
            max = x.abs();
        }
    }
    if max == f64::INFINITY {
        return f64::INFINITY;
    }
    if nan {
        return f64::NAN;
    }
    if max == 0.0 {
        return 0.0;
    }
    let (mut sum, mut compensation) = (0.0f64, 0.0f64);
    for &x in values {
        let n = x.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// `Math.hypot(a, b)` (`js_hypot`).
pub fn js_hypot2(a: f64, b: f64) -> f64 {
    js_hypot(&[a, b])
}

/// `Math.hypot(v[0], v[1], v[2])` (`js_hypot`): a vector's length as ScanRuler reads it.
pub fn js_hypot3(v: DVec3) -> f64 {
    js_hypot(&[v.x, v.y, v.z])
}

/// Unit vector, falling back to +Z for a zero-length input. Deliberately not a nullable
/// normalise: the fitting iterations need a usable direction even when a degenerate step
/// collapses a vector, and a null contract would force a failure branch into every numeric
/// loop.
pub fn normalize_or_z(v: DVec3) -> DVec3 {
    let len = js_hypot3(v);
    if !(len > 1e-20) {
        return DVec3::Z;
    }
    DVec3::new(v.x / len, v.y / len, v.z / len)
}

/// Any two unit vectors completing `n` into a right-handed frame — ScanRuler's `linalg.ts`
/// `orthoBasis`, normalised as it normalises (`normalize_or_z`). mm-reverse's
/// `linalg::ortho_basis` picks the same helper axis but normalises with glam's `length`,
/// which differs from `Math.hypot` in the last bit.
pub fn ortho_basis(n: DVec3) -> (DVec3, DVec3) {
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let helper = if ax <= ay && ax <= az {
        DVec3::X
    } else if ay <= az {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let u = normalize_or_z(cross(n, helper));
    let v = normalize_or_z(cross(n, u));
    (u, v)
}

/// Eigen decomposition of a symmetric 3×3 matrix (row-major, 9 entries) by cyclic Jacobi
/// rotations. Plane and cylinder fitting both reduce to "which direction does this cloud of
/// points or normals scatter least along", which is exactly the smallest eigenvector.
pub fn symmetric_eigen3(input: &[f64; 9]) -> Eigen3 {
    let mut a = *input;
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    const PAIRS: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

    for _sweep in 0..32 {
        let off = a[1].abs() + a[2].abs() + a[5].abs();
        let diag = a[0].abs() + a[4].abs() + a[8].abs();
        // 1e-14 relative is as converged as float64 rotations get; a tighter gate would
        // never fire and every call would burn all the sweeps.
        if off <= 1e-14 * diag || off == 0.0 {
            break;
        }
        for (p, q) in PAIRS {
            let apq = a[p * 3 + q];
            if apq == 0.0 {
                continue;
            }
            // The rotation angle that zeroes the (p, q) entry.
            let theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * apq);
            let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
            let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let akp = a[k * 3 + p];
                let akq = a[k * 3 + q];
                a[k * 3 + p] = c * akp - s * akq;
                a[k * 3 + q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let apk = a[p * 3 + k];
                let aqk = a[q * 3 + k];
                a[p * 3 + k] = c * apk - s * aqk;
                a[q * 3 + k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let vkp = v[k * 3 + p];
                let vkq = v[k * 3 + q];
                v[k * 3 + p] = c * vkp - s * vkq;
                v[k * 3 + q] = s * vkp + c * vkq;
            }
        }
    }

    let mut order = [0usize, 1, 2];
    // A stable sort by the diagonal, ascending, as the JavaScript comparator `a - b` sorts.
    order.sort_by(|&i, &j| js_cmp(a[i * 3 + i], a[j * 3 + j]));
    let values = [a[order[0] * 4], a[order[1] * 4], a[order[2] * 4]];
    let column = |i: usize| normalize_or_z(DVec3::new(v[i], v[3 + i], v[6 + i]));
    Eigen3 { values, vectors: [column(order[0]), column(order[1]), column(order[2])] }
}

/// Two numbers ordered as JavaScript's `(a, b) => a - b` comparator orders them, made a
/// total order: equal values (−0 and +0 among them) keep their order, and NaN goes last.
pub fn js_cmp(a: f64, b: f64) -> std::cmp::Ordering {
    a.partial_cmp(&b).unwrap_or_else(|| a.is_nan().cmp(&b.is_nan()))
}

/// The centroid and principal axes of a point cloud.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PrincipalFrame {
    pub centroid: DVec3,
    /// Unit axes in descending order of scatter, right-handed.
    pub axes: [DVec3; 3],
    /// Scatter along each axis, descending: a flat plate has a near-zero third, a bar two
    /// near-equal smaller ones.
    pub spread: [f64; 3],
}

/// The centroid and principal axes of a point cloud.
pub fn principal_frame(xyz: &[DVec3]) -> PrincipalFrame {
    let n = xyz.len();
    if n == 0 {
        return PrincipalFrame { centroid: DVec3::ZERO, axes: [DVec3::X, DVec3::Y, DVec3::Z], spread: [0.0; 3] };
    }
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for p in xyz {
        mx += p.x;
        my += p.y;
        mz += p.z;
    }
    let nf = n as f64;
    mx /= nf;
    my /= nf;
    mz /= nf;

    let mut c = [0.0f64; 9];
    for p in xyz {
        let x = p.x - mx;
        let y = p.y - my;
        let z = p.z - mz;
        c[0] += x * x;
        c[1] += x * y;
        c[2] += x * z;
        c[4] += y * y;
        c[5] += y * z;
        c[8] += z * z;
    }
    c[3] = c[1];
    c[6] = c[2];
    c[7] = c[5];
    for e in c.iter_mut() {
        *e /= nf;
    }

    // symmetric_eigen3 returns ascending; the dominant axis is last.
    let eig = symmetric_eigen3(&c);
    let a0 = normalize_or_z(eig.vectors[2]);
    let mut a1 = eig.vectors[1];
    // Re-orthogonalised against a0, then completed right-handed, so the frame is a proper
    // rotation even when two eigenvalues are nearly equal and Jacobi has returned a slightly
    // skewed pair.
    let d = a0.x * a1.x + a0.y * a1.y + a0.z * a1.z;
    a1 = normalize_or_z(DVec3::new(a1.x - d * a0.x, a1.y - d * a0.y, a1.z - d * a0.z));
    let a2 = normalize_or_z(cross(a0, a1));

    PrincipalFrame {
        centroid: DVec3::new(mx, my, mz),
        axes: [a0, a1, a2],
        spread: [eig.values[2].max(0.0).sqrt(), eig.values[1].max(0.0).sqrt(), eig.values[0].max(0.0).sqrt()],
    }
}

/// The cross product, written out as ScanRuler's `cross` writes it.
pub fn cross(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen3_decomposes_a_symmetric_matrix() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let e = symmetric_eigen3(&a);
        assert!(e.values[0] <= e.values[1] && e.values[1] <= e.values[2]);
        for (lambda, vec) in e.values.iter().zip(e.vectors) {
            let v = [vec.x, vec.y, vec.z];
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i * 3 + j] * v[j]).sum();
                assert!((av - lambda * v[i]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn hypot_as_v8_computes_it() {
        // Node's Math.hypot, to the last bit — where sqrt(x² + y² + z²) is one off.
        assert_eq!(js_hypot(&[3.0, 4.0]), 5.0);
        assert_eq!(js_hypot(&[1e-300, 1e-300]), 1.4142135623730952e-300);
        assert_eq!(js_hypot(&[1.0, 2.0, 3.0]), 3.741657386773941);
        assert_eq!(js_hypot(&[0.1, 0.2, 0.3]), 0.37416573867739417);
        assert_eq!(js_hypot(&[8.53, 7.16, 3.6]), 11.70412320509315);
        assert_eq!((8.53f64 * 8.53 + 7.16 * 7.16 + 3.6 * 3.6).sqrt(), 11.704123205093152);
        assert_eq!(js_hypot3(DVec3::new(8.69, 0.65, 1.62)), 8.863577155979408);
        assert_eq!(js_hypot2(4.48, 5.02), 6.728357897734038);
        assert_eq!(js_hypot(&[1e200, 1e200, 1e200]), 1.7320508075688773e200);
        assert_eq!(js_hypot(&[0.0, 0.0]), 0.0);
        assert_eq!(js_hypot(&[f64::INFINITY, f64::NAN]), f64::INFINITY);
        assert!(js_hypot(&[f64::NAN, 1.0]).is_nan());
    }

    #[test]
    fn principal_frame_of_a_plate() {
        let mut pts = Vec::new();
        for i in 0..20 {
            for j in 0..10 {
                pts.push(DVec3::new(i as f64, j as f64 * 0.5, 0.0));
            }
        }
        let f = principal_frame(&pts);
        assert!(f.axes[0].x.abs() > 0.999);
        assert!(f.axes[2].z.abs() > 0.999);
        assert!(f.spread[2] < 1e-9);
        // Right-handed.
        assert!((cross(f.axes[0], f.axes[1]) - f.axes[2]).length() < 1e-12);
    }
}
