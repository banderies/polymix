// SPDX-License-Identifier: AGPL-3.0-only
//! The dense linear algebra the 3D fits add to `crate::linalg`: the symmetric 3×3
//! eigenproblem by Jacobi rotation — a plane's normal, a cylinder's axis from its normals and
//! a torus's spine plane are each an extreme eigenvector of a small matrix — and the
//! point-set helpers the fits share (a centroid, projection onto a plane, an RMS over a
//! residual, the extent of a patch). Lengths go through `Math.hypot` as V8 rounds it
//! (`js_hypot`), and the frame across a direction is ScanRuler's own `orthoBasis`, so the
//! fits start from the very numbers ScanRuler's do.
//!
//! Ported from ScanRuler (`src/core/fit/linalg.ts`, with `normalize` from `src/core/vec.ts`
//! and `extentOfIndexed` from `src/core/geometry/bounds.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only. Elimination, Cholesky and the n×n eigenproblem are already in
//! `crate::fit2d` and `crate::linalg`; `crate::linalg::ortho_basis` rounds differently from
//! ScanRuler's and is not used here.

use glam::{DVec2, DVec3};

/// Eigenvalues of a symmetric 3×3 matrix in ascending order, with matching unit
/// eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Eigen3 {
    pub values: [f64; 3],
    pub vectors: [DVec3; 3],
}

/// Eigen decomposition of a symmetric 3×3 matrix (row-major, 9 entries) by cyclic Jacobi
/// rotations. Plane and cylinder fitting both reduce to "which direction does this cloud of
/// points / normals scatter least along", which is exactly the smallest eigenvector.
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
            // Rotation angle that zeroes the (p, q) entry.
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

    // A stable sort that, as the comparator `(i, j) => a[i] − a[j]` does, takes equal (and
    // incomparable) diagonals as tied.
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| a[i * 3 + i].partial_cmp(&a[j * 3 + j]).unwrap_or(std::cmp::Ordering::Equal));
    let values = order.map(|i| a[i * 3 + i]);
    let vectors = order.map(|i| normalize_or_z(DVec3::new(v[i], v[3 + i], v[6 + i])));
    Eigen3 { values, vectors }
}

/// The eigen decomposition of a 3×3 scatter matrix given by its six distinct entries.
pub fn eigen_of_scatter(xx: f64, xy: f64, xz: f64, yy: f64, yz: f64, zz: f64) -> Eigen3 {
    symmetric_eigen3(&[xx, xy, xz, xy, yy, yz, xz, yz, zz])
}

/// `Math.hypot(...values)` exactly as V8 computes it: infinity if any value is infinite, else
/// NaN if any is NaN, else every magnitude scaled by the largest, the squares summed with
/// Kahan's compensation, and the root scaled back. This rounds differently from both
/// `sqrt(x² + y² + z²)` and `f64::hypot`, and the fits normalise every direction through it,
/// so it is reproduced bit for bit.
pub(crate) fn js_hypot(values: &[f64]) -> f64 {
    let mut max = 0.0f64;
    let mut nan = false;
    for &v in values {
        if v.is_nan() {
            nan = true;
        } else if v.abs() > max {
            max = v.abs();
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
    let mut sum = 0.0f64;
    let mut compensation = 0.0f64;
    for &v in values {
        let n = v.abs() / max;
        let summand = n * n - compensation;
        let preliminary = sum + summand;
        compensation = (preliminary - sum) - summand;
        sum = preliminary;
    }
    sum.sqrt() * max
}

/// `Math.hypot(a, b)` as V8 computes it (see `js_hypot`).
pub(crate) fn hypot2(a: f64, b: f64) -> f64 {
    js_hypot(&[a, b])
}

/// `|v|`, as `Math.hypot` of three components reads it (see `js_hypot`).
pub fn hypot3(x: f64, y: f64, z: f64) -> f64 {
    js_hypot(&[x, y, z])
}

/// Unit vector, falling back to +Z for a zero-length input. Deliberately not the nullable
/// `normalize`: the fitting iterations need a usable direction even when a degenerate step
/// collapses a vector, and a nullable contract would force a failure branch into every
/// numeric loop.
pub fn normalize_or_z(v: DVec3) -> DVec3 {
    let len = hypot3(v.x, v.y, v.z);
    if !(len > 1e-20) {
        return DVec3::Z;
    }
    DVec3::new(v.x / len, v.y / len, v.z / len)
}

/// Any two unit vectors completing `n` into a right-handed frame: `u = n × helper` and
/// `v = n × u`, the helper the axis `n` has least of. ScanRuler's own `orthoBasis` — its
/// lengths through `Math.hypot` and its zero test at 1e-20 — rather than
/// `crate::linalg::ortho_basis`, whose rounding differs in the last place: every fit's frame
/// is built from it, and the iterations amplify that difference.
pub fn ortho_basis(n: DVec3) -> (DVec3, DVec3) {
    let (ax, ay, az) = (n.x.abs(), n.y.abs(), n.z.abs());
    let helper = if ax <= ay && ax <= az {
        DVec3::X
    } else if ay <= az {
        DVec3::Y
    } else {
        DVec3::Z
    };
    let u = normalize_or_z(n.cross(helper));
    let v = normalize_or_z(n.cross(u));
    (u, v)
}

/// Unit vector, or `None` for a (near) zero-length input.
pub fn normalize(a: DVec3) -> Option<DVec3> {
    let l = hypot3(a.x, a.y, a.z);
    if l < 1e-12 {
        return None;
    }
    Some(DVec3::new(a.x / l, a.y / l, a.z / l))
}

/// The mean of the points `idx` names.
pub fn centroid_of(positions: &[DVec3], idx: &[u32]) -> DVec3 {
    let n = idx.len() as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for &i in idx {
        let p = positions[i as usize];
        mx += p.x;
        my += p.y;
        mz += p.z;
    }
    DVec3::new(mx / n, my / n, mz / n)
}

/// The points `idx` names as coordinates in the plane through `centre` that `u` and `v`
/// span — where a fit across an axis looks for its circle.
pub fn project_onto_plane(positions: &[DVec3], idx: &[u32], centre: DVec3, u: DVec3, v: DVec3) -> Vec<DVec2> {
    idx.iter()
        .map(|&i| {
            let q = positions[i as usize] - centre;
            DVec2::new(q.dot(u), q.dot(v))
        })
        .collect()
}

/// The root mean square of a residual over the points `idx` names, those it has no finite
/// answer for left out — infinity when none is left.
pub fn rms_over(positions: &[DVec3], idx: &[u32], residual: impl Fn(DVec3) -> f64) -> f64 {
    let mut sum_sq = 0.0;
    let mut m = 0usize;
    for &i in idx {
        let e = residual(positions[i as usize]);
        if !e.is_finite() {
            continue;
        }
        sum_sq += e * e;
        m += 1;
    }
    if m == 0 {
        f64::INFINITY
    } else {
        (sum_sq / m as f64).sqrt()
    }
}

/// The least and greatest coordinate on each axis — the box inside out, ±infinity, when
/// there are no points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Extent {
    pub min: DVec3,
    pub max: DVec3,
}

impl Extent {
    /// The length of the box's diagonal.
    pub fn diagonal(&self) -> f64 {
        let d = self.max - self.min;
        (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
    }
}

/// The extent of the points `idx` names — a patch, or a mesh's triangle corners, repeats
/// and all.
pub fn extent_of_indexed(positions: &[DVec3], idx: &[u32]) -> Extent {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &i in idx {
        let p = positions[i as usize].to_array();
        for k in 0..3 {
            let x = p[k];
            if x < min[k] {
                min[k] = x;
            }
            if x > max[k] {
                max[k] = x;
            }
        }
    }
    Extent { min: DVec3::from_array(min), max: DVec3::from_array(max) }
}

/// The smaller of two numbers, NaN when either is — `Math.min`'s rule, which `f64::min`
/// (ignoring a NaN) does not keep.
pub(crate) fn js_min(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.min(b)
    }
}

/// The larger of two numbers, NaN when either is — `Math.max`'s rule, which `f64::max`
/// (ignoring a NaN) does not keep.
pub(crate) fn js_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `Math.round`: the nearest integer, halves rounded up (towards +infinity).
pub(crate) fn js_round(x: f64) -> f64 {
    let f = x.floor();
    if x - f >= 0.5 {
        f + 1.0
    } else {
        f
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eigen3_matches_the_matrix() {
        let a = [4.0, 1.0, 0.5, 1.0, 3.0, 0.2, 0.5, 0.2, 2.0];
        let e = symmetric_eigen3(&a);
        assert!(e.values[0] <= e.values[1] && e.values[1] <= e.values[2]);
        for (lambda, vec) in e.values.iter().zip(e.vectors) {
            let v = vec.to_array();
            for i in 0..3 {
                let av: f64 = (0..3).map(|j| a[i * 3 + j] * v[j]).sum();
                assert!((av - lambda * v[i]).abs() < 1e-10);
            }
            assert!((vec.length() - 1.0).abs() < 1e-12);
        }
    }

    /// `Math.hypot` as Node (V8) returns it, for arguments where the plain root of the sum
    /// of squares rounds differently.
    #[test]
    fn hypot_rounds_as_v8_does() {
        assert_eq!(hypot3(-1.499929562676665, -0.10538663557050126, -0.002331840334149003), 1.5036291010178093);
        assert_eq!(hypot3(-0.3721488126889565, 0.09830171223650763, 0.0015687755898427104), 0.384916129138908);
        assert_eq!(hypot3(-1.076598454069625, -0.46340584939504315, -0.004621107824901635), 1.1721051860631697);
        assert_eq!(hypot3(0.9122360634674485, -0.3494937675304216, 0.0005824911620386375), 0.9768934784965804);
        assert_eq!(hypot2(std::f64::consts::FRAC_1_SQRT_2, std::f64::consts::FRAC_1_SQRT_2), 1.0000000000000002);
        assert_eq!(hypot2(1e200, 1e200), 1.414213562373095e200);
        assert_eq!(js_hypot(&[1.0, 2.0, 3.0, 4.0, 5.0]), 7.416198487095663);
        assert!(hypot3(f64::NAN, f64::INFINITY, 0.0).is_infinite());
        assert!(hypot3(f64::NAN, 1.0, 0.0).is_nan());
        assert_eq!(hypot3(0.0, -0.0, 0.0), 0.0);
        // The frame ScanRuler builds across a direction.
        let (u, v) = ortho_basis(DVec3::new(0.3, -0.5, 0.81));
        assert!(u.dot(v).abs() < 1e-15 && (u.length() - 1.0).abs() < 1e-15);
        assert_eq!(ortho_basis(DVec3::ZERO), (DVec3::Z, DVec3::Z));
    }

    #[test]
    fn js_rounding_rules() {
        assert_eq!(js_round(2.5), 3.0);
        assert_eq!(js_round(-2.5), -2.0);
        assert_eq!(js_round(-2.6), -3.0);
        assert!(js_min(f64::NAN, 1.0).is_nan());
        assert_eq!(js_min(2.0, 1.0), 1.0);
    }
}
