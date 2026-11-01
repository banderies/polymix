// SPDX-License-Identifier: AGPL-3.0-only
//! The B-splines of a patch network: cubic, clamped, with uniform knots over [0, 1].
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkSpline.ts`, and `clampedKnots` from
//! `plugins/reverse/core/surfaceFit.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A curve of `spans` spans has spans + 3 poles, a patch of nu × nv spans (nu + 3) × (nv + 3),
//! row-major with u along the rows' index. Two curves or two patch sides with the same span
//! count have the same knots, so a side whose poles are a curve's poles is that curve
//! exactly. The basis and its first derivative are Cox–de Boor's (Piegl and Tiller, The
//! NURBS Book, A2.3); the knots are Fit Surface's.
//!
//! ScanRuler keeps each span count's knots in a cache; here a knot is worked out where it is
//! read (it is i / spans inside, 0 and 1 at the clamped ends), which gives the same numbers
//! to the bit without the cache.

use glam::DVec3;

pub const DEGREE: usize = 3;

/// The clamped knots of a B-spline of `n` poles and degree `p` over [a, b], the inner ones
/// uniform.
pub fn clamped_knots(n: usize, p: usize, a: f64, b: f64) -> Vec<f64> {
    let m = n + p + 1;
    let mut knots = vec![0.0; m];
    let inner = n as i64 - p as i64 - 1;
    for k in knots.iter_mut().take(p + 1) {
        *k = a;
    }
    for i in 1..=inner.max(0) as usize {
        knots[p + i] = a + ((b - a) * i as f64) / (inner + 1) as f64;
    }
    for k in knots.iter_mut().skip(m.saturating_sub(p + 1)) {
        *k = b;
    }
    knots
}

/// The clamped uniform knots of a cubic of `spans` spans over [0, 1].
pub fn knots_of(spans: usize) -> Vec<f64> {
    clamped_knots(spans + DEGREE, DEGREE, 0.0, 1.0)
}

/// Knot `k` of [`knots_of`]`(spans)`, as it would read it.
#[inline]
fn knot(spans: usize, k: usize) -> f64 {
    if k <= DEGREE {
        0.0
    } else if k >= spans + DEGREE {
        1.0
    } else {
        (k - DEGREE) as f64 / spans as f64
    }
}

/// The span of t among the knots of `spans` spans, without a search: the knots inside are
/// i / spans, so it is DEGREE + ⌊t spans⌋, a step off where t lies within rounding of a knot,
/// which the knots themselves put right. The ends, and NaN, as a search has them.
fn span_of(spans: usize, t: f64) -> usize {
    if !(t < 1.0) {
        return spans + DEGREE - 1;
    }
    if t <= 0.0 {
        return DEGREE;
    }
    let mut s = (spans + DEGREE - 1).min(DEGREE + (t * spans as f64).floor() as usize);
    while s > DEGREE && t < knot(spans, s) {
        s -= 1;
    }
    while s < spans + DEGREE - 1 && t >= knot(spans, s + 1) {
        s += 1;
    }
    s
}

/// The four basis functions non-zero at t and their first derivatives, written to n0 and n1;
/// the span they are of, so pole span − 3 + i takes n0[i]. A2.3 written out for the cubic,
/// without its arrays: the fit evaluates it a few million times a pass, two to a patch
/// point. Its steps are the loop's, in its order, so its numbers are the loop's to the bit.
pub fn basis_with_derivative(spans: usize, t: f64, n0: &mut [f64; 4], n1: &mut [f64; 4]) -> usize {
    let spans = spans.max(1);
    let tt = t.clamp(0.0, 1.0);
    let tt = if t.is_nan() { t } else { tt };
    let s = span_of(spans, tt);
    let k = |i: usize| knot(spans, i);
    let (l1, l2, l3) = (tt - k(s), tt - k(s - 1), tt - k(s - 2));
    let (r1, r2, r3) = (k(s + 1) - tt, k(s + 2) - tt, k(s + 3) - tt);
    // Degree 1 (e), degree 2 (f), then degree 3 over the knot differences a, b, c.
    let mut temp = 1.0 / (r1 + l1);
    let e0 = r1 * temp;
    let e1 = l1 * temp;
    temp = e0 / (r1 + l2);
    let f0 = r1 * temp;
    let mut saved = l2 * temp;
    temp = e1 / (r2 + l1);
    let f1 = saved + r2 * temp;
    let f2 = l1 * temp;
    let (a, b, c) = (r1 + l3, r2 + l2, r3 + l1);
    temp = f0 / a;
    n0[0] = r1 * temp;
    saved = l3 * temp;
    temp = f1 / b;
    n0[1] = saved + r2 * temp;
    saved = l2 * temp;
    temp = f2 / c;
    n0[2] = saved + r3 * temp;
    n0[3] = l1 * temp;
    // The derivatives from the degree-2 functions: 3 (f[r − 1] / a[r − 1] − f[r] / a[r]),
    // the loop's 0 − x kept where x may be +0.
    let (fa, fb, fc) = (f0 / a, f1 / b, f2 / c);
    n1[0] = 3.0 * (0.0 - fa);
    n1[1] = 3.0 * (fa - fb);
    n1[2] = 3.0 * (fb - fc);
    n1[3] = 3.0 * fc;
    s
}

/// The basis functions alone, as [`basis_with_derivative`]'s first half: the Cox–de Boor
/// recurrence without its arrays a call.
pub fn basis_at(spans: usize, t: f64, out: &mut [f64; 4]) -> usize {
    let spans = spans.max(1);
    let tt = if t.is_nan() { t } else { t.clamp(0.0, 1.0) };
    let s = span_of(spans, tt);
    let mut left = [0.0f64; DEGREE + 1];
    let mut right = [0.0f64; DEGREE + 1];
    out[0] = 1.0;
    for j in 1..=DEGREE {
        left[j] = tt - knot(spans, s + 1 - j);
        right[j] = knot(spans, s + j) - tt;
        let mut saved = 0.0;
        for r in 0..j {
            let temp = out[r] / (right[r + 1] + left[j - r]);
            out[r] = saved + right[r + 1] * temp;
            saved = left[j - r] * temp;
        }
        out[j] = saved;
    }
    s
}

/// A curve's point at t, poles flat (x, y, z per pole).
pub fn curve_point(poles: &[f64], spans: usize, t: f64) -> DVec3 {
    let mut b0 = [0.0; 4];
    let s = basis_at(spans, t, &mut b0);
    let mut out = [0.0f64; 3];
    for (i, w) in b0.iter().enumerate() {
        let at = (s - DEGREE + i) * 3;
        if at + 2 >= poles.len() {
            continue;
        }
        out[0] += w * poles[at];
        out[1] += w * poles[at + 1];
        out[2] += w * poles[at + 2];
    }
    DVec3::from_array(out)
}

/// A curve's derivative at t.
pub fn curve_derivative(poles: &[f64], spans: usize, t: f64) -> DVec3 {
    let mut b0 = [0.0; 4];
    let mut b1 = [0.0; 4];
    let s = basis_with_derivative(spans, t, &mut b0, &mut b1);
    let mut out = [0.0f64; 3];
    for (i, w) in b1.iter().enumerate() {
        let at = (s - DEGREE + i) * 3;
        if at + 2 >= poles.len() {
            continue;
        }
        out[0] += w * poles[at];
        out[1] += w * poles[at + 1];
        out[2] += w * poles[at + 2];
    }
    DVec3::from_array(out)
}

/// A patch: its spans each way and its poles, (nu + 3) rows of (nv + 3), x, y, z each.
#[derive(Debug, Clone, PartialEq)]
pub struct PatchNet {
    pub nu: usize,
    pub nv: usize,
    pub poles: Vec<f64>,
}

/// A patch's point and first derivatives at (u, v): (p, du, dv).
pub fn patch_point(net: &PatchNet, u: f64, v: f64) -> (DVec3, DVec3, DVec3) {
    let (mut u0, mut u1, mut v0, mut v1) = ([0.0; 4], [0.0; 4], [0.0; 4], [0.0; 4]);
    let su = basis_with_derivative(net.nu, u, &mut u0, &mut u1);
    let sv = basis_with_derivative(net.nv, v, &mut v0, &mut v1);
    let cols = net.nv + DEGREE;
    let poles = &net.poles;
    // The sums in locals, written out once; the derivatives' are cheap beside the basis and
    // summed whether asked for or not.
    let (mut px, mut py, mut pz) = (0.0, 0.0, 0.0);
    let (mut ux, mut uy, mut uz) = (0.0, 0.0, 0.0);
    let (mut vx, mut vy, mut vz) = (0.0, 0.0, 0.0);
    for a in 0..=DEGREE {
        let row = (su - DEGREE + a) * cols;
        let ua = u0[a];
        let ud = u1[a];
        for b in 0..=DEGREE {
            let at = (row + sv - DEGREE + b) * 3;
            if at + 2 >= poles.len() {
                continue;
            }
            let (x, y, z) = (poles[at], poles[at + 1], poles[at + 2]);
            let w = ua * v0[b];
            px += w * x;
            py += w * y;
            pz += w * z;
            let wu = ud * v0[b];
            ux += wu * x;
            uy += wu * y;
            uz += wu * z;
            let wv = ua * v1[b];
            vx += wv * x;
            vy += wv * y;
            vz += wv * z;
        }
    }
    (DVec3::new(px, py, pz), DVec3::new(ux, uy, uz), DVec3::new(vx, vy, vz))
}

/// The patch's unit normal at (u, v): du × dv.
pub fn patch_normal(net: &PatchNet, u: f64, v: f64) -> DVec3 {
    let (_, du, dv) = patch_point(net, u, v);
    let out = DVec3::new(du.y * dv.z - du.z * dv.y, du.z * dv.x - du.x * dv.z, du.x * dv.y - du.y * dv.x);
    let l = out.length();
    let l = if l != 0.0 && !l.is_nan() { l } else { 1.0 };
    out / l
}

/// The Greville abscissae of a cubic of `spans` spans: where each pole "is" along [0, 1].
pub fn greville_of(spans: usize) -> Vec<f64> {
    let n = spans + DEGREE;
    (0..n).map(|i| (knot(spans, i + 1) + knot(spans, i + 2) + knot(spans, i + 3)) / 3.0).collect()
}

/// The weights of the `order`-th divided difference over x[from …], scaled by order! so it
/// reads as the derivative (as Fit Surface's).
pub fn divided_weights(x: &[f64], from: usize, order: usize) -> Vec<f64> {
    let mut fact = 1.0;
    for k in 2..=order {
        fact *= k as f64;
    }
    (0..=order)
        .map(|k| {
            let mut denom = 1.0;
            for j in 0..=order {
                if j != k {
                    denom *= x[from + k] - x[from + j];
                }
            }
            fact / denom
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The general recurrence over the knots, as Fit Surface's `basisFunctions` has it, for
    /// the written-out cubic to be checked against.
    fn reference(spans: usize, t: f64) -> (usize, [f64; 4]) {
        let knots = knots_of(spans);
        let n = spans + DEGREE;
        // findSpan: the last knot interval holding t.
        let s = if t >= knots[n] {
            n - 1
        } else {
            let mut s = DEGREE;
            while t >= knots[s + 1] {
                s += 1;
            }
            s
        };
        let mut out = [0.0; 4];
        let mut left = [0.0; 4];
        let mut right = [0.0; 4];
        out[0] = 1.0;
        for j in 1..=DEGREE {
            left[j] = t - knots[s + 1 - j];
            right[j] = knots[s + j] - t;
            let mut saved = 0.0;
            for r in 0..j {
                let temp = out[r] / (right[r + 1] + left[j - r]);
                out[r] = saved + right[r + 1] * temp;
                saved = left[j - r] * temp;
            }
            out[j] = saved;
        }
        (s, out)
    }

    #[test]
    fn the_written_out_cubic_is_the_recurrence() {
        for spans in [2usize, 3, 7, 20] {
            for i in 0..=200 {
                let t = i as f64 / 200.0;
                let (s, want) = reference(spans, t);
                let mut n0 = [0.0; 4];
                let mut n1 = [0.0; 4];
                assert_eq!(basis_with_derivative(spans, t, &mut n0, &mut n1), s);
                let mut b = [0.0; 4];
                assert_eq!(basis_at(spans, t, &mut b), s);
                for k in 0..4 {
                    assert_eq!(n0[k], want[k], "spans {spans}, t {t}");
                    assert_eq!(b[k], want[k], "spans {spans}, t {t}");
                }
                assert!((n1.iter().sum::<f64>()).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn a_curve_on_a_line_runs_at_its_greville_points() {
        let spans = 5;
        let g = greville_of(spans);
        let poles: Vec<f64> = g.iter().flat_map(|&x| [x * 10.0, 2.0, -x]).collect();
        for i in 0..=50 {
            let t = i as f64 / 50.0;
            let p = curve_point(&poles, spans, t);
            assert!((p - DVec3::new(10.0 * t, 2.0, -t)).length() < 1e-12);
            let d = curve_derivative(&poles, spans, t);
            assert!((d - DVec3::new(10.0, 0.0, -1.0)).length() < 1e-9);
        }
        assert_eq!(divided_weights(&[0.0, 1.0, 2.0, 3.0], 0, 3), vec![-1.0, 3.0, -3.0, 1.0]);
    }

    #[test]
    fn a_flat_patch_has_its_plane_normal() {
        let (nu, nv) = (3, 4);
        let gu = greville_of(nu);
        let gv = greville_of(nv);
        let mut poles = Vec::new();
        for &u in &gu {
            for &v in &gv {
                poles.extend_from_slice(&[u * 20.0, v * 10.0, 1.0]);
            }
        }
        let net = PatchNet { nu, nv, poles };
        let (p, du, dv) = patch_point(&net, 0.3, 0.6);
        assert!((p - DVec3::new(6.0, 6.0, 1.0)).length() < 1e-12);
        assert!((du - DVec3::new(20.0, 0.0, 0.0)).length() < 1e-9);
        assert!((dv - DVec3::new(0.0, 10.0, 0.0)).length() < 1e-9);
        assert!((patch_normal(&net, 0.3, 0.6) - DVec3::Z).length() < 1e-12);
    }
}
