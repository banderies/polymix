//! A small radix-2 complex FFT (lengths that are powers of two), enough for histograms.

use std::f64::consts::PI;

/// In-place transform of (`re`, `im`); `inverse` divides by the length.
pub fn fft(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    assert!(n.is_power_of_two() && im.len() == n, "FFT length must be a power of two");
    let mut j = 0;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            re.swap(i, j);
            im.swap(i, j);
        }
    }
    let mut len = 2;
    while len <= n {
        let ang = if inverse { 2.0 } else { -2.0 } * PI / len as f64;
        let (wr, wi) = (ang.cos(), ang.sin());
        for start in (0..n).step_by(len) {
            let (mut cr, mut ci) = (1.0, 0.0);
            for k in 0..len / 2 {
                let (a, b) = (start + k, start + k + len / 2);
                let (vr, vi) = (re[b] * cr - im[b] * ci, re[b] * ci + im[b] * cr);
                (re[b], im[b]) = (re[a] - vr, im[a] - vi);
                re[a] += vr;
                im[a] += vi;
                (cr, ci) = (cr * wr - ci * wi, cr * wi + ci * wr);
            }
        }
        len <<= 1;
    }
    if inverse {
        let s = 1.0 / n as f64;
        re.iter_mut().chain(im.iter_mut()).for_each(|x| *x *= s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_the_direct_transform_and_inverts() {
        let n = 16;
        let x: Vec<f64> = (0..n).map(|k| ((k * 7) % 5) as f64 - 1.5).collect();
        let (mut re, mut im) = (x.clone(), vec![0.0; n]);
        fft(&mut re, &mut im, false);
        for f in 0..n {
            let (dr, di) = (0..n).fold((0.0, 0.0), |(a, b), k| {
                let t = -2.0 * PI * (f * k) as f64 / n as f64;
                (a + x[k] * t.cos(), b + x[k] * t.sin())
            });
            assert!((re[f] - dr).abs() < 1e-9 && (im[f] - di).abs() < 1e-9);
        }
        fft(&mut re, &mut im, true);
        assert!(re.iter().zip(&x).all(|(a, b)| (a - b).abs() < 1e-9));
    }
}
