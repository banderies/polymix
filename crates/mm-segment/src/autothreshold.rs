//! Automatic thresholds from a histogram: the eleven methods 3D Slicer's Threshold effect
//! offers, each written here from its paper.
//!
//! Every method looks at a histogram of counts and picks a bin `t`: bins `0..=t` are the
//! background (darker class) and bins `t + 1..` the foreground (brighter class). Only splits
//! with something on both sides are considered, so `t` runs from the first non-empty bin to
//! the one before the last. Where the grey level of a bin matters it is the bin's number
//! (from 1 for Li, whose logarithms need positive levels). Methods that score each split
//! (Otsu, Huang, Kittler–Illingworth, maximum entropy, Moments, Rényi, Shanbhag, Yen) score
//! only the distinct splits (after each non-empty bin) and, as a run of empty bins changes
//! nothing, put the threshold in the middle of the empty run after the best one. Huang and
//! Shanbhag take time quadratic in the number of non-empty bins, the rest linear.
//!
//! The criteria behave as published, quirks included: the entropy ones (maximum entropy,
//! Rényi, Yen) can move a thinly filled edge bin of a big class into a small one rather than
//! cut in the valley, and Shanbhag balances how spread out the two classes are.
//!
//! - **Otsu**: the most variance between the classes, `P0 P1 (μ0 − μ1)²`. N. Otsu, "A
//!   threshold selection method from gray-level histograms", IEEE Trans. Systems, Man, and
//!   Cybernetics 9(1):62–66, 1979.
//! - **Huang**: the least fuzziness. Each level belongs to its class by `1 / (1 + |g − μ| / C)`
//!   (μ its class mean, C the range of levels present); the fuzziness is the mean of Shannon's
//!   function `−μ ln μ − (1 − μ) ln(1 − μ)` of those memberships. L.-K. Huang & M.-J. J. Wang,
//!   "Image thresholding by minimizing the measure of fuzziness", Pattern Recognition
//!   28(1):41–51, 1995.
//! - **IsoData**: iterative intermeans; from the mean level, the threshold moves to halfway
//!   between the two class means until it stays put. T. W. Ridler & S. Calvard, "Picture
//!   thresholding using an iterative selection method", IEEE Trans. Systems, Man, and
//!   Cybernetics 8(8):630–632, 1978.
//! - **Kittler–Illingworth**: minimum error; fits a Gaussian to each class and minimises
//!   `Σ P_k (ln σ_k² − 2 ln P_k)`. Each class variance has the bin width's own variance (1/12)
//!   added, as if each bin's counts were spread evenly over it, so a class in a single bin
//!   stays finite. J. Kittler & J. Illingworth, "Minimum error thresholding", Pattern
//!   Recognition 19(1):41–47, 1986.
//! - **Li**: minimum cross entropy between the image and its two-level version (C. H. Li &
//!   C. K. Lee, "Minimum cross entropy thresholding", Pattern Recognition 26(4):617–625, 1993),
//!   found by Li & Tam's iteration: the threshold moves to the logarithmic mean of the class
//!   means, `(μ1 − μ0) / (ln μ1 − ln μ0)`, where the criterion's derivative vanishes (C. H. Li
//!   & P. K. S. Tam, "An iterative algorithm for minimum cross entropy thresholding", Pattern
//!   Recognition Letters 19(8):771–776, 1998). It starts from the mean level and finds the
//!   nearest such point, not necessarily the global minimum.
//! - **Maximum entropy**: the most Shannon entropy in the two class distributions together.
//!   J. N. Kapur, P. K. Sahoo & A. K. C. Wong, "A new method for gray-level picture
//!   thresholding using the entropy of the histogram", Computer Vision, Graphics, and Image
//!   Processing 29(3):273–285, 1985.
//! - **Moments**: the two-level image keeping the first three moments; its share `p0` of dark
//!   pixels is solved in closed form and the threshold is the split whose background share is
//!   closest to `p0`. W.-H. Tsai, "Moment-preserving thresholding: a new approach", Computer
//!   Vision, Graphics, and Image Processing 29(3):377–393, 1985.
//! - **Rényi**: the most Rényi entropy of order α in the two classes together. Order 1 is
//!   maximum entropy and order 2 is Yen; Sahoo et al. take thresholds for α below 1, at 1 and
//!   above 1 (here α = 0.5, 1, 2), sort them as t1 ≤ t2 ≤ t3 and blend them as
//!   `t1 (P(t1) + ω β1 / 4) + t2 ω β2 / 4 + t3 (1 − P(t3) + ω β3 / 4)`, with P the background
//!   share, `ω = P(t3) − P(t1)` and `β = (0, 1, 3)` when only t1 and t2 are close, `(3, 1, 0)`
//!   when only t2 and t3 are, and `(1, 2, 1)` otherwise; "close" is within 5 levels of 256,
//!   scaled here to the bin count. P. K. Sahoo, C. Wilkins & J. Yeager, "Threshold selection
//!   using Renyi's entropy", Pattern Recognition 30(1):71–84, 1997 (blend as restated by
//!   M. Sezgin & B. Sankur, "Survey over image thresholding techniques and quantitative
//!   performance evaluation", Journal of Electronic Imaging 13(1), 2004).
//! - **Shanbhag**: each class as a fuzzy set whose membership falls to ½ at the threshold: a
//!   background level `g` belongs by `½ + ½ P(g..=t) / P0`, a foreground level by
//!   `½ + ½ P(t+1..=g) / P1`. The threshold is where the two classes' information
//!   `−Σ (p_g / P_k) ln(membership)` is most nearly equal. A. G. Shanbhag, "Utilization of
//!   information measure as a means of image thresholding", CVGIP: Graphical Models and Image
//!   Processing 56(5):414–419, 1994.
//! - **Triangle**: a line from the histogram's peak to the zero just past the far end of its
//!   longer tail; the threshold is the bin furthest below the line (the knee of the tail), on
//!   the peak's side. Either tail works. G. W. Zack, W. E. Rogers & S. A. Latt, "Automatic
//!   measurement of sister chromatid exchange frequency", Journal of Histochemistry &
//!   Cytochemistry 25(7):741–753, 1977.
//! - **Yen**: the maximum correlation criterion, `−ln Σ (p_g / P0)² − ln Σ (p_g / P1)²` over
//!   each class. J.-C. Yen, F.-J. Chang & S. Chang, "A new criterion for automatic multilevel
//!   thresholding", IEEE Trans. Image Processing 4(3):370–378, 1995.

use rayon::prelude::*;

/// Automatic threshold methods (as offered by 3D Slicer's Threshold effect).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AutoThreshold {
    /// Most variance between the classes (Otsu 1979).
    Otsu,
    /// Least fuzziness, with Shannon's function (Huang & Wang 1995).
    Huang,
    /// Iterative intermeans (Ridler & Calvard 1978).
    IsoData,
    /// Minimum error between two fitted Gaussians (Kittler & Illingworth 1986).
    KittlerIllingworth,
    /// Minimum cross entropy, iterative (Li & Lee 1993, Li & Tam 1998).
    Li,
    /// Most entropy in the two classes (Kapur, Sahoo & Wong 1985).
    MaxEntropy,
    /// Moment preserving (Tsai 1985).
    Moments,
    /// Rényi entropy of three orders, blended (Sahoo, Wilkins & Yeager 1997).
    Renyi,
    /// Balanced fuzzy information (Shanbhag 1994).
    Shanbhag,
    /// The knee of the longer tail (Zack, Rogers & Latt 1977).
    Triangle,
    /// Maximum correlation (Yen, Chang & Chang 1995).
    Yen,
}

impl AutoThreshold {
    /// Every method, in menu order.
    pub const ALL: [AutoThreshold; 11] = [
        Self::Otsu,
        Self::Huang,
        Self::IsoData,
        Self::KittlerIllingworth,
        Self::Li,
        Self::MaxEntropy,
        Self::Moments,
        Self::Renyi,
        Self::Shanbhag,
        Self::Triangle,
        Self::Yen,
    ];

    /// The name to show.
    pub fn label(self) -> &'static str {
        match self {
            Self::Otsu => "Otsu",
            Self::Huang => "Huang (fuzzy entropy)",
            Self::IsoData => "IsoData (Ridler–Calvard)",
            Self::KittlerIllingworth => "Kittler–Illingworth (minimum error)",
            Self::Li => "Li (minimum cross entropy)",
            Self::MaxEntropy => "Maximum entropy (Kapur)",
            Self::Moments => "Moments (Tsai)",
            Self::Renyi => "Rényi entropy",
            Self::Shanbhag => "Shanbhag",
            Self::Triangle => "Triangle",
            Self::Yen => "Yen",
        }
    }

    /// A short lowercase key for scripts and agents.
    pub fn key(self) -> &'static str {
        match self {
            Self::Otsu => "otsu",
            Self::Huang => "huang",
            Self::IsoData => "isodata",
            Self::KittlerIllingworth => "kittler",
            Self::Li => "li",
            Self::MaxEntropy => "maxentropy",
            Self::Moments => "moments",
            Self::Renyi => "renyi",
            Self::Shanbhag => "shanbhag",
            Self::Triangle => "triangle",
            Self::Yen => "yen",
        }
    }

    /// The method with this [`key`](Self::key) (ignoring case).
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.key().eq_ignore_ascii_case(key.trim()))
    }
}

/// The threshold bin for a histogram of counts: voxels in bins > the returned index are the
/// foreground (brighter class). `None` when the histogram cannot be split (empty, or all counts in one bin).
pub fn auto_threshold(method: AutoThreshold, hist: &[u64]) -> Option<usize> {
    let h = Hist::new(hist)?;
    let t = match method {
        AutoThreshold::Otsu => h.best_split(|a| {
            let ([p0, s0, _], [p1, s1, _]) = (h.below[a], h.above[a]);
            p0 * p1 * (s0 / p0 - s1 / p1).powi(2)
        })?,
        AutoThreshold::Huang => h.huang()?,
        AutoThreshold::IsoData => h.iterate(0.0, |m0, m1| 0.5 * (m0 + m1)),
        AutoThreshold::KittlerIllingworth => h.best_split(|a| {
            let cost = |[p, s, s2]: [f64; 3]| {
                let m = s / p;
                p * (((s2 / p - m * m).max(0.0) + 1.0 / 12.0).ln() - 2.0 * p.ln())
            };
            -cost(h.below[a]) - cost(h.above[a])
        })?,
        AutoThreshold::Li => h.iterate(1.0, |m0, m1| (m1 - m0) / (m1.ln() - m0.ln())),
        AutoThreshold::MaxEntropy => h.entropy(1.0)?,
        AutoThreshold::Moments => h.moments()?,
        AutoThreshold::Renyi => h.renyi()?,
        AutoThreshold::Shanbhag => h.shanbhag()?,
        AutoThreshold::Triangle => h.triangle(),
        AutoThreshold::Yen => h.entropy(2.0)?,
    };
    Some(t.clamp(h.first(), h.last() - 1))
}

/// The threshold value for raw values (finite ones only; NaN/inf ignored), using a `bins`-bin
/// histogram over their range: values above it are the foreground. `None` as above.
pub fn auto_threshold_values(method: AutoThreshold, values: &[f32], bins: usize) -> Option<f32> {
    let range = |(a, b): (f32, f32), (c, d): (f32, f32)| (a.min(c), b.max(d));
    let none = || (f32::INFINITY, f32::NEG_INFINITY);
    let (lo, hi) = values.par_iter().filter(|v| v.is_finite()).fold(none, |r, &v| range(r, (v, v))).reduce(none, range);
    if bins < 2 || hi <= lo {
        return None;
    }
    let (lo, scale) = (lo as f64, bins as f64 / (hi as f64 - lo as f64));
    // Bin t holds (lo + t w, lo + (t + 1) w] (bin 0 also lo itself), so values in bins up to t
    // are exactly those at or below bin t's upper edge.
    let bin = |v: f32| ((((v as f64 - lo) * scale).ceil() as usize).saturating_sub(1)).min(bins - 1);
    let hist = values
        .par_chunks(1 << 16)
        .fold(
            || vec![0u64; bins],
            |mut h, chunk| {
                chunk.iter().filter(|v| v.is_finite()).for_each(|&v| h[bin(v)] += 1);
                h
            },
        )
        .reduce(
            || vec![0u64; bins],
            |mut a, b| {
                a.iter_mut().zip(b).for_each(|(x, y)| *x += y);
                a
            },
        );
    let t = auto_threshold(method, &hist)?;
    Some((lo + (t + 1) as f64 / scale) as f32)
}

/// A histogram as probabilities, with the class sums every method shares.
struct Hist {
    /// Probability of each bin.
    p: Vec<f64>,
    /// The non-empty bins, in order (at least two).
    nz: Vec<usize>,
    /// Per bin `a`: probability, first and second moment of the bin number over bins `0..=a`
    /// (`below`) and over the bins after `a` (`above`, summed from the top end, so a small
    /// bright class keeps its precision).
    below: Vec<[f64; 3]>,
    above: Vec<[f64; 3]>,
}

impl Hist {
    fn new(hist: &[u64]) -> Option<Self> {
        let nz: Vec<usize> = (0..hist.len()).filter(|&g| hist[g] > 0).collect();
        if nz.len() < 2 {
            return None;
        }
        let total = hist.iter().map(|&c| c as f64).sum::<f64>();
        let p: Vec<f64> = hist.iter().map(|&c| c as f64 / total).collect();
        let term = |g: usize| [p[g], g as f64 * p[g], g as f64 * g as f64 * p[g]];
        let add = |a: [f64; 3], b: [f64; 3]| [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
        let below = (0..p.len()).scan([0.0; 3], |s, g| { *s = add(*s, term(g)); Some(*s) }).collect();
        let mut above = vec![[0.0; 3]; p.len()];
        for g in (0..p.len() - 1).rev() {
            above[g] = add(above[g + 1], term(g + 1));
        }
        Some(Self { p, nz, below, above })
    }

    fn first(&self) -> usize {
        self.nz[0]
    }

    fn last(&self) -> usize {
        self.nz[self.nz.len() - 1]
    }

    /// The class means (of the bin number) when bins `0..=a` are the background.
    fn means(&self, a: usize) -> (f64, f64) {
        (self.below[a][1] / self.below[a][0], self.above[a][1] / self.above[a][0])
    }

    /// `(Σ f(p_g) over g ≤ a, over g > a)` per bin `a`, for the non-empty bins only.
    fn split_sums(&self, f: impl Fn(f64) -> f64) -> (Vec<f64>, Vec<f64>) {
        let v: Vec<f64> = self.p.iter().map(|&p| if p > 0.0 { f(p) } else { 0.0 }).collect();
        let below: Vec<f64> = v.iter().scan(0.0, |s, &x| { *s += x; Some(*s) }).collect();
        let mut above = vec![0.0; v.len()];
        for g in (0..v.len() - 1).rev() {
            above[g] = above[g + 1] + v[g + 1];
        }
        (below, above)
    }

    /// The best-scoring split (highest `score(a)`, `a` the last background bin), placed in the
    /// middle of the empty bins that follow it, where every split scores the same.
    fn best_split(&self, score: impl Fn(usize) -> f64) -> Option<usize> {
        argmax(0..self.nz.len() - 1, |j| score(self.nz[j])).map(|j| (self.nz[j] + self.nz[j + 1] - 1) / 2)
    }

    /// Fixed-point iteration from the mean level: the background is the bins whose level
    /// (bin number + `offset`) is at most `next(μ0, μ1)` of the class mean levels.
    fn iterate(&self, offset: f64, next: impl Fn(f64, f64) -> f64) -> usize {
        let (first, last) = (self.first() as f64, (self.last() - 1) as f64);
        let at = |level: f64| (level - offset).floor().clamp(first, last) as usize;
        let mut t = at(self.below[self.p.len() - 1][1] + offset);
        for _ in 0..1000 {
            let (m0, m1) = self.means(t);
            let n = at(next(m0 + offset, m1 + offset));
            if n == t {
                break;
            }
            t = n;
        }
        t
    }

    fn huang(&self) -> Option<usize> {
        let c = (self.last() - self.first()) as f64;
        let shannon = |m: f64| if m >= 1.0 { 0.0 } else { -m * m.ln() - (1.0 - m) * (1.0 - m).ln() };
        self.best_split(|a| {
            let (m0, m1) = self.means(a);
            let fuzz = |g: usize| shannon(1.0 / (1.0 + (g as f64 - if g <= a { m0 } else { m1 }).abs() / c)) * self.p[g];
            -self.nz.iter().map(|&g| fuzz(g)).sum::<f64>()
        })
    }

    /// The split with the most Rényi entropy of order `alpha` in the two classes together
    /// (Shannon's at 1).
    fn entropy(&self, alpha: f64) -> Option<usize> {
        let shannon = alpha == 1.0;
        let (lo, hi) = self.split_sums(|p| if shannon { p * p.ln() } else { p.powf(alpha) });
        // A class with probability q and sum s of p ln p (or of p^α).
        let h = |s: f64, q: f64| if shannon { q.ln() - s / q } else { (s.ln() - alpha * q.ln()) / (1.0 - alpha) };
        self.best_split(|a| h(lo[a], self.below[a][0]) + h(hi[a], self.above[a][0]))
    }

    fn moments(&self) -> Option<usize> {
        // Central moments of the level scaled to [0, 1), for conditioning.
        let l = self.p.len() as f64;
        let mean = self.below[self.p.len() - 1][1] / l;
        let m = |k: i32| self.nz.iter().map(|&g| self.p[g] * (g as f64 / l - mean).powi(k)).sum::<f64>();
        let (m2, m3) = (m(2), m(3));
        // Tsai's two levels z0 < z1 solve z² + c1 z + c0 = 0 with c0 = (m1 m3 − m2²) / (m2 − m1²)
        // and c1 = (m1 m2 − m3) / (m2 − m1²); with m1 = 0 that is z² − (m3 / m2) z − m2, whose
        // roots are always real and apart, and p0 = (z1 − m1) / (z1 − z0).
        let c = m3 / m2;
        let root = (c * c + 4.0 * m2).sqrt();
        let p0 = 0.5 * (c + root) / root;
        self.best_split(|a| -(self.below[a][0] - p0).abs())
    }

    fn renyi(&self) -> Option<usize> {
        let mut t = [self.entropy(0.5)?, self.entropy(1.0)?, self.entropy(2.0)?];
        t.sort_unstable();
        let (p1, p3) = (self.below[t[0]][0], self.below[t[2]][0]);
        let w = p3 - p1;
        let near = 5.0 * self.p.len() as f64 / 256.0;
        let beta = match ((t[1] - t[0]) as f64 <= near, (t[2] - t[1]) as f64 <= near) {
            (true, false) => [0.0, 1.0, 3.0],
            (false, true) => [3.0, 1.0, 0.0],
            _ => [1.0, 2.0, 1.0],
        };
        let [a, b, c] = t.map(|t| t as f64);
        let blend = a * (p1 + 0.25 * w * beta[0]) + 0.25 * b * w * beta[1] + c * (1.0 - p3 + 0.25 * w * beta[2]);
        Some(blend.round() as usize)
    }

    fn shanbhag(&self) -> Option<usize> {
        self.best_split(|a| {
            let (p0, p1, cum) = (self.below[a][0], self.above[a][0], |g: usize| self.below[g][0]);
            let (mut info0, mut info1) = (0.0, 0.0);
            for &g in &self.nz {
                let q = self.p[g];
                if g <= a {
                    // The background's share from g (included) up to the threshold.
                    info0 -= q / p0 * (0.5 + 0.5 * (cum(a) - cum(g) + q) / p0).ln();
                } else {
                    info1 -= q / p1 * (0.5 + 0.5 * (cum(g) - cum(a)) / p1).ln();
                }
            }
            -(info0 - info1).abs()
        })
    }

    fn triangle(&self) -> usize {
        let (first, last) = (self.first(), self.last());
        let peak = self.nz.iter().map(|&g| self.p[g]).fold(0.0, f64::max);
        let lo_peak = self.nz.iter().copied().find(|&g| self.p[g] == peak).unwrap_or(first);
        let hi_peak = self.nz.iter().copied().rfind(|&g| self.p[g] == peak).unwrap_or(last);
        // How far bin g lies below the line from (top, peak) to (end, 0), up to a constant
        // factor; searched outwards from the peak, so the nearest wins ties.
        let below_line = |g: usize, top: usize, end: f64| peak * (end - g as f64) / (end - top as f64) - self.p[g];
        if last - hi_peak >= lo_peak - first {
            // Bright tail: the knee stays with the peak, in the background.
            let end = last as f64 + 1.0;
            argmax(hi_peak + 1..=last, |g| below_line(g, hi_peak, end)).unwrap_or(hi_peak)
        } else {
            // Dark tail: the knee stays with the peak, in the foreground.
            let end = first as f64 - 1.0;
            argmax((first..lo_peak).rev(), |g| below_line(g, lo_peak, end)).map_or(lo_peak, |g| g.saturating_sub(1))
        }
    }
}

/// The first of `items` with the highest finite `score`.
fn argmax(items: impl Iterator<Item = usize>, score: impl Fn(usize) -> f64) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for i in items {
        let s = score(i);
        if s.is_finite() && best.is_none_or(|(b, _)| s > b) {
            best = Some((s, i));
        }
    }
    best.map(|(_, i)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two Gaussian humps: 3000 high at bin 70 (σ 12) and 1500 high at bin 170 (σ 15).
    fn bimodal() -> Vec<u64> {
        let hump = |g: usize, m: f64, s: f64, a: f64| a * (-(g as f64 - m).powi(2) / (2.0 * s * s)).exp();
        (0..256).map(|g| (hump(g, 70.0, 12.0, 3000.0) + hump(g, 170.0, 15.0, 1500.0)).round() as u64).collect()
    }

    /// A peak at bin 40 with a long exponential tail towards the top.
    fn skewed() -> Vec<u64> {
        (0..256).map(|g| { let x = g as f64 - 40.0; 5000.0 * if x < 0.0 { (-x * x / 50.0).exp() } else { (-x / 35.0).exp() } }.round() as u64).collect()
    }

    #[test]
    fn bimodal_splits_between_the_humps() {
        use AutoThreshold::*;
        let h = bimodal();
        let valley = (95..=145).min_by_key(|&g| h[g]).unwrap();
        for m in AutoThreshold::ALL {
            let t = auto_threshold(m, &h).unwrap();
            // Between the humps, clear of both peaks.
            assert!(t > 80 && t < 160, "{}: {t}", m.key());
            match m {
                Otsu | Huang | IsoData | KittlerIllingworth | Li | Moments => assert!(t.abs_diff(valley) <= 12, "{} not in the valley ({valley}): {t}", m.key()),
                // The knee at the foot of the taller hump.
                Triangle => assert!(t > 85 && t < valley, "triangle: {t}"),
                // The entropy criteria favour the foot of the taller hump (Kapur, Rényi, Yen),
                // Shanbhag the foot of the wider one (it balances how spread the classes are).
                MaxEntropy | Renyi | Yen => assert!(t < valley, "{}: {t}", m.key()),
                Shanbhag => assert!(t > valley, "shanbhag: {t}"),
            }
        }
    }

    #[test]
    fn triangle_cuts_the_tail_on_either_side() {
        let h = skewed();
        let t = auto_threshold(AutoThreshold::Triangle, &h).unwrap();
        assert!(t > 40 && t < 160, "{t}");
        let mut r = h.clone();
        r.reverse();
        let u = auto_threshold(AutoThreshold::Triangle, &r).unwrap();
        assert!(u < 215 && u > 95, "{u}");
        // Mirrored, the knee bin stays with the peak.
        assert_eq!(u, 255 - t - 1);
    }

    #[test]
    fn values_split_two_clusters_ignoring_non_finite() {
        // Integer intensities, one per bin: 3000 voxels evenly over 0..=29 and 1020 over
        // 226..=255. (Evenly filled: the entropy criteria would move a thinly filled edge bin of
        // the big cluster into the small class, a property of the criteria themselves.)
        let mut v: Vec<f32> = (0..3000).map(|k| (k % 30) as f32).collect();
        v.extend((0..1020).map(|k| (226 + k % 30) as f32));
        v.extend([f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::NAN]);
        for m in AutoThreshold::ALL {
            let t = auto_threshold_values(m, &v, 256).unwrap();
            assert!(v.iter().filter(|x| x.is_finite()).all(|&x| (x > t) == (x > 100.0)), "{}: {t}", m.key());
        }
        // Score-based methods land in the middle of the gap.
        assert_eq!(auto_threshold_values(AutoThreshold::Otsu, &v, 256), Some(127.5));
        // A threshold between two values is above the low one.
        for m in AutoThreshold::ALL {
            let t = auto_threshold_values(m, &[1.0, 2.0, f32::NAN], 2).unwrap();
            assert!((1.0..2.0).contains(&t), "{}: {t}", m.key());
        }
    }

    #[test]
    fn degenerate_inputs() {
        for m in AutoThreshold::ALL {
            for h in [&[][..], &[0; 10], &[0, 0, 7, 0], &[5]] {
                assert_eq!(auto_threshold(m, h), None, "{} {h:?}", m.key());
            }
            for h in [&[5u64, 7][..], &[7, 5], &[1, 1], &[0, 5, 0, 0, 7, 0], &[u64::MAX / 4, 1, 0, 1]] {
                let (first, last) = (h.iter().position(|&c| c > 0).unwrap(), h.iter().rposition(|&c| c > 0).unwrap());
                let t = auto_threshold(m, h).unwrap();
                assert!(t >= first && t < last, "{} {h:?}: {t}", m.key());
            }
            assert_eq!(auto_threshold_values(m, &[], 256), None);
            assert_eq!(auto_threshold_values(m, &[3.0; 10], 256), None);
            assert_eq!(auto_threshold_values(m, &[f32::NAN, f32::INFINITY, 1.0], 256), None);
            assert_eq!(auto_threshold_values(m, &[1.0, 2.0], 1), None);
        }
        // Score-based methods centre the threshold in the empty bins between two spikes.
        assert_eq!(auto_threshold(AutoThreshold::Otsu, &[0, 5, 0, 0, 7, 0]), Some(2));
    }

    #[test]
    fn keys_are_unique() {
        for (i, a) in AutoThreshold::ALL.iter().enumerate() {
            assert_eq!(AutoThreshold::from_key(a.key()), Some(*a));
            assert!(!a.label().is_empty());
            for b in &AutoThreshold::ALL[i + 1..] {
                assert!(a != b && a.key() != b.key() && a.label() != b.label());
            }
        }
        assert_eq!(AutoThreshold::from_key(" Kittler "), Some(AutoThreshold::KittlerIllingworth));
        assert_eq!(AutoThreshold::from_key("nope"), None);
    }
}
