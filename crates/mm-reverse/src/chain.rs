// SPDX-License-Identifier: AGPL-3.0-only
//! A slice chain as the fitting sees it: its points with a loop's closing duplicate dropped,
//! whether it closes on itself, its point spacing and its scatter. Ported from ScanRuler
//! (`viewChain` in `src/core/flat/grow.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::DVec2;

/// The scatter floor as a fraction of the point spacing, so a synthetic chain with no noise
/// still leaves the band a width.
const SCATTER_FLOOR: f64 = 0.05;
/// The scatter is read off the chain by its second differences: noise of σ per point gives
/// the deviation of a point from its neighbours' chord a spread of σ·√1.5, and the median of
/// the absolute value is 0.6745 of the spread.
fn scatter_from_median() -> f64 {
    1.0 / (0.6745 * 1.5f64.sqrt())
}
const SAMPLE: usize = 2048;

#[derive(Debug, Clone)]
pub struct ChainView {
    pub points: Vec<DVec2>,
    pub closed: bool,
    /// Median distance between neighbouring points, millimetres.
    pub spacing: f64,
    /// The chain's noise, millimetres.
    pub scatter: f64,
}

fn median(values: &mut [f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.total_cmp(b));
    values[values.len() >> 1]
}

/// Read a chain the way the growing and the segmentation do, so the two agree on its
/// spacing and scatter.
pub fn view_chain(chain: &[DVec2]) -> ChainView {
    let n0 = chain.len();
    let step = (n0 / SAMPLE).max(1);
    let mut gaps = Vec::new();
    let mut i = step;
    while i < n0 {
        gaps.push((chain[i] - chain[i - step]).length() / step as f64);
        i += step;
    }
    let mut spacing = median(&mut gaps);
    if spacing == 0.0 || !spacing.is_finite() {
        spacing = 1.0;
    }

    let mut points: &[DVec2] = chain;
    let mut closed = false;
    if n0 >= 4 {
        let gap = (chain[0] - chain[n0 - 1]).length();
        if gap <= 2.0 * spacing {
            closed = true;
            // A section's loop comes back onto its own first point.
            if gap < 0.01 * spacing {
                points = &chain[..n0 - 1];
            }
        }
    }

    let n = points.len();
    let mut seconds = Vec::new();
    let step2 = (n / SAMPLE).max(1);
    let mut i = 1;
    while i + 1 < n {
        let a = points[i + 1] - points[i - 1];
        let len = a.length();
        if len >= 1e-9 {
            let b = points[i] - points[i - 1];
            seconds.push((a.x * b.y - a.y * b.x).abs() / len);
        }
        i += step2;
    }
    let scatter = (median(&mut seconds) * scatter_from_median()).max(SCATTER_FLOOR * spacing);
    ChainView { points: points.to_vec(), closed, spacing, scatter }
}
