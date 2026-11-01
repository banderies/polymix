// SPDX-License-Identifier: AGPL-3.0-only
//! Thinning a chain: Ramer–Douglas–Peucker over a run of points, keeping every point the
//! polyline would otherwise stray from by more than the tolerance. Iterative rather than
//! recursive, because a chain round a large part is a hundred thousand points long. Ported
//! from ScanRuler (`src/core/flat/simplify.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec2;

/// The indices of `points[from..to]` worth keeping so that the polyline through them lies
/// within `tolerance` of every point dropped. Ascending; both ends always stay. A tolerance
/// of zero keeps everything.
pub fn simplify_run(points: &[DVec2], from: usize, to: usize, tolerance: f64) -> Vec<usize> {
    if to <= from {
        return Vec::new();
    }
    let n = to - from;
    if n <= 2 || !(tolerance > 0.0) {
        return (from..to).collect();
    }
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut stack: Vec<(usize, usize)> = vec![(0, n - 1)];
    while let Some((a, b)) = stack.pop() {
        if b - a < 2 {
            continue;
        }
        let pa = points[from + a];
        let d = points[from + b] - pa;
        let len = d.length();
        let mut worst = None;
        let mut worst_d = tolerance;
        for i in a + 1..b {
            let p = points[from + i] - pa;
            // Distance to the chord — or, when the ends coincide (a loop's two ends), to the
            // shared end point.
            let dist = if len > 1e-12 { (p.x * d.y - p.y * d.x).abs() / len } else { p.length() };
            if dist > worst_d {
                worst_d = dist;
                worst = Some(i);
            }
        }
        if let Some(w) = worst {
            keep[w] = true;
            stack.push((a, w));
            stack.push((w, b));
        }
    }
    keep.iter().enumerate().filter(|(_, k)| **k).map(|(i, _)| from + i).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_the_corner_of_an_l() {
        let mut pts: Vec<DVec2> = (0..=10).map(|i| DVec2::new(i as f64, 0.0)).collect();
        pts.extend((1..=10).map(|i| DVec2::new(10.0, i as f64)));
        assert_eq!(simplify_run(&pts, 0, pts.len(), 0.01), vec![0, 10, 20]);
    }
}
