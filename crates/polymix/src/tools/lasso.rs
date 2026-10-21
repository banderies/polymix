//! Screen-space selection curves for the Select tool.
//!
//! * A **closed** curve (its ends meet) is an ordinary lasso: everything inside is selected.
//! * An **open** curve is a "laser" cut (Meshmixer's Laser Lasso): its ends are extended
//!   straight along their end directions until the curve splits the whole screen in two,
//!   and the side holding less of the model is selected. Like Meshmixer, the laser cuts
//!   through the model, so faces behind the visible surface are included.

use glam::DVec2;
use std::f64::consts::TAU;

/// Ends closer than this (in points) close the curve…
const CLOSE_GAP: f64 = 20.0;
/// …or closer than this fraction of the curve length.
const CLOSE_GAP_FRACTION: f64 = 0.15;
/// How far back along the curve the end direction is measured (smooths hand jitter).
const END_DIR_LENGTH: f64 = 20.0;
/// Distance the ends are extended to (far outside any viewport).
const FAR: f64 = 1e6;
/// Horizontal bands of the inside test's edge index.
const BANDS: usize = 1024;

#[derive(Clone, Debug)]
pub struct ScreenCurve {
    /// Polygon used for the inside test (the curve itself when closed; the curve with its
    /// extensions and a closing arc at infinity when open).
    poly: Vec<DVec2>,
    min: DVec2,
    max: DVec2,
    pub closed: bool,
    bands: Bands,
}

/// The polygon's edges sorted into horizontal bands, so the inside test of a point only
/// looks at the edges whose height range can reach it. Edge `i` runs from `poly[i]` to the
/// point before it (the last for `i` = 0), as the even-odd walk takes them.
#[derive(Clone, Debug)]
struct Bands {
    /// Bottom of band 0, and bands per unit of height.
    y0: f64,
    scale: f64,
    count: usize,
    /// Band `b` holds `edges[start[b]..start[b + 1]]`.
    start: Vec<u32>,
    edges: Vec<u32>,
}

impl Bands {
    /// Bands over `lo..hi` (where points will be tested), the first and last extending to the
    /// infinities. Any monotone band function would do: an edge that can flip the test for
    /// `p` has `p.y` within its height range, so `p`'s band lies within the edge's bands.
    fn new(poly: &[DVec2], lo: f64, hi: f64) -> Self {
        let count = if hi - lo > 0.0 && (hi - lo).is_finite() { BANDS } else { 1 };
        let scale = if count > 1 { count as f64 / (hi - lo) } else { 0.0 };
        let mut bands = Self { y0: lo, scale, count, start: vec![0; count + 1], edges: Vec::new() };
        // Edges with a NaN coordinate never flip the test (every comparison with NaN fails).
        let span = |i: usize| {
            let (a, b) = (poly[i], poly[if i == 0 { poly.len() - 1 } else { i - 1 }]);
            (!(a.is_nan() || b.is_nan())).then(|| (bands.band(a.y.min(b.y)), bands.band(a.y.max(b.y))))
        };
        let spans: Vec<Option<(usize, usize)>> = (0..poly.len()).map(span).collect();
        for &(b0, b1) in spans.iter().flatten() {
            for n in &mut bands.start[b0 + 1..=b1 + 1] {
                *n += 1;
            }
        }
        for b in 0..count {
            bands.start[b + 1] += bands.start[b];
        }
        let mut fill = bands.start.clone();
        bands.edges = vec![0; bands.start[count] as usize];
        for (i, s) in spans.iter().enumerate() {
            if let Some((b0, b1)) = *s {
                for at in &mut fill[b0..=b1] {
                    bands.edges[*at as usize] = i as u32;
                    *at += 1;
                }
            }
        }
        bands
    }

    /// The band of height `y`: the same floor and clamp for edge ends and tested points (NaN
    /// lands in band 0, where it flips nothing).
    fn band(&self, y: f64) -> usize {
        (((y - self.y0) * self.scale).floor().max(0.0) as usize).min(self.count - 1)
    }

    fn edges(&self, y: f64) -> &[u32] {
        let b = self.band(y);
        &self.edges[self.start[b] as usize..self.start[b + 1] as usize]
    }
}

fn length(pts: &[DVec2]) -> f64 {
    pts.windows(2).map(|w| w[0].distance(w[1])).sum()
}

/// Unit direction pointing out of the curve at its last point.
fn end_direction(pts: &[DVec2]) -> DVec2 {
    let end = *pts.last().unwrap();
    let mut walked = 0.0;
    let mut back = pts[0];
    for w in pts.windows(2).rev() {
        walked += w[0].distance(w[1]);
        back = w[0];
        if walked >= END_DIR_LENGTH {
            break;
        }
    }
    (end - back).try_normalize().unwrap_or(DVec2::X)
}

/// Whether a stroke with these points would be treated as a closed lasso.
pub fn is_closed(pts: &[DVec2]) -> bool {
    if pts.len() < 3 {
        return false;
    }
    let gap = pts[0].distance(*pts.last().unwrap());
    let len = length(pts);
    gap <= CLOSE_GAP.max(CLOSE_GAP_FRACTION * len) && len > 3.0 * gap
}

/// Far ends of an open curve, for drawing the laser's extensions.
pub fn extensions(pts: &[DVec2], reach: f64) -> Option<[(DVec2, DVec2); 2]> {
    if pts.len() < 2 {
        return None;
    }
    let rev: Vec<DVec2> = pts.iter().rev().copied().collect();
    let (s, e) = (pts[0], *pts.last().unwrap());
    Some([(s, s + end_direction(&rev) * reach), (e, e + end_direction(pts) * reach)])
}

impl ScreenCurve {
    /// `None` for strokes too short to mean anything.
    pub fn new(pts: &[DVec2]) -> Option<Self> {
        if pts.len() < 2 || length(pts) < 6.0 {
            return None;
        }
        let closed = is_closed(pts);
        let mut poly = pts.to_vec();
        if !closed {
            let [(_, s_far), (_, e_far)] = extensions(pts, FAR)?;
            // Close the region with a large arc around the curve's centroid, from the far end
            // back to the far start (either way round is fine: callers pick the side).
            let c = pts.iter().copied().sum::<DVec2>() / pts.len() as f64;
            let a0 = (e_far - c).y.atan2((e_far - c).x);
            let mut a1 = (s_far - c).y.atan2((s_far - c).x);
            while a1 <= a0 {
                a1 += TAU;
            }
            poly.push(e_far);
            const ARC_STEPS: usize = 48;
            for k in 1..ARC_STEPS {
                let a = a0 + (a1 - a0) * k as f64 / ARC_STEPS as f64;
                poly.push(c + DVec2::new(a.cos(), a.sin()) * (2.0 * FAR));
            }
            poly.push(s_far);
        }
        let min = poly.iter().copied().fold(DVec2::splat(f64::MAX), DVec2::min);
        let max = poly.iter().copied().fold(DVec2::splat(f64::MIN), DVec2::max);
        // Points are tested where the curve was drawn (in the view), not out where an open
        // curve's extensions reach: the bands cover the drawn height, and as much again above
        // and below as the curve is wide or tall (the rest of the model on screen).
        let (lo, hi) = pts.iter().fold((DVec2::splat(f64::MAX), DVec2::splat(f64::MIN)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
        let pad = (hi - lo).max_element();
        let bands = Bands::new(&poly, lo.y - pad, hi.y + pad);
        Some(Self { poly, min, max, closed, bands })
    }

    /// Even-odd inside test (self-intersecting strokes behave like a classic lasso), over
    /// the edges of the point's band only.
    pub fn contains(&self, p: DVec2) -> bool {
        if p.x < self.min.x || p.y < self.min.y || p.x > self.max.x || p.y > self.max.y {
            return false;
        }
        let poly = &self.poly;
        let mut inside = false;
        for &i in self.bands.edges(p.y) {
            let i = i as usize;
            let (a, b) = (poly[i], poly[if i == 0 { poly.len() - 1 } else { i - 1 }]);
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                inside = !inside;
            }
        }
        inside
    }

    /// The inside test over every edge (what the bands stand in for).
    #[cfg(test)]
    fn contains_every_edge(&self, p: DVec2) -> bool {
        if p.x < self.min.x || p.y < self.min.y || p.x > self.max.x || p.y > self.max.y {
            return false;
        }
        let poly = &self.poly;
        let mut inside = false;
        let mut j = poly.len() - 1;
        for i in 0..poly.len() {
            let (a, b) = (poly[i], poly[j]);
            if (a.y > p.y) != (b.y > p.y) && p.x < (b.x - a.x) * (p.y - a.y) / (b.y - a.y) + a.x {
                inside = !inside;
            }
            j = i;
        }
        inside
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(x: f64, y: f64) -> DVec2 {
        DVec2::new(x, y)
    }

    #[test]
    fn closed_loop_selects_inside() {
        let pts: Vec<DVec2> = (0..=40).map(|k| {
            let a = k as f64 / 40.0 * TAU;
            v(100.0 + 50.0 * a.cos(), 100.0 + 50.0 * a.sin())
        }).collect();
        let c = ScreenCurve::new(&pts).unwrap();
        assert!(c.closed);
        assert!(c.contains(v(100.0, 100.0)));
        assert!(!c.contains(v(200.0, 100.0)));
    }

    #[test]
    fn straight_laser_splits_the_plane() {
        // Horizontal line across the middle: points above and below land on opposite sides,
        // including points far beyond the drawn ends.
        let pts: Vec<DVec2> = (0..=10).map(|k| v(k as f64 * 10.0, 50.0)).collect();
        let c = ScreenCurve::new(&pts).unwrap();
        assert!(!c.closed);
        let above = c.contains(v(50.0, 10.0));
        assert_ne!(above, c.contains(v(50.0, 90.0)));
        assert_eq!(above, c.contains(v(-500.0, 10.0)));
        assert_eq!(above, c.contains(v(900.0, 10.0)));
        assert_eq!(!above, c.contains(v(-500.0, 90.0)));
        assert_eq!(!above, c.contains(v(900.0, 90.0)));
    }

    #[test]
    fn curved_laser_follows_the_curve() {
        // A "U" opening upwards, ends far apart (open): the inside of the U is one side.
        let mut pts = vec![v(0.0, 0.0), v(0.0, 100.0), v(100.0, 100.0), v(100.0, 0.0)];
        // Densify.
        let mut dense = Vec::new();
        for w in pts.windows(2) {
            for k in 0..10 {
                dense.push(w[0].lerp(w[1], k as f64 / 10.0));
            }
        }
        dense.push(*pts.last().unwrap());
        pts = dense;
        let c = ScreenCurve::new(&pts).unwrap();
        assert!(!c.closed);
        let inner = c.contains(v(50.0, 50.0));
        // Above the U (beyond the extended arms) is still the inner side.
        assert_eq!(inner, c.contains(v(50.0, -400.0)));
        assert_ne!(inner, c.contains(v(50.0, 150.0)));
        assert_ne!(inner, c.contains(v(-50.0, 50.0)));
        assert_ne!(inner, c.contains(v(150.0, 50.0)));
    }

    #[test]
    fn banded_test_matches_every_edge() {
        // A small fixed-seed generator (xorshift), for repeatable scribbles.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut curves: Vec<Vec<DVec2>> = vec![
            // Closed and open circles, a wavy laser, a vertical laser (its ends run up and down).
            (0..=60).map(|k| v(300.0 + 120.0 * (k as f64 / 60.0 * TAU).cos(), 250.0 + 90.0 * (k as f64 / 60.0 * TAU).sin())).collect(),
            (0..=40).map(|k| v(300.0 + 120.0 * (k as f64 / 60.0 * TAU).cos(), 250.0 + 90.0 * (k as f64 / 60.0 * TAU).sin())).collect(),
            (0..300).map(|k| v(10.0 + 2.6 * k as f64, 300.0 + 60.0 * (k as f64 * 0.07).sin() + 9.0 * (k as f64 * 0.31).cos())).collect(),
            (0..80).map(|k| v(400.0 + 3.0 * (k as f64 * 0.4).sin(), 20.0 + 7.0 * k as f64)).collect(),
            // Flat: every point at one height.
            (0..50).map(|k| v(5.0 * k as f64, 100.0)).collect(),
            // On whole pixels, so tested points land exactly on vertex heights.
            vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 100.0), v(50.0, 40.0), v(0.0, 100.0), v(0.0, 3.0)],
        ];
        // Self-crossing scribbles, closed and open.
        for n in [30, 200, 1200] {
            let pts: Vec<DVec2> = (0..n).map(|_| v(50.0 + 500.0 * rnd(), 40.0 + 400.0 * rnd())).collect();
            curves.push(pts.clone());
            let mut closed = pts;
            closed.push(closed[0] + v(3.0, 2.0));
            curves.push(closed);
        }
        let mut with_nan = curves[2].clone();
        with_nan[100] = v(f64::NAN, 300.0);
        with_nan[150].y = f64::NAN;
        curves.push(with_nan);
        let mut tested = 0;
        for (k, pts) in curves.iter().enumerate() {
            let Some(c) = ScreenCurve::new(pts) else { continue };
            let mut probe: Vec<DVec2> = (0..4000).map(|_| v(-100.0 + 900.0 * rnd(), -100.0 + 700.0 * rnd())).collect();
            // Exactly at vertex heights and positions, at whole pixels, far out, and NaN.
            probe.extend(pts.iter().flat_map(|p| [*p, v(p.x + 0.5, p.y), v(p.x - 7.0, p.y), v(p.x, p.y.floor())]));
            probe.extend((0..60).flat_map(|i| (0..60).map(move |j| v(i as f64 * 10.0, j as f64 * 10.0))));
            probe.extend([v(-5e5, 300.0), v(5e5, 10.0), v(300.0, -1.5e6), v(300.0, 1.5e6), v(f64::NAN, 200.0), v(200.0, f64::NAN), v(f64::INFINITY, 0.0)]);
            for p in probe {
                assert_eq!(c.contains(p), c.contains_every_edge(p), "curve {k}, point {p:?}");
                tested += 1;
            }
        }
        assert!(tested > 100_000);
    }

    #[test]
    fn nearly_closed_counts_as_closed() {
        let pts = vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 100.0), v(0.0, 100.0), v(0.0, 12.0)];
        assert!(is_closed(&pts));
        let open = vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 100.0)];
        assert!(!is_closed(&open));
        assert!(ScreenCurve::new(&[v(0.0, 0.0), v(2.0, 0.0)]).is_none());
    }
}
