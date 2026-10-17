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

#[derive(Clone, Debug)]
pub struct ScreenCurve {
    /// Polygon used for the inside test (the curve itself when closed; the curve with its
    /// extensions and a closing arc at infinity when open).
    poly: Vec<DVec2>,
    min: DVec2,
    max: DVec2,
    pub closed: bool,
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
        Some(Self { poly, min, max, closed })
    }

    /// Even-odd inside test (self-intersecting strokes behave like a classic lasso).
    pub fn contains(&self, p: DVec2) -> bool {
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
    fn nearly_closed_counts_as_closed() {
        let pts = vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 100.0), v(0.0, 100.0), v(0.0, 12.0)];
        assert!(is_closed(&pts));
        let open = vec![v(0.0, 0.0), v(100.0, 0.0), v(100.0, 100.0)];
        assert!(!is_closed(&open));
        assert!(ScreenCurve::new(&[v(0.0, 0.0), v(2.0, 0.0)]).is_none());
    }
}
