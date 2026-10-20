//! Shapes drawn on a slice, as voxels: a polygon filled in its slice (Draw) or extruded
//! through a range of slices (Scissors), and the region of a slice traced from a voxel (Level
//! tracing); and shapes through points placed in the volume: a tube along them (Draw tube)
//! and the closed blob they span (Surface cut, their convex hull).
//!
//! A slice is given by its voxel axes `[across, down, through]`; polygons are in voxel
//! coordinates along `across` and `down` (voxel centres at whole numbers), and a voxel is
//! inside when its centre is (even–odd rule).

use crate::mask::Mask;
use glam::DVec3;
use mm_volume::load::LoadedVolume;

/// The voxels of each row `down` of a slice inside `polygon`: (down, first across, last
/// across), inclusive, clamped to the slice of `size` (across, down) voxels.
fn spans(polygon: &[(f64, f64)], size: [usize; 2]) -> Vec<(usize, usize, usize)> {
    let n = polygon.len();
    if n < 3 {
        return Vec::new();
    }
    let (lo, hi) = polygon.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.1), b.max(p.1)));
    let (lo, hi) = (lo.ceil().max(0.0), hi.floor().min(size[1] as f64 - 1.0));
    let mut out = Vec::new();
    if lo.is_nan() || hi.is_nan() || lo > hi {
        return out;
    }
    let mut xs: Vec<f64> = Vec::new();
    for d in lo as usize..=hi as usize {
        let y = d as f64;
        xs.clear();
        for i in 0..n {
            let (a, b) = (polygon[i], polygon[(i + 1) % n]);
            // Half-open in y, so a vertex on the row counts once.
            if (a.1 <= y && y < b.1) || (b.1 <= y && y < a.1) {
                xs.push(a.0 + (y - a.1) / (b.1 - a.1) * (b.0 - a.0));
            }
        }
        xs.sort_by(|p, q| p.total_cmp(q));
        for [x0, x1] in xs.as_chunks::<2>().0 {
            let first = x0.ceil().max(0.0);
            let last = x1.floor().min(size[0] as f64 - 1.0);
            if first <= last {
                out.push((d, first as usize, last as usize));
            }
        }
    }
    out
}

/// The voxels inside `polygon` on slices `from..=to` along `axes[2]` (see the module
/// documentation).
pub fn polygon_prism(dims: [usize; 3], axes: [usize; 3], polygon: &[(f64, f64)], from: usize, to: usize) -> Mask {
    let mut m = Mask::new(dims);
    let stride = [1, dims[0], dims[0] * dims[1]];
    let to = to.min(dims[axes[2]].saturating_sub(1));
    for (d, a0, a1) in spans(polygon, [dims[axes[0]], dims[axes[1]]]) {
        for s in from..=to {
            let base = d * stride[axes[1]] + s * stride[axes[2]];
            for a in a0..=a1 {
                m.data[base + a * stride[axes[0]]] = 1;
            }
        }
    }
    m
}

/// The region of the slice through `seed` (across `axes[2]`) connected to it by the 4
/// neighbours in the slice, whose values are at least the seed's (with `darker`: at most):
/// the structure an iso-line through the seed's value outlines. Voxel indices.
pub fn trace_level(vol: &LoadedVolume, axes: [usize; 3], seed: [usize; 3], darker: bool) -> Vec<usize> {
    let dims = vol.dims;
    if (0..3).any(|a| seed[a] >= dims[a]) {
        return Vec::new();
    }
    let stride = [1, dims[0], dims[0] * dims[1]];
    let (na, nd) = (dims[axes[0]], dims[axes[1]]);
    let base = seed[axes[2]] * stride[axes[2]];
    let at = |a: usize, d: usize| base + a * stride[axes[0]] + d * stride[axes[1]];
    let level = vol.values[at(seed[axes[0]], seed[axes[1]])];
    let inside = |v: f32| if darker { v <= level } else { v >= level };
    let mut seen = vec![false; na * nd];
    let mut stack = vec![(seed[axes[0]], seed[axes[1]])];
    seen[seed[axes[1]] * na + seed[axes[0]]] = true;
    let mut out = Vec::new();
    while let Some((a, d)) = stack.pop() {
        out.push(at(a, d));
        let mut visit = |a: usize, d: usize| {
            if !seen[d * na + a] && inside(vol.values[at(a, d)]) {
                seen[d * na + a] = true;
                stack.push((a, d));
            }
        };
        if a > 0 {
            visit(a - 1, d);
        }
        if a + 1 < na {
            visit(a + 1, d);
        }
        if d > 0 {
            visit(a, d - 1);
        }
        if d + 1 < nd {
            visit(a, d + 1);
        }
    }
    out
}

/// A circle of radius `r` (voxels across; `aspect` = down voxels per across voxel, for
/// anisotropic pixels) about `c`, as a polygon of `n` points.
pub fn circle(c: (f64, f64), r: f64, aspect: f64, n: usize) -> Vec<(f64, f64)> {
    (0..n.max(3))
        .map(|k| {
            let t = std::f64::consts::TAU * k as f64 / n.max(3) as f64;
            (c.0 + r * t.cos(), c.1 + r * aspect * t.sin())
        })
        .collect()
}

/// The voxels within `radius_mm` of a smooth curve through `points` (voxel coordinates,
/// `spacing` mm apart): a Catmull–Rom spline through them (centripetal, so it neither
/// overshoots nor loops between uneven points), or just the points' ball for one point.
pub fn tube(dims: [usize; 3], spacing: [f64; 3], points: &[DVec3], radius_mm: f64) -> Mask {
    let mut m = Mask::new(dims);
    if points.is_empty() || radius_mm.is_nan() || radius_mm <= 0.0 {
        return m;
    }
    let mm = |v: DVec3| DVec3::new(v.x * spacing[0], v.y * spacing[1], v.z * spacing[2]);
    let p: Vec<DVec3> = points.iter().map(|&v| mm(v)).collect();
    // The curve, sampled at most half the smallest spacing apart.
    let step = spacing.iter().cloned().fold(f64::INFINITY, f64::min) * 0.5;
    let mut curve = vec![p[0]];
    for i in 0..p.len().saturating_sub(1) {
        let (p0, p1, p2, p3) = (p[i.saturating_sub(1)], p[i], p[i + 1], p[(i + 2).min(p.len() - 1)]);
        let n = ((p2 - p1).length() / step).ceil().max(1.0) as usize;
        for s in 1..=n {
            curve.push(catmull_rom(p0, p1, p2, p3, s as f64 / n as f64));
        }
    }
    // A ball at each sample (their union is the tube).
    let r2 = radius_mm * radius_mm;
    for c in curve {
        let lo = [0, 1, 2].map(|a| ((c[a] - radius_mm) / spacing[a]).floor().max(0.0) as usize);
        let hi = [0, 1, 2].map(|a| (((c[a] + radius_mm) / spacing[a]).ceil().max(0.0) as usize).min(dims[a].saturating_sub(1)));
        for k in lo[2]..=hi[2] {
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    if mm(DVec3::new(i as f64, j as f64, k as f64)).distance_squared(c) <= r2 {
                        let n = m.index(i, j, k);
                        m.data[n] = 1;
                    }
                }
            }
        }
    }
    m
}

/// A point of the centripetal Catmull–Rom spline between `p1` and `p2` (Barry & Goldman's
/// pyramidal form, knots spaced by the square root of the chord lengths), `t` in 0..1.
fn catmull_rom(p0: DVec3, p1: DVec3, p2: DVec3, p3: DVec3, t: f64) -> DVec3 {
    let knot = |a: DVec3, b: DVec3| a.distance(b).sqrt().max(1e-9);
    let (t0, t1) = (0.0, knot(p0, p1));
    let t2 = t1 + knot(p1, p2);
    let t3 = t2 + knot(p2, p3);
    let t = t1 + (t2 - t1) * t;
    let lerp = |a: DVec3, b: DVec3, ta: f64, tb: f64| if (tb - ta).abs() < 1e-12 { a } else { a * ((tb - t) / (tb - ta)) + b * ((t - ta) / (tb - ta)) };
    let (a1, a2, a3) = (lerp(p0, p1, t0, t1), lerp(p1, p2, t1, t2), lerp(p2, p3, t2, t3));
    let (b1, b2) = (lerp(a1, a2, t0, t2), lerp(a2, a3, t1, t3));
    lerp(b1, b2, t1, t2)
}

/// The voxels inside the convex hull of `points` (voxel coordinates, `spacing` mm apart): the
/// closed blob through points placed on a structure's surface. Empty when they span no volume.
pub fn hull(dims: [usize; 3], spacing: [f64; 3], points: &[DVec3]) -> Mask {
    let mut m = Mask::new(dims);
    let mm = |v: DVec3| DVec3::new(v.x * spacing[0], v.y * spacing[1], v.z * spacing[2]);
    let pts: Vec<DVec3> = points.iter().map(|&v| mm(v)).collect();
    let Some(h) = mm_edit::fit::convex_hull(&pts) else { return m };
    // Inside every face's plane (a small tolerance keeps voxels on a face).
    let planes: Vec<(DVec3, f64)> = h.triangle_ids().filter_map(|t| {
        let [a, b, c] = h.tri_positions(t);
        let n = (b - a).cross(c - a).try_normalize()?;
        Some((n, n.dot(a)))
    }).collect();
    if planes.is_empty() {
        return m;
    }
    let tol = 1e-6 * spacing.iter().cloned().fold(0.0, f64::max);
    let (lo, hi) = pts.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, b), &p| (a.min(p), b.max(p)));
    let lo = [0, 1, 2].map(|a| (lo[a] / spacing[a]).floor().max(0.0) as usize);
    let hi = [0, 1, 2].map(|a| ((hi[a] / spacing[a]).ceil().max(0.0) as usize).min(dims[a].saturating_sub(1)));
    for k in lo[2]..=hi[2] {
        for j in lo[1]..=hi[1] {
            for i in lo[0]..=hi[0] {
                let p = mm(DVec3::new(i as f64, j as f64, k as f64));
                if planes.iter().all(|(n, d)| n.dot(p) <= d + tol) {
                    let n = m.index(i, j, k);
                    m.data[n] = 1;
                }
            }
        }
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    #[test]
    fn polygons_fill_by_voxel_centres() {
        let dims = [10, 8, 6];
        // A square from 1.5 to 4.5 across (voxels 2, 3, 4) and 0.5 to 2.5 down (1, 2), on
        // slice 3 along k.
        let sq = [(1.5, 0.5), (4.5, 0.5), (4.5, 2.5), (1.5, 2.5)];
        let m = polygon_prism(dims, [0, 1, 2], &sq, 3, 3);
        assert_eq!(m.count(), 6);
        assert!(m.get(2, 1, 3) && m.get(4, 2, 3) && !m.get(1, 1, 3) && !m.get(2, 0, 3) && !m.get(2, 1, 2));
        // Through slices 1–4 and in another orientation (across j, down k, through i).
        let m = polygon_prism(dims, [1, 2, 0], &sq, 1, 4);
        assert_eq!(m.count(), 6 * 4);
        assert!(m.get(1, 2, 1) && m.get(4, 4, 2));
        // A triangle, clipped by the edge of the slice; fewer than 3 points: nothing.
        let m = polygon_prism(dims, [0, 1, 2], &[(-5.0, -5.0), (20.0, -5.0), (-5.0, 20.0)], 0, 0);
        assert!(m.count() > 40 && m.get(0, 0, 0) && !m.get(9, 7, 0));
        assert_eq!(polygon_prism(dims, [0, 1, 2], &[(1.0, 1.0), (3.0, 3.0)], 0, 5).count(), 0);
        // A circle of radius 3 has about π · 9 voxels.
        let m = polygon_prism([20, 20, 1], [0, 1, 2], &circle((10.0, 10.0), 3.0, 1.0, 64), 0, 0);
        assert!((m.count() as f64 - std::f64::consts::PI * 9.0).abs() < 5.0, "{}", m.count());
    }

    #[test]
    fn tubes_and_hulls() {
        // A straight tube of radius 2 along i from 5 to 25 (1 mm voxels): about π r² per
        // slice across it, and nothing beyond its ends' balls.
        let dims = [40, 20, 20];
        let t = tube(dims, [1.0; 3], &[DVec3::new(5.0, 10.0, 10.0), DVec3::new(25.0, 10.0, 10.0)], 2.0);
        let slice = (0..20).flat_map(|k| (0..20).map(move |j| (j, k))).filter(|&(j, k)| t.get(15, j, k)).count();
        assert_eq!(slice, 13, "the voxels of a radius-2 disc");
        assert!(t.get(5, 10, 10) && t.get(3, 10, 10) && !t.get(2, 10, 10) && !t.get(28, 10, 10));
        // Through three points it bends but passes each of them.
        let pts = [DVec3::new(5.0, 5.0, 10.0), DVec3::new(20.0, 15.0, 10.0), DVec3::new(35.0, 5.0, 10.0)];
        let t = tube(dims, [1.0; 3], &pts, 1.0);
        assert!(pts.iter().all(|p| t.get(p.x as usize, p.y as usize, p.z as usize)));
        assert!(!t.get(20, 5, 10), "it does not cut the corner");
        // Anisotropic voxels: radius in mm.
        let t = tube([40, 20, 10], [1.0, 1.0, 2.0], &[DVec3::new(5.0, 10.0, 5.0), DVec3::new(25.0, 10.0, 5.0)], 2.0);
        assert!(t.get(15, 10, 6) && !t.get(15, 10, 7));
        // The hull of a cube's corners is the cube; of three points, nothing.
        let corners: Vec<DVec3> = (0..8).map(|c| DVec3::new(if c & 1 == 0 { 2.0 } else { 6.0 }, if c & 2 == 0 { 2.0 } else { 6.0 }, if c & 4 == 0 { 2.0 } else { 6.0 })).collect();
        assert_eq!(hull([10, 10, 10], [1.0; 3], &corners).count(), 125);
        assert_eq!(hull([10, 10, 10], [1.0; 3], &corners[..3]).count(), 0);
    }

    #[test]
    fn level_tracing_outlines_the_connected_bright_region() {
        // Two bright squares in slice 1 of a 10 × 10 × 3 volume, and bright voxels in slice 0.
        let mut values = vec![0.0f32; 300];
        let at = |i: usize, j: usize, k: usize| (k * 10 + j) * 10 + i;
        for j in 2..5 {
            for i in 2..5 {
                values[at(i, j, 1)] = 50.0;
                values[at(i + 5, j + 4, 1)] = 80.0;
                values[at(i, j, 0)] = 90.0;
            }
        }
        values[at(3, 3, 1)] = 70.0;
        let g = Geometry::from_axes([10, 10, 3, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [10, 10, 3], values, g);
        // From an edge voxel (50): its square only (the other is not connected, slice 0 is
        // another slice).
        let r = trace_level(&vol, [0, 1, 2], [2, 2, 1], false);
        assert_eq!(r.len(), 9);
        // From the brighter centre (70): just it.
        assert_eq!(trace_level(&vol, [0, 1, 2], [3, 3, 1], false).len(), 1);
        // Darker from the background: everything at 0 in the slice.
        assert_eq!(trace_level(&vol, [0, 1, 2], [0, 0, 1], true).len(), 100 - 18);
        assert!(trace_level(&vol, [0, 1, 2], [10, 0, 0], false).is_empty());
    }
}
