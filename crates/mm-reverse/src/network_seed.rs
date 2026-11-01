// SPDX-License-Identifier: AGPL-3.0-only
//! A point well inside a loop drawn on the scan: the seed a patch placed by its corners is
//! filled at (PN4).
//!
//! Ported from ScanRuler (`plugins/reverse/core/networkSeed.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only.
//!
//! The cell reader takes a region by the point inside it, and the patch's corners move
//! afterwards (dragged, or a neighbour's put on its side) so the seed is put as far from the
//! loop as it can be, not just over its first edge.
//!
//! The loop is seen along the way it faces: its vector area, the normal integrated over any
//! surface it bounds, turned to agree with the scan's normals along it. A plain mean of the
//! normals at its points is not that: the points lie closer on some sides than others, and
//! round a strip of 270° of a cylinder the strip's two straight ends outweighed its arcs and
//! the mean pointed out of the gap. Seen so, the loop is flattened onto the plane across it,
//! and the point of that polygon farthest from its sides is found on a grid: the middle of a
//! square, the inside of an L's corner, not its centroid, which an L or a U puts outside.
//! That point is taken back onto the scan along the normal: of the scan's crossings with the
//! line there, the one nearest the plane among those facing the loop's way. The facing
//! matters on a shell: a loop round the outside of a housing has its plane below the skin,
//! often under the inner wall too, and the inner wall (nearer, facing in) is the wrong one
//! (a 60° cap of a shell 1.5 mm thick).
//!
//! A loop the flattening folds over (a strip round more than half a cylinder) has no such
//! polygon; nor has one whose seed, so found, comes out within MARGIN_EDGES of the loop. Then
//! the seed is stepped in from the loop's sides, across them on the scan, to the farthest
//! that stays inside the polygon there.

use glam::DVec3;

use crate::scan_path::ScanPaths;

/// The grid the flattened loop is searched on, a side, and once more round the best of it.
const GRID: usize = 24;
/// How far from the loop a seed must be, in the scan's edges.
const MARGIN_EDGES: f64 = 2.0;
/// The loop's points used at most: a few hundred carry its shape.
const MAX_POINTS: usize = 400;

/// A point on the scan inside a closed loop of points on it (x, y, z each, the last joined to
/// the first), or `None` for a loop too small.
pub fn seed_inside(scan: &ScanPaths, outline: &[f64]) -> Option<DVec3> {
    let all = outline.len() / 3;
    if all < 3 {
        return None;
    }
    let stride = all.div_ceil(MAX_POINTS).max(1);
    let pts: Vec<DVec3> = (0..all).step_by(stride).map(|i| DVec3::new(outline[i * 3], outline[i * 3 + 1], outline[i * 3 + 2])).collect();
    let m = pts.len();
    if m < 3 {
        return None;
    }
    // The loop's middle, and the way it faces: its vector area (Newell's), turned the way the
    // scan's normals along it lean, each by the length of loop it stands for.
    let mut c = DVec3::ZERO;
    for p in &pts {
        c += *p / m as f64;
    }
    let mut n = DVec3::ZERO;
    for i in 0..m {
        n += cross(pts[i] - c, pts[(i + 1) % m] - c) / 2.0;
    }
    let nl = n.length();
    if !(nl > 0.0) {
        return None;
    }
    n /= nl;
    let mut lean = 0.0;
    for i in 0..m {
        if let Some(on) = scan.closest_inside(pts[i]) {
            lean += scan.triangle_normal(on.triangle).dot(n) * (pts[(i + 1) % m] - pts[i]).length();
        }
    }
    if lean < 0.0 {
        n = -n;
    }
    // A frame across the normal, and the loop flattened into it.
    let u = unit(if n.x.abs() < 0.9 { cross(n, DVec3::X) } else { cross(n, DVec3::Y) });
    let v = cross(n, u);
    let xs: Vec<f64> = pts.iter().map(|p| (*p - c).dot(u)).collect();
    let ys: Vec<f64> = pts.iter().map(|p| (*p - c).dot(v)).collect();
    let margin = MARGIN_EDGES * scan.edge();
    if let Some(flat) = farthest_inside(&xs, &ys) {
        if flat.2 > margin {
            let base = DVec3::new(c.x + flat.0 * u.x + flat.1 * v.x, c.y + flat.0 * u.y + flat.1 * v.y, c.z + flat.0 * u.z + flat.1 * v.z);
            if let Some(hit) = along_normal(scan, base, n, &pts) {
                if nearest_of(hit, &pts) > margin {
                    return Some(scan.closest_inside(hit).map(|c| c.point).unwrap_or(hit));
                }
            }
        }
    }
    stepped_in(scan, &pts, c, n, u, v, &xs, &ys)
}

/// The point of a polygon farthest from its sides, on a grid and once more on a finer one
/// round the best: (x, y, distance); `None` for a polygon with nothing inside it.
fn farthest_inside(xs: &[f64], ys: &[f64]) -> Option<(f64, f64, f64)> {
    let (mut x0, mut x1, mut y0, mut y1) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for i in 0..xs.len() {
        x0 = x0.min(xs[i]);
        x1 = x1.max(xs[i]);
        y0 = y0.min(ys[i]);
        y1 = y1.max(ys[i]);
    }
    let mut best: Option<(f64, f64, f64)> = None;
    let search = |ax: f64, bx: f64, ay: f64, by: f64, best: &mut Option<(f64, f64, f64)>| {
        for i in 0..GRID {
            for j in 0..GRID {
                let x = ax + ((i as f64 + 0.5) / GRID as f64) * (bx - ax);
                let y = ay + ((j as f64 + 0.5) / GRID as f64) * (by - ay);
                if !inside_polygon(xs, ys, x, y) {
                    continue;
                }
                let d = polygon_distance(xs, ys, x, y);
                if best.is_none_or(|b| d > b.2) {
                    *best = Some((x, y, d));
                }
            }
        }
    };
    search(x0, x1, y0, y1, &mut best);
    let (x, y, _) = best?;
    let hx = (x1 - x0) / GRID as f64;
    let hy = (y1 - y0) / GRID as f64;
    search(x - hx, x + hx, y - hy, y + hy, &mut best);
    best
}

/// Where the line through `base` along `n` crosses the scan: the crossing nearest `base`
/// among the triangles facing `n`'s way, or among all.
fn along_normal(scan: &ScanPaths, base: DVec3, n: DVec3, pts: &[DVec3]) -> Option<DVec3> {
    // As far each way as the loop is wide.
    let mut reach = 0.0f64;
    for p in pts {
        reach = reach.max((*p - base).length());
    }
    reach += 4.0 * scan.edge();
    let mut lo = DVec3::ZERO;
    let mut hi = DVec3::ZERO;
    for k in 0..3 {
        lo[k] = (base[k] - reach * n[k]).min(base[k] + reach * n[k]) - scan.edge();
        hi[k] = (base[k] - reach * n[k]).max(base[k] + reach * n[k]) + scan.edge();
    }
    let mut facing: Option<f64> = None;
    let mut any: Option<f64> = None;
    for tri in scan.triangles_in(lo, hi) {
        let [a, b, d] = scan.triangle(tri);
        let Some(t) = line_triangle(base, n, scan.vertex(a), scan.vertex(b), scan.vertex(d)) else { continue };
        if any.is_none_or(|x| t.abs() < x.abs()) {
            any = Some(t);
        }
        if scan.triangle_normal(tri).dot(n) > 0.2 && facing.is_none_or(|x| t.abs() < x.abs()) {
            facing = Some(t);
        }
    }
    let t = facing.or(any)?;
    Some(base + t * n)
}

/// Möller–Trumbore for a line: where along `dir` from `o` it crosses the triangle, or `None`.
fn line_triangle(o: DVec3, dir: DVec3, a: DVec3, b: DVec3, c: DVec3) -> Option<f64> {
    let e1 = b - a;
    let e2 = c - a;
    let h = cross(dir, e2);
    let det = e1.dot(h);
    if det.abs() < 1e-12 {
        return None;
    }
    let s = o - a;
    let uu = s.dot(h) / det;
    if uu < 0.0 || uu > 1.0 {
        return None;
    }
    let q = cross(s, e1);
    let vv = dir.dot(q) / det;
    if vv < 0.0 || uu + vv > 1.0 {
        return None;
    }
    Some(e2.dot(q) / det)
}

/// The fallback: from each of a dozen points round the loop, a step in across it on the
/// scan (left of the way it runs, seen from outside, for a loop running anticlockwise, right
/// for one running clockwise) of a fifth of the loop's width, then halved until the
/// flattened point is inside; the one farthest from the loop.
#[allow(clippy::too_many_arguments)]
fn stepped_in(scan: &ScanPaths, pts: &[DVec3], c: DVec3, n: DVec3, u: DVec3, v: DVec3, xs: &[f64], ys: &[f64]) -> Option<DVec3> {
    let m = pts.len();
    let mut area = 0.0;
    for i in 0..m {
        area += xs[i] * ys[(i + 1) % m] - xs[(i + 1) % m] * ys[i];
    }
    let sign = if area >= 0.0 { 1.0 } else { -1.0 };
    let mut width = 0.0f64;
    for p in pts {
        width = width.max((*p - pts[0]).length());
    }
    let mut best: Option<(DVec3, f64)> = None;
    for j in 0..12 {
        let i = (j * m) / 12;
        let p = pts[i];
        let t = pts[(i + 1) % m] - pts[(i + m - 1) % m];
        let normal = scan.closest_inside(p).map(|on| scan.triangle_normal(on.triangle)).unwrap_or(n);
        let left = unit(cross(normal, t));
        let mut step = width / 5.0;
        while step > scan.edge() {
            let q = p + sign * step * left;
            let Some(back) = scan.closest_inside(q) else { break };
            let r = back.point;
            if !inside_polygon(xs, ys, (r - c).dot(u), (r - c).dot(v)) {
                step /= 2.0;
                continue;
            }
            let d = nearest_of(r, pts);
            if best.is_none_or(|b| d > b.1) {
                best = Some((r, d));
            }
            break;
        }
    }
    match best {
        Some((at, _)) => Some(at),
        None => scan.closest_inside(pts[0]).map(|c| c.point),
    }
}

fn inside_polygon(xs: &[f64], ys: &[f64], x: f64, y: f64) -> bool {
    let mut inside = false;
    let mut j = xs.len().wrapping_sub(1);
    for i in 0..xs.len() {
        if (ys[i] > y) != (ys[j] > y) && x < ((xs[j] - xs[i]) * (y - ys[i])) / (ys[j] - ys[i]) + xs[i] {
            inside = !inside;
        }
        j = i;
    }
    inside
}

fn polygon_distance(xs: &[f64], ys: &[f64], x: f64, y: f64) -> f64 {
    let mut best = f64::INFINITY;
    let mut j = xs.len().wrapping_sub(1);
    for i in 0..xs.len() {
        let dx = xs[i] - xs[j];
        let dy = ys[i] - ys[j];
        let l2 = dx * dx + dy * dy;
        let t = if l2 > 0.0 { (((x - xs[j]) * dx + (y - ys[j]) * dy) / l2).clamp(0.0, 1.0) } else { 0.0 };
        best = best.min((xs[j] + t * dx - x).hypot(ys[j] + t * dy - y));
        j = i;
    }
    best
}

fn nearest_of(p: DVec3, pts: &[DVec3]) -> f64 {
    pts.iter().map(|q| (p - *q).length()).fold(f64::INFINITY, f64::min)
}

fn cross(a: DVec3, b: DVec3) -> DVec3 {
    DVec3::new(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x)
}

fn unit(a: DVec3) -> DVec3 {
    let l = a.length();
    if l > 0.0 {
        a / l
    } else {
        DVec3::Z
    }
}

/// Ported from ScanRuler's `plugins/reverse/tests/reNetworkSeed.test.ts`: where a patch
/// placed by its corners is filled. Its seed well inside the loop: the middle of a square,
/// the bar of a U and not its notch, the outer skin of a shell and not the inner wall nearer
/// the loop's plane, the middle of a strip round 270° of a cylinder. And the corners a patch
/// was placed with are its corners, whatever the boundary turns there.
#[cfg(test)]
mod tests {
    use std::f64::consts::PI;

    use super::*;
    use crate::network::read_network;
    use crate::network_meshes::{curve, grid_scan, network, sphere_scan, GridOptions, TestMesh};

    /// A closed loop through corners, a point every half millimetre along each straight side.
    fn closed_loop(corners: &[DVec3]) -> Vec<f64> {
        let mut out = Vec::new();
        for i in 0..corners.len() {
            let a = corners[i];
            let b = corners[(i + 1) % corners.len()];
            let n = ((b - a).length() / 0.5).ceil().max(1.0) as usize;
            for k in 0..n {
                let f = |ak: f64, bk: f64| ak + ((bk - ak) * k as f64) / n as f64;
                out.extend_from_slice(&[f(a.x, b.x), f(a.y, b.y), f(a.z, b.z)]);
            }
        }
        out
    }

    fn inside_poly(poly: &[[f64; 2]], x: f64, y: f64) -> bool {
        let mut inside = false;
        let mut j = poly.len() - 1;
        for i in 0..poly.len() {
            let ([xi, yi], [xj, yj]) = (poly[i], poly[j]);
            if (yi > y) != (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi {
                inside = !inside;
            }
            j = i;
        }
        inside
    }

    fn flat() -> ScanPaths {
        let sheet = grid_scan(100, 100, &|u, v| DVec3::new(100.0 * u, 100.0 * v, 0.0), &GridOptions { noise: 0.01, ..Default::default() });
        ScanPaths::new(sheet.positions, sheet.indices)
    }

    #[test]
    fn lies_in_the_middle_of_a_square() {
        let seed = seed_inside(&flat(), &closed_loop(&[DVec3::new(20.0, 20.0, 0.0), DVec3::new(60.0, 20.0, 0.0), DVec3::new(60.0, 60.0, 0.0), DVec3::new(20.0, 60.0, 0.0)])).expect("a seed");
        assert!((seed.x - 40.0).hypot(seed.y - 40.0) < 2.0);
        assert!(seed.z.abs() < 0.1);
    }

    #[test]
    fn lies_in_a_u_s_bar_or_arms_not_in_its_notch_where_its_centroid_is() {
        let u: [[f64; 2]; 8] = [[20.0, 20.0], [80.0, 20.0], [80.0, 80.0], [60.0, 80.0], [60.0, 40.0], [40.0, 40.0], [40.0, 80.0], [20.0, 80.0]];
        let corners: Vec<DVec3> = u.iter().map(|&[x, y]| DVec3::new(x, y, 0.0)).collect();
        let seed = seed_inside(&flat(), &closed_loop(&corners)).expect("a seed");
        assert!(inside_poly(&u, seed.x, seed.y));
        // Ten millimetres off every side: the bar and the arms are twenty wide.
        assert!((seed.x - 20.0).min(80.0 - seed.x).min(seed.y - 20.0) > 5.0, "{seed}");
    }

    #[test]
    fn lies_on_a_shell_s_outer_skin_not_on_the_inner_wall_its_loop_s_plane_is_under() {
        // A shell 1.5 mm thick: the outer sphere's normals out, the inner one's in, to the
        // hollow. A loop 30° round the outer one's +x pole has its middle 5.4 mm in, under the
        // inner wall.
        let outer = sphere_scan(40.0, 0.01, 5);
        let inner = sphere_scan(38.5, 0.01, 5);
        let mut flipped = inner.indices.clone();
        for t in flipped.chunks_exact_mut(3) {
            t.swap(1, 2);
        }
        let offset = (outer.positions.len() / 3) as u32;
        let mut shell = TestMesh { positions: outer.positions.clone(), indices: outer.indices.clone() };
        shell.positions.extend_from_slice(&inner.positions);
        shell.indices.extend(flipped.iter().map(|i| i + offset));
        let scan = ScanPaths::new(shell.positions, shell.indices);
        let mut ring = Vec::new();
        for k in 0..360 {
            let t = (k as f64 * PI) / 180.0;
            ring.extend_from_slice(&[40.0 * (PI / 6.0).cos(), 40.0 * (PI / 6.0).sin() * t.cos(), 40.0 * (PI / 6.0).sin() * t.sin()]);
        }
        let seed = seed_inside(&scan, &ring).expect("a seed");
        assert!((seed.length() - 40.0).abs() < 0.1, "{seed}");
        assert!(seed.x > 39.0, "{seed}");
    }

    #[test]
    fn lies_in_the_middle_of_a_strip_round_270_degrees_of_a_cylinder_which_folds_over_seen_along_any_one_way() {
        let r = 20.0;
        let m = grid_scan(180, 40, &|u, v| DVec3::new(r * (2.0 * PI * u).cos(), r * (2.0 * PI * u).sin(), 40.0 * v), &GridOptions { wrap: true, noise: 0.01, ..Default::default() });
        let scan = ScanPaths::new(m.positions, m.indices);
        let on = |deg: f64, z: f64| [r * ((deg * PI) / 180.0).cos(), r * ((deg * PI) / 180.0).sin(), z];
        let mut out = Vec::new();
        let mut a = 0.0;
        while a < 270.0 {
            out.extend_from_slice(&on(a, 5.0));
            a += 2.0;
        }
        let mut z = 5.0;
        while z < 35.0 {
            out.extend_from_slice(&on(270.0, z));
            z += 0.5;
        }
        let mut a = 270.0;
        while a > 0.0 {
            out.extend_from_slice(&on(a, 35.0));
            a -= 2.0;
        }
        let mut z = 35.0;
        while z > 5.0 {
            out.extend_from_slice(&on(0.0, z));
            z -= 0.5;
        }
        let seed = seed_inside(&scan, &out).expect("a seed");
        let angle = (seed.y.atan2(seed.x) * 180.0) / PI;
        let deg = if angle < 0.0 { angle + 360.0 } else { angle };
        assert!((seed.x.hypot(seed.y) - r).abs() < 0.1);
        assert!(deg > 5.0 && deg < 265.0, "{deg}");
        assert!(seed.z > 7.0 && seed.z < 33.0, "{}", seed.z);
    }

    /// ScanRuler's "the corners a patch was placed with": the corners the cells are read with
    /// and handed to the fit, and the patch fitted through them.
    #[test]
    fn the_corners_a_patch_was_placed_with_are_its_corners() {
        let m = grid_scan(100, 100, &|u, v| DVec3::new(100.0 * u, 100.0 * v, 0.0), &GridOptions { noise: 0.01, ..Default::default() });
        let scan = ScanPaths::new(m.positions, m.indices);
        // A square with a node in its bottom side's middle (5): by the rule its corners are
        // 1, 2, 3 and 4. Placed with 1, 5, 3 and 4 for its corners, they are those; 2 is a
        // bend in the side from 5 to 3.
        let mut f = network(
            &[(1, DVec3::new(20.0, 20.0, 0.0)), (5, DVec3::new(40.0, 20.0, 0.0)), (2, DVec3::new(60.0, 20.0, 0.0)), (3, DVec3::new(60.0, 60.0, 0.0)), (4, DVec3::new(20.0, 60.0, 0.0))],
            &[curve(1, 5), curve(5, 2), curve(2, 3), curve(3, 4), curve(4, 1)],
            &[DVec3::new(40.0, 40.0, 0.0)],
        );
        let corners_of = |input: crate::network::NetworkFitInput<'_>| -> Vec<Vec<u32>> {
            input
                .patches
                .iter()
                .map(|p| {
                    let mut c = p.corners.clone();
                    c.sort_unstable();
                    c
                })
                .collect()
        };
        let by_rule = read_network(&scan, &f, None, corners_of);
        assert_eq!(by_rule.fit.expect("a patch")[0], vec![1, 2, 3, 4]);
        f.cells[0].corners = Some(vec![1, 5, 3, 4]);
        let placed = read_network(&scan, &f, None, corners_of);
        assert!(matches!(placed.outcomes[0], crate::network::CellOutcome::Filled { sides: 4, .. }), "{:?}", placed.outcomes[0]);
        assert_eq!(placed.fit.expect("a patch")[0], vec![1, 3, 4, 5]);
        let fitted = crate::network_fit::read_network_fitted(&scan, &f, None).map_err(|e| e.0).expect("a fit");
        let fit = fitted.fit.expect("a patch");
        let mut corners = fit.patches[0].corners.clone();
        corners.sort_unstable();
        assert_eq!(corners, vec![1, 3, 4, 5]);
        assert!(fit.patches[0].rms < 0.02, "rms {}", fit.patches[0].rms);
    }
}
