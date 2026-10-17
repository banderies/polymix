//! Small internal helpers: a fast deterministic hasher, sparse/dense vertex index maps,
//! best-fit plane frames and 2D polygon predicates.

use glam::{DVec2, DVec3};
use mm_core::INVALID;
use std::hash::{BuildHasherDefault, Hasher};

/// FxHash-style multiplicative hasher (deterministic, fast for integer keys).
#[derive(Default, Clone, Copy)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, x: u64) {
        self.hash = (self.hash.rotate_left(5) ^ x).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }
}

impl Hasher for FxHasher {
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.add(b as u64);
        }
    }
    fn write_u32(&mut self, i: u32) {
        self.add(i as u64);
    }
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    fn finish(&self) -> u64 {
        self.hash
    }
}

pub(crate) type FxBuild = BuildHasherDefault<FxHasher>;
pub(crate) type FxHashMap<K, V> = std::collections::HashMap<K, V, FxBuild>;
pub(crate) type FxHashSet<K> = std::collections::HashSet<K, FxBuild>;

/// Map from (sparse, recycled) mesh vertex ids to dense local indices. Uses a dense array
/// when the local set is a sizeable fraction of the id range, a hash map otherwise.
pub(crate) enum LocalIndex {
    Dense(Vec<u32>),
    Sparse(FxHashMap<u32, u32>),
}

impl LocalIndex {
    pub fn new(max_id: usize, expected: usize) -> Self {
        if expected.saturating_mul(16) >= max_id {
            LocalIndex::Dense(vec![INVALID; max_id])
        } else {
            let mut m = FxHashMap::default();
            m.reserve(expected);
            LocalIndex::Sparse(m)
        }
    }
    #[inline]
    pub fn get(&self, v: u32) -> Option<u32> {
        match self {
            LocalIndex::Dense(d) => d.get(v as usize).copied().filter(|&x| x != INVALID),
            LocalIndex::Sparse(m) => m.get(&v).copied(),
        }
    }
    #[inline]
    pub fn insert(&mut self, v: u32, local: u32) {
        match self {
            LocalIndex::Dense(d) => d[v as usize] = local,
            LocalIndex::Sparse(m) => {
                m.insert(v, local);
            }
        }
    }
}

/// Orthonormal frame of a best-fit plane.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PlaneFrame {
    pub origin: DVec3,
    pub normal: DVec3,
    pub u: DVec3,
    pub v: DVec3,
}

impl PlaneFrame {
    /// Plane through the centroid of a closed polygon, with Newell's normal (so the polygon
    /// winds counter-clockwise when seen from the normal). Falls back to the least-variance
    /// direction of the points when the polygon has (near) zero vector area.
    pub fn from_polygon(pts: &[DVec3]) -> Self {
        let n = pts.len().max(1) as f64;
        let c = pts.iter().copied().sum::<DVec3>() / n;
        let mut nrm = newell_normal(pts, c);
        let scale: f64 = pts.iter().map(|p| p.distance_squared(c)).sum::<f64>() / n;
        if nrm.length_squared() <= 1e-24 * scale * scale.max(1e-300) || !nrm.is_finite() {
            nrm = least_variance_direction(pts, c);
        }
        let normal = nrm.normalize_or(DVec3::Z);
        let (u, v) = mm_core::geom::orthonormal_basis(normal);
        PlaneFrame { origin: c, normal, u, v }
    }
    #[inline]
    pub fn project(&self, p: DVec3) -> DVec2 {
        let d = p - self.origin;
        DVec2::new(d.dot(self.u), d.dot(self.v))
    }
    #[inline]
    pub fn height(&self, p: DVec3) -> f64 {
        (p - self.origin).dot(self.normal)
    }
    #[inline]
    pub fn project_to_plane(&self, p: DVec3) -> DVec3 {
        p - self.normal * self.height(p)
    }
}

/// Newell's polygon normal (length = twice the vector area).
pub(crate) fn newell_normal(pts: &[DVec3], c: DVec3) -> DVec3 {
    let n = pts.len();
    let mut s = DVec3::ZERO;
    for i in 0..n {
        s += (pts[i] - c).cross(pts[(i + 1) % n] - c);
    }
    s
}

/// Eigenvector of the smallest eigenvalue of the covariance of `pts` (cyclic Jacobi).
fn least_variance_direction(pts: &[DVec3], c: DVec3) -> DVec3 {
    let mut a = [[0.0f64; 3]; 3];
    for p in pts {
        let d = *p - c;
        let d = [d.x, d.y, d.z];
        for i in 0..3 {
            for j in 0..3 {
                a[i][j] += d[i] * d[j];
            }
        }
    }
    let (vals, vecs) = jacobi_eigen3(a);
    let mut k = 0;
    for i in 1..3 {
        if vals[i] < vals[k] {
            k = i;
        }
    }
    DVec3::new(vecs[0][k], vecs[1][k], vecs[2][k])
}

/// Eigen-decomposition of a symmetric 3x3 matrix; eigenvectors are the columns.
fn jacobi_eigen3(mut a: [[f64; 3]; 3]) -> ([f64; 3], [[f64; 3]; 3]) {
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..50 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1e-300 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1e-300 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let cs = 1.0 / (t * t + 1.0).sqrt();
            let sn = t * cs;
            for row in a.iter_mut() {
                let (akp, akq) = (row[p], row[q]);
                row[p] = cs * akp - sn * akq;
                row[q] = sn * akp + cs * akq;
            }
            let (rp, rq) = (a[p], a[q]);
            for k in 0..3 {
                a[p][k] = cs * rp[k] - sn * rq[k];
                a[q][k] = sn * rp[k] + cs * rq[k];
            }
            for row in v.iter_mut() {
                let vkp = row[p];
                let vkq = row[q];
                row[p] = cs * vkp - sn * vkq;
                row[q] = sn * vkp + cs * vkq;
            }
        }
    }
    ([a[0][0], a[1][1], a[2][2]], v)
}

#[inline]
pub(crate) fn orient2(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b - a).perp_dot(c - a)
}

#[inline]
fn on_segment(a: DVec2, b: DVec2, p: DVec2, eps: f64) -> bool {
    p.x >= a.x.min(b.x) - eps && p.x <= a.x.max(b.x) + eps && p.y >= a.y.min(b.y) - eps && p.y <= a.y.max(b.y) + eps
}

/// Closed segments (p1,p2) and (p3,p4) intersect or touch. `eps` is an area tolerance.
fn segments_intersect(p1: DVec2, p2: DVec2, p3: DVec2, p4: DVec2, eps: f64, len_eps: f64) -> bool {
    let d1 = orient2(p3, p4, p1);
    let d2 = orient2(p3, p4, p2);
    let d3 = orient2(p1, p2, p3);
    let d4 = orient2(p1, p2, p4);
    if ((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) && ((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps)) {
        return true;
    }
    (d1.abs() <= eps && on_segment(p3, p4, p1, len_eps))
        || (d2.abs() <= eps && on_segment(p3, p4, p2, len_eps))
        || (d3.abs() <= eps && on_segment(p1, p2, p3, len_eps))
        || (d4.abs() <= eps && on_segment(p1, p2, p4, len_eps))
}

/// True if the closed 2D polygon has no self-intersections, no zero-length edges and no
/// back-tracking edges. Uses a uniform grid, so it is roughly linear in the vertex count.
pub(crate) fn polygon_is_simple(q: &[DVec2]) -> bool {
    let n = q.len();
    if n < 3 {
        return false;
    }
    let mut lo = DVec2::splat(f64::INFINITY);
    let mut hi = DVec2::splat(f64::NEG_INFINITY);
    let mut total_len = 0.0;
    for i in 0..n {
        lo = lo.min(q[i]);
        hi = hi.max(q[i]);
        total_len += q[i].distance(q[(i + 1) % n]);
    }
    let diag = (hi - lo).length();
    if diag.is_nan() || diag <= 0.0 || !diag.is_finite() {
        return false;
    }
    let len_eps = 1e-12 * diag;
    let eps = 1e-12 * diag * diag;
    for i in 0..n {
        if q[i].distance(q[(i + 1) % n]) <= len_eps {
            return false;
        }
    }
    // Grid with about n cells, cell size at least the mean edge length.
    let mean = total_len / n as f64;
    let ext = hi - lo;
    let mut h = mean.max((ext.x * ext.y / n as f64).sqrt()).max(diag * 1e-6);
    let mut nx = ((ext.x / h).ceil() as usize).max(1);
    let mut ny = ((ext.y / h).ceil() as usize).max(1);
    while nx * ny > 4 * n + 16 {
        h *= 1.5;
        nx = ((ext.x / h).ceil() as usize).max(1);
        ny = ((ext.y / h).ceil() as usize).max(1);
    }
    let cell = |p: f64, l: f64, k: usize| (((p - l) / h).floor().max(0.0) as usize).min(k - 1);
    let mut grid: Vec<Vec<u32>> = vec![Vec::new(); nx * ny];
    for i in 0..n {
        let a = q[i];
        let b = q[(i + 1) % n];
        let (x0, x1) = (cell(a.x.min(b.x), lo.x, nx), cell(a.x.max(b.x), lo.x, nx));
        let (y0, y1) = (cell(a.y.min(b.y), lo.y, ny), cell(a.y.max(b.y), lo.y, ny));
        for y in y0..=y1 {
            for x in x0..=x1 {
                grid[y * nx + x].push(i as u32);
            }
        }
    }
    for c in &grid {
        for (ii, &i) in c.iter().enumerate() {
            let i = i as usize;
            let (a, b) = (q[i], q[(i + 1) % n]);
            for &j in &c[ii + 1..] {
                let j = j as usize;
                let (c0, d0) = (q[j], q[(j + 1) % n]);
                let adjacent_next = (i + 1) % n == j;
                let adjacent_prev = (j + 1) % n == i;
                if adjacent_next || adjacent_prev {
                    // Shared vertex; reject only collinear back-tracking.
                    let (p, s, r) = if adjacent_next { (a, b, d0) } else { (c0, a, b) };
                    let cr = orient2(p, s, r);
                    if cr.abs() <= eps && (s - p).dot(r - s) < 0.0 {
                        return false;
                    }
                    continue;
                }
                if segments_intersect(a, b, c0, d0, eps, len_eps) {
                    return false;
                }
            }
        }
    }
    true
}

/// Signed area of a 2D polygon (positive when counter-clockwise).
pub(crate) fn signed_area2(q: &[DVec2]) -> f64 {
    let n = q.len();
    let mut s = 0.0;
    for i in 0..n {
        s += q[i].perp_dot(q[(i + 1) % n]);
    }
    0.5 * s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_polygon_checks() {
        let sq = [DVec2::new(0.0, 0.0), DVec2::new(1.0, 0.0), DVec2::new(1.0, 1.0), DVec2::new(0.0, 1.0)];
        assert!(polygon_is_simple(&sq));
        let bow = [DVec2::new(0.0, 0.0), DVec2::new(1.0, 1.0), DVec2::new(1.0, 0.0), DVec2::new(0.0, 1.0)];
        assert!(!polygon_is_simple(&bow));
        let circle: Vec<DVec2> = (0..500)
            .map(|i| {
                let t = i as f64 / 500.0 * std::f64::consts::TAU;
                DVec2::new(t.cos(), t.sin())
            })
            .collect();
        assert!(polygon_is_simple(&circle));
        assert!(signed_area2(&circle) > 3.0);
    }

    #[test]
    fn plane_frame_of_tilted_square() {
        let pts = [DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 1.0), DVec3::new(1.0, 1.0, 1.0), DVec3::new(0.0, 1.0, 0.0)];
        let f = PlaneFrame::from_polygon(&pts);
        let expect = DVec3::new(-1.0, 0.0, 1.0).normalize();
        assert!(f.normal.dot(expect) > 0.999);
        for p in pts {
            assert!(f.height(p).abs() < 1e-12);
        }
    }

    #[test]
    fn jacobi_eigen_finds_flat_direction() {
        let pts: Vec<DVec3> = (0..20).map(|i| DVec3::new((i as f64).cos() * 3.0, (i as f64 * 0.7).sin(), 0.0)).collect();
        let d = least_variance_direction(&pts, DVec3::ZERO);
        assert!(d.z.abs() > 0.999);
    }
}
