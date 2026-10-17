//! Small geometric primitives: bounding boxes, rays, planes, triangle queries.

use glam::DVec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: DVec3,
    pub max: DVec3,
}

impl Aabb {
    pub const EMPTY: Aabb = Aabb { min: DVec3::splat(f64::INFINITY), max: DVec3::splat(f64::NEG_INFINITY) };

    pub fn from_points(pts: impl IntoIterator<Item = DVec3>) -> Self {
        let mut b = Self::EMPTY;
        for p in pts {
            b.include(p);
        }
        b
    }
    #[inline]
    pub fn include(&mut self, p: DVec3) {
        self.min = self.min.min(p);
        self.max = self.max.max(p);
    }
    #[inline]
    pub fn union(&self, o: &Aabb) -> Aabb {
        Aabb { min: self.min.min(o.min), max: self.max.max(o.max) }
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.min.x > self.max.x
    }
    #[inline]
    pub fn center(&self) -> DVec3 {
        (self.min + self.max) * 0.5
    }
    #[inline]
    pub fn extents(&self) -> DVec3 {
        if self.is_empty() {
            DVec3::ZERO
        } else {
            self.max - self.min
        }
    }
    pub fn diagonal(&self) -> f64 {
        self.extents().length()
    }
    pub fn expanded(&self, r: f64) -> Aabb {
        Aabb { min: self.min - DVec3::splat(r), max: self.max + DVec3::splat(r) }
    }
    #[inline]
    pub fn contains(&self, p: DVec3) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
    #[inline]
    pub fn distance_squared(&self, p: DVec3) -> f64 {
        let d = (self.min - p).max(p - self.max).max(DVec3::ZERO);
        d.length_squared()
    }
    #[inline]
    pub fn intersects(&self, o: &Aabb) -> bool {
        self.min.cmple(o.max).all() && o.min.cmple(self.max).all()
    }
    /// Slab test; returns entry distance if the ray hits within [0, t_max].
    #[inline]
    pub fn ray_hit(&self, origin: DVec3, inv_dir: DVec3, t_max: f64) -> Option<f64> {
        let t1 = (self.min - origin) * inv_dir;
        let t2 = (self.max - origin) * inv_dir;
        let tmin = t1.min(t2).max_element().max(0.0);
        let tmax = t1.max(t2).min_element().min(t_max);
        if tmin <= tmax {
            Some(tmin)
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ray {
    pub origin: DVec3,
    /// Unit direction.
    pub dir: DVec3,
}

impl Ray {
    pub fn new(origin: DVec3, dir: DVec3) -> Self {
        Self { origin, dir: dir.normalize() }
    }
    pub fn at(&self, t: f64) -> DVec3 {
        self.origin + self.dir * t
    }
}

/// Plane `{x : normal·x = d}` with unit normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Plane {
    pub normal: DVec3,
    pub d: f64,
}

impl Plane {
    pub fn from_point_normal(p: DVec3, n: DVec3) -> Self {
        let n = n.normalize();
        Self { normal: n, d: n.dot(p) }
    }
    #[inline]
    pub fn signed_distance(&self, p: DVec3) -> f64 {
        self.normal.dot(p) - self.d
    }
    pub fn origin(&self) -> DVec3 {
        self.normal * self.d
    }
    pub fn project(&self, p: DVec3) -> DVec3 {
        p - self.normal * self.signed_distance(p)
    }
}

/// Möller–Trumbore ray/triangle intersection (two-sided). Returns (t, u, v) where the hit
/// point is `a*(1-u-v) + b*u + c*v`.
#[inline]
pub fn ray_triangle(ray: &Ray, a: DVec3, b: DVec3, c: DVec3) -> Option<(f64, f64, f64)> {
    let e1 = b - a;
    let e2 = c - a;
    let p = ray.dir.cross(e2);
    let det = e1.dot(p);
    if det.abs() < 1e-30 {
        return None;
    }
    let inv = 1.0 / det;
    let s = ray.origin - a;
    let u = s.dot(p) * inv;
    if !(-1e-12..=1.0 + 1e-12).contains(&u) {
        return None;
    }
    let q = s.cross(e1);
    let v = ray.dir.dot(q) * inv;
    if v < -1e-12 || u + v > 1.0 + 1e-12 {
        return None;
    }
    let t = e2.dot(q) * inv;
    if t < 0.0 {
        return None;
    }
    Some((t, u, v))
}

/// Closest point on triangle (a,b,c) to p. Returns (point, barycentric (wa, wb, wc)).
/// From Ericson, "Real-Time Collision Detection", 5.1.5.
pub fn closest_point_on_triangle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> (DVec3, DVec3) {
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = ab.dot(ap);
    let d2 = ac.dot(ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return (a, DVec3::new(1.0, 0.0, 0.0));
    }
    let bp = p - b;
    let d3 = ab.dot(bp);
    let d4 = ac.dot(bp);
    if d3 >= 0.0 && d4 <= d3 {
        return (b, DVec3::new(0.0, 1.0, 0.0));
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return (a + ab * v, DVec3::new(1.0 - v, v, 0.0));
    }
    let cp = p - c;
    let d5 = ab.dot(cp);
    let d6 = ac.dot(cp);
    if d6 >= 0.0 && d5 <= d6 {
        return (c, DVec3::new(0.0, 0.0, 1.0));
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return (a + ac * w, DVec3::new(1.0 - w, 0.0, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return (b + (c - b) * w, DVec3::new(0.0, 1.0 - w, w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    (a + ab * v + ac * w, DVec3::new(1.0 - v - w, v, w))
}

/// Build an orthonormal frame (u, v) perpendicular to unit vector n.
pub fn orthonormal_basis(n: DVec3) -> (DVec3, DVec3) {
    let n = n.normalize();
    let sign = if n.z >= 0.0 { 1.0 } else { -1.0 };
    let a = -1.0 / (sign + n.z);
    let b = n.x * n.y * a;
    let u = DVec3::new(1.0 + sign * n.x * n.x * a, sign * b, -sign * n.x);
    let v = DVec3::new(b, sign + n.y * n.y * a, -n.y);
    (u, v)
}

/// Solid angle subtended by triangle (a,b,c) at p (Van Oosterom & Strackee), signed by
/// orientation. Summing over a closed, outward-oriented mesh gives 4π inside, 0 outside.
#[inline]
pub fn triangle_solid_angle(p: DVec3, a: DVec3, b: DVec3, c: DVec3) -> f64 {
    let a = a - p;
    let b = b - p;
    let c = c - p;
    let la = a.length();
    let lb = b.length();
    let lc = c.length();
    let num = a.dot(b.cross(c));
    let den = la * lb * lc + a.dot(b) * lc + b.dot(c) * la + c.dot(a) * lb;
    2.0 * num.atan2(den)
}
