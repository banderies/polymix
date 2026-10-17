//! Symmetric 4x4 error quadrics (Garland & Heckbert 1997).
//!
//! `Q(x) = xᵀ A x + 2 bᵀ x + c`; a plane `n·x + d = 0` (unit `n`) with weight `w`
//! contributes `w (n·x + d)²`, i.e. `w` times the squared distance to the plane.

use mm_core::{DMat3, DVec3};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) struct Quadric {
    /// xx, xy, xz, yy, yz, zz
    a: [f64; 6],
    b: DVec3,
    c: f64,
}

impl Quadric {
    pub fn from_plane(n: DVec3, d: f64, w: f64) -> Self {
        Quadric {
            a: [w * n.x * n.x, w * n.x * n.y, w * n.x * n.z, w * n.y * n.y, w * n.y * n.z, w * n.z * n.z],
            b: n * (w * d),
            c: w * d * d,
        }
    }

    /// Plane through `p` with unit normal `n`.
    pub fn from_point_normal(p: DVec3, n: DVec3, w: f64) -> Self {
        Self::from_plane(n, -n.dot(p), w)
    }

    #[inline]
    pub fn add(&mut self, o: &Quadric) {
        for i in 0..6 {
            self.a[i] += o.a[i];
        }
        self.b += o.b;
        self.c += o.c;
    }

    #[inline]
    pub fn sum(&self, o: &Quadric) -> Quadric {
        let mut q = *self;
        q.add(o);
        q
    }

    #[inline]
    fn mul_a(&self, v: DVec3) -> DVec3 {
        let a = &self.a;
        DVec3::new(
            a[0] * v.x + a[1] * v.y + a[2] * v.z,
            a[1] * v.x + a[3] * v.y + a[4] * v.z,
            a[2] * v.x + a[4] * v.y + a[5] * v.z,
        )
    }

    /// Error at `p` (clamped at 0 against round-off).
    #[inline]
    pub fn eval(&self, p: DVec3) -> f64 {
        (p.dot(self.mul_a(p)) + 2.0 * self.b.dot(p) + self.c).max(0.0)
    }

    /// Minimizer of the quadric if `A` is well conditioned.
    pub fn solve(&self) -> Option<DVec3> {
        let a = &self.a;
        let m = DMat3::from_cols(
            DVec3::new(a[0], a[1], a[2]),
            DVec3::new(a[1], a[3], a[4]),
            DVec3::new(a[2], a[4], a[5]),
        );
        let det = m.determinant();
        // Scale-free conditioning test: det relative to (trace/3)^3 is the product of
        // the normalized eigenvalues for a PSD matrix.
        let tr = (a[0] + a[3] + a[5]) / 3.0;
        if tr.is_nan() || tr <= 0.0 || det.abs() < 1e-10 * tr * tr * tr {
            return None;
        }
        let x = m.inverse() * (-self.b);
        x.is_finite().then_some(x)
    }

    /// Minimizer on the segment `[p0, p1]`.
    pub fn min_on_segment(&self, p0: DVec3, p1: DVec3) -> DVec3 {
        let d = p1 - p0;
        let ad = self.mul_a(d);
        let den = d.dot(ad);
        let t = if den > 1e-300 { (-(p0.dot(ad) + self.b.dot(d)) / den).clamp(0.0, 1.0) } else { 0.5 };
        p0 + d * t
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn corner_of_three_planes() {
        let mut q = Quadric::default();
        q.add(&Quadric::from_point_normal(DVec3::new(1.0, 0.0, 0.0), DVec3::X, 1.0));
        q.add(&Quadric::from_point_normal(DVec3::new(0.0, 2.0, 0.0), DVec3::Y, 1.0));
        q.add(&Quadric::from_point_normal(DVec3::new(0.0, 0.0, 3.0), DVec3::Z, 1.0));
        let x = q.solve().unwrap();
        assert!((x - DVec3::new(1.0, 2.0, 3.0)).length() < 1e-12);
        assert!(q.eval(x) < 1e-12);
        assert!((q.eval(DVec3::ZERO) - 14.0).abs() < 1e-12);
        let flat = Quadric::from_point_normal(DVec3::ZERO, DVec3::Z, 1.0);
        assert!(flat.solve().is_none());
        let p = flat.min_on_segment(DVec3::new(0.0, 0.0, -1.0), DVec3::new(0.0, 0.0, 3.0));
        assert!(p.length() < 1e-12);
    }
}
