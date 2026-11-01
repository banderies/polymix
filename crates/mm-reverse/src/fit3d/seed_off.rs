// SPDX-License-Identifier: AGPL-3.0-only
//! Where a seed lies against the surface a fit found from it and others: how far off the
//! surface, along its normal there, and how far the seed's own normal turns from that one —
//! what tells a click on another surface, which grew nothing, from one at the rim of the
//! region the fit grew.
//!
//! Ported from ScanRuler (`src/core/fit/seedOff.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

use super::cone::{cone_normal_align, cone_residual};
use super::linalg3::{hypot3, js_min};
use super::torus::{torus_normal_align, torus_residual};
use super::types::{Cone, FitData, Torus};

/// A seed read against a fitted surface.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeedOff {
    /// The seed's distance off the surface, along the surface's normal — a plane's as the
    /// fit gives it, a round surface's outward from its axis or centre.
    pub off_mm: f64,
    /// The turn between the seed's normal and the surface's there, 0 to 90°, either way
    /// round.
    pub deg: f64,
}

/// The angle a |cos| stands for, degrees.
fn deg_of(cos: f64) -> f64 {
    (js_min(1.0, cos.abs()).acos() * 180.0) / std::f64::consts::PI
}

/// Where `point`, with its vertex normal `normal`, lies against the fit's surface. `None`
/// for a kind with no surface, and on an axis or a centre, where the surface has no normal
/// to say.
pub fn seed_off(fit: &FitData, point: DVec3, normal: DVec3) -> Option<SeedOff> {
    let len = hypot3(normal.x, normal.y, normal.z);
    if !(len > 0.0) {
        return None;
    }
    let n = DVec3::new(normal.x / len, normal.y / len, normal.z / len);
    match fit {
        FitData::Plane(f) => Some(SeedOff { off_mm: (point - f.center).dot(f.normal), deg: deg_of(n.dot(f.normal)) }),
        FitData::Sphere(f) => {
            let d = point - f.center;
            let r = hypot3(d.x, d.y, d.z);
            (r > 1e-12).then(|| SeedOff { off_mm: r - f.radius, deg: deg_of(d.dot(n) / r) })
        }
        FitData::Cylinder(f) => {
            let d = point - f.center;
            let t = d.dot(f.axis);
            let w = DVec3::new(d.x - t * f.axis.x, d.y - t * f.axis.y, d.z - t * f.axis.z);
            let rho = hypot3(w.x, w.y, w.z);
            (rho > 1e-12).then(|| SeedOff { off_mm: rho - f.radius, deg: deg_of(w.dot(n) / rho) })
        }
        FitData::Cone(f) => {
            let c = Cone { point: f.center, axis: f.axis, r: f.radius, phi: (f.half_angle * std::f64::consts::PI) / 180.0 };
            let off = cone_residual(&c, point);
            off.is_finite().then(|| SeedOff { off_mm: off, deg: deg_of(cone_normal_align(&c, point, n)) })
        }
        FitData::Torus(f) => {
            let t = Torus { center: f.center, axis: f.axis, major: f.major_radius, minor: f.minor_radius };
            Some(SeedOff { off_mm: torus_residual(&t, point), deg: deg_of(torus_normal_align(&t, point, n)) })
        }
        FitData::Point(_) | FitData::Line(_) | FitData::Circle(_) => None,
    }
}

#[cfg(test)]
mod tests {
    // commandsSegment.test.ts — 'reads a seed against each kind of surface'.
    use super::*;
    use crate::fit3d::types::{ConeFit, CylinderFit, FitBase, PointFit, SphereFit, TorusFit};

    fn base() -> FitBase {
        FitBase { sigma: 0.01, used_points: 100, region_size: 100, ..FitBase::default() }
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn reads_a_seed_against_each_kind_of_surface() {
        // A bore of radius 5 along z, a seed 0.2 mm into the metal facing the axis.
        let bore = FitData::Cylinder(CylinderFit { base: base(), center: DVec3::ZERO, axis: DVec3::Z, radius: 5.0, length: 10.0, coverage: 360.0 });
        let r = seed_off(&bore, DVec3::new(5.2, 0.0, 1.0), DVec3::new(-1.0, 0.0, 0.0)).unwrap();
        assert!(close(r.off_mm, 0.2, 9));
        assert!(close(r.deg, 0.0, 4));
        // A ball, the seed on it but its normal 30° off.
        let s30 = std::f64::consts::PI / 6.0;
        let ball = FitData::Sphere(SphereFit { base: base(), center: DVec3::ZERO, radius: 10.0, held: None });
        let r = seed_off(&ball, DVec3::new(0.0, 0.0, 10.0), DVec3::new(s30.sin(), 0.0, s30.cos())).unwrap();
        assert!(close(r.off_mm, 0.0, 9));
        assert!(close(r.deg, 30.0, 6));
        // A 45° cone, radius 5 at its centre: a point 1 mm out square to its slant.
        let s = std::f64::consts::FRAC_1_SQRT_2;
        let cone = FitData::Cone(ConeFit {
            base: base(),
            center: DVec3::ZERO,
            axis: DVec3::Z,
            half_angle: 45.0,
            radius: 5.0,
            radius1: 4.0,
            radius2: 6.0,
            length: 2.0,
            coverage: 360.0,
        });
        let r = seed_off(&cone, DVec3::new(5.0 + s, 0.0, -s), DVec3::new(s, 0.0, -s)).unwrap();
        assert!(close(r.off_mm, 1.0, 9));
        assert!(close(r.deg, 0.0, 4));
        // A torus: on the tube's outer equator.
        let ring = FitData::Torus(TorusFit {
            base: base(),
            center: DVec3::ZERO,
            axis: DVec3::Z,
            major_radius: 20.0,
            minor_radius: 2.0,
            tube_coverage: 360.0,
            spine_coverage: 360.0,
        });
        let r = seed_off(&ring, DVec3::new(22.0, 0.0, 0.0), DVec3::X).unwrap();
        assert!(close(r.off_mm, 0.0, 9));
        assert!(close(r.deg, 0.0, 4));
        // A point has no surface.
        let point = FitData::Point(PointFit { base: FitBase::default(), center: DVec3::ZERO });
        assert!(seed_off(&point, DVec3::X, DVec3::X).is_none());
    }
}
