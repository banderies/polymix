// SPDX-License-Identifier: AGPL-3.0-only
//! How much of a surface of revolution a fit rests on: the arc it covers round the axis and
//! how far it reaches along it. A scan only ever sees part of a bore or a shaft, so the
//! cylinder and the cone report both — the arc says how well the radius is pinned down, the
//! axial reach where the element starts and ends.
//!
//! Ported from ScanRuler (`src/core/fit/extents.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use glam::DVec3;

use super::linalg3::{js_max, ortho_basis};

/// Angles beyond which the coverage estimate stops sampling — plenty to resolve the widest
/// gap in any real scan.
const COVERAGE_SAMPLES: usize = 20_000;

/// The arc a patch covers round an axis and its reach along it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxialExtents {
    /// Degrees of the way round the axis the patch reaches: 360° less the widest gap.
    pub coverage: f64,
    /// The least and greatest axial position, from the axis point along the axis.
    pub min_t: f64,
    pub max_t: f64,
}

/// How much of the way around the axis a fitted patch reaches, in degrees, plus the axial
/// extent of the patch — shared by the cylinder and the cone, whose fits both rest on part
/// of a surface of revolution. A scan only ever sees part of one; both numbers say how much
/// of it this fit actually rests on. Coverage is 360° minus the widest empty gap, which —
/// unlike counting occupied bins — does not shrink just because the mesh is coarse.
/// `point` is a point on the axis, `axis` its unit direction.
pub fn axial_extents(positions: &[DVec3], region: &[u32], point: DVec3, axis: DVec3) -> AxialExtents {
    let (u, v) = ortho_basis(axis);
    let mut min_t = f64::INFINITY;
    let mut max_t = f64::NEG_INFINITY;

    let step = region.len().div_ceil(COVERAGE_SAMPLES).max(1);
    let mut angles = Vec::new();
    for (i, &vi) in region.iter().enumerate() {
        let q = positions[vi as usize] - point;
        let t = q.x * axis.x + q.y * axis.y + q.z * axis.z;
        if t < min_t {
            min_t = t;
        }
        if t > max_t {
            max_t = t;
        }
        if i % step == 0 {
            angles.push(q.dot(v).atan2(q.dot(u)));
        }
    }
    if angles.len() < 2 {
        return AxialExtents { coverage: 0.0, min_t, max_t };
    }

    angles.sort_unstable_by(|a, b| a.total_cmp(b));
    let last = angles.len() - 1;
    let mut widest = angles[0] + 2.0 * std::f64::consts::PI - angles[last];
    for i in 1..angles.len() {
        let gap = angles[i] - angles[i - 1];
        if gap > widest {
            widest = gap;
        }
    }
    let coverage = js_max(0.0, 360.0 - (widest * 180.0) / std::f64::consts::PI);
    AxialExtents { coverage, min_t, max_t }
}

#[cfg(test)]
mod tests {
    // Not in ScanRuler's tests (it checks coverage through the seed fits): a half turn of a
    // wall reads 180° less one step, and its axial reach is the band sampled.
    use super::*;

    #[test]
    fn a_half_turn_covers_about_180_degrees() {
        let mut pts = Vec::new();
        for i in 0..=90 {
            let a = (i as f64 / 90.0) * std::f64::consts::PI;
            for z in [-2.0, 3.0] {
                pts.push(DVec3::new(5.0 * a.cos(), 5.0 * a.sin(), z));
            }
        }
        let region: Vec<u32> = (0..pts.len() as u32).collect();
        let e = axial_extents(&pts, &region, DVec3::ZERO, DVec3::Z);
        assert!((e.coverage - 180.0).abs() < 1e-9);
        assert_eq!((e.min_t, e.max_t), (-2.0, 3.0));
        let one = axial_extents(&pts, &[0], DVec3::ZERO, DVec3::Z);
        assert_eq!(one.coverage, 0.0);
    }
}
