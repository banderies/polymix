// SPDX-License-Identifier: AGPL-3.0-only
//! The loops a profile is made of, as the kernel takes them: lines, arcs and circles in the
//! part's coordinates. Ported from ScanRuler (`plugins/reverse/core/kernelTypes.ts` — the
//! `LoopPiece` and `Profile` types; `plugins/reverse/core/loops.ts` — `piecePoints`;
//! `plugins/reverse/core/kernelMath.ts` — `circleThrough`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! An arc is given by three points on it rather than by centre and angles, because that is
//! the one description no two kernels read differently. ScanRuler's loop pieces also carry a
//! spline (a clamped cubic B-spline); none of the geometry ported here makes or reads one, so
//! it is left to the module that ports the sketch's splines.

use std::f64::consts::{PI, TAU};

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// One piece of a profile loop, in the part's coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum LoopPiece {
    Line { a: DVec3, b: DVec3 },
    /// From `a` through `m` to `b`.
    Arc { a: DVec3, m: DVec3, b: DVec3 },
    Circle { center: DVec3, normal: DVec3, radius: f64 },
}

impl LoopPiece {
    /// The piece run the other way: a line's or an arc's ends swapped; a circle as it is.
    pub fn reversed(&self) -> LoopPiece {
        match *self {
            LoopPiece::Line { a, b } => LoopPiece::Line { a: b, b: a },
            LoopPiece::Arc { a, m, b } => LoopPiece::Arc { a: b, m, b: a },
            c => c,
        }
    }

    /// The kind's name as ScanRuler writes it: `line`, `arc` or `circle`.
    pub fn kind_name(&self) -> &'static str {
        match self {
            LoopPiece::Line { .. } => "line",
            LoopPiece::Arc { .. } => "arc",
            LoopPiece::Circle { .. } => "circle",
        }
    }
}

/// A closed region to make a face of: an outer loop and the loops of any holes in it, every
/// loop closed end to start.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Profile {
    pub outer: Vec<LoopPiece>,
    pub holes: Vec<Vec<LoopPiece>>,
}

/// The circle through three points: its centre, radius and the axis about which `a → b → c`
/// runs counter-clockwise. `None` for three points in a line.
pub fn circle_through(a: DVec3, b: DVec3, c: DVec3) -> Option<(DVec3, f64, DVec3)> {
    let ab = b - a;
    let ac = c - a;
    let n = ab.cross(ac);
    let nn = n.dot(n);
    if nn < 1e-18 {
        return None;
    }
    // The circumcentre in the plane of the three, by the standard formula.
    let t = (n.cross(ab) * ac.dot(ac) - n.cross(ac) * ab.dot(ab)) / (2.0 * nn);
    let center = a + t;
    Some((center, t.length(), normalize3(n).unwrap_or(DVec3::Z)))
}

/// A unit vector, or `None` for one too short to have a direction (under 1e-12).
pub(crate) fn normalize3(a: DVec3) -> Option<DVec3> {
    let l = a.length();
    if l < 1e-12 {
        None
    } else {
        Some(a / l)
    }
}

/// The default step `piece_points` samples an arc or a circle at, radians.
pub const PIECE_STEP: f64 = PI / 64.0;

/// Points along a loop's piece, its ends included: a line's two ends, an arc or a circle every
/// `step` radians at most (a circle's last point is its first).
pub fn piece_points(piece: &LoopPiece, step: f64) -> Vec<DVec3> {
    let (center, radius, axis, start, sweep) = match *piece {
        LoopPiece::Line { a, b } => return vec![a, b],
        LoopPiece::Circle { center, normal, radius } => {
            let axis = normalize3(normal).unwrap_or(DVec3::Z);
            let side = if axis.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
            let start = center + normalize3(axis.cross(side)).unwrap_or(DVec3::X) * radius;
            (center, radius, axis, start, None)
        }
        LoopPiece::Arc { a, m, b } => match circle_through(a, m, b) {
            Some((center, radius, axis)) => (center, radius, axis, a, Some(b)),
            None => return vec![a, m, b],
        },
    };
    let u = normalize3(start - center).unwrap_or(DVec3::X);
    let v = axis.cross(u);
    // The axis circle_through gives runs a → m → b counter-clockwise, so the arc's sweep is
    // b's angle about it.
    let angle_of = |p: DVec3| {
        let a = (p - center).dot(v).atan2((p - center).dot(u));
        if a <= 0.0 {
            a + TAU
        } else {
            a
        }
    };
    let sweep = match sweep {
        Some(b) => angle_of(b),
        None => TAU,
    };
    let n = ((sweep / step).ceil() as usize).max(2);
    (0..=n)
        .map(|i| {
            let t = sweep * i as f64 / n as f64;
            center + u * (radius * t.cos()) + v * (radius * t.sin())
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn samples_an_arc_from_its_three_points() {
        let arc = LoopPiece::Arc { a: DVec3::new(10.0, 0.0, 0.0), m: DVec3::new(0.0, -10.0, 0.0), b: DVec3::new(-10.0, 0.0, 0.0) };
        let pts = piece_points(&arc, PIECE_STEP);
        assert!((pts[0] - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-12);
        assert!((pts[pts.len() - 1] - DVec3::new(-10.0, 0.0, 0.0)).length() < 1e-9);
        for p in &pts {
            assert!((p.length() - 10.0).abs() < 1e-9);
            assert!(p.y <= 1e-9);
        }
        let circle = LoopPiece::Circle { center: DVec3::ZERO, normal: DVec3::Z, radius: 3.0 };
        let ring = piece_points(&circle, PIECE_STEP);
        assert!((ring[0] - ring[ring.len() - 1]).length() < 1e-9);
    }
}
