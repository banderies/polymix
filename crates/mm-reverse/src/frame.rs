// SPDX-License-Identifier: AGPL-3.0-only
//! Where a section cuts and where a sketch lies: a plane and the two in-plane axes the 2D
//! sheet is laid out along. Ported from ScanRuler (`src/core/section/frame.ts`), © Stefan
//! Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! `u × v = normal`, so laid flat the sheet is seen from the side the normal points to. A
//! sketch's sheet follows the rule FreeCAD, Fusion, Onshape and SolidWorks converged on: its
//! zero is the part's origin projected square onto the plane, and its X a coordinate axis
//! flattened into it.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::linalg::ortho_basis;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SectionFrame {
    pub origin: DVec3,
    pub normal: DVec3,
    /// The sheet's X, ScanRuler's `basisU`.
    #[serde(rename = "basisU", alias = "u")]
    pub u: DVec3,
    /// The sheet's Y, ScanRuler's `basisV`.
    #[serde(rename = "basisV", alias = "v")]
    pub v: DVec3,
}

/// One of the part's coordinate axes, written `x`, `y`, `z` as ScanRuler writes them.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WorldAxis {
    #[serde(rename = "x", alias = "X")]
    X,
    #[serde(rename = "y", alias = "Y")]
    Y,
    #[serde(rename = "z", alias = "Z")]
    Z,
}

impl WorldAxis {
    pub const ALL: [WorldAxis; 3] = [WorldAxis::X, WorldAxis::Y, WorldAxis::Z];

    pub fn dir(self) -> DVec3 {
        match self {
            WorldAxis::X => DVec3::X,
            WorldAxis::Y => DVec3::Y,
            WorldAxis::Z => DVec3::Z,
        }
    }

    /// The coordinate plane across the axis, as CAD names it: the XY plane is the one the Z
    /// axis stands on.
    pub fn plane_name(self) -> &'static str {
        match self {
            WorldAxis::X => "YZ plane",
            WorldAxis::Y => "XZ plane",
            WorldAxis::Z => "XY plane",
        }
    }
}

/// A line a section slides along: a point, a direction, and the in-plane X to keep.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CutAxis {
    pub origin: DVec3,
    pub dir: DVec3,
    pub u: DVec3,
}

/// A coordinate axis as a cut axis through `through` (the origin, or the scan's centre).
pub fn world_cut_axis(axis: WorldAxis, through: DVec3) -> CutAxis {
    let dir = axis.dir();
    let u = match axis {
        WorldAxis::X => DVec3::Y,
        WorldAxis::Y => DVec3::Z,
        WorldAxis::Z => DVec3::X,
    };
    CutAxis { origin: through - dir * through.dot(dir), dir, u }
}

fn in_plane_x(normal: DVec3, want: Option<DVec3>) -> DVec3 {
    if let Some(w) = want {
        let flat = w - normal * w.dot(normal);
        if flat.length() > 1e-12 {
            return flat.normalize();
        }
    }
    ortho_basis(normal).0
}

/// The coordinate axis a sketch's X is taken from: the one next round from the axis the
/// normal is nearest (Z's plane gets X, X's gets Y, Y's gets Z).
pub fn sketch_x_axis(normal: DVec3) -> DVec3 {
    let a = normal.abs();
    let nearest = if a.x >= a.y && a.x >= a.z {
        0
    } else if a.y >= a.z {
        1
    } else {
        2
    };
    match (nearest + 1) % 3 {
        0 => DVec3::X,
        1 => DVec3::Y,
        _ => DVec3::Z,
    }
}

/// A plane through an axis, turned `angle_deg` about it (ScanRuler's `radialFrame`): v runs up
/// the axis — turned to point along the positive side of the coordinate axis it lies nearest —
/// and u out along the radius. Zero degrees is the coordinate axis next round from that one laid
/// square to the axis, and the angle turns from it by the right hand about the axis. The origin
/// is the point on the axis given.
pub fn radial_frame(origin: DVec3, dir: DVec3, angle_deg: f64) -> Option<SectionFrame> {
    let mut up = dir.try_normalize()?;
    if !angle_deg.is_finite() {
        return None;
    }
    let mut k = 0;
    for i in 1..3 {
        if up[i].abs() > up[k].abs() {
            k = i;
        }
    }
    if up[k] < 0.0 {
        up = -up;
    }
    let zero = in_plane_x(up, Some(sketch_x_axis(up)));
    let quarter = up.cross(zero);
    let a = angle_deg.to_radians();
    let u = (zero * a.cos() + quarter * a.sin()).try_normalize()?;
    Some(SectionFrame { origin, normal: u.cross(up), u, v: up })
}

impl SectionFrame {
    /// The frame `offset` millimetres along an axis.
    pub fn along(axis: CutAxis, offset: f64, seed_u: Option<DVec3>) -> Option<SectionFrame> {
        let normal = axis.dir.try_normalize()?;
        let origin = axis.origin + normal * if offset.is_finite() { offset } else { 0.0 };
        let u = in_plane_x(normal, seed_u.or(Some(axis.u)));
        Some(SectionFrame { origin, normal, u, v: normal.cross(u) })
    }

    /// The frame a sketch takes on a plane: the sheet's zero is the part's origin projected
    /// onto it, its X a coordinate axis flattened into it (`seed_u` keeps one in use).
    pub fn sketch_on(plane_origin: DVec3, plane_normal: DVec3, seed_u: Option<DVec3>) -> Option<SectionFrame> {
        let normal = plane_normal.try_normalize()?;
        let origin = normal * plane_origin.dot(normal);
        let u = in_plane_x(normal, Some(seed_u.unwrap_or_else(|| sketch_x_axis(normal))));
        Some(SectionFrame { origin, normal, u, v: normal.cross(u) })
    }

    /// The offset along an axis that puts the plane through `p`.
    pub fn offset_through(axis: CutAxis, p: DVec3) -> Option<f64> {
        let n = axis.dir.try_normalize()?;
        Some((p - axis.origin).dot(n))
    }

    /// A part point laid flat on the sheet.
    pub fn to_sheet(&self, p: DVec3) -> DVec2 {
        let d = p - self.origin;
        DVec2::new(d.dot(self.u), d.dot(self.v))
    }

    /// A sheet point stood up in the part.
    pub fn to_world(&self, p: DVec2) -> DVec3 {
        self.origin + self.u * p.x + self.v * p.y
    }

    /// The same plane moved `d` along its normal.
    pub fn offset(&self, d: f64) -> SectionFrame {
        SectionFrame { origin: self.origin + self.normal * d, ..*self }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sketch_on_xy_has_the_parts_own_axes() {
        let f = SectionFrame::sketch_on(DVec3::new(5.0, 7.0, 20.0), DVec3::Z, None).unwrap();
        assert_eq!(f.origin, DVec3::new(0.0, 0.0, 20.0));
        assert_eq!(f.u, DVec3::X);
        assert_eq!(f.v, DVec3::Y);
        let p = DVec3::new(3.0, 4.0, 20.0);
        assert!((f.to_world(f.to_sheet(p)) - p).length() < 1e-12);
    }
}
