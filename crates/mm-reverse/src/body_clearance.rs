// SPDX-License-Identifier: AGPL-3.0-only
//! How two solid bodies stand to each other: the least distance between them, or — where they
//! touch or run into each other — none, and the volume they share; and what every pair of solid
//! bodies shares, meshed, for the viewport to draw in red where parts run into each other.
//!
//! Ported from ScanRuler (`plugins/reverse/core/kernel.ts` `clearance`, `app/reSession.ts`
//! `clearance`, `core/overlapMesh.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//! ScanRuler measures the gap with Open CASCADE's distance and the shared solid with Manifold on
//! the display meshes; here both run on the SolidCraft kernel: its exact body distance and its
//! boolean intersection, the shared solid tessellated as a body is.

use glam::DVec3;
use solidcraft_kernel::{self as kernel, BoolOp, Body, DistanceShape};

use crate::build::{body_mesh, dv3, BodyMesh};
use crate::display_units::{fmt_len, fmt_vol, DisplayUnits, FmtOptions};
use crate::model::BodyId;

/// How two solids stand: the gap between them, mm, and — where it is none — the volume they
/// share, mm³.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clearance {
    pub gap: f64,
    pub overlap: f64,
}

fn kerr(e: kernel::KernelError) -> String {
    e.to_string()
}

/// The chord a body's distance mesh is cut at: a fine one, off the body's size.
fn chord_of(body: &Body) -> f64 {
    let bb = body.outline_bounds();
    let size = if bb.is_empty() { 10.0 } else { (dv3(bb.max) - dv3(bb.min)).length() };
    (size * 2e-4).clamp(0.001, 0.05)
}

/// The least distance between two solids, or — where they touch or run into each other — none,
/// and the volume they share.
pub fn clearance(a: &Body, b: &Body) -> Result<Clearance, String> {
    let (ca, cb) = (chord_of(a), chord_of(b));
    let ma = a.display_mesh(ca).map_err(kerr)?;
    let mb = b.display_mesh(cb).map_err(kerr)?;
    let sa = DistanceShape::new(a, &ma, ca);
    let sb = DistanceShape::new(b, &mb, cb);
    let gap = kernel::body_distance(&sa, &sb, f64::INFINITY).map_err(kerr)?.map_or(0.0, |d| d.distance);
    if gap > 1e-6 {
        return Ok(Clearance { gap, overlap: 0.0 });
    }
    let overlap = match kernel::boolean(a, b, BoolOp::Intersect).map_err(kerr)? {
        Some(common) => kernel::measure(&common).map(|m| m.volume.max(0.0)).unwrap_or(0.0),
        None => 0.0,
    };
    Ok(Clearance { gap: 0.0, overlap })
}

/// The clearance in the status line's words.
pub fn clearance_text(a: &str, b: &str, c: &Clearance, units: DisplayUnits) -> String {
    if c.gap > 0.0 {
        format!("{a} and {b} stand {} apart at the closest.", fmt_len(c.gap, units, 3, FmtOptions::default()))
    } else if c.overlap > 1e-6 {
        format!("{a} and {b} run into each other: {} in common.", fmt_vol(c.overlap, units, 3, FmtOptions::default()))
    } else {
        format!("{a} and {b} touch, with no clearance and no overlap.")
    }
}

/// The solid two bodies share, meshed in part coordinates, and its volume.
#[derive(Clone)]
pub struct Overlap {
    pub a: BodyId,
    pub b: BodyId,
    pub mesh: BodyMesh,
    pub volume: f64,
}

/// Whether two boxes (min, max) meet, give or take a hair.
fn boxes_meet(a: (DVec3, DVec3), b: (DVec3, DVec3)) -> bool {
    let eps = 1e-6;
    a.0.x <= b.1.x + eps && b.0.x <= a.1.x + eps && a.0.y <= b.1.y + eps && b.0.y <= a.1.y + eps && a.0.z <= b.1.z + eps && b.0.z <= a.1.z + eps
}

/// What every pair of the given solids shares, where they share a volume — those that only touch
/// or stand apart are left out, as is a pair the kernel cannot intersect. `stale` is asked between
/// pairs: once it says the answer is not wanted any more, what was found so far comes back.
pub fn overlaps(bodies: &[(BodyId, &Body)], stale: &dyn Fn() -> bool) -> Vec<Overlap> {
    let boxes: Vec<Option<(DVec3, DVec3)>> = bodies
        .iter()
        .map(|(_, b)| {
            let bb = b.outline_bounds();
            (!bb.is_empty()).then(|| (dv3(bb.min), dv3(bb.max)))
        })
        .collect();
    let mut out = Vec::new();
    for i in 0..bodies.len() {
        for j in i + 1..bodies.len() {
            if stale() {
                return out;
            }
            let (Some(ba), Some(bb)) = (boxes[i], boxes[j]) else { continue };
            if !boxes_meet(ba, bb) {
                continue;
            }
            let Ok(Some(common)) = kernel::boolean(bodies[i].1, bodies[j].1, BoolOp::Intersect) else { continue };
            let volume = kernel::measure(&common).map(|m| m.volume).unwrap_or(0.0);
            if volume <= 1e-6 {
                continue;
            }
            if let Ok(mesh) = body_mesh(&common) {
                out.push(Overlap { a: bodies[i].0, b: bodies[j].0, mesh, volume });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn block(min: [f64; 3], max: [f64; 3]) -> Body {
        let lo = solidcraft_geom::Vec3::new(min[0], min[1], min[2]);
        let hi = solidcraft_geom::Vec3::new(max[0], max[1], max[2]);
        kernel::box_solid(lo, hi).expect("a box")
    }

    #[test]
    fn two_blocks_apart_touching_and_overlapping() {
        let a = block([0.0, 0.0, 0.0], [10.0, 10.0, 10.0]);
        let apart = block([13.0, 0.0, 0.0], [20.0, 10.0, 10.0]);
        let c = clearance(&a, &apart).expect("measured");
        assert!((c.gap - 3.0).abs() < 1e-4, "{c:?}");
        assert_eq!(clearance_text("Solid 1", "Solid 2", &c, DisplayUnits::Mm), "Solid 1 and Solid 2 stand 3.000 mm apart at the closest.");
        let into = block([8.0, 0.0, 0.0], [12.0, 10.0, 10.0]);
        let c = clearance(&a, &into).expect("measured");
        assert_eq!(c.gap, 0.0);
        assert!((c.overlap - 200.0).abs() < 1e-3, "{c:?}");
        assert_eq!(clearance_text("Solid 1", "Solid 2", &c, DisplayUnits::Mm), "Solid 1 and Solid 2 run into each other: 200.000 mm³ in common.");
        assert_eq!(clearance_text("A", "B", &Clearance { gap: 0.0, overlap: 0.0 }, DisplayUnits::Mm), "A and B touch, with no clearance and no overlap.");
        let found = overlaps(&[(BodyId::new(1), &a), (BodyId::new(2), &apart), (BodyId::new(3), &into)], &|| false);
        assert_eq!(found.len(), 1);
        assert_eq!((found[0].a, found[0].b), (BodyId::new(1), BodyId::new(3)));
        assert!((found[0].volume - 200.0).abs() < 1e-3);
        assert!(!found[0].mesh.triangles.is_empty());
    }
}
