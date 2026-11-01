// SPDX-License-Identifier: AGPL-3.0-only
//! The Extract feature's build: a measured element as a body at its extents — a plane as a
//! slab of the thickness asked, a cylinder or a cone as a solid of its length, a sphere as a
//! ball, a torus as a ring — or a shadow section as the block it stands as: its outline stood
//! up through the part's thickness, following the shadow as it is taken again off the scan. A
//! new body, or joined to or cut from what it meets. Ported from ScanRuler
//! (`plugins/reverse/core/resolve.ts` primitiveOf and the extract step, `primitives.ts`,
//! `shadowPieces.ts` shadowPrimitive), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only. In
//! Polymix a section element keeps its frame alone; its shadow is taken off the scan the build
//! reads, as 3D Measure takes it.

use std::f64::consts::TAU;

use glam::{DVec2, DVec3};
use solidcraft_geom::{Loop2, Region2, Seg2, Vec2 as SVec2};
use solidcraft_kernel::{self as kernel, Body};

use super::{kernel_plane, kerr, primitive_of, primitive_solid, sv3, Builder};
use crate::frame::SectionFrame;
use crate::loop_piece::{circle_through, LoopPiece, Profile};
use crate::model::{ElementGeometry, ExtractFeature, Feature, Part};
use crate::shadow::{shadow_mesh, ShadowOptions};
use crate::shadow_pieces::{shadow_primitive, ShadowPrism, ShadowSource};

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &ExtractFeature) -> Result<(), String> {
    if let Some(sec) = x.section {
        let prism = shadow_prism(b, part, sec)?;
        let solid = prism_solid(&prism)?;
        b.place(x.op, x.target, vec![solid])?;
        return Ok(());
    }
    let element = x.element.and_then(|id| part.element(id));
    let primitive = primitive_of(element, x.thickness)?;
    let solid = primitive_solid(&primitive)?;
    b.place(x.op, x.target, vec![solid])?;
    Ok(())
}

/// A section's shadow as the block it stands as, taken off the scan the build reads — or why
/// there is none: the section gone, no section, no scan yet to take it off.
fn shadow_prism(b: &Builder, part: &Part, id: crate::model::ElementId) -> Result<ShadowPrism, String> {
    let Some(e) = part.element(id) else { return shadow_primitive(None).map_err(|e| e.0) };
    let ElementGeometry::Section { frame } = &e.geometry else {
        return shadow_primitive(Some(&ShadowSource { name: Some(&e.name), is_shadow: false, shadow: None })).map_err(|e| e.0);
    };
    let cut = b.surf.scan.as_ref().map(|scan| shadow_mesh(scan.positions(), scan.triangles(), frame, &ShadowOptions::default()));
    shadow_primitive(Some(&ShadowSource { name: Some(&e.name), is_shadow: true, shadow: cut.as_ref().map(|c| &c.shadow) })).map_err(|e| e.0)
}

/// Profiles stood along their plane's normal from `from` to `to` (ScanRuler's prism of a
/// shadow's profiles): each outer loop with its holes, one solid of them all.
pub(crate) fn prism_solid(prism: &ShadowPrism) -> Result<Body, String> {
    let Some(first) = prism.profiles.first().and_then(|p| p.outer.first()) else { return Err("The shadow has no outline.".into()) };
    let origin = match first {
        LoopPiece::Line { a, .. } | LoopPiece::Arc { a, .. } => *a,
        LoopPiece::Circle { center, .. } => *center,
    };
    let frame = SectionFrame::sketch_on(origin, prism.dir, None).ok_or("The shadow's plane has no direction.")?;
    let plane = kernel_plane(&frame)?;
    let regions: Vec<Region2> = prism.profiles.iter().map(|p| region_of(&frame, p)).collect::<Result<_, _>>()?;
    // The plane sits through the outline: the offsets are the shadow's, from its own origin.
    let shift = (origin - frame.origin).dot(frame.normal);
    let (mut from, mut to) = (prism.from - shift, prism.to - shift);
    if plane.normal().dot(sv3(prism.dir)) < 0.0 {
        (from, to) = (-to, -from);
    }
    let mut solids = kernel::extrude(&plane, &regions, from, to).map_err(kerr)?;
    let mut acc = solids.pop().ok_or("The shadow's block could not be made.")?;
    for s in solids {
        acc = kernel::boolean(&acc, &s, kernel::BoolOp::Union).map_err(kerr)?.unwrap_or(acc);
    }
    Ok(acc)
}

/// A profile of loop pieces in the part as a region on a frame's sheet.
fn region_of(frame: &SectionFrame, p: &Profile) -> Result<Region2, String> {
    Ok(Region2 { outer: loop_of(frame, &p.outer)?.ccw(), holes: p.holes.iter().map(|h| loop_of(frame, h).map(|l| l.ccw().reversed())).collect::<Result<_, _>>()? })
}

fn s2(p: DVec2) -> SVec2 {
    SVec2::new(p.x, p.y)
}

/// A loop of pieces as the kernel's segments on the sheet.
fn loop_of(frame: &SectionFrame, pieces: &[LoopPiece]) -> Result<Loop2, String> {
    let mut segs = Vec::new();
    for piece in pieces {
        match *piece {
            LoopPiece::Line { a, b } => segs.push(Seg2::Line { a: s2(frame.to_sheet(a)), b: s2(frame.to_sheet(b)) }),
            LoopPiece::Circle { center, radius, .. } => segs.extend(Loop2::circle(s2(frame.to_sheet(center)), radius).segs),
            LoopPiece::Arc { a, m, b } => {
                let (pa, pm, pb) = (frame.to_sheet(a), frame.to_sheet(m), frame.to_sheet(b));
                let Some((c3, r, _)) = circle_through(DVec3::new(pa.x, pa.y, 0.0), DVec3::new(pm.x, pm.y, 0.0), DVec3::new(pb.x, pb.y, 0.0)) else {
                    segs.push(Seg2::Line { a: s2(pa), b: s2(pb) });
                    continue;
                };
                let c = DVec2::new(c3.x, c3.y);
                let ang = |p: DVec2| (p.y - c.y).atan2(p.x - c.x);
                let (sa, sm, sb) = (ang(pa), ang(pm), ang(pb));
                let ccw = |from: f64, to: f64| (to - from).rem_euclid(TAU);
                // Counter-clockwise from a through m to b, or clockwise.
                let sweep = if ccw(sa, sm) <= ccw(sa, sb) { ccw(sa, sb) } else { -(TAU - ccw(sa, sb)) };
                segs.push(Seg2::Arc { center: s2(c), radius: r, start: sa, sweep });
            }
        }
    }
    if segs.is_empty() {
        return Err("The shadow has an empty loop.".into());
    }
    Ok(Loop2 { segs })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{build, build_scanned, BuildScan};
    use crate::model::{BodyId, BodyOp, Element, FeatureKind};
    use std::sync::Arc;

    /// A scan of a 20 × 10 × 6 box with a Ø4 hole through it along Z, as a closed mesh.
    fn holed_box() -> Arc<BuildScan> {
        let solid = kernel::box_solid(solidcraft_geom::Vec3::new(0.0, 0.0, 0.0), solidcraft_geom::Vec3::new(20.0, 10.0, 6.0)).unwrap();
        let bore = kernel::cylinder(solidcraft_geom::Vec3::new(10.0, 5.0, -1.0), solidcraft_geom::Vec3::new(0.0, 0.0, 1.0), 2.0, 8.0).unwrap();
        let cut = kernel::boolean(&solid, &bore, kernel::BoolOp::Cut).unwrap().unwrap();
        let m = cut.tessellate(0.01).unwrap();
        let pos: Vec<DVec3> = m.positions.iter().map(|p| DVec3::new(p.x, p.y, p.z)).collect();
        let (pos, tris) = crate::build::welded(&pos, &m.triangles, 1e-9);
        Arc::new(BuildScan::new(pos, tris))
    }

    // reShadowExtract.test.ts: stands as its block through the part, and says why not for a cut
    // or one not taken yet.
    #[test]
    fn stands_a_shadow_as_its_block_through_the_part() {
        let scan = holed_box();
        let mut part = Part::default();
        let frame = SectionFrame::sketch_on(DVec3::new(0.0, 0.0, 3.0), DVec3::Z, None).unwrap();
        part.elements.push(Element { id: 1, name: "Shadow 1".into(), geometry: ElementGeometry::Section { frame }, stats: None });
        let id = part.add(FeatureKind::Extract(ExtractFeature { element: None, section: Some(1), thickness: 2.0, op: BodyOp::New, target: None, exprs: Default::default() }));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let b = built.bodies.iter().find(|b| b.id == BodyId::new(id)).unwrap();
        let v = super::super::volume(&b.body).unwrap();
        // The shadow lies outside the part by its margin, a raster cell or so all round.
        let want = 20.0 * 10.0 * 6.0 - std::f64::consts::PI * 4.0 * 6.0;
        assert!((v - want).abs() / want < 0.05, "{v} against {want}");
        // No scan: the shadow is not taken yet.
        let built = build(&part);
        assert_eq!(built.errors.get(&id).map(String::as_str), Some("Shadow 1 is still being taken — the body is built once it is."));
        // Gone.
        part.elements.clear();
        let built = build(&part);
        assert_eq!(built.errors.get(&id).map(String::as_str), Some("The section this body is taken from is gone."));
    }

    // reShadowExtract.test.ts: a 40 × 20 outline with its corners rounded to 3 mm, round a Ø8
    // hole, 10 mm thick, stands as its block — its sides the outline's own pieces.
    #[test]
    fn stands_a_shadows_outline_as_a_block_its_sides_the_outlines_pieces() {
        use crate::shadow::{ShadowLoop, ShadowPiece, ShadowShape, Span};
        let r = 3.0;
        let q = std::f64::consts::FRAC_PI_2;
        let arc = |cx: f64, cy: f64, a0: f64| ShadowPiece::Arc { from: DVec2::new(cx + r * a0.cos(), cy + r * a0.sin()), to: DVec2::new(cx + r * (a0 + q).cos(), cy + r * (a0 + q).sin()), center: DVec2::new(cx, cy), radius: r, sweep: q };
        let line = |a: (f64, f64), b: (f64, f64)| ShadowPiece::Line { from: DVec2::new(a.0, a.1), to: DVec2::new(b.0, b.1) };
        let frame = SectionFrame::sketch_on(DVec3::ZERO, DVec3::Z, None).unwrap();
        let shape = ShadowShape {
            loops: vec![
                ShadowLoop { hole: false, pieces: vec![line((-17.0, -10.0), (17.0, -10.0)), arc(17.0, -7.0, -q), line((20.0, -7.0), (20.0, 7.0)), arc(17.0, 7.0, 0.0), line((17.0, 10.0), (-17.0, 10.0)), arc(-17.0, 7.0, q), line((-20.0, 7.0), (-20.0, -7.0)), arc(-17.0, -7.0, 2.0 * q)] },
                ShadowLoop { hole: true, pieces: vec![ShadowPiece::Circle { center: DVec2::ZERO, radius: 4.0 }] },
            ],
            area: 0.0,
            play: 1.0,
            tolerance: 0.1,
            smooth_deg: 30.0,
            margin: Span { min: 1.0, max: 1.1 },
            cell: 0.05,
            widened: false,
            extent: Span { min: -5.0, max: 5.0 },
            frame,
        };
        let area = 40.0 * 20.0 - (4.0 - std::f64::consts::PI) * 9.0 - std::f64::consts::PI * 16.0;
        assert!(shadow_primitive(None).unwrap_err().0.contains("gone"));
        assert!(shadow_primitive(Some(&ShadowSource { name: Some("Plan"), is_shadow: false, shadow: None })).unwrap_err().0.starts_with("Plan is a cut"));
        assert!(shadow_primitive(Some(&ShadowSource { name: Some("Outline"), is_shadow: true, shadow: None })).unwrap_err().0.contains("still being taken"));
        let prism = shadow_primitive(Some(&ShadowSource { name: Some("Outline"), is_shadow: true, shadow: Some(&shape) })).unwrap();
        let solid = prism_solid(&prism).unwrap();
        let v = super::super::volume(&solid).unwrap();
        // The kernel measures its volume off a fine mesh: to a few millionths.
        assert!((v - area * 10.0).abs() < 0.05, "{v} against {}", area * 10.0);
        let t = super::super::read_topology(&solid, &vec![0; solid.face_count()]);
        let kinds: Vec<crate::topo::FaceKind> = t.faces.iter().map(|f| f.fingerprint.kind).collect();
        // Four rounds and the hole's wall — which the kernel keeps as two halves, as it keeps
        // every whole bore.
        assert_eq!(kinds.iter().filter(|k| **k == crate::topo::FaceKind::Cylinder).count(), 6, "{kinds:?}");
        assert_eq!(kinds.iter().filter(|k| **k == crate::topo::FaceKind::Plane).count(), 6, "{kinds:?}");
    }
}
