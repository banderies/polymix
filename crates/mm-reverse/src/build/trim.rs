// SPDX-License-Identifier: AGPL-3.0-only
//! The Trim feature's build: a set of surfaces closed into a solid — a box round the scan cut
//! by every tool (the fitted surfaces and sheets of the workspace, solid bodies, and measured
//! planes, cylinders, cones, spheres and tori reaching past the part), and of the faces that
//! makes, the ones lying on the scan — within the tolerance, over the overlap share of their
//! samples — kept and sewn. A set that closes is a solid, its faces named by the surfaces they
//! lie on; one that does not leaves a surface body of the kept faces, its open edges reported
//! to be drawn red. A surface fitted as a plane or a cylinder cuts as the whole plane or
//! cylinder, as a measured element does. Ported from ScanRuler (`plugins/reverse/core/build.ts`
//! trimOf, trimBox, trimPrimitive, fittedFor and the trim step; `resolve.ts` the trim's
//! checks), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, on the SolidCraft kernel: the box
//! is cut into cells by each tool in turn (a plane splits, a solid parts what lies inside it
//! from what lies outside, a sheet cuts where it crosses — see `surf_split`), and the cells'
//! faces are the faces the tools cut — each once, though two cells share it.

use glam::DVec3;
use solidcraft_geom::{Plane, Vec3 as SVec3};
use solidcraft_kernel::{self as kernel, Body};

use super::surf::{on_scan_share, TrimReport};
use super::surf_kernel::{MeshIndex, ShapeMesh};
use super::surf_split::{cut_by, fuse_all, on_tolerance, Tool};
use super::{diagonal_of, kerr, mesh_deflection, primitive_of, primitive_solid, probe_inside, sv3, BodyId, BodyState, Builder};
use crate::model::{ElementGeometry, Feature, Part, Primitive, TrimFeature, TrimPrimitive, TrimTool};

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &TrimFeature) -> Result<(), String> {
    // The tools as resolved: bodies by id, measured elements as primitives.
    let bodies: Vec<BodyId> = x.tools.iter().filter_map(|t| if let TrimTool::Body { id } = t { Some(*id) } else { None }).collect();
    let mut primitives: Vec<TrimPrimitive> = Vec::new();
    for t in &x.tools {
        let TrimTool::Element { id } = t else { continue };
        let Some(e) = part.element(*id) else { return Err("An element this trim cuts with is gone.".into()) };
        primitives.push(match &e.geometry {
            ElementGeometry::Plane { center, normal, basis_u, basis_v, .. } => TrimPrimitive::Plane { origin: *center, normal: *normal, basis_u: *basis_u, basis_v: *basis_v },
            g => match primitive_of(Some(e), 1.0) {
                Ok(Primitive::Cylinder { origin, axis, radius, length }) => TrimPrimitive::Cylinder { origin, axis, radius, length },
                Ok(Primitive::Cone { origin, axis, radius1, radius2, length }) => TrimPrimitive::Cone { origin, axis, radius1, radius2, length },
                Ok(Primitive::Sphere { center, radius }) => TrimPrimitive::Sphere { center, radius },
                Ok(Primitive::Torus { center, axis, major_radius, minor_radius }) => TrimPrimitive::Torus { center, axis, major_radius, minor_radius },
                _ => return Err(format!("A {} cannot cut space for a trim.", g.kind())),
            },
        });
    }
    if bodies.len() + primitives.len() == 0 {
        return Err("The trim has no surfaces to close.".into());
    }
    if !(x.tolerance > 0.0) || !(x.overlap > 0.0 && x.overlap <= 1.0) {
        return Err("The trim needs a tolerance above zero and an overlap between 0 and 1.".into());
    }
    let scan = b.scan_for("trim against")?;
    // A surface fitted as a plane or a cylinder cuts as the whole plane or cylinder, like a
    // measured element: its drawn patch ends where its region did, grown by a share, and two
    // such patches need not reach each other at a corner the scan rounded off.
    let mut whole: Vec<TrimPrimitive> = Vec::new();
    let mut tools: Vec<(BodyId, BodyState)> = Vec::new();
    for id in &bodies {
        let (at, state) = b.body_for(*id).map_err(|_| "A surface this trim closes is gone, or failed to build.".to_string())?;
        match b.surf.fitted.get(&at).filter(|_| !b.surf.reshaped.contains_key(&at)) {
            Some(p) => whole.push(p.clone()),
            None => tools.push((at, state)),
        }
    }
    let (min, max) = trim_box(scan.bounds(), &tools.iter().map(|t| &t.1.shape).collect::<Vec<_>>())?;
    let diagonal = (max - min).length();
    let box_solid = kernel::box_solid(sv3(min), sv3(max)).map_err(kerr)?;
    let mut cells = vec![box_solid];
    for p in whole.iter().chain(&primitives) {
        cells = match p {
            TrimPrimitive::Plane { origin, normal, .. } => {
                let plane = super::plane_through(*origin, *normal)?;
                split_cells_by_plane(&cells, &plane)?
            }
            other => {
                let solid = trim_primitive(other, diagonal)?;
                cut_cells(&cells, &Tool::Solid(&solid))?
            }
        };
    }
    for (id, t) in &tools {
        let whole = !b.surf.reshaped.contains_key(id);
        let tool = if t.surface { Tool::Sheet { body: &t.shape, chains: b.surf.chains.get(id).cloned(), field: b.surf.fits.get(id).filter(|_| whole).cloned() } } else { Tool::Solid(&t.shape) };
        cells = cut_cells(&cells, &tool)?;
    }
    // Every face the tools cut the box into, each once though two cells share it.
    let mut faces: Vec<(Body, DVec3, f64, Option<MeshIndex>)> = Vec::new();
    let tol = on_tolerance(diagonal, diagonal);
    for cell in &cells {
        let (pieces, rest) = kernel::unstitch(cell, &[]).map_err(kerr)?;
        for face in pieces.into_iter().chain(rest) {
            let Ok(m) = ShapeMesh::of(&face, 1.0) else { continue };
            let Some(t) = m.triangles.get(m.triangles.len() / 2) else { continue };
            let [a, bb, c] = t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
            let at = (a + bb + c) / 3.0;
            let area = super::surf_kernel::surface_area(&face);
            let twin = faces.iter().any(|(_, _, ar, idx)| (ar - area).abs() <= 1e-3 * area.max(1.0) && idx.as_ref().is_some_and(|i| i.nearest(at, tol).is_some()));
            if !twin {
                faces.push((face, at, area, MeshIndex::of(&m)));
            }
        }
    }
    let total = faces.len();
    let mut kept: Vec<Body> = Vec::new();
    for (face, _, _, _) in &faces {
        let Ok(m) = ShapeMesh::of(face, 4.0) else { continue };
        if on_scan_share(&scan, &m.positions, &m.triangles, x.tolerance, None) >= x.overlap {
            kept.push(face.clone());
        }
    }
    if kept.is_empty() {
        return Err(format!("None of the {total} faces the tools cut lies on the scan within {} mm — widen the tolerance, or add the surfaces the part still lacks.", crate::surface_fit::js_num(x.tolerance)));
    }
    let (shape, closed) = sew(&kept, x.tolerance, &cells)?;
    let open_edges = if closed { Vec::new() } else { ShapeMesh::of(&shape, 1.0).map(|m| m.open_edges()).unwrap_or_default() };
    // Each face traced to the surface it lies on: its centroid within the tolerance of a
    // surface body's mesh makes that surface's feature its maker; the faces of the measured
    // elements are the trim's own.
    let origins = makers(b, &shape, &bodies, x.tolerance);
    let id = b.fresh_id()?;
    let mut state = BodyState::new(shape, origins);
    state.surface = !closed;
    b.set_body(id, state);
    b.touched(vec![id]);
    b.report().trim = Some(TrimReport { faces: total, kept: kept.len(), closed, open_edges: open_edges.clone() });
    if !closed {
        b.report().where_lines = open_edges;
    }
    Ok(())
}

/// The box a trim cuts up: round the scan, grown a tenth each way (a millimetre at least), or
/// round the tools when there is no scan (ScanRuler's `trimBox`).
fn trim_box(scan: Option<(DVec3, DVec3)>, tools: &[&Body]) -> Result<(DVec3, DVec3), String> {
    let (mut min, mut max) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    match scan {
        Some((lo, hi)) => {
            min = lo;
            max = hi;
        }
        None => {
            for t in tools {
                let bb = t.outline_bounds();
                if !bb.is_empty() {
                    min = min.min(super::dv3(bb.min));
                    max = max.max(super::dv3(bb.max));
                }
            }
        }
    }
    if !min.x.is_finite() {
        return Err("Nothing to trim round — no scan and no tool has an extent.".into());
    }
    let grow = (0.1 * (max - min).length()).max(1.0);
    Ok((min - DVec3::splat(grow), max + DVec3::splat(grow)))
}

/// A measured element as a trim tool, reaching past the box (ScanRuler's `trimPrimitive`): a
/// cylinder or a cone stretched along its axis to the box's diagonal and a half each way (a cone
/// no further than its apex), a sphere and a torus as they are.
fn trim_primitive(p: &TrimPrimitive, diagonal: f64) -> Result<Body, String> {
    let reach = diagonal * 1.5;
    match p {
        TrimPrimitive::Plane { .. } => Err("A plane cuts as a plane.".into()),
        TrimPrimitive::Cylinder { origin, axis, radius, .. } => {
            let a = axis.try_normalize().ok_or("The axis has no direction.")?;
            kernel::cylinder(sv3(*origin - a * reach), sv3(a), *radius, 2.0 * reach).map_err(kerr)
        }
        TrimPrimitive::Cone { origin, axis, radius1, radius2, length } => {
            // The radii run linearly along the axis; each end goes out to the reach, or to where
            // the radius would reach zero.
            let slope = (radius2 - radius1) / length.max(1e-6);
            let r = |t: f64| radius1 + slope * (t + length / 2.0);
            let (mut from, mut to) = (-reach, reach);
            if r(from) < 1e-3 {
                from = -length / 2.0 - (radius1 - 1e-3) / (-slope).max(1e-9);
            }
            if r(to) < 1e-3 {
                to = length / 2.0 + (radius2 - 1e-3) / slope.max(1e-9);
            }
            from = from.max(-reach);
            to = to.min(reach);
            let n = axis.try_normalize().unwrap_or(DVec3::Z);
            primitive_solid(&Primitive::Cone { origin: *origin + n * ((from + to) / 2.0), axis: n, radius1: r(from), radius2: r(to), length: to - from })
        }
        TrimPrimitive::Sphere { center, radius } => primitive_solid(&Primitive::Sphere { center: *center, radius: *radius }),
        TrimPrimitive::Torus { center, axis, major_radius, minor_radius } => primitive_solid(&Primitive::Torus { center: *center, axis: *axis, major_radius: *major_radius, minor_radius: *minor_radius }),
    }
}

/// Cells split by a whole plane: each cell the plane crosses, into its two sides.
fn split_cells_by_plane(cells: &[Body], plane: &Plane) -> Result<Vec<Body>, String> {
    let n = plane.normal();
    let o = plane.origin;
    let mut out = Vec::new();
    for c in cells {
        let bb = c.outline_bounds();
        let corners = [bb.min.x, bb.max.x].iter().flat_map(|x| [bb.min.y, bb.max.y].into_iter().flat_map(move |y| [bb.min.z, bb.max.z].into_iter().map(move |z| SVec3::new(*x, y, z)))).map(|p| (p - o).dot(n)).collect::<Vec<f64>>();
        let tol = 1e-6 * diagonal_of(c).max(1.0);
        if corners.iter().all(|d| *d > tol) || corners.iter().all(|d| *d < -tol) {
            out.push(c.clone());
            continue;
        }
        let parts = kernel::split_by_plane(c, plane).map_err(kerr)?;
        if parts.is_empty() {
            out.push(c.clone());
        }
        for p in parts {
            out.extend(super::surf_split::pieces_of(Some(p)));
        }
    }
    Ok(out)
}

/// Cells cut by a tool: each into its pieces.
fn cut_cells(cells: &[Body], tool: &Tool) -> Result<Vec<Body>, String> {
    let mut out = Vec::new();
    for c in cells {
        out.extend(cut_by(c, tool)?);
    }
    Ok(out)
}

/// The kept faces sewn: a solid where they close — the cells inside them made one, so a face is
/// one face however many cells it bounded — a surface body of them otherwise.
fn sew(kept: &[Body], tolerance: f64, cells: &[Body]) -> Result<(Body, bool), String> {
    let refs: Vec<&Body> = kept.iter().collect();
    let sewn = kernel::stitch(&refs, tolerance).map_err(kerr)?;
    if sewn.is_surface() {
        return Ok((sewn, false));
    }
    // Closed: the cells inside it, made one.
    let mesh = ShapeMesh::of(&sewn, 1.0)?;
    if let Some(index) = MeshIndex::of(&mesh) {
        let inside: Vec<Body> = cells.iter().filter(|c| probe_inside(c).is_some_and(|p| index.inside(p))).cloned().collect();
        if !inside.is_empty() {
            if let Ok(one) = fuse_all(&inside) {
                if one.lumps().map(|l| l.len()).unwrap_or(0) == 1 && super::surf_kernel::mesh_volume(&one) > 0.0 {
                    return Ok((one, true));
                }
            }
        }
    }
    let v = super::surf_kernel::mesh_volume(&sewn);
    Ok((sewn, v.abs() > 0.0))
}

/// Which feature made each face of a trim: the surface body its faces lie on, the trim itself
/// for a measured element's.
fn makers(b: &Builder, shape: &Body, bodies: &[BodyId], tolerance: f64) -> Vec<i64> {
    let n = shape.face_count();
    let own = b.step as i64;
    let mut out = vec![own; n];
    let Ok(m) = ShapeMesh::of(shape, 1.0) else { return out };
    let sag = mesh_deflection(diagonal_of(shape));
    let points = m.face_points(n);
    let sources: Vec<(i64, MeshIndex)> = bodies
        .iter()
        .filter_map(|id| {
            let s = b.body(*id)?;
            let mi = MeshIndex::of(&ShapeMesh::of(&s.shape, 1.0).ok()?)?;
            Some((id.maker() as i64, mi))
        })
        .collect();
    for (face, p) in points.into_iter().enumerate() {
        let Some(p) = p else { continue };
        let mut best = tolerance + sag;
        for (maker, idx) in &sources {
            if let Some((_, d, _, _)) = idx.nearest(p, best) {
                if d < best {
                    best = d;
                    out[face] = *maker;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::{build_scanned, volume, BuildScan};
    use crate::model::{Element, ElementGeometry, FeatureKind};
    use std::sync::Arc;

    /// A scan of a box's six sides, `n` × `n` a side.
    fn box_scan(min: DVec3, max: DVec3) -> Arc<BuildScan> {
        let mut pos = Vec::new();
        let mut tris = Vec::new();
        let mut quad = |a: DVec3, u: DVec3, v: DVec3| {
            let n = 8u32;
            let base = pos.len() as u32;
            for i in 0..=n {
                for j in 0..=n {
                    pos.push(a + u * (i as f64 / n as f64) + v * (j as f64 / n as f64));
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let k = base + i * (n + 1) + j;
                    tris.push([k, k + n + 1, k + n + 2]);
                    tris.push([k, k + n + 2, k + 1]);
                }
            }
        };
        let d = max - min;
        let (x, y, z) = (DVec3::X * d.x, DVec3::Y * d.y, DVec3::Z * d.z);
        quad(min, x, y);
        quad(min + z, x, y);
        quad(min, x, z);
        quad(min + y, x, z);
        quad(min, y, z);
        quad(min + x, y, z);
        Arc::new(BuildScan::new(pos, tris))
    }

    fn plane(id: u32, center: DVec3, normal: DVec3) -> Element {
        let (u, v) = crate::linalg::ortho_basis(normal);
        Element { id, name: format!("Plane {id}"), geometry: ElementGeometry::Plane { center, normal, basis_u: u, basis_v: v, extent_u: 5.0, extent_v: 5.0 }, stats: None }
    }

    fn trim(tools: Vec<TrimTool>) -> FeatureKind {
        FeatureKind::Trim(TrimFeature { tools, tolerance: 0.3, overlap: 0.5, exprs: Default::default() })
    }

    #[test]
    fn closes_six_measured_planes_round_a_box_into_its_solid() {
        let scan = box_scan(DVec3::new(0.0, 0.0, 0.0), DVec3::new(20.0, 10.0, 5.0));
        let mut part = Part::default();
        part.elements = vec![
            plane(1, DVec3::new(0.0, 5.0, 2.5), DVec3::X),
            plane(2, DVec3::new(20.0, 5.0, 2.5), DVec3::X),
            plane(3, DVec3::new(10.0, 0.0, 2.5), DVec3::Y),
            plane(4, DVec3::new(10.0, 10.0, 2.5), DVec3::Y),
            plane(5, DVec3::new(10.0, 5.0, 0.0), DVec3::Z),
            plane(6, DVec3::new(10.0, 5.0, 5.0), DVec3::Z),
        ];
        let id = part.add(trim((1..=6).map(|id| TrimTool::Element { id }).collect()));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let r = built.surf.get(&id).and_then(|r| r.trim.clone()).unwrap();
        assert!(r.closed, "{r:?}");
        assert_eq!(r.kept, 6, "{r:?}");
        let b = &built.bodies[0];
        assert!(!b.surface);
        let v = volume(&b.body).unwrap();
        assert!((v - 1000.0).abs() < 1e-3, "{v}");
        assert_eq!(b.body.face_count(), 6);
    }

    #[test]
    fn leaves_an_open_set_a_surface_body_with_its_open_edges() {
        // Five planes of the six: the top is missing.
        let scan = box_scan(DVec3::new(0.0, 0.0, 0.0), DVec3::new(20.0, 10.0, 5.0));
        let mut part = Part::default();
        part.elements = vec![
            plane(1, DVec3::new(0.0, 5.0, 2.5), DVec3::X),
            plane(2, DVec3::new(20.0, 5.0, 2.5), DVec3::X),
            plane(3, DVec3::new(10.0, 0.0, 2.5), DVec3::Y),
            plane(4, DVec3::new(10.0, 10.0, 2.5), DVec3::Y),
            plane(5, DVec3::new(10.0, 5.0, 0.0), DVec3::Z),
        ];
        let id = part.add(trim((1..=5).map(|id| TrimTool::Element { id }).collect()));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let r = built.surf.get(&id).and_then(|r| r.trim.clone()).unwrap();
        assert!(!r.closed);
        assert!(!r.open_edges.is_empty());
        assert!(built.bodies[0].surface);
        // With no tool it fails by name; with no scan too.
        let mut part = Part::default();
        let none = part.add(trim(Vec::new()));
        let built = crate::build::build(&part);
        assert_eq!(built.errors.get(&none).map(String::as_str), Some("The trim has no surfaces to close."));
    }
}
