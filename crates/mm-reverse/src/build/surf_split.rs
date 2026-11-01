// SPDX-License-Identifier: AGPL-3.0-only
//! A solid cut into fragments by sheets and solids — what Open CASCADE's splitter does for
//! ScanRuler's Split and Trim (`plugins/reverse/core/kernel.ts` split, splitFaces), made on
//! SolidCraft's public calls. Ported in its outcome, not its calls, from ScanRuler
//! (`plugins/reverse/core/build.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! SolidCraft cuts a solid by a plane or by a face's whole surface (`split_body`), and joins,
//! cuts and intersects solids; it has no splitter that cuts by a bounded sheet. So a sheet cuts
//! as a solid on one side of it does — a sketch's chain closed far round one side and stood
//! through the sheet's height, a freeform sheet thickened far to one side, a flat or round face
//! by its whole surface — the solid's inside and outside each the fragments on that side. What
//! the cutter cut where the sheet is not (its far walls, a plane run on past the sheet) is no
//! cut of the sheet's: the fragments either side of such a face are made one again. A sheet
//! whose edge runs inside the solid does not part it, as the splitter's does not; it is left
//! whole. A solid tool parts it into what lies inside the tool and what lies outside.

use std::sync::Arc;

use glam::{DVec2, DVec3};
use solidcraft_geom::{Loop2, Plane, Region2, Seg2, Vec2 as SVec2};
use solidcraft_kernel::{self as kernel, BoolOp, Body, FaceSurface};

use super::surf_kernel::{along_polylines, mesh_volume, MeshIndex, ShapeMesh};
use super::{diagonal_of, kerr, mesh_deflection};
use crate::surface_fit::SurfaceFitResult;

/// A sketch sheet as it was drawn: its plane, its chains of pieces on it and the offsets it
/// stands between — what its cutter solids are made of, while the sheet is still as it was made.
pub(crate) struct SheetChains {
    pub(crate) made: Body,
    pub(crate) plane: Plane,
    pub(crate) chains: Vec<Vec<Seg2>>,
    pub(crate) from: f64,
    pub(crate) to: f64,
}

/// What a solid is cut with.
pub(crate) enum Tool<'a> {
    /// A sheet: its shape, its sketch's chains while it is as drawn, and the height field it
    /// is the face of where it is a fitted surface's whole face (as fitted, or carried on).
    Sheet { body: &'a Body, chains: Option<Arc<SheetChains>>, field: Option<Arc<SurfaceFitResult>> },
    /// A solid body.
    Solid(&'a Body),
}

/// How near a point must lie to a face to lie on it, for a face read off two meshes of
/// diagonals `a` and `b`: twice both chords, and a micron.
pub(crate) fn on_tolerance(a: f64, b: f64) -> f64 {
    2.0 * (mesh_deflection(a) + mesh_deflection(b)) + 1e-3
}

/// The pieces of a solid, each a body of its own, largest first; none of an empty result.
pub(crate) fn pieces_of(shape: Option<Body>) -> Vec<Body> {
    let Some(s) = shape.filter(|s| s.face_count() > 0) else { return Vec::new() };
    let mut pieces: Vec<(Body, f64)> = match s.lumps() {
        Ok(l) if l.len() > 1 => l.into_iter().map(|p| {
            let v = mesh_volume(&p);
            (p, v)
        }).collect(),
        _ => {
            let v = mesh_volume(&s);
            vec![(s, v)]
        }
    };
    pieces.retain(|p| p.1 > 1e-9);
    pieces.sort_by(|a, b| b.1.total_cmp(&a.1));
    pieces.into_iter().map(|p| p.0).collect()
}

/// The solid's inside and outside of `cutter`: the fragments each side (the kernel tells
/// solids apart that do not meet before it intersects them).
fn both_sides(piece: &Body, cutter: &Body) -> Result<Vec<Body>, String> {
    let inside = kernel::boolean(piece, cutter, BoolOp::Intersect).map_err(kerr)?;
    let outside = kernel::boolean(piece, cutter, BoolOp::Cut).map_err(kerr)?;
    let mut out = pieces_of(inside);
    out.extend(pieces_of(outside));
    if out.is_empty() {
        out.push(piece.clone());
    }
    Ok(out)
}

/// The pieces a solid falls into cut by one tool. A sheet that does not reach across the solid
/// — an edge of it runs through the solid's inside — leaves it whole.
pub(crate) fn cut_by(piece: &Body, tool: &Tool) -> Result<Vec<Body>, String> {
    match tool {
        Tool::Solid(t) => both_sides(piece, t),
        Tool::Sheet { body, chains, field } => cut_by_sheet(piece, body, chains.as_deref(), field.as_deref()),
    }
}

/// Whether a sheet reaches across a solid: none of its open edges runs through the solid's
/// inside — deeper than a hair of the two meshes.
pub(crate) fn sheet_crosses(piece_index: &MeshIndex, sheet: &ShapeMesh, depth: f64) -> bool {
    let open = sheet.open_edges();
    along_polylines(&open, 240).iter().all(|p| !piece_index.deep_inside(*p, depth))
}

fn cut_by_sheet(piece: &Body, sheet: &Body, chains: Option<&SheetChains>, field: Option<&SurfaceFitResult>) -> Result<Vec<Body>, String> {
    let piece_mesh = ShapeMesh::of(piece, 1.0)?;
    let Some(piece_index) = MeshIndex::of(&piece_mesh) else { return Ok(vec![piece.clone()]) };
    let sheet_mesh = ShapeMesh::of(sheet, 1.0)?;
    let (dp, ds) = (diagonal_of(piece), diagonal_of(sheet));
    let tol = on_tolerance(dp, ds);
    // A sheet that misses the solid, or stops inside it, parts nothing.
    let bounds = |m: &ShapeMesh| m.positions.iter().fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let ((plo, phi), (slo, shi)) = (bounds(&piece_mesh), bounds(&sheet_mesh));
    let apart = shi.cmplt(plo - DVec3::splat(tol)).any() || slo.cmpgt(phi + DVec3::splat(tol)).any();
    if apart || !sheet_crosses(&piece_index, &sheet_mesh, tol) {
        return Ok(vec![piece.clone()]);
    }
    let fragments = match chains.filter(|c| c.made.same(sheet)) {
        // A sketch's sheet: each chain's cutter in turn.
        Some(c) => {
            let reach = dp * 2.0 + ds + 10.0;
            let mut frags = vec![piece.clone()];
            for chain in &c.chains {
                let cutter = chain_cutter(&c.plane, chain, c.from, c.to, reach)?;
                let mut next = Vec::new();
                for f in &frags {
                    next.extend(both_sides(f, &cutter)?);
                }
                frags = next;
            }
            frags
        }
        None => {
            let one = sheet.face_count() == 1;
            let exact = one && matches!(kernel::face_surface(sheet, 0), Some(FaceSurface::Plane { .. }) | Some(FaceSurface::Cylinder { .. }));
            let cutter = match field.filter(|_| one && !exact) {
                Some(f) => field_cutter(f, &piece_mesh.positions).ok(),
                None => None,
            };
            if let Some(cutter) = cutter {
                both_sides(piece, &cutter)?
            } else if exact {
                kernel::split_body(piece, &kernel::SplitTool::Face { body: sheet, face: 0 }).map_err(kerr)?
            } else {
                match slab_of(sheet, dp + ds) {
                    Some(slab) => both_sides(piece, &slab)?,
                    None if one => kernel::split_body(piece, &kernel::SplitTool::Face { body: sheet, face: 0 }).map_err(kerr)?,
                    None => return Err("The sheet does not thicken far enough either way to cut with, and a sheet of several faces cuts no other way yet — split by its faces one at a time.".into()),
                }
            }
        }
    };
    let Some(sheet_index) = MeshIndex::of(&sheet_mesh) else { return Ok(fragments) };
    regroup(fragments, &piece_index, &sheet_index, tol)
}

/// A sheet thickened far to one side — against its normal, else along it: the solid a sheet
/// cuts as. `None` where it folds both ways.
fn slab_of(sheet: &Body, depth: f64) -> Option<Body> {
    let d = depth * 1.5 + 1.0;
    for t in [-d, d] {
        if let Ok(b) = kernel::guard("thicken", || kernel::thicken(sheet, t, false)) {
            if !b.is_surface() && b.face_count() > 0 {
                return Some(b);
            }
        }
    }
    None
}

/// The solid under a fitted surface's face — its height field over the fit's frame, down past
/// the solid it cuts (`points`) — the face itself its top, the face's edges carried straight
/// down against the fit's normal as its walls, cut off flat by a floor that closes it: what a
/// freeform surface cuts as. Sewn of the face, the ruled walls (which stand on the face's own
/// edges) and the floor.
pub(crate) fn field_cutter(fit: &SurfaceFitResult, points: &[DVec3]) -> Result<Body, String> {
    let top = super::surf_kernel::fit_face(fit)?;
    let low_points = points.iter().map(|q| (*q - fit.origin).dot(fit.normal)).fold(f64::INFINITY, f64::min);
    let (low_net, high_net) = fit.heights.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, h| (a.0.min(*h), a.1.max(*h)));
    let size = (fit.u_range[1] - fit.u_range[0]).max(fit.v_range[1] - fit.v_range[0]);
    let floor = low_points.min(low_net) - 1.0 - 0.05 * size;
    // The walls run on past the floor, and are cut off flat there.
    let depth = high_net - floor + 1.0 + 0.05 * size;
    let mids = |b: &Body| -> Result<Vec<solidcraft_geom::Vec3>, String> {
        let m = ShapeMesh::of(b, 1.0)?;
        Ok(m.open_edges().iter().map(|l| super::sv3(super::surf_kernel::point_along(l, l.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>() / 2.0))).collect())
    };
    let ruled = kernel::Ruled { distance: depth, mode: kernel::RuledMode::Direction(super::sv3(-fit.normal)), flip: false, corners: kernel::RuledCorners::Mitre, face: None };
    let walls = kernel::ruled_surface(&top, &mids(&top)?, &ruled).map_err(kerr)?;
    let tol = 1e-6 * size.max(1.0) + 1e-7;
    let shell = kernel::stitch(&[&top, &walls], tol).map_err(kerr)?;
    let floor_plane = Plane::new(super::sv3(fit.origin + fit.normal * floor), super::sv3(fit.axis_u), super::sv3(fit.axis_v)).ok_or("The surface's frame is flat.")?;
    let above = super::sv3(fit.origin + fit.normal * (high_net + 1.0));
    let shell = kernel::trim(&shell, &kernel::SplitTool::Plane(floor_plane), above).map_err(kerr)?;
    let floor_face = kernel::patch_edges(&shell, &mids(&shell)?).map_err(kerr)?;
    let sewn = kernel::stitch(&[&shell, &floor_face], tol).map_err(kerr)?;
    if sewn.is_surface() {
        return Err("The surface's cutter did not close.".into());
    }
    Ok(sewn)
}

/// A chain of a sketch's pieces closed far round one side and stood from `from` to `to` along
/// the plane's normal: the solid an open chain's sheet cuts as — the chain itself the one wall
/// of it inside the reach; a closed chain stood as it is.
pub(crate) fn chain_cutter(plane: &Plane, chain: &[Seg2], from: f64, to: f64, reach: f64) -> Result<Body, String> {
    let (Some(first), Some(last)) = (chain.first(), chain.last()) else { return Err("The sheet has no pieces.".into()) };
    let p0 = first.start();
    let p1 = last.end();
    let closed = (p1 - p0).len() < 1e-6;
    let outer = if closed {
        Loop2 { segs: chain.to_vec() }
    } else {
        // The chain run on along its ends' tangents to a square far round it, and round the
        // square from where its end runs out to where its start runs out.
        let pts: Vec<DVec2> = chain.iter().flat_map(|s| [v2(s.start()), v2(s.end()), v2(s.mid())]).collect();
        let centre = pts.iter().fold(DVec2::ZERO, |a, p| a + *p) / pts.len().max(1) as f64;
        let size = pts.iter().fold(0.0f64, |m, p| m.max((*p - centre).length()));
        let half = reach.max(size * 4.0);
        let t0 = v2(first.start_tangent()).try_normalize().ok_or("A piece of the sheet has no direction.")?;
        let t1 = v2(last.end_tangent()).try_normalize().ok_or("A piece of the sheet has no direction.")?;
        let (x1, side1) = exit(v2(p1), t1, centre, half);
        let (x0, side0) = exit(v2(p0), -t0, centre, half);
        let corners = [DVec2::new(half, -half), DVec2::new(half, half), DVec2::new(-half, half), DVec2::new(-half, -half)].map(|c| c + centre);
        let mut segs: Vec<Seg2> = chain.to_vec();
        let mut at = v2(p1);
        let push = |segs: &mut Vec<Seg2>, at: &mut DVec2, to: DVec2| {
            if (to - *at).length() > 1e-9 {
                segs.push(Seg2::Line { a: s2(*at), b: s2(to) });
                *at = to;
            }
        };
        push(&mut segs, &mut at, x1);
        // Counter-clockwise round the square, side by side (0 right, 1 top, 2 left, 3
        // bottom; side k runs to corner k + 1), to the side the start runs out on.
        if !(side1 == side0 && ahead(x1, x0, side1, centre)) {
            let mut side = side1;
            for _ in 0..4 {
                push(&mut segs, &mut at, corners[(side + 1) % 4]);
                side = (side + 1) % 4;
                if side == side0 {
                    break;
                }
            }
        }
        push(&mut segs, &mut at, x0);
        push(&mut segs, &mut at, v2(p0));
        Loop2 { segs }
    };
    let region = Region2 { outer: outer.ccw(), holes: Vec::new() };
    let mut made = kernel::extrude(plane, &[region], from.min(to), from.max(to)).map_err(kerr)?;
    made.pop().ok_or_else(|| "The sheet's cutter could not be made.".to_string())
}

fn v2(p: SVec2) -> DVec2 {
    DVec2::new(p.x, p.y)
}

fn s2(p: DVec2) -> SVec2 {
    SVec2::new(p.x, p.y)
}

/// Where a ray from inside a square (centre `c`, half side `h`) runs out of it, and the side
/// (0 right, 1 top, 2 left, 3 bottom).
fn exit(p: DVec2, d: DVec2, c: DVec2, h: f64) -> (DVec2, usize) {
    let q = p - c;
    let mut best = (f64::INFINITY, 0usize);
    for (side, (axis, sign)) in [(0usize, (0usize, 1.0f64)), (1, (1, 1.0)), (2, (0, -1.0)), (3, (1, -1.0))] {
        let dk = if axis == 0 { d.x } else { d.y };
        let qk = if axis == 0 { q.x } else { q.y };
        if dk * sign > 1e-12 {
            let t = (sign * h - qk) / dk;
            if t >= 0.0 && t < best.0 {
                best = (t, side);
            }
        }
    }
    (p + d * best.0, best.1)
}

/// On one side of the square, whether `to` lies ahead of `from` going counter-clockwise.
fn ahead(from: DVec2, to: DVec2, side: usize, c: DVec2) -> bool {
    let (a, b) = (from - c, to - c);
    match side {
        0 => b.y > a.y,
        1 => b.x < a.x,
        2 => b.y < a.y,
        _ => b.x > a.x,
    }
}

/// Fragments made one again across the faces the cutter made where the sheet is not: a face of
/// a fragment lying neither on the solid as it was (`before`) nor on the sheet is a wall of
/// the cutter, and the fragments either side of it one piece.
fn regroup(fragments: Vec<Body>, before: &MeshIndex, sheet: &MeshIndex, tol: f64) -> Result<Vec<Body>, String> {
    if fragments.len() < 2 {
        return Ok(fragments);
    }
    let meshes: Vec<ShapeMesh> = fragments.iter().map(|f| ShapeMesh::of(f, 1.0)).collect::<Result<_, _>>()?;
    let indices: Vec<Option<MeshIndex>> = meshes.iter().map(MeshIndex::of).collect();
    let n = fragments.len();
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        p[i] = r;
        r
    }
    for (i, m) in meshes.iter().enumerate() {
        let faces = m.tri_face.iter().copied().max().map_or(0, |f| f as usize + 1);
        for f in 0..faces {
            let tris = m.face_triangles(f as u32);
            let samples: Vec<DVec3> = tris.iter().step_by((tris.len() / 5).max(1)).map(|t| t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default())).map(|[a, b, c]| (a + b + c) / 3.0).collect();
            if samples.is_empty() {
                continue;
            }
            let wall = samples.iter().filter(|p| before.nearest(**p, tol).is_none() && sheet.nearest(**p, tol).is_none()).count();
            if wall * 2 <= samples.len() {
                continue;
            }
            let p = samples[samples.len() / 2];
            for (j, idx) in indices.iter().enumerate() {
                if j == i {
                    continue;
                }
                if idx.as_ref().is_some_and(|x| x.nearest(p, tol).is_some()) {
                    let (a, b) = (root(&mut parent, i), root(&mut parent, j));
                    if a != b {
                        parent[a] = b;
                    }
                }
            }
        }
    }
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let mut of_root: std::collections::HashMap<usize, usize> = std::collections::HashMap::new();
    for i in 0..n {
        let r = root(&mut parent, i);
        let g = *of_root.entry(r).or_insert_with(|| {
            groups.push(Vec::new());
            groups.len() - 1
        });
        groups[g].push(i);
    }
    let mut out = Vec::new();
    for g in groups {
        let bodies: Vec<Body> = g.iter().map(|i| fragments[*i].clone()).collect();
        out.extend(pieces_of(Some(fuse_all(&bodies)?)));
    }
    Ok(out)
}

/// A sheet cut by its tools — sheets or solids — into its pieces (ScanRuler's kernel
/// `sheetPieces`): its faces split along each tool's surface, then taken together where they
/// meet across an edge that lies on no tool's faces, each group a piece — a tool's surface run
/// on past the tool cuts nothing. A sheet the tools do not cut is one piece.
pub(crate) fn sheet_pieces(sheet: &Body, tools: &[Tool]) -> Result<Vec<Body>, String> {
    let mut current = sheet.clone();
    let mut tool_meshes: Vec<MeshIndex> = Vec::new();
    for tool in tools {
        let (body, chains) = match tool {
            Tool::Sheet { body, chains, .. } => (*body, chains.as_ref().filter(|c| c.made.same(body))),
            Tool::Solid(body) => (*body, None),
        };
        if let Some(index) = ShapeMesh::of(body, 1.0).ok().as_ref().and_then(MeshIndex::of) {
            tool_meshes.push(index);
        }
        let cutters: Vec<kernel::SplitTool> = match chains {
            Some(c) => vec![kernel::SplitTool::Curves { plane: c.plane, curves: c.chains.clone(), along: None }],
            None => (0..body.face_count()).map(|face| kernel::SplitTool::Face { body, face }).collect(),
        };
        for cutter in &cutters {
            current = split_where_crossed(&current, cutter)?;
        }
    }
    let m = ShapeMesh::of(&current, 1.0)?;
    let n = current.face_count();
    let tol = on_tolerance(diagonal_of(&current), diagonal_of(sheet));
    let mut parent: Vec<usize> = (0..n).collect();
    fn root(p: &mut [usize], i: usize) -> usize {
        let mut r = i;
        while p[r] != r {
            r = p[r];
        }
        p[i] = r;
        r
    }
    for (e, on) in m.edges.iter().zip(&m.edge_faces) {
        let mut faces: Vec<usize> = on.iter().map(|f| *f as usize).filter(|f| *f < n).collect();
        faces.dedup();
        if faces.len() < 2 || e.is_empty() {
            continue;
        }
        let mid = super::surf_kernel::point_along(e, e.windows(2).map(|w| (w[1] - w[0]).length()).sum::<f64>() / 2.0);
        if tool_meshes.iter().any(|t| t.nearest(mid, tol).is_some()) {
            continue;
        }
        for f in &faces[1..] {
            let (a, b) = (root(&mut parent, *f), root(&mut parent, faces[0]));
            if a != b {
                parent[a] = b;
            }
        }
    }
    let points = m.face_points(n);
    let mut groups: std::collections::BTreeMap<usize, Vec<usize>> = std::collections::BTreeMap::new();
    for f in 0..n {
        let r = root(&mut parent, f);
        groups.entry(r).or_default().push(f);
    }
    if groups.len() < 2 {
        return Ok(vec![current]);
    }
    let mut out = Vec::new();
    for faces in groups.values() {
        let at: Vec<solidcraft_geom::Vec3> = faces.iter().filter_map(|f| points.get(*f).copied().flatten()).map(super::sv3).collect();
        if at.is_empty() {
            continue;
        }
        out.push(kernel::copy_faces(&current, &at).map_err(kerr)?);
    }
    Ok(out)
}

/// A sheet's faces split where a tool's surface crosses them — those it misses left as they
/// are; the sheet as it was where it crosses none.
fn split_where_crossed(sheet: &Body, tool: &kernel::SplitTool) -> Result<Body, String> {
    let mut faces: Vec<usize> = (0..sheet.face_count()).collect();
    while !faces.is_empty() {
        match kernel::split_faces(sheet, &faces, tool) {
            Ok(b) => return Ok(b),
            Err(e) => {
                let said = e.to_string();
                let missed = said.rsplit("does not cross face ").next().filter(|_| said.contains("does not cross face ")).and_then(|x| x.trim().trim_end_matches('.').parse::<usize>().ok());
                match missed {
                    Some(i) if faces.contains(&i) => faces.retain(|f| *f != i),
                    _ if said.contains("does not cross") || said.contains("misses") => return Ok(sheet.clone()),
                    _ => return Err(kerr(e)),
                }
            }
        }
    }
    Ok(sheet.clone())
}

/// Solids made one (ScanRuler's `fuseAll`).
pub(crate) fn fuse_all(bodies: &[Body]) -> Result<Body, String> {
    let mut it = bodies.iter();
    let Some(first) = it.next() else { return Err("Nothing to fuse.".into()) };
    let mut acc = first.clone();
    for b in it {
        acc = kernel::boolean(&acc, b, BoolOp::Union).map_err(kerr)?.ok_or("The fragments did not fuse.")?;
    }
    Ok(acc)
}

#[cfg(test)]
mod tests {
    use super::*;
    use solidcraft_geom::{Vec2, Vec3};

    fn xy() -> Plane {
        Plane::new(Vec3::new(0.0, 0.0, 0.0), Vec3::new(1.0, 0.0, 0.0), Vec3::new(0.0, 1.0, 0.0)).unwrap()
    }

    fn vol(b: &Body) -> f64 {
        super::super::volume(b).unwrap_or(f64::NAN)
    }

    #[test]
    fn cuts_a_box_by_a_flat_sheet_and_leaves_it_whole_where_the_sheet_stops_inside() {
        let bx = kernel::box_solid(Vec3::new(-10.0, -10.0, 0.0), Vec3::new(10.0, 10.0, 10.0)).unwrap();
        let line = |a: (f64, f64), b: (f64, f64)| vec![Seg2::Line { a: Vec2::new(a.0, a.1), b: Vec2::new(b.0, b.1) }];
        let across = kernel::extrude_sheet(&xy(), &[line((-30.0, 2.0), (30.0, 2.0))], -10.0, 30.0).unwrap();
        let mut parts = cut_by(&bx, &Tool::Sheet { body: &across, chains: None, field: None }).unwrap();
        parts.sort_by(|a, b| vol(a).total_cmp(&vol(b)));
        assert_eq!(parts.len(), 2);
        assert!((vol(&parts[0]) - 1600.0).abs() < 1e-3 && (vol(&parts[1]) - 2400.0).abs() < 1e-3);
        // A sheet stopping halfway across parts nothing.
        let short = kernel::extrude_sheet(&xy(), &[line((-30.0, 2.0), (0.0, 2.0))], -10.0, 30.0).unwrap();
        assert_eq!(cut_by(&bx, &Tool::Sheet { body: &short, chains: None, field: None }).unwrap().len(), 1);
    }

    #[test]
    fn cuts_an_l_block_by_a_sheet_across_one_arm_only() {
        // An L: 20 × 20 less its top right 10 × 10; a sheet at x = 5 across the bottom arm alone,
        // from y = -5 to 9 — its plane run on would part the top arm too.
        let a = kernel::box_solid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(20.0, 10.0, 10.0)).unwrap();
        let b = kernel::box_solid(Vec3::new(0.0, 0.0, 0.0), Vec3::new(10.0, 20.0, 10.0)).unwrap();
        let l = kernel::boolean(&a, &b, BoolOp::Union).unwrap().unwrap();
        let sheet = kernel::extrude_sheet(&xy(), &[vec![Seg2::Line { a: Vec2::new(15.0, -5.0), b: Vec2::new(15.0, 11.0) }]], -5.0, 15.0).unwrap();
        let mut parts = cut_by(&l, &Tool::Sheet { body: &sheet, chains: None, field: None }).unwrap();
        parts.sort_by(|a, b| vol(a).total_cmp(&vol(b)));
        assert_eq!(parts.len(), 2, "{:?}", parts.iter().map(vol).collect::<Vec<_>>());
        assert!((vol(&parts[0]) - 500.0).abs() < 1e-3, "{}", vol(&parts[0]));
    }

    #[test]
    fn cuts_by_an_open_chains_cutter_and_a_closed_ones() {
        let plane = xy();
        let bx = kernel::box_solid(Vec3::new(-10.0, -10.0, 0.0), Vec3::new(10.0, 10.0, 10.0)).unwrap();
        // An L chain through the box: the corner quarter x > 2, y < 3 comes off.
        let chain = vec![Seg2::Line { a: Vec2::new(2.0, -30.0), b: Vec2::new(2.0, 3.0) }, Seg2::Line { a: Vec2::new(2.0, 3.0), b: Vec2::new(30.0, 3.0) }];
        let sheet = kernel::extrude_sheet(&plane, &[chain.clone()], -5.0, 15.0).unwrap();
        let chains = Arc::new(SheetChains { made: sheet.clone(), plane, chains: vec![chain], from: -5.0, to: 15.0 });
        let mut parts = cut_by(&bx, &Tool::Sheet { body: &sheet, chains: Some(chains), field: None }).unwrap();
        parts.sort_by(|a, b| vol(a).total_cmp(&vol(b)));
        assert_eq!(parts.len(), 2);
        assert!((vol(&parts[0]) - 8.0 * 13.0 * 10.0).abs() < 1e-3, "{}", vol(&parts[0]));
        // A circle's tube through it: a plug and the box round it.
        let circle = Loop2::circle(Vec2::new(0.0, 0.0), 3.0).segs;
        let tube = kernel::extrude_sheet(&plane, &[circle.clone()], -5.0, 15.0).unwrap();
        let chains = Arc::new(SheetChains { made: tube.clone(), plane, chains: vec![circle], from: -5.0, to: 15.0 });
        let mut parts = cut_by(&bx, &Tool::Sheet { body: &tube, chains: Some(chains), field: None }).unwrap();
        parts.sort_by(|a, b| vol(a).total_cmp(&vol(b)));
        assert_eq!(parts.len(), 2);
        assert!((vol(&parts[0]) - std::f64::consts::PI * 9.0 * 10.0).abs() < 0.5, "{}", vol(&parts[0]));
    }

    #[test]
    fn stands_a_fitted_surface_on_a_floor_as_the_solid_it_cuts_as() {
        use crate::surface_fit::{fit_surface, Net, SurfaceFitParams, PULL};
        let mut pts = Vec::new();
        for i in 0..21 {
            for j in 0..21 {
                let (x, y) = (-15.0 + 1.5 * i as f64, -15.0 + 1.5 * j as f64);
                pts.push(DVec3::new(x, y, 3.0 + 0.03 * x * x - 0.01 * y * y));
            }
        }
        let params = SurfaceFitParams { net: Some(Net { u: 6, v: 5 }), expand_pct: 5.0, ..SurfaceFitParams::default() };
        let fit = fit_surface(&pts, &params, None, PULL).unwrap();
        let floor_at = [DVec3::new(0.0, 0.0, -20.0)];
        let cutter = field_cutter(&fit, &floor_at).unwrap();
        assert!(!cutter.is_surface());
        let v = super::super::volume(&cutter).unwrap();
        assert!(v > 0.0, "{v}");
        // Carried on a third further each way, it still stands.
        let carried = super::super::surf_kernel::carried_fit(&fit, [fit.u_range[0] - 10.0, fit.u_range[1] + 10.0], [fit.v_range[0] - 10.0, fit.v_range[1] + 10.0]).unwrap();
        let cutter = field_cutter(&carried, &floor_at).unwrap();
        assert!(!cutter.is_surface());
    }
}
