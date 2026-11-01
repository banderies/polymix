// SPDX-License-Identifier: AGPL-3.0-only
//! The Combine (boolean) feature's build: two bodies made separately into one — the tool joined
//! to the body, cut out of it, or what they share kept — the tool going into the body or, kept,
//! standing beside it as it was. A body left in pieces goes in with all of them: a tool a
//! pattern left in copies apart cuts every window, and the body's own pieces are combined
//! together (`Builder::family_of`). A sheet as the body is trimmed by a solid tool — intersect
//! keeps what lies inside it, cut what lies outside — or, joined, sewn to the tool sheets edge to
//! edge. Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the boolean step, `build.ts`
//! applyToBody, toolFamily, bodyFamily, combineSheet, tookSomething), © Stefan Hermann (CNC
//! Kitchen), AGPL-3.0-only.

use solidcraft_kernel::{self as kernel, BoolOp, Body, SplitTool};

use super::{kerr, origins_after, volume, BodyState, Builder};
use crate::model::{BodyId, BooleanFeature, BooleanOp, Feature, Part};

/// How far apart the pieces of one sheet may lie and still be sewn into one (ScanRuler's
/// `SHEET_SEW_MM`).
const SHEET_SEW_MM: f64 = 1e-3;

/// The tolerances two sheets joined are sewn at, the tight one first (`SHEET_JOIN_MM`).
const SHEET_JOIN_MM: [f64; 3] = [SHEET_SEW_MM, 0.01, 0.05];

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, c: &BooleanFeature) -> Result<(), String> {
    // The body first, as ScanRuler's build asks for it before the step: a body gone says so.
    let (bid, body) = b.body_for(c.body)?;
    if b.resolve(c.tool) == bid {
        return Err("The body this one is combined with has gone into it already.".into());
    }
    // The tool in every piece, but the body it goes into.
    let family: Vec<BodyId> = b.family_of(c.tool).into_iter().filter(|id| *id != bid).collect();
    let tools: Vec<(BodyId, BodyState)> = family.iter().filter_map(|id| b.body(*id).map(|s| (*id, s.clone()))).collect();
    if tools.is_empty() {
        return Err("The body this one is combined with is gone, or failed to build.".into());
    }
    let name = b.name_of(part, bid);
    if body.surface {
        let made = combine_sheet(b, c.op, &body.shape, &name, &tools, part)?;
        // A solid tool stays a body as it was, as with keep; a sheet joined is sewn in.
        let consumed: Vec<BodyId> = if c.op == BooleanOp::Join && !c.keep { family.clone() } else { Vec::new() };
        let mut operands: Vec<(&Body, &Vec<i64>)> = vec![(&body.shape, &body.origins)];
        operands.extend(tools.iter().map(|(_, t)| (&t.shape, &t.origins)));
        let origins = origins_after(&made, &operands, b.step);
        for id in &consumed {
            b.remove_body(*id);
            b.alias(*id, bid);
        }
        // A sheet stays one body, its pieces apart and all.
        let ids = b.settle_sheet(bid, made, origins);
        b.touched(ids);
        return Ok(());
    }
    if let Some((sid, _)) = tools.iter().find(|(_, t)| t.surface) {
        return Err(format!("{} is a sheet, and {name} a solid — a solid is combined with solids. Cut a body with a sheet by Split, or close sheets into a solid by Trim; to trim the sheet by the solid, make the sheet the body.", b.name_of(part, *sid)));
    }
    // The pieces as one tool, and the body's pieces as one body, held together for the
    // operation alone.
    let tool_shapes: Vec<Body> = tools.iter().map(|(_, t)| t.shape.clone()).collect();
    let tool = fuse_all(&tool_shapes)?;
    let others: Vec<(BodyId, BodyState)> = b.family_of(c.body).into_iter().filter(|id| *id != bid && !family.contains(id)).filter_map(|id| b.body(id).filter(|s| !s.surface).map(|s| (id, s.clone()))).collect();
    let mut into_shapes = vec![body.shape.clone()];
    into_shapes.extend(others.iter().map(|(_, s)| s.shape.clone()));
    let into_shape = fuse_all(&into_shapes)?;
    let made = match c.op {
        BooleanOp::Join => kernel::boolean(&into_shape, &tool, BoolOp::Union).map_err(kerr)?.ok_or("The boolean left no closed solid.")?,
        BooleanOp::Cut => {
            let what = if tools.len() == 1 { b.name_of(part, tools[0].0) } else { "The tool".to_string() };
            let r = kernel::boolean(&into_shape, &tool, BoolOp::Cut).map_err(kerr)?;
            took_something(&into_shape, r, &what, &name)?
        }
        BooleanOp::Intersect => {
            let into_bodies: Vec<&Body> = std::iter::once(&body.shape).chain(others.iter().map(|(_, s)| &s.shape)).collect();
            kernel::boolean(&into_shape, &tool, BoolOp::Intersect).map_err(kerr)?.ok_or_else(|| empty_intersect(&into_bodies, &tool_shapes.iter().collect::<Vec<_>>()))?
        }
    };
    // Which feature first made each face: the body's pieces and the tool's, as they stood.
    let mut operands: Vec<(&Body, &Vec<i64>)> = Vec::new();
    let into_origins: Vec<i64>;
    if others.is_empty() {
        operands.push((&body.shape, &body.origins));
    } else {
        let mut parts: Vec<(&Body, &Vec<i64>)> = vec![(&body.shape, &body.origins)];
        parts.extend(others.iter().map(|(_, s)| (&s.shape, &s.origins)));
        into_origins = origins_after(&into_shape, &parts, b.step);
        operands.push((&into_shape, &into_origins));
    }
    let tool_origins: Vec<i64>;
    if tools.len() == 1 {
        operands.push((&tools[0].1.shape, &tools[0].1.origins));
    } else {
        let parts: Vec<(&Body, &Vec<i64>)> = tools.iter().map(|(_, t)| (&t.shape, &t.origins)).collect();
        tool_origins = origins_after(&tool, &parts, b.step);
        operands.push((&tool, &tool_origins));
    }
    let origins = origins_after(&made, &operands, b.step);
    // A kept tool stays a body as it was; the result was made of its shape, which is not used
    // up. The body's other pieces are in the result.
    let mut consumed: Vec<BodyId> = if c.keep { Vec::new() } else { family.clone() };
    consumed.extend(others.iter().map(|(id, _)| *id));
    for id in &consumed {
        b.remove_body(*id);
        b.alias(*id, bid);
    }
    let mut ids = b.settle(bid, made, origins, true)?;
    if c.keep {
        ids.extend(family.iter().copied().filter(|id| b.body(*id).is_some()));
    }
    b.touched(ids);
    Ok(())
}

/// What a body is called in the build's words: the name of the feature that made it, as
/// ScanRuler's builder names a body after its step — or its id where that is gone.
pub(crate) fn body_name(part: &Part, id: BodyId) -> String {
    part.feature(id.maker()).map_or_else(|| id.to_string(), |f| f.name.clone())
}

/// Shapes as one: the one alone, bodies apart joined without intersection work, the rest
/// joined one by one (ScanRuler's `fuseAll`).
pub(crate) fn fuse_all(shapes: &[Body]) -> Result<Body, String> {
    match shapes {
        [] => Err("Nothing to combine.".into()),
        [one] => Ok(one.clone()),
        _ => {
            if let Ok(Some(apart)) = kernel::union_apart(shapes) {
                return Ok(apart);
            }
            let mut acc = shapes[0].clone();
            for s in &shapes[1..] {
                acc = kernel::boolean(&acc, s, BoolOp::Union).map_err(kerr)?.ok_or("The pieces could not be held together.")?;
            }
            Ok(acc)
        }
    }
}

/// A cut that takes nothing out of the body fails, saying so (ScanRuler's `tookSomething`).
fn took_something(body: &Body, result: Option<Body>, what: &str, name: &str) -> Result<Body, String> {
    let was = volume(body).unwrap_or(0.0);
    let left = result.as_ref().and_then(volume).unwrap_or(0.0);
    if (was - left).abs() <= 1e-7 * was.max(1.0) {
        return Err(format!("{what} meets nothing of {name} — the cut took nothing out. Check where it stands against the body and how far it reaches."));
    }
    result.ok_or_else(|| "The boolean left no closed solid.".to_string())
}

/// An intersect that left nothing, in words (ScanRuler's `emptyIntersectWords`): no box of the
/// body's pieces meets a box of the tool's — they do not overlap — or one does, and the kernel
/// found no common volume where there may well be one: a body whose skin folds over itself.
fn empty_intersect(into: &[&Body], tools: &[&Body]) -> String {
    let boxes = |list: &[&Body]| list.iter().map(|b| super::offset_body::true_box(b)).collect::<Vec<_>>();
    let (xs, ys) = (boxes(into), boxes(tools));
    let meet = xs.iter().any(|x| ys.iter().any(|y| boxes_meet(x, y)));
    if meet {
        "The bounds overlap but the kernel found no common volume — a body may fold over itself.".into()
    } else {
        "The bodies do not overlap.".into()
    }
}

/// Whether two boxes meet, a tenth of a micrometre allowed (ScanRuler's `boxesMeet`).
fn boxes_meet(a: &solidcraft_geom::Aabb3, b: &solidcraft_geom::Aabb3) -> bool {
    const MARGIN: f64 = 1e-4;
    let (amin, amax, bmin, bmax) = ([a.min.x, a.min.y, a.min.z], [a.max.x, a.max.y, a.max.z], [b.min.x, b.min.y, b.min.z], [b.max.x, b.max.y, b.max.z]);
    (0..3).all(|i| !(amax[i] < bmin[i] - MARGIN || bmax[i] < amin[i] - MARGIN))
}

/// How many pieces apart a sheet is in: its faces gathered across the edges they share. The
/// kernel keeps a sewn sheet in one shell however many pieces it is in, so the pieces are
/// counted off the faces themselves (ScanRuler's `sewSheet` pieces).
pub(crate) fn sheet_pieces(sheet: &Body) -> usize {
    let nf = sheet.face_count();
    if nf <= 1 {
        return nf.max(1);
    }
    let size = super::diagonal_of(sheet).max(1e-3);
    let Ok(mesh) = sheet.display_mesh((size * 1e-3).max(1e-4)) else { return 1 };
    let mut root: Vec<usize> = (0..nf).collect();
    fn find(root: &mut [usize], mut i: usize) -> usize {
        while root[i] != i {
            root[i] = root[root[i]];
            i = root[i];
        }
        i
    }
    for faces in &mesh.edge_faces {
        let mut it = faces.iter().map(|f| *f as usize).filter(|f| *f < nf);
        let Some(first) = it.next() else { continue };
        for other in it {
            let (a, b) = (find(&mut root, first), find(&mut root, other));
            if a != b {
                root[a] = b;
            }
        }
    }
    (0..nf).filter(|i| find(&mut root, *i) == *i).count()
}

/// A Combine on a sheet (ScanRuler's `combineSheet`): `intersect` keeps the sheet inside a solid
/// tool and `cut` outside it — its faces split where the tool's faces cross them and those on
/// the side asked for kept, sewn as they were — and `join` sews the sheet and the tool sheets
/// into one, no face dropped: two sheets trimmed to each other along where they cross meet there
/// edge to edge.
fn combine_sheet(b: &mut Builder, op: BooleanOp, sheet: &Body, name: &str, tools: &[(BodyId, BodyState)], part: &Part) -> Result<Body, String> {
    let named = if tools.len() == 1 { b.name_of(part, tools[0].0) } else { "the tool".to_string() };
    if op == BooleanOp::Join {
        if let Some((sid, _)) = tools.iter().find(|(_, t)| !t.surface) {
            return Err(format!("{name} is a sheet, and {} a solid — a sheet joins sheets. Thicken the sheet first, then join the two.", b.name_of(part, *sid)));
        }
        let mut all: Vec<&Body> = vec![sheet];
        all.extend(tools.iter().map(|(_, t)| &t.shape));
        let mut pieces = 0;
        for tol in SHEET_JOIN_MM {
            let sewn = kernel::stitch(&all, tol).map_err(kerr)?;
            let n = sheet_pieces(&sewn);
            if n == 1 {
                return Ok(sewn);
            }
            pieces = n;
        }
        return Err(format!("{name} and {named} do not meet edge to edge — sewn within {} mm they stay {pieces} sheets apart. Trim each by the other first (Split, keep scan), so that both end where they cross.", SHEET_JOIN_MM[SHEET_JOIN_MM.len() - 1]));
    }
    if let Some((sid, _)) = tools.iter().find(|(_, t)| t.surface) {
        return Err(format!("{} is a sheet: a Combine trims a sheet by a solid — intersect keeps it inside, cut outside. To trim a sheet by another sheet, Split it.", b.name_of(part, *sid)));
    }
    let shapes: Vec<Body> = tools.iter().map(|(_, t)| t.shape.clone()).collect();
    let tool = fuse_all(&shapes)?;
    let inside = op == BooleanOp::Intersect;
    let trimmed = trim_sheet(sheet, &tool, inside)?.ok_or_else(|| {
        format!("Nothing of {name} lies {} {named} — {}.", if inside { "inside" } else { "outside" }, if inside { "the sheet has to run into the solid" } else { "the solid takes all of it" })
    })?;
    let n = sheet_pieces(&trimmed);
    if n > 1 {
        b.warn(format!("{name} came out as {n} pieces apart — a cutter as it is, but a Thicken takes one piece: split it again to keep one, or thicken each."));
    }
    Ok(trimmed)
}

/// A sheet trimmed by a solid: every face of the sheet split where a face of the solid crosses
/// it, and the pieces lying inside the solid (or outside it) kept — `None` when no piece does.
fn trim_sheet(sheet: &Body, tool: &Body, inside: bool) -> Result<Option<Body>, String> {
    let mut cur = sheet.clone();
    for j in 0..tool.face_count() {
        let st = SplitTool::Face { body: tool, face: j };
        // Per face of the sheet: not asked yet (0), missed by this face of the tool (1), or a
        // piece it cut (2) — a piece is not cut again by the surface it ends on.
        let mut state = vec![0u8; cur.face_count()];
        let mut asked = 0;
        while let Some(i) = state.iter().position(|s| *s == 0) {
            asked += 1;
            if asked > 4096 {
                break;
            }
            match kernel::split_faces_with_map(&cur, &[i], &st) {
                Ok((split, origin)) if split.face_count() > cur.face_count() && origin.len() == split.face_count() => {
                    state = origin.iter().map(|o| if *o == i { 2 } else { state.get(*o).copied().unwrap_or(1) }).collect();
                    cur = split;
                }
                _ => state[i] = 1,
            }
        }
    }
    let size = super::diagonal_of(&cur).max(1e-3);
    let mesh = cur.tessellate((size * 1e-3).max(1e-4)).map_err(kerr)?;
    let nf = cur.face_count();
    // A point on each face: the middle of its largest triangle.
    let mut best: Vec<Option<(f64, solidcraft_geom::Vec3)>> = vec![None; nf];
    for (t, f) in mesh.triangles.iter().zip(&mesh.tri_face) {
        let Some([a, b2, c]) = mesh.tri(t) else { continue };
        let area = (b2 - a).cross(c - a).len();
        if let Some(slot) = best.get_mut(*f as usize) {
            if slot.is_none_or(|(x, _)| area > x) {
                *slot = Some((area, (a + b2 + c) * (1.0 / 3.0)));
            }
        }
    }
    let mut drop = Vec::new();
    for (i, p) in best.iter().enumerate() {
        let keep = p.is_some_and(|(_, p)| tool.contains(p) == inside);
        if !keep {
            drop.push(i);
        }
    }
    if drop.len() == nf {
        return Ok(None);
    }
    if drop.is_empty() {
        return Ok(Some(cur));
    }
    let left = kernel::remove_faces(&cur, &drop).map_err(kerr)?;
    // Sewn as the pieces lie.
    Ok(Some(kernel::stitch(&[&left], SHEET_SEW_MM).unwrap_or(left)))
}
