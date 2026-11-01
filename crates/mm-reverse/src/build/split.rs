// SPDX-License-Identifier: AGPL-3.0-only
//! The Split feature's build: a body cut by sheets or solids into fragments, and the ones kept
//! — those lying on the scan (a fragment whose faces are mostly on the scan is part of it; a
//! pocket's plug has the pocket's roof where the scan has its floor, and falls out), the
//! largest or the smallest, or what lies between the sheets: the wall between a part's outer
//! and inner skin, each fitted to the scan. The kept fragments are made one body again. Within
//! the fitted sheets' regions, only the body inside their outlines is cut and the rest put back
//! as it was. A fitted surface that stops short of the body is carried on to its sides. A sheet
//! as the body is trimmed: its pieces kept by the scan or by area and sewn into one sheet.
//! Ported from ScanRuler (`plugins/reverse/core/build.ts` the split step, splitSheet,
//! keepFragments, betweenFragments, sheetsApart, withinRegions, offScanOf, reachingTools,
//! fittedFor; `resolve.ts` the split's checks), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only, on the SolidCraft kernel (see `surf_split` for how a sheet cuts there).

use std::sync::Arc;

use glam::{DVec2, DVec3};
use solidcraft_geom::Plane;
use solidcraft_kernel::{self as kernel, BoolOp, Body};

use super::surf::{area_samples, on_scan_share, BuildScan, OffScanReport, Reshaped};
use super::surf_kernel::{carried_fit, surface_area, traced_origins, MeshIndex, ShapeMesh};
use super::surf_split::{cut_by, fuse_all, sheet_pieces, Tool};
use super::{diagonal_of, kerr, mesh_deflection, polygon_region, sv3, BodyId, BodyState, Builder};
use crate::model::{Feature, Part, SplitFeature, SplitKeep, TrimPrimitive};
use crate::surface_fit::{carried_height, grid_area, grid_of, js_num, reach_over, reaching_grid_of, SurfaceFitResult, SurfacePlane, GRID_AREA_SLACK, GRID_PER_SPAN};
use crate::surface_region::{field_at, off_past_region, region_outline};

/// How near two sheets a split keeps the wall between may come, mm, and be said to meet there.
const MEET_MM: f64 = 1e-3;

/// How far apart the pieces of one sheet may lie and still be sewn into one, mm.
const SHEET_SEW_MM: f64 = 1e-3;

/// A body as a sentence names it: the feature that made it, as ScanRuler's builder names its
/// bodies.
fn named(part: &Part, id: BodyId) -> String {
    part.feature(id.maker()).map_or_else(|| format!("Body {id}"), |f| f.name.clone())
}

/// An area in words: whole mm², or in powers of ten past a million.
fn area_words(x: f64) -> String {
    if x >= 1e6 {
        let e = x.log10().floor();
        format!("{:.1}e{}", x / 10f64.powf(e), e as i64)
    } else {
        format!("{}", x.round())
    }
}

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &SplitFeature) -> Result<(), String> {
    if x.tools.is_empty() {
        return Err("Pick a sheet or a body to split with.".into());
    }
    if x.tools.contains(&x.body) {
        return Err("A body cannot split itself.".into());
    }
    if x.keep == SplitKeep::Between && x.tools.len() < 2 {
        return Err("Keeping what lies between the sheets takes two sheets or more — the skins the wall lies between.".into());
    }
    let (body_id, body) = b.body_for(x.body)?;
    let name = named(part, x.body);
    if body.surface {
        return split_sheet(b, part, x, body_id, &body, &name);
    }
    let mut tools: Vec<(BodyId, BodyState)> = Vec::new();
    for t in &x.tools {
        let (id, s) = b.body_for(*t).map_err(|_| "A sheet this split cuts with is gone, or failed to build.".to_string())?;
        tools.push((id, s));
    }
    let between = x.keep == SplitKeep::Between;
    // Within the sheets' regions: what is cut is the body inside their outlines, the rest put
    // back as it was after.
    let within = match x.within {
        Some(margin) => Some(within_regions(b, part, &x.tools, margin, &body.shape, &name)?),
        None => None,
    };
    let target = within.as_ref().map_or(body.shape.clone(), |w| w.inner.clone());
    let shapes: Vec<&Body> = tools.iter().map(|t| &t.1.shape).collect();
    if between {
        sheets_apart(part, &x.tools, &tools.iter().map(|t| (t.1.surface, &t.1.shape)).collect::<Vec<_>>(), &body.shape, &name, false)?;
    }
    let plain: Vec<Cutter> = tools.iter().map(|(id, t)| Cutter::of(b, *id, t)).collect();
    let mut fragments = split_all(&target, &plain)?;
    let crossed_words = format!("The sheets cross inside {name}: the corners they cut it into each have a face on every sheet, on different sides of them, so none is the wall between. Thicken one skin instead — fit it past the part's edge, trim it by Split or Combine, then Thicken.");
    let mut against: Vec<Body> = shapes.iter().map(|s| (*s).clone()).collect();
    let mut wall = if between { Some(between_fragments(&fragments, &against)?) } else { None };
    if wall.as_ref().is_some_and(|w| w.crossed) {
        return Err(crossed_words);
    }
    if fragments.len() < 2 || wall.as_ref().is_some_and(|w| w.kept.is_empty()) {
        // A fitted surface ends where its region did, grown by a share: short of the body's
        // sides, it is carried on to them and the cut tried again with it.
        if let Some(reaching) = reaching_tools(b, part, &x.tools, &target, &name)? {
            if between {
                let list: Vec<(bool, &Body)> = reaching.iter().zip(&tools).map(|(r, t)| (t.1.surface, &r.0)).collect();
                sheets_apart(part, &x.tools, &list, &body.shape, &name, true)?;
            }
            let carried: Vec<Cutter> = reaching.iter().zip(&tools).map(|(r, (id, t))| if r.2 { Cutter { shape: r.0.clone(), sheet: t.surface, chains: None, field: r.1.clone() } } else { Cutter::of(b, *id, t) }).collect();
            fragments = split_all(&target, &carried)?;
            against = reaching.into_iter().map(|r| r.0).collect();
            wall = None;
        }
    }
    if fragments.len() < 2 {
        return Err("The sheet does not cut the body into parts — it has to cross the body from side to side.".into());
    }
    let kept: Vec<Body> = if between {
        let w = match wall {
            Some(w) => w,
            None => between_fragments(&fragments, &against)?,
        };
        if w.crossed {
            return Err(crossed_words);
        }
        w.kept
    } else {
        let leave_out = within.as_ref().map(|w| &w.wall);
        keep_fragments(b, &fragments, x, false, leave_out.map(|f| f as &dyn Fn(DVec3) -> bool))
    };
    if kept.is_empty() {
        return Err(match x.keep {
            SplitKeep::Scan => "No fragment lies on the scan within the tolerance — widen it, or keep the largest.".into(),
            SplitKeep::Between => "No fragment lies between the sheets — none has a face on every one of them. Each sheet has to cross the body from side to side, and the body to reach past them all.".into(),
            _ => "No fragment to keep.".into(),
        });
    }
    // The kept fragments made one again with what lies outside the regions (ScanRuler's
    // `fuseAll`): one kept and nothing outside is that fragment; otherwise the body less the
    // fragments let go — the same solid, without fusing pieces that share faces.
    let removed: Vec<&Body> = fragments.iter().filter(|f| !kept.iter().any(|k| k.same(f))).collect();
    let acc = match (kept.as_slice(), &within) {
        ([one], None) => one.clone(),
        _ if removed.is_empty() => body.shape.clone(),
        _ => {
            let mut acc = body.shape.clone();
            for r in &removed {
                acc = kernel::boolean(&acc, r, BoolOp::Cut).map_err(kerr)?.ok_or("The split left nothing of the body.")?;
            }
            acc
        }
    };
    // The kernel tracks no history for a split: each face of the kept fragments is traced by a
    // point on it to the face of the body it lies on, and keeps that face's maker. Only the
    // faces on the tools are the split's.
    let origins = traced_origins(&acc, &[(&body.shape, &body.origins)], b.step as i64);
    let off = off_scan_of(b, part, x, &acc, &origins);
    let ids = b.settle(body_id, acc, origins, false)?;
    b.touched(ids);
    if !off.is_empty() {
        b.report().off_scan = off;
    }
    Ok(())
}

/// A tool as a split cuts with it: its shape, whether it is a sheet, a sketch sheet's chains,
/// and the height field of a fitted surface's whole face.
struct Cutter {
    shape: Body,
    sheet: bool,
    chains: Option<Arc<super::surf_split::SheetChains>>,
    field: Option<Arc<SurfaceFitResult>>,
}

impl Cutter {
    /// A body as it stands.
    fn of(b: &Builder, id: BodyId, s: &BodyState) -> Cutter {
        let whole = !b.surf.reshaped.contains_key(&id);
        Cutter {
            shape: s.shape.clone(),
            sheet: s.surface,
            chains: if s.surface { b.surf.chains.get(&id).cloned() } else { None },
            field: if s.surface && whole { b.surf.fits.get(&id).cloned() } else { None },
        }
    }

    fn tool(&self) -> Tool<'_> {
        if self.sheet {
            Tool::Sheet { body: &self.shape, chains: self.chains.clone(), field: self.field.clone() }
        } else {
            Tool::Solid(&self.shape)
        }
    }
}

/// The body cut by every tool in turn: each fragment by the next.
fn split_all(target: &Body, tools: &[Cutter]) -> Result<Vec<Body>, String> {
    let mut fragments = vec![target.clone()];
    for c in tools {
        let tool = c.tool();
        let mut next = Vec::new();
        for f in &fragments {
            next.extend(cut_by(f, &tool)?);
        }
        fragments = next;
    }
    Ok(fragments)
}

/// The fragments of a split to keep: those on the scan, or the largest or the smallest by
/// volume — a sheet's pieces by their area (ScanRuler's `keepFragments`). A fragment is on the
/// scan when the share of its faces' sample points within the tolerance of the scan is at least
/// the overlap. With no scan to read, the largest.
fn keep_fragments(b: &Builder, fragments: &[Body], x: &SplitFeature, sheet: bool, leave_out: Option<&dyn Fn(DVec3) -> bool>) -> Vec<Body> {
    if let (SplitKeep::Scan, Some(scan)) = (x.keep, b.surf.scan.as_ref()) {
        return fragments.iter().filter(|f| scan_share(scan, f, x.tolerance, leave_out) >= x.overlap).cloned().collect();
    }
    let sizes: Vec<f64> = fragments.iter().map(|f| if sheet { surface_area(f) } else { super::surf_kernel::mesh_volume(f) }).collect();
    let pick = if x.keep == SplitKeep::Smallest {
        sizes.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).map(|p| p.0)
    } else {
        sizes.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|p| p.0)
    };
    pick.map(|i| vec![fragments[i].clone()]).unwrap_or_default()
}

/// A shape's share on the scan, off a coarse mesh of it.
pub(crate) fn scan_share(scan: &BuildScan, shape: &Body, tolerance: f64, leave_out: Option<&dyn Fn(DVec3) -> bool>) -> f64 {
    match ShapeMesh::of(shape, 4.0) {
        Ok(m) => on_scan_share(scan, &m.positions, &m.triangles, tolerance, leave_out),
        Err(_) => 0.0,
    }
}

/// What lies between the sheets.
struct Between {
    kept: Vec<Body>,
    crossed: bool,
}

/// The fragments with a face on every tool, lying between them (ScanRuler's
/// `betweenFragments`): each fragment's faces looked for on each tool, and the side of each
/// tool it lies on read off how its face there looks against the sheet's normal. Two skins
/// that cross inside the body cut it into corners each with a face on both, on different
/// sides: none is kept, and `crossed` says why.
fn between_fragments(fragments: &[Body], tools: &[Body]) -> Result<Between, String> {
    if tools.len() < 2 {
        return Ok(Between { kept: Vec::new(), crossed: false });
    }
    let tool_meshes: Vec<(MeshIndex, f64)> = tools
        .iter()
        .filter_map(|t| {
            let m = ShapeMesh::of(t, 1.0).ok()?;
            Some((MeshIndex::of(&m)?, mesh_deflection(diagonal_of(t))))
        })
        .collect();
    if tool_meshes.len() != tools.len() {
        return Ok(Between { kept: Vec::new(), crossed: false });
    }
    let mut kept = Vec::new();
    let mut sides: Vec<String> = Vec::new();
    for f in fragments {
        let m = ShapeMesh::of(f, 1.0)?;
        let sag = mesh_deflection(diagonal_of(f));
        let faces = f.face_count();
        let mut points: Vec<(DVec3, DVec3)> = Vec::new();
        for face in 0..faces as u32 {
            let tris = m.face_triangles(face);
            if let Some(t) = tris.get(tris.len() / 2) {
                let [a, b, c] = t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
                points.push(((a + b + c) / 3.0, (b - a).cross(c - a).normalize_or_zero()));
            }
        }
        let mut side = String::new();
        let mut on_all = true;
        for (index, tsag) in &tool_meshes {
            let near = 2.0 * (sag + tsag) + 1e-3;
            match points.iter().find_map(|(p, n)| index.nearest(*p, near).map(|(_, _, _, sn)| (n, sn))) {
                Some((n, sn)) => side.push(if n.dot(sn) > 0.0 { '-' } else { '+' }),
                None => {
                    on_all = false;
                    break;
                }
            }
        }
        if !on_all {
            continue;
        }
        kept.push(f.clone());
        if !sides.contains(&side) {
            sides.push(side);
        }
    }
    Ok(if sides.len() > 1 { Between { kept: Vec::new(), crossed: true } } else { Between { kept, crossed: false } })
}

/// Before a split keeps what lies between its sheets: that no two of them meet inside the body
/// — come within MEET_MM, and cross at a point inside it (ScanRuler's `sheetsApart`). Where they
/// do, refused, saying where. A solid among the tools is crossed by a sheet as a matter of
/// course. The crossing is read off the sheets' meshes: where an edge of one's triangles passes
/// through a triangle of the other.
fn sheets_apart(part: &Part, ids: &[BodyId], shapes: &[(bool, &Body)], body: &Body, name: &str, carried: bool) -> Result<(), String> {
    let body_mesh = ShapeMesh::of(body, 1.0)?;
    let Some(inside) = MeshIndex::of(&body_mesh) else { return Ok(()) };
    for i in 0..shapes.len() {
        for j in i + 1..shapes.len() {
            if !shapes[i].0 || !shapes[j].0 {
                continue;
            }
            let (a, c) = (ShapeMesh::of(shapes[i].1, 1.0)?, ShapeMesh::of(shapes[j].1, 1.0)?);
            let mut at = None;
            for (x, y) in [(&a, &c), (&c, &a)] {
                if at.is_some() {
                    break;
                }
                at = crossing_inside(x, y, &inside);
            }
            let Some(at) = at else { continue };
            let word = |k: usize| ids.get(k).map_or_else(|| "a sheet".to_string(), |id| named(part, *id));
            return Err(format!(
                "{} and {}{} meet at ({}, {}, {}), inside {name} — the wall between them closes there, so no one solid lies between the two. Thicken one skin instead: fit it past the part's edge, trim it to shape by Split or Combine (both take a sheet), then Thicken.",
                word(i),
                word(j),
                if carried { format!(", carried on to reach {name},") } else { String::new() },
                fixed2(at.x),
                fixed2(at.y),
                fixed2(at.z)
            ));
        }
    }
    Ok(())
}

fn fixed2(v: f64) -> String {
    format!("{v:.2}")
}

/// A point where an edge of `a`'s triangles passes through a triangle of `b`, inside the body.
fn crossing_inside(a: &ShapeMesh, b: &ShapeMesh, inside: &MeshIndex) -> Option<DVec3> {
    let tri = |m: &ShapeMesh, t: &[u32; 3]| t.map(|v| m.positions.get(v as usize).copied().unwrap_or_default());
    let reach = |m: &ShapeMesh| m.positions.iter().fold((DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY)), |(lo, hi), p| (lo.min(*p), hi.max(*p)));
    let (blo, bhi) = reach(b);
    let tris_b: Vec<[DVec3; 3]> = b.triangles.iter().map(|t| tri(b, t)).collect();
    for t in &a.triangles {
        let p = tri(a, t);
        let (lo, hi) = (p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2]));
        if hi.cmplt(blo - DVec3::splat(MEET_MM)).any() || lo.cmpgt(bhi + DVec3::splat(MEET_MM)).any() {
            continue;
        }
        for (s, e) in [(p[0], p[1]), (p[1], p[2]), (p[2], p[0])] {
            for q in &tris_b {
                if let Some(x) = segment_triangle(s, e, q) {
                    if inside.deep_inside(x, 1e-6) {
                        return Some(x);
                    }
                }
            }
        }
    }
    None
}

/// Where a segment passes through a triangle (Möller–Trumbore), if it does.
fn segment_triangle(s: DVec3, e: DVec3, t: &[DVec3; 3]) -> Option<DVec3> {
    let d = e - s;
    let (e1, e2) = (t[1] - t[0], t[2] - t[0]);
    let h = d.cross(e2);
    let det = e1.dot(h);
    if det.abs() < 1e-14 {
        return None;
    }
    let f = 1.0 / det;
    let sv = s - t[0];
    let u = f * sv.dot(h);
    if !(0.0..=1.0).contains(&u) {
        return None;
    }
    let q = sv.cross(e1);
    let v = f * d.dot(q);
    if v < 0.0 || u + v > 1.0 {
        return None;
    }
    let w = f * e2.dot(q);
    (0.0..=1.0).contains(&w).then(|| s + d * w)
}

/// A split within its sheets' regions: what the sheets cut — the body outside it stays as it
/// was — and a point on a prism's side told apart.
struct Within {
    inner: Body,
    wall: Box<dyn Fn(DVec3) -> bool>,
}

/// A split within its sheets' regions (ScanRuler's `withinRegions`): each fitted sheet's
/// region outline grown by `margin` stood up along the sheet's height direction through the
/// body, as a prism — the body inside the prisms is what the sheets cut (`inner`); the body
/// outside them stays as it was (the fragments let go are cut from the body, not the kept ones
/// fused with what lies outside). `wall` tells a point on a prism's side — no
/// face of the part, but where the cut stops — which keep "scan" does not read. Refused for a
/// tool with no region, and where the outlines miss the body.
fn within_regions(b: &Builder, part: &Part, ids: &[BodyId], margin: f64, body: &Body, name: &str) -> Result<Within, String> {
    let points = ShapeMesh::of(body, 4.0)?.positions;
    let mut prisms = Vec::new();
    let mut walls: Vec<(Arc<SurfaceFitResult>, crate::surface_region::RegionField, f64)> = Vec::new();
    for id in ids {
        let tool = named(part, *id);
        let fit = b.surf.fits.get(&b.resolve(*id)).filter(|f| f.footprint.is_some()).cloned().ok_or_else(|| format!("{tool} has no region to keep the cut within — within takes freeform fitted surfaces (Fit Surface). Leave within out to cut with the whole of it."))?;
        let outline = region_outline(&fit, margin).ok_or_else(|| format!("{tool}'s region has no outline to keep the cut within."))?;
        let along: Vec<f64> = points.iter().map(|p| (*p - fit.origin).dot(fit.normal)).collect();
        let (min, max) = along.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), x| (lo.min(*x), hi.max(*x)));
        let lo = min - 1.0;
        let plane = Plane::new(sv3(fit.origin + fit.normal * lo), sv3(fit.axis_u), sv3(fit.axis_v)).ok_or("The fitted surface has no frame.")?;
        let height = max + 1.0 - lo;
        let flip = plane.normal().dot(sv3(fit.normal)) < 0.0;
        for l in &outline.loops {
            let region = polygon_region(l);
            let mut made = kernel::extrude(&plane, &[region], if flip { -height } else { 0.0 }, if flip { 0.0 } else { height }).map_err(kerr)?;
            if let Some(p) = made.pop() {
                prisms.push(p);
            }
        }
        walls.push((fit.clone(), outline.field, outline.level));
    }
    let prism = fuse_all(&prisms)?;
    let inner = kernel::boolean(body, &prism, BoolOp::Intersect).map_err(kerr)?.filter(|s| super::surf_kernel::mesh_volume(s) > 1e-9);
    let Some(inner) = inner else {
        return Err(format!("The regions' outlines, grown by {} mm, miss {name} — there is nothing within them to cut.", js_num(margin)));
    };
    let wall = move |p: DVec3| {
        walls.iter().any(|(fit, field, level)| {
            let d = p - fit.origin;
            (field_at(field, d.dot(fit.axis_u), d.dot(fit.axis_v)) - level).abs() <= 2.0 * field.cell
        })
    };
    Ok(Within { inner, wall: Box::new(wall) })
}

/// A split's sheets carried on to reach over the body (ScanRuler's `reachingTools`): a freeform
/// surface along the slope at its edge, a surface fitted as a plane as that plane, wide enough
/// — over the body's own points, not its box's corners. A sheet drawn from a sketch, or a
/// cylinder, stays as it is. A freeform carried on is held to its grid — its area within
/// GRID_AREA_SLACK of the grid's — and refused, named, where the kernel's surface through the
/// grid swings wild. `None` when none was carried on; each tool with whether it was.
fn reaching_tools(b: &Builder, part: &Part, ids: &[BodyId], shape: &Body, into: &str) -> Result<Option<Vec<(Body, Option<Arc<SurfaceFitResult>>, bool)>>, String> {
    let diagonal = diagonal_of(shape);
    let margin = 0.05 * diagonal + 1.0;
    let points = ShapeMesh::of(shape, 4.0)?.positions;
    let mut out = Vec::new();
    let mut any = false;
    for id in ids {
        let (at, state) = b.body_for(*id)?;
        let joined = b.surf.reshaped.get(&at) == Some(&Reshaped::Joined);
        let mut carried: Option<Body> = None;
        let mut field: Option<Arc<SurfaceFitResult>> = None;
        if let Some(fit) = b.surf.fits.get(&at).filter(|_| !joined) {
            let reach = reach_over(&fit.plane(), &points, margin);
            // A sheet trimmed since it was fitted is carried on from the whole of its fit.
            let whole = reach.is_none() && surface_area(&state.shape) * GRID_AREA_SLACK < grid_area(&grid_of(fit, GRID_PER_SPAN));
            if reach.is_some() || whole {
                // Carried on as a cubic over the reach, held to the grid of the carried
                // heights: its area within GRID_AREA_SLACK of the grid's.
                let (net, grid) = match reach {
                    Some(r) => (Arc::new(carried_fit(fit, r.u, r.v)?), reaching_grid_of(fit, r.u, r.v, GRID_PER_SPAN)),
                    None => (fit.clone(), grid_of(fit, GRID_PER_SPAN)),
                };
                let face = super::surf_kernel::fit_face(&net)?;
                let (area, want) = (surface_area(&face), grid_area(&grid));
                if !(area / want > GRID_AREA_SLACK) {
                    carried = Some(face);
                    field = Some(net);
                }
                if carried.is_none() {
                    let tool = named(part, *id);
                    return Err(format!("{tool} carried on to {into} comes out wild ({} mm² against the {} it should have) — fit it with an Expand by that reaches, or trim {into} to where the sheet is.", area_words(area), area_words(want)));
                }
            }
        } else if let Some(TrimPrimitive::Plane { origin, basis_u, basis_v, .. }) = b.surf.fitted.get(&at).filter(|_| !joined) {
            // The plane's rectangle over the body's points, in its own axes.
            if let Some(r) = reach_over(&SurfacePlane { origin: *origin, axis_u: *basis_u, axis_v: *basis_v, u_range: [0.0, 0.0], v_range: [0.0, 0.0] }, &points, margin) {
                let plane = Plane::new(sv3(*origin + *basis_u * r.u[0] + *basis_v * r.v[0]), sv3(*basis_u), sv3(*basis_v)).ok_or("The fitted plane has no direction.")?;
                let region = polygon_region(&[DVec2::new(0.0, 0.0), DVec2::new(r.u[1] - r.u[0], 0.0), DVec2::new(r.u[1] - r.u[0], r.v[1] - r.v[0]), DVec2::new(0.0, r.v[1] - r.v[0])]);
                carried = Some(kernel::patch_region(&plane, &region).map_err(kerr)?);
            }
        }
        let made = carried.is_some();
        any |= made;
        out.push((carried.unwrap_or_else(|| state.shape.clone()), field, made));
    }
    Ok(any.then_some(out))
}

/// How much of a split's cut face lies off the scan past the regions of the fitted sheets it was
/// cut by (ScanRuler's `offScanOf`): the faces the split made sampled as a fragment is, each
/// point given to the fitted sheet it lies on, as fitted or carried on. Said on the row where a
/// sheet's cut face lies further off than the split's tolerance.
fn off_scan_of(b: &mut Builder, part: &Part, x: &SplitFeature, shape: &Body, origins: &[i64]) -> Vec<OffScanReport> {
    let fitted: Vec<(String, Arc<SurfaceFitResult>)> = x.tools.iter().filter_map(|id| b.surf.fits.get(&b.resolve(*id)).filter(|f| f.footprint.is_some()).map(|f| (named(part, *id), f.clone()))).collect();
    if fitted.is_empty() {
        return Vec::new();
    }
    let Some(scan) = b.surf.scan.clone() else { return Vec::new() };
    let diagonal = diagonal_of(shape);
    let reach = (0.25 * diagonal).max(1.0);
    let Ok(m) = ShapeMesh::of(shape, 1.0) else { return Vec::new() };
    let own: Vec<[u32; 3]> = m.triangles.iter().zip(&m.tri_face).filter(|(_, f)| origins.get(**f as usize) == Some(&(b.step as i64))).map(|(t, _)| *t).collect();
    let on = (1e-4 * diagonal).max(0.01);
    let mut by_tool: Vec<(Vec<DVec3>, Vec<f64>)> = vec![(Vec::new(), Vec::new()); fitted.len()];
    for (q, area, _) in area_samples(&m.positions, &own, 1) {
        let mut best: Option<usize> = None;
        let mut gap = on;
        for (j, (_, fit)) in fitted.iter().enumerate() {
            let d = q - fit.origin;
            let off = (d.dot(fit.normal) - carried_height(fit, d.dot(fit.axis_u), d.dot(fit.axis_v))).abs();
            if off <= gap {
                best = Some(j);
                gap = off;
            }
        }
        if let Some(j) = best {
            by_tool[j].0.push(q);
            by_tool[j].1.push(area);
        }
    }
    let mut out = Vec::new();
    for ((tool, fit), (points, areas)) in fitted.iter().zip(&by_tool) {
        let Some(off) = off_past_region(fit, points, areas, |p| scan.closest(p, reach), x.tolerance) else { continue };
        let at = (off.at * 100.0).round() / 100.0;
        out.push(OffScanReport { tool: tool.clone(), max: (off.max * 1000.0).round() / 1000.0, area: (off.area * 10.0).round() / 10.0, at });
        if off.max > x.tolerance {
            b.warn(format!(
                "The cut face lies up to {:.2} mm off the scan past {tool}'s region, at ({}, {}, {}) — {} mm² of it further off than the {} mm tolerance: the sheet past its region is not the part's surface there. Split within its region (within), or fit it over more of the scan.",
                off.max,
                js_num(at.x),
                js_num(at.y),
                js_num(at.z),
                area_words(off.area),
                js_num(x.tolerance)
            ));
        }
    }
    out
}

/// A sheet split (ScanRuler's `splitSheet`): cut by its tools — sheets or solids — into
/// pieces, the pieces kept as for a body's fragments but the largest and the smallest by area,
/// and the kept ones sewn into one sheet. What a fitted skin laid past the part's edge is
/// trimmed back by before it is thickened. A fitted tool that stops short of the sheet's edges
/// is carried on to them. Between is a solid's: refused. The faces kept are the sheet's own,
/// traced by where they lie.
fn split_sheet(b: &mut Builder, part: &Part, x: &SplitFeature, body_id: BodyId, body: &BodyState, name: &str) -> Result<(), String> {
    if x.keep == SplitKeep::Between {
        return Err(format!("{name} is a sheet: keep \"between\" takes the wall between two skins out of a solid. A sheet split keeps its pieces on the scan, the largest or the smallest."));
    }
    if x.within.is_some() {
        return Err(format!("{name} is a sheet: within keeps a solid's cut to the sheets' regions. Trim the sheet by sheets or solids as they are, or fit it over less."));
    }
    let mut tools: Vec<(BodyId, BodyState)> = Vec::new();
    for t in &x.tools {
        tools.push(b.body_for(*t).map_err(|_| "A sheet or body this split cuts with is gone, or failed to build.".to_string())?);
    }
    let make = |list: &[Cutter]| -> Result<Vec<Body>, String> {
        let tools: Vec<Tool> = list.iter().map(Cutter::tool).collect();
        sheet_pieces(&body.shape, &tools)
    };
    let plain: Vec<Cutter> = tools.iter().map(|(id, t)| Cutter::of(b, *id, t)).collect();
    let mut pieces = make(&plain)?;
    if pieces.len() < 2 {
        if let Some(reaching) = reaching_tools(b, part, &x.tools, &body.shape, name)? {
            let carried: Vec<Cutter> = reaching.into_iter().zip(&tools).map(|((s, field, made), (id, t))| if made { Cutter { shape: s, sheet: t.surface, chains: None, field } } else { Cutter::of(b, *id, t) }).collect();
            pieces = make(&carried)?;
        }
    }
    if pieces.len() < 2 {
        return Err(format!("The tools do not cut {name} into pieces — a tool has to cross the sheet from edge to edge."));
    }
    let kept = keep_fragments(b, &pieces, x, true, None);
    if kept.is_empty() {
        return Err(if x.keep == SplitKeep::Scan { format!("No piece of {name} lies on the scan within the tolerance — widen it, or keep the largest.") } else { "No piece to keep.".into() });
    }
    let sewn = if kept.len() == 1 {
        kept[0].clone()
    } else {
        let refs: Vec<&Body> = kept.iter().collect();
        kernel::stitch(&refs, SHEET_SEW_MM).map_err(kerr)?
    };
    let apart = sewn.lumps().map(|l| l.len()).unwrap_or(1);
    if apart > 1 {
        b.warn(format!("{name} came out as {apart} pieces apart — a cutter as it is, but a Thicken takes one piece: split it again to keep one, or thicken each."));
    }
    let origins = traced_origins(&sewn, &[(&body.shape, &body.origins)], b.step as i64);
    let mut state = BodyState::new(sewn, origins);
    state.surface = true;
    state.label = body.label.clone();
    b.set_body(body_id, state);
    // A sheet trimmed keeps its fit, and is no longer the whole patch it was fitted as.
    let was = b.surf.reshaped.get(&body_id).copied();
    b.surf.reshaped.insert(body_id, if was == Some(Reshaped::Joined) { Reshaped::Joined } else { Reshaped::Trimmed });
    b.surf.chains.remove(&body_id);
    b.touched(vec![body_id]);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::surface::tests::{patch_scan, surface};
    use crate::build::{build, build_scanned, default_regions, volume, BuildScan};
    use crate::frame::WorldAxis;
    use crate::model::{BodyOp, ExtrudeExtent, ExtrudeFeature, FeatureKind, SheetFeature, SketchFeature};
    use crate::regions::find_regions;
    use crate::sketch::{Shape, Sketch};

    fn block(part: &mut Part, x0: f64, y0: f64, w: f64, h: f64, z0: f64, z1: f64) -> u32 {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(x0, y0), DVec2::new(x0 + w, y0 + h));
        let regions = default_regions(&find_regions(&s));
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Extrude(ExtrudeFeature::new(sk, regions, ExtrudeExtent::Distance { from: z0, to: z1 }, BodyOp::New)))
    }

    fn sheet(part: &mut Part, shapes: &[Shape], from: f64, to: f64) -> u32 {
        let mut s = Sketch::default();
        for sh in shapes {
            s.add_shape(sh.clone());
        }
        let sk = part.add(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 0.0, s)));
        part.add(FeatureKind::Sheet(SheetFeature { sketch_id: sk, from, to, exprs: Default::default() }))
    }

    fn split(body: u32, tools: &[u32], keep: SplitKeep) -> FeatureKind {
        FeatureKind::Split(SplitFeature { body: BodyId::new(body), tools: tools.iter().map(|t| BodyId::new(*t)).collect(), keep, tolerance: 0.3, overlap: 0.5, within: None, exprs: Default::default() })
    }

    fn vol_of(built: &crate::build::Built, id: u32) -> f64 {
        built.bodies.iter().find(|b| b.id == BodyId::new(id)).and_then(|b| volume(&b.body)).unwrap_or(f64::NAN)
    }

    #[test]
    fn keeps_the_largest_or_the_smallest_fragment() {
        for (keep, want) in [(SplitKeep::Largest, 2400.0), (SplitKeep::Smallest, 1600.0)] {
            let mut part = Part::default();
            let bx = block(&mut part, -10.0, -10.0, 20.0, 20.0, 0.0, 10.0);
            let cut = sheet(&mut part, &[Shape::Line { a: DVec2::new(-30.0, 2.0), b: DVec2::new(30.0, 2.0) }], -10.0, 30.0);
            let sp = part.add(split(bx, &[cut], keep));
            let built = build(&part);
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            assert!((vol_of(&built, bx) - want).abs() < 1e-3, "{keep:?} {}", vol_of(&built, bx));
            // The sheet stands, a surface of its own.
            assert!(built.bodies.iter().any(|b| b.id == BodyId::new(cut) && b.surface));
            assert_eq!(built.touched.get(&sp), Some(&vec![BodyId::new(bx)]));
        }
    }

    // reSplitSides.test.ts: a box split at x 4 keeps five faces as the extrude's, and the one on
    // the sheet as the split's.
    #[test]
    fn keeps_the_faces_makers_the_one_on_the_sheet_the_splits() {
        let mut part = Part::default();
        let bx = block(&mut part, 0.0, 0.0, 10.0, 10.0, 0.0, 10.0);
        let cut = sheet(&mut part, &[Shape::Line { a: DVec2::new(4.0, -5.0), b: DVec2::new(4.0, 15.0) }], -5.0, 15.0);
        let sp = part.add(split(bx, &[cut], SplitKeep::Largest));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let b = built.bodies.iter().find(|b| b.id == BodyId::new(bx)).unwrap();
        let makers: Vec<i64> = b.topology.faces.iter().map(|f| f.fingerprint.origin).collect();
        assert_eq!(makers.iter().filter(|m| **m == bx as i64).count(), 5, "{makers:?}");
        assert_eq!(makers.iter().filter(|m| **m == sp as i64).count(), 1, "{makers:?}");
    }

    #[test]
    fn refuses_a_sheet_that_does_not_cross_and_the_checks_before() {
        let mut part = Part::default();
        let bx = block(&mut part, -10.0, -10.0, 20.0, 20.0, 0.0, 10.0);
        let short = sheet(&mut part, &[Shape::Line { a: DVec2::new(-30.0, 2.0), b: DVec2::new(0.0, 2.0) }], -10.0, 30.0);
        let a = part.add(split(bx, &[short], SplitKeep::Largest));
        let b = part.add(split(bx, &[], SplitKeep::Largest));
        let c = part.add(split(bx, &[bx], SplitKeep::Largest));
        let d = part.add(split(bx, &[short], SplitKeep::Between));
        let built = build(&part);
        assert_eq!(built.errors.get(&a).map(String::as_str), Some("The sheet does not cut the body into parts — it has to cross the body from side to side."));
        assert_eq!(built.errors.get(&b).map(String::as_str), Some("Pick a sheet or a body to split with."));
        assert_eq!(built.errors.get(&c).map(String::as_str), Some("A body cannot split itself."));
        assert_eq!(built.errors.get(&d).map(String::as_str), Some("Keeping what lies between the sheets takes two sheets or more — the skins the wall lies between."));
    }

    // reThicken.test.ts: a solid between two skins — keeps the one solid between them, its two
    // big faces on the skins; and says so when no fragment has a face on every skin.
    #[test]
    fn keeps_the_one_solid_between_two_curved_skins() {
        let arc = |r: f64| Shape::Arc { c: DVec2::ZERO, r, start: (-40f64).to_radians(), sweep: 80f64.to_radians() };
        let mut part = Part::default();
        let s30 = sheet(&mut part, &[arc(30.0)], -5.0, 15.0);
        let s32 = sheet(&mut part, &[arc(32.0)], -5.0, 15.0);
        let bx = block(&mut part, 25.0, -8.0, 15.0, 16.0, 0.0, 10.0);
        let sp = part.add(split(bx, &[s32, s30], SplitKeep::Between));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let mut area = 0.0;
        let n = 4000;
        for i in 0..n {
            let y = -8.0 + 16.0 * (i as f64 + 0.5) / n as f64;
            area += ((32.0f64 * 32.0 - y * y).sqrt() - (30.0f64 * 30.0 - y * y).sqrt()) * (16.0 / n as f64);
        }
        let v = vol_of(&built, bx);
        assert!((v - area * 10.0).abs() < 0.05, "{v} against {}", area * 10.0);
        let wall = built.bodies.iter().find(|b| b.id == BodyId::new(bx)).unwrap();
        let mut rounds: Vec<f64> = wall.topology.faces.iter().filter(|f| f.fingerprint.kind == crate::topo::FaceKind::Cylinder).map(|f| (f.fingerprint.radius.unwrap_or(0.0) * 1000.0).round() / 1000.0).collect();
        rounds.sort_by(|a, b| a.total_cmp(b));
        assert_eq!(rounds, vec![30.0, 32.0]);
        let _ = sp;
        // A block that crosses the outer skin only.
        let mut part = Part::default();
        let s30 = sheet(&mut part, &[arc(30.0)], -5.0, 15.0);
        let s32 = sheet(&mut part, &[arc(32.0)], -5.0, 15.0);
        let bx = block(&mut part, 31.0, -8.0, 10.0, 16.0, 0.0, 10.0);
        let sp = part.add(split(bx, &[s30, s32], SplitKeep::Between));
        let built = build(&part);
        assert!(built.errors.get(&sp).is_some_and(|e| e.starts_with("No fragment lies between the sheets")), "{:?}", built.errors);
    }

    // reThicken.test.ts: keeps the wall between two fitted freeform skins.
    #[test]
    fn keeps_the_wall_between_two_fitted_freeform_skins() {
        let (outer, tris, region) = patch_scan(15.0, |x, y| 7.0 + 0.01 * (x * x + y * y));
        let (inner, _, _) = patch_scan(15.0, |x, y| 5.0 + 0.01 * (x * x + y * y));
        let n = outer.len() as u32;
        let mut positions = outer.clone();
        positions.extend(inner);
        let mut triangles = tris.clone();
        triangles.extend(tris.iter().map(|t| t.map(|v| v + n)));
        let scan = Arc::new(BuildScan::new(positions, triangles));
        let mut part = Part::default();
        let a = part.add(surface(region.clone(), 5, 10.0));
        let b = part.add(surface(region.iter().map(|v| v + n).collect(), 5, 10.0));
        let bx = block(&mut part, -10.0, -10.0, 20.0, 20.0, 0.0, 12.0);
        part.add(split(bx, &[a, b], SplitKeep::Between));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let v = vol_of(&built, bx);
        assert!((v / 800.0 - 1.0).abs() < 0.01, "{v}");
    }

    #[test]
    fn keeps_the_fragments_on_the_scan() {
        // The scan: the top of a 20 × 20 block at z = 10 and its four sides down to z = 0 —
        // a box with no floor.
        let mut pos = Vec::new();
        let mut tris = Vec::new();
        let mut quad = |a: DVec3, b: DVec3, c: DVec3, d: DVec3| {
            let n = 10;
            let base = pos.len() as u32;
            for i in 0..=n {
                for j in 0..=n {
                    let (s, t) = (i as f64 / n as f64, j as f64 / n as f64);
                    pos.push(a + (b - a) * s + (d - a) * t + (c - b - d + a) * s * t);
                }
            }
            for i in 0..n as u32 {
                for j in 0..n as u32 {
                    let k = base + i * (n as u32 + 1) + j;
                    tris.push([k, k + n as u32 + 1, k + n as u32 + 2]);
                    tris.push([k, k + n as u32 + 2, k + 1]);
                }
            }
        };
        let p = |x: f64, y: f64, z: f64| DVec3::new(x, y, z);
        quad(p(-10.0, -10.0, 10.0), p(10.0, -10.0, 10.0), p(10.0, 10.0, 10.0), p(-10.0, 10.0, 10.0));
        quad(p(-10.0, -10.0, 0.0), p(10.0, -10.0, 0.0), p(10.0, -10.0, 10.0), p(-10.0, -10.0, 10.0));
        quad(p(-10.0, 10.0, 0.0), p(10.0, 10.0, 0.0), p(10.0, 10.0, 10.0), p(-10.0, 10.0, 10.0));
        quad(p(-10.0, -10.0, 0.0), p(-10.0, 10.0, 0.0), p(-10.0, 10.0, 10.0), p(-10.0, -10.0, 10.0));
        quad(p(10.0, -10.0, 0.0), p(10.0, 10.0, 0.0), p(10.0, 10.0, 10.0), p(10.0, -10.0, 10.0));
        let scan = Arc::new(BuildScan::new(pos, tris));
        // A block running on past the scan to x = 30, split at x = 10 where the scan's wall
        // stands: the fragment inside the scan lies on it, the one past it does not.
        let mut part = Part::default();
        let bx = block(&mut part, -10.0, -10.0, 40.0, 20.0, 0.0, 10.0);
        let cut = sheet(&mut part, &[Shape::Line { a: DVec2::new(10.0, -30.0), b: DVec2::new(10.0, 30.0) }], -30.0, 30.0);
        part.add(split(bx, &[cut], SplitKeep::Scan));
        let built = build_scanned(&part, Some(scan));
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert!((vol_of(&built, bx) - 4000.0).abs() < 1e-3, "{}", vol_of(&built, bx));
    }

    // reSplitReach.test.ts: a split by a fitted surface that stops short of the body.
    mod reach {
        use super::*;

        /// The block 40 × 40 × 10 about the z axis, split by a surface through a small patch of
        /// the scan in its middle.
        fn split_by(height: fn(f64, f64) -> f64, plane: bool) -> (crate::build::Built, u32) {
            let (pos, tris, region) = patch_scan(3.0, height);
            let scan = Arc::new(BuildScan::new(pos, tris));
            let mut part = Part::default();
            let bx = block(&mut part, -20.0, -20.0, 40.0, 40.0, 0.0, 10.0);
            let mut k = surface(region, 4, 10.0);
            if let FeatureKind::Surface(sf) = &mut k {
                if plane {
                    sf.shape = Some(crate::model::SurfaceShape::Plane);
                    sf.fit.net = None;
                }
            }
            let sf = part.add(k);
            part.add(split(bx, &[sf], SplitKeep::Largest));
            (build_scanned(&part, Some(scan)), bx)
        }

        #[test]
        fn carries_a_plane_on_to_the_bodys_sides() {
            let (built, bx) = split_by(|x, _| 4.0 + 0.1 * x, true);
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            // Above z = 4 + 0.1 x over the block: 40 × 40 × 6, the larger part.
            assert!((vol_of(&built, bx) - 9600.0).abs() < 0.5, "{}", vol_of(&built, bx));
        }

        #[test]
        fn carries_a_freeform_surface_on_along_its_slope_at_the_edge() {
            let height = |x: f64, y: f64| 5.0 + 0.01 * (x * x + y * y);
            let (built, bx) = split_by(height, false);
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            let v = vol_of(&built, bx);
            assert!(v > 8000.0 && v < 13000.0, "{v}");
            let b = built.bodies.iter().find(|b| b.id == BodyId::new(bx)).unwrap();
            let m = ShapeMesh::of(&b.body, 1.0).unwrap();
            let top = m.positions.iter().fold(f64::NEG_INFINITY, |t, p| t.max(p.z));
            assert!(top < 10.0, "{top}");
            // At the block's side the cut stands where the slope takes it, about 6.2 mm up —
            // the bowl itself would be at 9 there.
            let at_side = m.positions.iter().filter(|p| (p.x - 20.0).abs() < 1e-3 && p.y.abs() < 1.0).fold(f64::NEG_INFINITY, |t, p| t.max(p.z));
            assert!((at_side - 6.2).abs() < 0.15, "{at_side}");
            // Over the patch the cut is the fitted surface itself.
            let near: Vec<&DVec3> = m.positions.iter().filter(|p| p.x.hypot(p.y) < 2.5 && p.z > 1.0).collect();
            assert!(!near.is_empty());
            for p in near {
                assert!((p.z - height(p.x, p.y)).abs() < 0.01, "{p}");
            }
        }

        #[test]
        fn still_refuses_a_sheet_drawn_short_from_a_sketch() {
            let mut part = Part::default();
            let bx = block(&mut part, -20.0, -20.0, 40.0, 40.0, 0.0, 10.0);
            let cut = sheet(&mut part, &[Shape::Line { a: DVec2::new(-5.0, 0.0), b: DVec2::new(5.0, 0.0) }], -1.0, 11.0);
            let sp = part.add(split(bx, &[cut], SplitKeep::Largest));
            let built = build(&part);
            assert!(built.errors.get(&sp).is_some_and(|e| e.contains("cross the body from side to side")), "{:?}", built.errors);
        }
    }

    // reSplitSides.test.ts: a split between two sheets refuses two that cross inside the body,
    // saying where; apart, the wall between them is kept.
    #[test]
    fn refuses_two_sheets_that_cross_inside_the_body_saying_where() {
        let line = |a: (f64, f64), b: (f64, f64)| Shape::Line { a: DVec2::new(a.0, a.1), b: DVec2::new(b.0, b.1) };
        let mut part = Part::default();
        let s2 = sheet(&mut part, &[line((-30.0, -10.0), (30.0, 10.0))], -5.0, 15.0);
        let s4 = sheet(&mut part, &[line((-30.0, 10.0), (30.0, -10.0))], -5.0, 15.0);
        for (id, name) in [(s2, "Skin 2"), (s4, "Skin 4")] {
            if let Some(f) = part.feature_mut(id) {
                f.name = name.into();
            }
        }
        let bx = block(&mut part, -20.0, -20.0, 40.0, 40.0, 0.0, 10.0);
        if let Some(f) = part.feature_mut(bx) {
            f.name = "Extrude 6".into();
        }
        let sp = part.add(split(bx, &[s2, s4], SplitKeep::Between));
        let t = std::time::Instant::now();
        let built = build(&part);
        let e = built.errors.get(&sp).cloned().unwrap_or_default();
        assert!(e.starts_with("Skin 2 and Skin 4 meet at (") && e.contains("), inside Extrude 6 — the wall between them closes there"), "{e}");
        assert!(e.ends_with("trim it to shape by Split or Combine (both take a sheet), then Thicken."), "{e}");
        // Refused before the kernel cuts: in seconds (ScanRuler's five, with room for a loaded machine).
        assert!(t.elapsed().as_secs_f64() < 30.0);
        // Apart, the wall between them is kept.
        let mut part = Part::default();
        let s2 = sheet(&mut part, &[line((-30.0, -2.0), (30.0, -2.0))], -5.0, 15.0);
        let s4 = sheet(&mut part, &[line((-30.0, 2.0), (30.0, 2.0))], -5.0, 15.0);
        let bx = block(&mut part, -20.0, -20.0, 40.0, 40.0, 0.0, 10.0);
        part.add(split(bx, &[s2, s4], SplitKeep::Between));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        assert!((vol_of(&built, bx) - 40.0 * 4.0 * 10.0).abs() < 1e-3, "{}", vol_of(&built, bx));
    }

    // reSheetTrim.test.ts: split by an open chain's sheet, keep scan — the overshoot past the
    // scan's edge goes, the skin stays the surface's; smallest keeps the overshoot.
    #[test]
    fn trims_a_skins_overshoot_past_the_scan_by_an_open_chains_sheet() {
        use crate::build::surface::tests::flange;
        let (pos, tris, region) = patch_scan(15.0, flange);
        let scan = Arc::new(BuildScan::new(pos, tris));
        let make = |keep: SplitKeep| {
            let mut part = Part::default();
            let skin = part.add(surface(region.clone(), 5, 20.0));
            let tip = sheet(&mut part, &[Shape::Line { a: DVec2::new(13.5, -40.0), b: DVec2::new(13.5, 0.0) }, Shape::Line { a: DVec2::new(13.5, 0.0), b: DVec2::new(14.5, 40.0) }], -20.0, 40.0);
            let sp = part.add(split(skin, &[tip], keep));
            let th = part.add(FeatureKind::Thicken(crate::model::ThickenFeature { sheet: BodyId::new(skin), thickness: 1.8, side: crate::model::ThickenSide::In, op: BodyOp::New, target: None, exprs: Default::default() }));
            (build_scanned(&part, Some(scan.clone())), skin, sp, th)
        };
        let (built, skin, _, th) = make(SplitKeep::Scan);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let sheet_body = built.bodies.iter().find(|b| b.id == BodyId::new(skin)).unwrap();
        assert!(sheet_body.surface);
        let m = ShapeMesh::of(&sheet_body.body, 1.0).unwrap();
        let (lo, hi) = m.positions.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |a, p| (a.0.min(p.x), a.1.max(p.x)));
        assert!(hi < 14.6 && hi > 13.4, "{hi}");
        assert!(lo < -17.0, "{lo}");
        assert_eq!(sheet_body.topology.faces.iter().map(|f| f.fingerprint.origin).collect::<Vec<_>>(), vec![skin as i64]);
        assert!(built.bodies.iter().any(|b| b.id == BodyId::new(th) && !b.surface));
        let (built, skin, _, _) = make(SplitKeep::Smallest);
        let sheet_body = built.bodies.iter().find(|b| b.id == BodyId::new(skin)).unwrap();
        let m = ShapeMesh::of(&sheet_body.body, 1.0).unwrap();
        let lo = m.positions.iter().fold(f64::INFINITY, |a, p| a.min(p.x));
        assert!(lo > 13.4, "{lo}");
    }

    // reSplitWithin.test.ts: a band of S-profile across a block's top, its region cut where the
    // band still curves.
    mod within {
        use super::*;

        fn s(y: f64) -> f64 {
            -1.0 * (-((y + 4.0) / 2.2).powi(2)).exp() + 2.5 * (-((y - 4.0) / 2.2).powi(2)).exp()
        }
        fn w(x: f64) -> f64 {
            if x.abs() <= 8.0 {
                1.0
            } else if x.abs() >= 14.0 {
                0.0
            } else {
                (((x.abs() - 8.0) / 6.0) * std::f64::consts::FRAC_PI_2).cos().powi(2)
            }
        }
        fn height(x: f64, y: f64) -> f64 {
            10.0 + s(y) * w(x)
        }
        fn in_region(x: f64, y: f64) -> bool {
            x.abs() <= 12.0 && (-6.5..=6.5).contains(&y)
        }

        fn scan_of() -> (Arc<BuildScan>, Vec<u32>) {
            let (nx, ny) = (121u32, 81u32);
            let mut pos = Vec::new();
            let mut tris = Vec::new();
            let mut region = Vec::new();
            for i in 0..nx {
                for j in 0..ny {
                    let (x, y) = (-30.0 + i as f64 * 0.5, -20.0 + j as f64 * 0.5);
                    if in_region(x, y) {
                        region.push(i * ny + j);
                    }
                    pos.push(DVec3::new(x as f32 as f64, y as f32 as f64, height(x, y) as f32 as f64));
                }
            }
            for i in 0..nx - 1 {
                for j in 0..ny - 1 {
                    let a = i * ny + j;
                    tris.push([a, a + ny, a + ny + 1]);
                    tris.push([a, a + ny + 1, a + 1]);
                }
            }
            (Arc::new(BuildScan::new(pos, tris)), region)
        }

        /// The block 60 × 40 × 13, the band fitted with a margin of 10 %, and a split of the
        /// block by it.
        fn part(within: Option<f64>) -> (Part, u32, u32, u32) {
            let (_, region) = scan_of();
            let mut part = Part::default();
            let bx = block(&mut part, -30.0, -20.0, 60.0, 40.0, 0.0, 13.0);
            let mut k = surface(region, 5, 10.0);
            if let FeatureKind::Surface(sf) = &mut k {
                sf.fit.net = None;
            }
            let band = part.add(k);
            if let Some(f) = part.feature_mut(band) {
                f.name = "Band".into();
            }
            let sp = part.add(FeatureKind::Split(SplitFeature { body: BodyId::new(bx), tools: vec![BodyId::new(band)], keep: SplitKeep::Largest, tolerance: 0.2, overlap: 0.5, within, exprs: Default::default() }));
            (part, bx, band, sp)
        }

        /// The body's top off the scan, in the region and outside it.
        fn top_off(body: &Body) -> (f64, f64) {
            let m = ShapeMesh::of(body, 1.0).unwrap();
            let (mut in_band, mut outside) = (0.0f64, 0.0f64);
            for p in &m.positions {
                if p.z < 5.0 || p.x.abs() > 29.9 || p.y.abs() > 19.9 {
                    continue;
                }
                let off = p.z - height(p.x, p.y);
                if in_region(p.x, p.y) {
                    in_band = in_band.max(off.abs());
                } else if off.abs() > outside.abs() {
                    outside = off;
                }
            }
            (in_band, outside)
        }

        #[test]
        fn the_fit_says_how_far_its_margin_stands_off_the_scan() {
            let (scan, _) = scan_of();
            let (mut part, _, band, sp) = part(None);
            part.remove(sp);
            let built = build_scanned(&part, Some(scan));
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            let (off, at) = built.surf.get(&band).and_then(|r| r.surface.as_ref()).and_then(|s| s.past_region).unwrap();
            assert!(off > 1.0, "{off}");
            assert!(at.x.abs() > 12.0 || at.y.abs() > 6.5, "{at}");
            assert!(built.warnings.get(&band).is_some_and(|w| w.iter().any(|w| w.starts_with("Past its region the surface stands up to "))), "{:?}", built.warnings);
        }

        #[test]
        fn a_split_carried_on_says_how_much_of_its_cut_face_lies_off_the_scan() {
            let (scan, _) = scan_of();
            let (part, bx, _, sp) = part(None);
            let built = build_scanned(&part, Some(scan));
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            let body = built.bodies.iter().find(|b| b.id == BodyId::new(bx)).unwrap();
            assert!(top_off(&body.body).1 < -4.5, "{:?}", top_off(&body.body));
            let off = &built.surf.get(&sp).unwrap().off_scan;
            assert_eq!(off.len(), 1);
            assert_eq!(off[0].tool, "Band");
            assert!(off[0].max > 4.5 && off[0].area > 100.0, "{off:?}");
            assert!(built.warnings.get(&sp).is_some_and(|w| w.iter().any(|w| w.starts_with("The cut face lies up to ") && w.contains(" mm off the scan past Band's region"))), "{:?}", built.warnings);
        }

        #[test]
        fn a_split_within_the_region_cuts_the_band_and_leaves_the_top_outside_it() {
            let (scan, _) = scan_of();
            let (part, bx, _, sp) = part(Some(1.0));
            let built = build_scanned(&part, Some(scan));
            assert!(built.errors.is_empty(), "{:?}", built.errors);
            let body = built.bodies.iter().find(|b| b.id == BodyId::new(bx)).unwrap();
            let (in_band, outside) = top_off(&body.body);
            assert!(in_band < 0.8, "{in_band}");
            assert!((outside - 3.0).abs() < 0.05, "{outside}");
            assert!(body.topology.faces.iter().any(|f| f.fingerprint.kind == crate::topo::FaceKind::Plane && f.fingerprint.origin == bx as i64 && (f.fingerprint.centroid.z - 13.0).abs() < 1e-6 && f.fingerprint.dir.z > 0.99));
            let v = volume(&body.body).unwrap();
            assert!(31_200.0 - v > 600.0 && 31_200.0 - v < 1_600.0, "{v}");
            let off = &built.surf.get(&sp).unwrap().off_scan;
            assert!(off.first().is_none_or(|o| o.max < 2.0), "{off:?}");
        }

        #[test]
        fn refuses_within_on_a_sheet_and_for_a_tool_with_no_region() {
            let (scan, _) = scan_of();
            let (mut part, bx, band, sp) = part(None);
            part.remove(sp);
            let trim = part.add(FeatureKind::Split(SplitFeature { body: BodyId::new(band), tools: vec![BodyId::new(bx)], keep: SplitKeep::Largest, tolerance: 0.2, overlap: 0.5, within: Some(1.0), exprs: Default::default() }));
            let other = block(&mut part, 40.0, 40.0, 10.0, 10.0, 0.0, 5.0);
            if let Some(f) = part.feature_mut(other) {
                f.name = "Other".into();
            }
            let no_region = part.add(FeatureKind::Split(SplitFeature { body: BodyId::new(bx), tools: vec![BodyId::new(other)], keep: SplitKeep::Largest, tolerance: 0.2, overlap: 0.5, within: Some(1.0), exprs: Default::default() }));
            let built = build_scanned(&part, Some(scan));
            assert!(built.errors.get(&trim).is_some_and(|e| e.contains("within keeps a solid's cut")), "{:?}", built.errors);
            assert!(built.errors.get(&no_region).is_some_and(|e| e.starts_with("Other has no region to keep the cut within")), "{:?}", built.errors);
        }
    }
}
