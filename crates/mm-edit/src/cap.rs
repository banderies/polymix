//! Planar capping of boundary loops.
//!
//! Loops are projected onto a plane and classified by signed area into outer boundaries and
//! holes; each hole is assigned to the smallest outer loop containing it, so nested loops
//! (annular cross-sections, islands inside holes, ...) produce polygon-with-holes caps. Each
//! cap region is triangulated by ear clipping, appended with the reverse loop orientation
//! (so caps are consistently oriented with the surface and face outward), turned into a
//! constrained Delaunay triangulation by Lawson edge flips, and optionally refined towards
//! a target edge length (long-edge splits + flips + in-plane smoothing; loop vertices fixed).

use crate::cut::CutFill;
use crate::triangulate::triangulate;
use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::mesh::{tri_edge_index, tri_other_vertex};
use mm_core::{DVec2, DVec3, Mesh, INVALID};
use std::collections::HashMap;

/// Output of [`cap_planar_loops`].
#[derive(Clone, Debug, Default)]
pub struct CapResult {
    /// All cap triangles.
    pub triangles: Vec<u32>,
    /// Face group of each cap region (one region = an outer loop plus its holes).
    pub groups: Vec<i32>,
    /// Vertices inserted inside caps (remeshing, fan fallback).
    pub interior_vertices: Vec<u32>,
    /// Indices (into the input) of loops that could not be capped.
    pub failed_loops: Vec<usize>,
}

/// Cap (near-)planar boundary loops. Loops may be given in either orientation. Loops are
/// grouped by the plane they lie in; within a plane, nested loops are capped as polygons
/// with holes. `normal` is an optional hint for the projection direction (default: each
/// group's own Newell normal). The cap orientation is always the one consistent with the
/// surrounding surface.
pub fn cap_planar_loops(
    mesh: &mut Mesh,
    loops: &[Vec<u32>],
    normal: Option<DVec3>,
    fill: CutFill,
) -> EditResult<CapResult> {
    let mut oriented = Vec::with_capacity(loops.len());
    for l in loops {
        oriented.push(orient_boundary_loop(mesh, l)?);
    }
    if let CutFill::FlatRemeshed { target_edge_length } = fill {
        if !(target_edge_length.is_finite() && target_edge_length > 0.0) {
            return Err(EditError::InvalidParameter("target_edge_length must be positive".into()));
        }
    }
    let hint = normal.filter(|n| n.length() > 1e-300).map(|n| n.normalize());
    // (normal, centroid, size) of each loop's cap.
    let planes: Vec<(DVec3, DVec3, f64)> = oriented
        .iter()
        .map(|l| {
            let mut p = positions(mesh, l);
            p.reverse();
            let a = area_vector(&p);
            (hint.unwrap_or_else(|| a.normalize_or_zero()), mean(&p), a.length().sqrt())
        })
        .collect();
    let mut clusters: Vec<Vec<usize>> = Vec::new();
    for i in 0..oriented.len() {
        let (ni, ci, si) = planes[i];
        let found = clusters.iter().position(|cl| {
            let (nj, cj, sj) = planes[cl[0]];
            nj.dot(ni).abs() > 0.99 && (ci - cj).dot(nj).abs() <= 0.02 * si.max(sj)
        });
        match found {
            Some(k) => clusters[k].push(i),
            None => clusters.push(vec![i]),
        }
    }
    let mut res = CapResult::default();
    for cl in clusters {
        let big = *cl.iter().max_by(|&&a, &&b| planes[a].2.partial_cmp(&planes[b].2).unwrap()).unwrap();
        let ls: Vec<Vec<u32>> = cl.iter().map(|&i| oriented[i].clone()).collect();
        let r = cap_oriented_loops(mesh, &ls, Some(planes[big].0), fill);
        res.triangles.extend(r.triangles);
        res.groups.extend(r.groups);
        res.interior_vertices.extend(r.interior_vertices);
        res.failed_loops.extend(r.failed_loops.iter().map(|&k| cl[k]));
    }
    res.failed_loops.sort_unstable();
    Ok(res)
}

struct LoopInfo {
    /// Cap traversal order (reverse of boundary orientation).
    verts: Vec<u32>,
    pts: Vec<DVec2>,
    area: f64,
    /// Mean offset along the projection normal.
    offset: f64,
}

struct Region {
    outer: usize,
    holes: Vec<usize>,
    inverted: bool,
}

/// A cap region inserted into the mesh, with its own 2D frame in which its triangles are
/// counter-clockwise.
struct CapRegion {
    id: u32,
    tris: Vec<u32>,
    interior: Vec<u32>,
    o: DVec3,
    u: DVec3,
    v: DVec3,
}

impl CapRegion {
    #[inline]
    fn p2(&self, mesh: &Mesh, x: u32) -> DVec2 {
        let d = mesh.vertex(x) - self.o;
        DVec2::new(d.dot(self.u), d.dot(self.v))
    }
}

#[inline]
fn cross2(a: DVec2, b: DVec2, c: DVec2) -> f64 {
    (b - a).perp_dot(c - a)
}

/// `d` strictly inside the circumcircle of counter-clockwise (a, b, c), with tolerance.
fn in_circle(a: DVec2, b: DVec2, c: DVec2, d: DVec2, tol: f64) -> bool {
    let (ad, bd, cd) = (a - d, b - d, c - d);
    let det = ad.length_squared() * bd.perp_dot(cd) - bd.length_squared() * ad.perp_dot(cd)
        + cd.length_squared() * ad.perp_dot(bd);
    det > tol
}

/// Cap loops given in boundary orientation. Never fails as a whole; loops that cannot be
/// capped are reported in `failed_loops`.
pub(crate) fn cap_oriented_loops(
    mesh: &mut Mesh,
    loops: &[Vec<u32>],
    normal: Option<DVec3>,
    fill: CutFill,
) -> CapResult {
    let mut res = CapResult::default();
    if matches!(fill, CutFill::None) || loops.is_empty() {
        return res;
    }
    let all_pts: Vec<DVec3> = loops.iter().flat_map(|l| l.iter().map(|&v| mesh.vertex(v))).collect();
    let o = mean(&all_pts);
    let n = match normal.filter(|n| n.length() > 1e-300) {
        Some(n) => n.normalize(),
        None => {
            let mut s = DVec3::ZERO;
            let mut best = DVec3::ZERO;
            for l in loops {
                let mut r = positions(mesh, l);
                r.reverse();
                let a = area_vector(&r);
                s += a;
                if a.length() > best.length() {
                    best = a;
                }
            }
            if s.length() > 1e-12 * best.length() && s.length() > 1e-300 {
                s.normalize()
            } else if best.length() > 1e-300 {
                best.normalize()
            } else {
                DVec3::Z
            }
        }
    };
    let (u, v) = frame(n);

    let infos: Vec<LoopInfo> = loops
        .iter()
        .map(|l| {
            let verts: Vec<u32> = l.iter().rev().copied().collect();
            let pts: Vec<DVec2> = verts
                .iter()
                .map(|&x| {
                    let d = mesh.vertex(x) - o;
                    DVec2::new(d.dot(u), d.dot(v))
                })
                .collect();
            let (area, _) = polygon_area_centroid_2d(&pts);
            let offset = verts.iter().map(|&x| (mesh.vertex(x) - o).dot(n)).sum::<f64>() / verts.len() as f64;
            LoopInfo { verts, pts, area, offset }
        })
        .collect();
    // The largest loop of a planar group is always an outer boundary: pick the frame
    // handedness accordingly (robust to a wrongly signed normal hint).
    let mut infos = infos;
    let mut v = v;
    if let Some(big) = infos.iter().max_by(|a, b| a.area.abs().partial_cmp(&b.area.abs()).unwrap()) {
        if big.area < 0.0 {
            v = -v;
            for li in infos.iter_mut() {
                li.area = -li.area;
                li.pts.iter_mut().for_each(|p| p.y = -p.y);
            }
        }
    }

    // Classify and nest.
    let outers: Vec<usize> = (0..infos.len()).filter(|&i| infos[i].area >= 0.0).collect();
    let mut regions: Vec<Region> =
        outers.iter().map(|&i| Region { outer: i, holes: Vec::new(), inverted: false }).collect();
    for h in (0..infos.len()).filter(|&i| infos[i].area < 0.0) {
        let hp = &infos[h].pts;
        let samples: Vec<DVec2> = (0..hp.len().min(9)).map(|k| hp[k * hp.len() / hp.len().min(9)]).collect();
        let mut best: Option<(usize, f64)> = None;
        for (ri, &oi) in outers.iter().enumerate() {
            let a = infos[oi].area;
            if a <= infos[h].area.abs() * (1.0 - 1e-9) {
                continue;
            }
            // Only loops in the same plane can nest.
            if (infos[oi].offset - infos[h].offset).abs() > 0.02 * a.sqrt() {
                continue;
            }
            let inside = samples.iter().filter(|&&p| point_in_polygon(&infos[oi].pts, p)).count();
            if 2 * inside > samples.len() && best.is_none_or(|(_, ba)| a < ba) {
                best = Some((ri, a));
            }
        }
        match best {
            Some((ri, _)) => regions[ri].holes.push(h),
            None => regions.push(Region { outer: h, holes: Vec::new(), inverted: true }),
        }
    }

    let mut owner: Vec<u32> = Vec::new();
    let base_group = new_group(mesh);
    let mut inserted: Vec<CapRegion> = Vec::new();
    for (ri, reg) in regions.iter().enumerate() {
        let group = base_group + ri as i32;
        let (ru, rv) = if reg.inverted { (u, -v) } else { (u, v) };
        let flip = if reg.inverted { -1.0 } else { 1.0 };
        // Points de-duplicated by mesh vertex id.
        let mut index: HashMap<u32, usize> = HashMap::new();
        let mut vids: Vec<u32> = Vec::new();
        let mut pts: Vec<DVec2> = Vec::new();
        let mut ring = |li: usize, index: &mut HashMap<u32, usize>| -> Vec<usize> {
            infos[li]
                .verts
                .iter()
                .zip(&infos[li].pts)
                .map(|(&vid, &p)| {
                    *index.entry(vid).or_insert_with(|| {
                        vids.push(vid);
                        pts.push(DVec2::new(p.x, p.y * flip));
                        vids.len() - 1
                    })
                })
                .collect()
        };
        let outer = ring(reg.outer, &mut index);
        let holes: Vec<Vec<usize>> = reg.holes.iter().map(|&h| ring(h, &mut index)).collect();
        let region_loops: Vec<usize> = std::iter::once(reg.outer).chain(reg.holes.iter().copied()).collect();

        let tri = {
            let m: &Mesh = mesh;
            triangulate(&pts, &outer, &holes, &mut |i, j| m.find_edge(vids[i], vids[j]).is_none())
        };
        let mut added: Vec<u32> = Vec::with_capacity(tri.triangles.len());
        let mut ok = tri.unbridged_holes.is_empty();
        if ok {
            for t in &tri.triangles {
                match mesh.append_triangle([vids[t[0]], vids[t[1]], vids[t[2]]], group) {
                    Ok(id) => added.push(id),
                    Err(_) => {
                        ok = false;
                        break;
                    }
                }
            }
        }
        if ok && !locally_consistent(mesh, &added) {
            ok = false;
        }
        let mut interior = Vec::new();
        if !ok {
            rollback(mesh, &added);
            added.clear();
            if reg.holes.is_empty() {
                // Fallback: fan around a new centre vertex (never conflicts with existing edges).
                let lp = &infos[reg.outer].verts;
                let p3 = positions(mesh, lp);
                let (_, c2) = polygon_area_centroid_2d(&infos[reg.outer].pts);
                let off = p3.iter().map(|p| (*p - o).dot(n)).sum::<f64>() / p3.len() as f64;
                let c = mesh.append_vertex(o + u * c2.x + v * c2.y + n * off);
                let mut good = true;
                for i in 0..lp.len() {
                    match mesh.append_triangle([lp[i], lp[(i + 1) % lp.len()], c], group) {
                        Ok(id) => added.push(id),
                        Err(_) => {
                            good = false;
                            break;
                        }
                    }
                }
                if good && locally_consistent(mesh, &added) {
                    interior.push(c);
                } else {
                    rollback(mesh, &added);
                    added.clear();
                    if mesh.is_vertex(c) && mesh.vertex_valence(c) == 0 {
                        mesh.remove_isolated_vertices();
                    }
                }
            }
        }
        if added.is_empty() {
            res.failed_loops.extend(region_loops);
            continue;
        }
        let id = inserted.len() as u32 + 1;
        if owner.len() < mesh.max_tid() {
            owner.resize(mesh.max_tid(), 0);
        }
        for &t in &added {
            owner[t as usize] = id;
        }
        res.groups.push(group);
        inserted.push(CapRegion { id, tris: added, interior, o, u: ru, v: rv });
    }

    for reg in inserted.iter_mut() {
        delaunay_flips(mesh, reg, &owner, None);
        if let CutFill::FlatRemeshed { target_edge_length } = fill {
            if target_edge_length.is_finite() && target_edge_length > 0.0 {
                refine(mesh, reg, &mut owner, target_edge_length);
            }
        }
        res.triangles.extend_from_slice(&reg.tris);
        res.interior_vertices.extend_from_slice(&reg.interior);
    }
    res
}

#[inline]
fn in_region(owner: &[u32], id: u32, t: u32) -> bool {
    t != INVALID && (t as usize) < owner.len() && owner[t as usize] == id
}

/// Interior edges of a region (both triangles in the region), each listed once.
fn region_edges(mesh: &Mesh, reg: &CapRegion, owner: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    for &t in &reg.tris {
        for e in mesh.tri_edges(t) {
            let [t0, t1] = mesh.edge_triangles(e);
            if t1 != INVALID && t0.min(t1) == t && in_region(owner, reg.id, t0) && in_region(owner, reg.id, t1) {
                out.push(e);
            }
        }
    }
    out
}

/// Flip `e` if that makes the pair locally Delaunay (or repairs a degenerate triangle).
fn try_flip(mesh: &mut Mesh, reg: &CapRegion, owner: &[u32], e: u32) -> bool {
    if !mesh.is_edge(e) {
        return false;
    }
    let [t0, t1] = mesh.edge_triangles(e);
    if !in_region(owner, reg.id, t0) || !in_region(owner, reg.id, t1) {
        return false;
    }
    let [a, b] = mesh.edge_vertices(e);
    let tv0 = mesh.triangle(t0);
    let Some(j) = tri_edge_index(tv0, a, b) else { return false };
    let (p, q, c) = (tv0[j], tv0[(j + 1) % 3], tv0[(j + 2) % 3]);
    let d = tri_other_vertex(mesh.triangle(t1), a, b);
    if d == INVALID || c == d {
        return false;
    }
    let (pp, pq, pc, pd) = (reg.p2(mesh, p), reg.p2(mesh, q), reg.p2(mesh, c), reg.p2(mesh, d));
    let l2 = pp
        .distance_squared(pq)
        .max(pp.distance_squared(pc))
        .max(pq.distance_squared(pc))
        .max(pp.distance_squared(pd))
        .max(pq.distance_squared(pd));
    let eps = 1e-12 * l2;
    // Flipped triangles: (c, p, d) and (d, q, c).
    if cross2(pc, pp, pd) <= eps || cross2(pd, pq, pc) <= eps {
        return false;
    }
    let degenerate = cross2(pp, pq, pc) <= eps || cross2(pq, pp, pd) <= eps;
    if !degenerate && !in_circle(pp, pq, pc, pd, 1e-10 * l2 * l2) {
        return false;
    }
    mesh.flip_edge(e).is_ok()
}

/// Lawson flipping restricted to a cap region (loop edges are never flipped).
fn delaunay_flips(mesh: &mut Mesh, reg: &CapRegion, owner: &[u32], seed: Option<Vec<u32>>) {
    let mut stack = seed.unwrap_or_else(|| region_edges(mesh, reg, owner));
    let mut budget = 50 * stack.len() + 10_000;
    while let Some(e) = stack.pop() {
        if budget == 0 {
            break;
        }
        budget -= 1;
        if try_flip(mesh, reg, owner, e) {
            for t in mesh.edge_triangles(e) {
                for x in mesh.tri_edges(t) {
                    if x != e {
                        stack.push(x);
                    }
                }
            }
        }
    }
}

/// Split long interior edges (longest first) with local Delaunay legalisation after each
/// pass, then smooth and re-Delaunay the whole region.
fn refine(mesh: &mut Mesh, reg: &mut CapRegion, owner: &mut Vec<u32>, target: f64) {
    let max_len = target * 4.0 / 3.0;
    let max_tris = 20_000_000usize;
    for _pass in 0..64 {
        let mut long: Vec<(f64, u32)> = region_edges(mesh, reg, owner)
            .into_iter()
            .map(|e| (mesh.edge_length(e), e))
            .filter(|&(l, _)| l > max_len)
            .collect();
        if long.is_empty() || reg.tris.len() > max_tris {
            break;
        }
        long.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut seed: Vec<u32> = Vec::new();
        let mut splits = 0usize;
        for (_, e) in long {
            if !mesh.is_edge(e) || mesh.edge_length(e) <= max_len {
                continue;
            }
            let [t0, t1] = mesh.edge_triangles(e);
            if !in_region(owner, reg.id, t0) || !in_region(owner, reg.id, t1) {
                continue;
            }
            let Ok(info) = mesh.split_edge(e) else { continue };
            if owner.len() < mesh.max_tid() {
                owner.resize(mesh.max_tid(), 0);
            }
            for t in info.t_new {
                if t != INVALID {
                    owner[t as usize] = reg.id;
                    reg.tris.push(t);
                }
            }
            reg.interior.push(info.v_new);
            splits += 1;
            // Link edges of the new vertex are the only ones that may now be non-Delaunay.
            for t in info.t_orig.into_iter().chain(info.t_new) {
                if t != INVALID {
                    seed.extend(mesh.tri_edges(t));
                }
            }
        }
        delaunay_flips(mesh, reg, owner, Some(seed));
        // Stragglers re-lengthened by smoothing are left to the final relaxation.
        if splits * 200 < reg.tris.len() {
            break;
        }
        // Relaxing every pass keeps the vertex density near the target (no over-refinement);
        // passes grow geometrically, so the total cost stays a small multiple of the last.
        smooth_interior(mesh, reg, owner, 1);
        delaunay_flips(mesh, reg, owner, None);
    }
    for _ in 0..2 {
        smooth_interior(mesh, reg, owner, 2);
        delaunay_flips(mesh, reg, owner, None);
    }
}

/// Move interior cap vertices towards the centroid of their neighbours, rejecting moves
/// that would invert or degenerate an incident triangle.
fn smooth_interior(mesh: &mut Mesh, reg: &CapRegion, owner: &[u32], iters: usize) {
    for _ in 0..iters {
        for &x in &reg.interior {
            if !mesh.is_vertex(x) {
                continue;
            }
            let tris = mesh.vertex_triangles(x);
            if tris.is_empty() || tris.iter().any(|&t| !in_region(owner, reg.id, t)) {
                continue;
            }
            let mut sum = DVec3::ZERO;
            let edges = mesh.vertex_edges(x);
            for &e in edges {
                sum += mesh.vertex(mesh.edge_other_vertex(e, x));
            }
            let target = sum / edges.len() as f64;
            let newp = mesh.vertex(x).lerp(target, 0.7);
            let d = newp - reg.o;
            let np = DVec2::new(d.dot(reg.u), d.dot(reg.v));
            let ok = tris.iter().all(|&t| {
                let tv = mesh.triangle(t);
                let p = |y: u32| if y == x { np } else { reg.p2(mesh, y) };
                let (a, b, c) = (p(tv[0]), p(tv[1]), p(tv[2]));
                let l2 = a.distance_squared(b).max(b.distance_squared(c)).max(c.distance_squared(a));
                cross2(a, b, c) > 1e-6 * l2
            });
            if ok {
                mesh.set_vertex(x, newp);
            }
        }
    }
}
