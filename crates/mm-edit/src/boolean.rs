//! Exact mesh booleans (Meshmixer's "Precise" solution mode).
//!
//! Unlike the volumetric boolean in `mm-solid`, the input surfaces are kept exactly: only
//! the triangles crossed by the other mesh are split, everything else is copied.
//!
//! 1. Candidate triangle pairs come from a BVH. Each pair is decided with exact orientation
//!    predicates (`robust`), so all decisions agree with each other: an edge of one mesh
//!    either crosses a triangle of the other or it does not, and every crossing becomes one
//!    shared point. The pieces of the intersection curve therefore join into closed
//!    polylines that both meshes are cut along.
//! 2. Each crossed triangle is cut along its part of the curve into polygons (chains from
//!    edge to edge split it; closed loops inside it become holes), which are triangulated.
//! 3. The patches between curve edges are classified inside / outside the other mesh by
//!    its generalized winding number, and the patches the operation keeps are stitched
//!    together along the shared curve points.
//!
//! Degenerate configurations (coplanar faces, a vertex exactly on the other surface, an
//! edge passing exactly through an edge) show up as zero predicates. The second mesh is
//! then nudged by a tiny amount (10⁻⁹ of the size, more if needed) and the computation
//! repeated, so touching coplanar faces end up either barely overlapping or barely apart.

use crate::triangulate::triangulate;
use crate::{EditError, EditResult};
use mm_core::{Aabb, Bvh, DVec2, DVec3, Mesh};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

/// Boolean operation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BooleanOp {
    /// Everything inside either mesh.
    Union,
    /// The first mesh minus the second.
    Difference,
    /// Everything inside both meshes.
    Intersection,
}

/// Result of [`mesh_boolean`].
#[derive(Clone, Debug)]
pub struct BooleanResult {
    pub mesh: Mesh,
    /// Output vertices on the intersection curve.
    pub seam_vertices: Vec<u32>,
    /// Added to the second mesh's face groups so they follow the first mesh's.
    pub group_offset: i32,
    /// The second mesh was nudged to break a degenerate configuration.
    pub perturbed: bool,
}

/// A curve point: an edge of one mesh crossing a triangle of the other.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum PointKey {
    /// Edge of A, triangle of B.
    A(u32, u32),
    /// Edge of B, triangle of A.
    B(u32, u32),
}

enum Pair {
    Miss,
    Hit(u32, u32, [PointKey; 2]),
    Degenerate,
}

/// Exact-predicate failure: retry with the second mesh nudged.
struct Degenerate;

#[inline]
fn c3(p: DVec3) -> robust::Coord3D<f64> {
    robust::Coord3D { x: p.x, y: p.y, z: p.z }
}

#[inline]
fn orient(a: DVec3, b: DVec3, c: DVec3, d: DVec3) -> i8 {
    let o = robust::orient3d(c3(a), c3(b), c3(c), c3(d));
    if o > 0.0 {
        1
    } else if o < 0.0 {
        -1
    } else {
        0
    }
}

/// Does segment `pq` (whose ends are strictly on opposite sides of the triangle's plane)
/// cross triangle `t`? `None` when it passes exactly through an edge or corner.
fn crosses(p: DVec3, q: DVec3, t: [DVec3; 3]) -> Option<bool> {
    let o = [orient(p, q, t[0], t[1]), orient(p, q, t[1], t[2]), orient(p, q, t[2], t[0])];
    let pos = o.iter().any(|&s| s > 0);
    let neg = o.iter().any(|&s| s < 0);
    if pos && neg {
        Some(false)
    } else if o.contains(&0) {
        None
    } else {
        Some(true)
    }
}

/// Decide triangle pair (`ta` of A, `tb` of B) exactly.
fn pair(pa: [DVec3; 3], ea: [u32; 3], ta: u32, pb: [DVec3; 3], eb: [u32; 3], tb: u32) -> Pair {
    let sa = pa.map(|p| orient(pb[0], pb[1], pb[2], p));
    if sa.iter().all(|&s| s > 0) || sa.iter().all(|&s| s < 0) {
        return Pair::Miss;
    }
    let sb = pb.map(|q| orient(pa[0], pa[1], pa[2], q));
    if sb.iter().all(|&s| s > 0) || sb.iter().all(|&s| s < 0) {
        return Pair::Miss;
    }
    if sa.contains(&0) || sb.contains(&0) {
        return Pair::Degenerate;
    }
    let mut keys: SmallVec<[PointKey; 4]> = SmallVec::new();
    for j in 0..3 {
        let k = (j + 1) % 3;
        if sa[j] != sa[k] {
            match crosses(pa[j], pa[k], pb) {
                None => return Pair::Degenerate,
                Some(true) => keys.push(PointKey::A(ea[j], tb)),
                Some(false) => {}
            }
        }
        if sb[j] != sb[k] {
            match crosses(pb[j], pb[k], pa) {
                None => return Pair::Degenerate,
                Some(true) => keys.push(PointKey::B(eb[j], ta)),
                Some(false) => {}
            }
        }
    }
    match keys.len() {
        0 => Pair::Miss,
        2 => Pair::Hit(ta, tb, [keys[0], keys[1]]),
        // Impossible with exact predicates in general position.
        _ => Pair::Degenerate,
    }
}

/// Where segment `pq` meets the plane of `t`, and the parameter along `pq`.
fn edge_plane_point(p: DVec3, q: DVec3, t: [DVec3; 3]) -> (DVec3, f64) {
    let n = (t[1] - t[0]).cross(t[2] - t[0]);
    let dp = n.dot(p - t[0]);
    let dq = n.dot(q - t[0]);
    let s = if dp == dq { 0.5 } else { (dp / (dp - dq)).clamp(0.0, 1.0) };
    (p + (q - p) * s, s)
}

/// One input mesh as seen by the boolean.
struct Operand<'a> {
    mesh: &'a Mesh,
    /// Positions used for the predicates (B is nudged).
    pos: Vec<DVec3>,
    /// Offset of this mesh's vertex ids in the global vertex table.
    base: u32,
}

impl Operand<'_> {
    fn tri_pos(&self, t: u32) -> [DVec3; 3] {
        self.mesh.triangle(t).map(|v| self.pos[v as usize])
    }
    fn tri_edges(&self, t: u32) -> [u32; 3] {
        let tv = self.mesh.triangle(t);
        [0, 1, 2].map(|j| self.mesh.find_edge(tv[j], tv[(j + 1) % 3]).expect("triangle edge"))
    }
}

/// A crossed triangle and its part of the curve.
#[derive(Default)]
struct TriCut {
    /// Points on each edge `j` (corner j → j+1), as (parameter from corner j, global id).
    edge_pts: [Vec<(f64, u32)>; 3],
    segs: Vec<[u32; 2]>,
}

/// Triangulate triangle `t` of `op` cut along `cut`. Output triangles use global ids and
/// keep the triangle's orientation.
fn split_triangle(op: &Operand, t: u32, cut: &mut TriCut, pos: &[DVec3]) -> Result<Vec<[u32; 3]>, Degenerate> {
    let corners = op.mesh.triangle(t).map(|v| v + op.base);
    let mut cycle: Vec<u32> = Vec::new();
    let mut on_edge: HashSet<u32> = HashSet::new();
    for (corner, pts) in corners.iter().zip(cut.edge_pts.iter_mut()) {
        cycle.push(*corner);
        pts.sort_by(|a, b| a.0.total_cmp(&b.0));
        for &(_, p) in pts.iter() {
            cycle.push(p);
            on_edge.insert(p);
        }
    }
    let mut adj: HashMap<u32, SmallVec<[u32; 2]>> = HashMap::new();
    for &[p, q] in &cut.segs {
        adj.entry(p).or_default().push(q);
        adj.entry(q).or_default().push(p);
    }
    for (p, nb) in &adj {
        if nb.len() != if on_edge.contains(p) { 1 } else { 2 } {
            return Err(Degenerate);
        }
    }
    // Chains from edge to edge, then closed loops inside the triangle.
    let mut seen: HashSet<u32> = HashSet::new();
    let mut chains: Vec<Vec<u32>> = Vec::new();
    let walk = |start: u32, seen: &mut HashSet<u32>| -> Vec<u32> {
        let mut path = vec![start];
        seen.insert(start);
        let mut prev = u32::MAX;
        let mut cur = start;
        while let Some(&next) = adj[&cur].iter().find(|&&n| n != prev) {
            if next == start || path.len() > adj.len() {
                break;
            }
            path.push(next);
            seen.insert(next);
            if on_edge.contains(&next) {
                break;
            }
            prev = cur;
            cur = next;
        }
        path
    };
    for &p in &cycle {
        if on_edge.contains(&p) && !seen.contains(&p) && adj.contains_key(&p) {
            chains.push(walk(p, &mut seen));
        }
    }
    let mut loops: Vec<Vec<u32>> = Vec::new();
    let mut interior: Vec<u32> = adj.keys().copied().filter(|p| !seen.contains(p)).collect();
    interior.sort_unstable();
    for p in interior {
        if !seen.contains(&p) {
            loops.push(walk(p, &mut seen));
        }
    }
    // Local 2D frame in which the triangle is counter-clockwise.
    let [c0, c1, c2] = corners.map(|v| pos[v as usize]);
    let e1 = (c1 - c0).normalize_or_zero();
    let e2 = (c1 - c0).cross(c2 - c0).cross(e1).normalize_or_zero();
    let flat = |v: u32| {
        let d = pos[v as usize] - c0;
        DVec2::new(d.dot(e1), d.dot(e2))
    };
    struct Piece {
        outer: Vec<u32>,
        holes: Vec<Vec<u32>>,
    }
    let mut pieces = vec![Piece { outer: cycle, holes: Vec::new() }];
    for ch in &chains {
        let (p, q) = (ch[0], *ch.last().unwrap());
        if ch.len() < 2 || !on_edge.contains(&q) {
            return Err(Degenerate);
        }
        let k = pieces.iter().position(|pc| pc.outer.contains(&p) && pc.outer.contains(&q)).ok_or(Degenerate)?;
        let outer = std::mem::take(&mut pieces[k].outer);
        let n = outer.len();
        let i = outer.iter().position(|&x| x == p).unwrap();
        let j = outer.iter().position(|&x| x == q).unwrap();
        let arc = |from: usize, to: usize| {
            let mut v = Vec::new();
            let mut x = from;
            loop {
                v.push(outer[x]);
                if x == to {
                    break;
                }
                x = (x + 1) % n;
            }
            v
        };
        let inner = &ch[1..ch.len() - 1];
        let mut first = arc(i, j);
        first.extend(inner.iter().rev());
        let mut second = arc(j, i);
        second.extend(inner.iter());
        if first.len() < 3 || second.len() < 3 {
            return Err(Degenerate);
        }
        pieces[k].outer = first;
        pieces.push(Piece { outer: second, holes: Vec::new() });
    }
    let area = |ring: &[u32]| {
        let pts: Vec<DVec2> = ring.iter().map(|&v| flat(v)).collect();
        (0..pts.len()).map(|i| pts[i].perp_dot(pts[(i + 1) % pts.len()])).sum::<f64>() * 0.5
    };
    let inside = |ring: &[u32], p: DVec2| {
        let pts: Vec<DVec2> = ring.iter().map(|&v| flat(v)).collect();
        let mut c = false;
        for i in 0..pts.len() {
            let (a, b) = (pts[i], pts[(i + 1) % pts.len()]);
            if (a.y > p.y) != (b.y > p.y) && p.x < a.x + (p.y - a.y) / (b.y - a.y) * (b.x - a.x) {
                c = !c;
            }
        }
        c
    };
    loops.sort_by(|a, b| area(b).abs().total_cmp(&area(a).abs()));
    for mut lp in loops {
        if lp.len() < 3 {
            return Err(Degenerate);
        }
        if area(&lp) < 0.0 {
            lp.reverse();
        }
        let probe = flat(lp[0]);
        let k = pieces
            .iter()
            .position(|pc| inside(&pc.outer, probe) && !pc.holes.iter().any(|h| inside(h, probe)))
            .ok_or(Degenerate)?;
        pieces[k].holes.push(lp.iter().rev().copied().collect());
        pieces.push(Piece { outer: lp, holes: Vec::new() });
    }
    // Triangulate each piece; diagonals may not repeat an edge already used in this triangle.
    let mut ids: Vec<u32> = Vec::new();
    let mut local: HashMap<u32, usize> = HashMap::new();
    let mut idx = |v: u32, ids: &mut Vec<u32>| {
        *local.entry(v).or_insert_with(|| {
            ids.push(v);
            ids.len() - 1
        })
    };
    let key = |a: u32, b: u32| (a.min(b), a.max(b));
    let mut used: HashSet<(u32, u32)> = HashSet::new();
    let mut rings: Vec<(Vec<usize>, Vec<Vec<usize>>)> = Vec::new();
    for pc in &pieces {
        for ring in std::iter::once(&pc.outer).chain(pc.holes.iter()) {
            for i in 0..ring.len() {
                used.insert(key(ring[i], ring[(i + 1) % ring.len()]));
            }
        }
        let outer: Vec<usize> = pc.outer.iter().map(|&v| idx(v, &mut ids)).collect();
        let holes: Vec<Vec<usize>> = pc.holes.iter().map(|h| h.iter().map(|&v| idx(v, &mut ids)).collect()).collect();
        rings.push((outer, holes));
    }
    let pts: Vec<DVec2> = ids.iter().map(|&v| flat(v)).collect();
    let mut out = Vec::new();
    for (outer, holes) in &rings {
        let mut diag_ok = |i: usize, j: usize| !used.contains(&key(ids[i], ids[j]));
        let tri = triangulate(&pts, outer, holes, &mut diag_ok);
        if !tri.unbridged_holes.is_empty() {
            return Err(Degenerate);
        }
        for [i, j, k] in tri.triangles {
            let g = [ids[i], ids[j], ids[k]];
            for e in 0..3 {
                used.insert(key(g[e], g[(e + 1) % 3]));
            }
            out.push(g);
        }
    }
    Ok(out)
}

/// The cut-up surface of one operand: every triangle (global ids) with its source group.
struct Pieces {
    tris: Vec<[u32; 3]>,
    groups: Vec<i32>,
}

fn cut_operand(op: &Operand, cuts: HashMap<u32, TriCut>, pos: &[DVec3], group_offset: i32) -> Result<Pieces, Degenerate> {
    let mut cuts: Vec<(u32, TriCut)> = cuts.into_iter().collect();
    cuts.sort_unstable_by_key(|c| c.0);
    let split: Vec<Result<Vec<[u32; 3]>, Degenerate>> = cuts.par_iter_mut().map(|(t, c)| split_triangle(op, *t, c, pos)).collect();
    let cut_set: HashSet<u32> = cuts.iter().map(|c| c.0).collect();
    let mut tris = Vec::with_capacity(op.mesh.triangle_count() + 4 * cuts.len());
    let mut groups = Vec::with_capacity(tris.capacity());
    for t in op.mesh.triangle_ids() {
        if !cut_set.contains(&t) {
            tris.push(op.mesh.triangle(t).map(|v| v + op.base));
            groups.push(op.mesh.tri_group(t) + group_offset);
        }
    }
    for ((t, _), r) in cuts.iter().zip(split) {
        let g = op.mesh.tri_group(*t) + group_offset;
        for tri in r? {
            tris.push(tri);
            groups.push(g);
        }
    }
    Ok(Pieces { tris, groups })
}

/// Inside flag per triangle of `side`: patches between seam edges are classified as a whole
/// by the other mesh's winding number at the centroids of their largest triangles.
fn classify(side: &Pieces, seam: &HashSet<(u32, u32)>, pos: &[DVec3], other: &mm_solid::WindingNumber) -> Vec<bool> {
    let n = side.tris.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let mut first: HashMap<(u32, u32), u32> = HashMap::with_capacity(n * 3 / 2);
    for (i, t) in side.tris.iter().enumerate() {
        for j in 0..3 {
            let (a, b) = (t[j], t[(j + 1) % 3]);
            let k = (a.min(b), a.max(b));
            if seam.contains(&k) {
                continue;
            }
            match first.get(&k) {
                Some(&o) => {
                    let (ra, rb) = (find(&mut parent, i as u32), find(&mut parent, o));
                    if ra != rb {
                        parent[ra as usize] = rb;
                    }
                }
                None => {
                    first.insert(k, i as u32);
                }
            }
        }
    }
    let area = |t: &[u32; 3]| {
        let [a, b, c] = t.map(|v| pos[v as usize]);
        (b - a).cross(c - a).length()
    };
    // Up to three largest triangles per patch vote.
    let mut best: HashMap<u32, SmallVec<[(f64, usize); 3]>> = HashMap::new();
    for i in 0..n {
        let r = find(&mut parent, i as u32);
        let e = best.entry(r).or_default();
        let a = area(&side.tris[i]);
        if e.len() < 3 {
            e.push((a, i));
        } else if let Some(m) = e.iter_mut().min_by(|x, y| x.0.total_cmp(&y.0)).filter(|m| m.0 < a) {
            *m = (a, i);
        }
    }
    let verdict: HashMap<u32, bool> = best
        .into_par_iter()
        .map(|(r, votes)| {
            let ins = votes
                .iter()
                .filter(|&&(_, i)| {
                    let c = side.tris[i].iter().map(|&v| pos[v as usize]).sum::<DVec3>() / 3.0;
                    other.eval(c).abs() > 0.5
                })
                .count();
            (r, 2 * ins > votes.len())
        })
        .collect();
    (0..n).map(|i| verdict[&find(&mut parent, i as u32)]).collect()
}

fn attempt(a: &Mesh, b: &Mesh, op: BooleanOp, shift: DVec3) -> Result<BooleanResult, Degenerate> {
    let na = a.max_vid() as u32;
    let oa = Operand { mesh: a, pos: (0..a.max_vid() as u32).map(|v| if a.is_vertex(v) { a.vertex(v) } else { DVec3::ZERO }).collect(), base: 0 };
    let ob = Operand {
        mesh: b,
        pos: (0..b.max_vid() as u32).map(|v| if b.is_vertex(v) { b.vertex(v) + shift } else { DVec3::ZERO }).collect(),
        base: na,
    };
    // Pairs: walk the smaller mesh's triangles against the larger mesh's BVH.
    let (small, large, small_is_a) = if a.triangle_count() <= b.triangle_count() { (&oa, &ob, true) } else { (&ob, &oa, false) };
    let large_bvh = Bvh::build(large.mesh);
    let large_shift = if small_is_a { shift } else { DVec3::ZERO };
    let pad = 1e-9 * (a.bounds().union(&b.bounds()).diagonal() + shift.length());
    let small_tris: Vec<u32> = small.mesh.triangle_ids().collect();
    let found: Vec<Pair> = small_tris
        .par_iter()
        .flat_map_iter(|&ts| {
            let ps = small.tri_pos(ts);
            let es = small.tri_edges(ts);
            let mut bb = Aabb::from_points(ps.map(|p| p - large_shift));
            bb.min -= DVec3::splat(pad);
            bb.max += DVec3::splat(pad);
            let mut out = Vec::new();
            large_bvh.query_aabb(&bb, |tl| {
                let pl = large.tri_pos(tl);
                let el = large.tri_edges(tl);
                let r = if small_is_a { pair(ps, es, ts, pl, el, tl) } else { pair(pl, el, tl, ps, es, ts) };
                if !matches!(r, Pair::Miss) {
                    out.push(r);
                }
            });
            out
        })
        .collect();
    if found.iter().any(|p| matches!(p, Pair::Degenerate)) {
        return Err(Degenerate);
    }
    // Curve points.
    let mut pos: Vec<DVec3> = oa.pos.clone();
    pos.resize((na + b.max_vid() as u32) as usize, DVec3::ZERO);
    // The nudged positions of B are used throughout so the seam matches exactly.
    for v in b.vertex_ids() {
        pos[(na + v) as usize] = ob.pos[v as usize];
    }
    let first_point = pos.len() as u32;
    let mut point_id: HashMap<PointKey, u32> = HashMap::new();
    let mut cuts_a: HashMap<u32, TriCut> = HashMap::new();
    let mut cuts_b: HashMap<u32, TriCut> = HashMap::new();
    let mut a_edge_pts: HashMap<u32, Vec<(f64, u32)>> = HashMap::new();
    let mut b_edge_pts: HashMap<u32, Vec<(f64, u32)>> = HashMap::new();
    let mut seam: HashSet<(u32, u32)> = HashSet::new();
    let mut hits: Vec<(u32, u32, [PointKey; 2])> = found
        .into_iter()
        .filter_map(|p| match p {
            Pair::Hit(ta, tb, k) => Some((ta, tb, k)),
            _ => None,
        })
        .collect();
    hits.sort_unstable();
    for &(ta, tb, keys) in &hits {
        let mut ids = [0u32; 2];
        for (slot, &k) in ids.iter_mut().zip(&keys) {
            *slot = *point_id.entry(k).or_insert_with(|| {
                let id = pos.len() as u32;
                let (p, e, s, list) = match k {
                    PointKey::A(e, tb) => {
                        let [u, v] = a.edge_vertices(e);
                        let (p, s) = edge_plane_point(oa.pos[u as usize], oa.pos[v as usize], ob.tri_pos(tb));
                        (p, e, s, &mut a_edge_pts)
                    }
                    PointKey::B(e, ta) => {
                        let [u, v] = b.edge_vertices(e);
                        let (p, s) = edge_plane_point(ob.pos[u as usize], ob.pos[v as usize], oa.tri_pos(ta));
                        (p, e, s, &mut b_edge_pts)
                    }
                };
                list.entry(e).or_default().push((s, id));
                pos.push(p);
                id
            });
        }
        cuts_a.entry(ta).or_default().segs.push(ids);
        cuts_b.entry(tb).or_default().segs.push(ids);
        seam.insert((ids[0].min(ids[1]), ids[0].max(ids[1])));
    }
    let seam_points: Vec<u32> = (first_point..pos.len() as u32).collect();
    // Edge points per crossed triangle, measured from each triangle edge's first corner.
    let fill_edges = |opd: &Operand, cuts: &mut HashMap<u32, TriCut>, edge_pts: &HashMap<u32, Vec<(f64, u32)>>| {
        for (&t, c) in cuts.iter_mut() {
            let tv = opd.mesh.triangle(t);
            for (j, e) in opd.tri_edges(t).into_iter().enumerate() {
                if let Some(list) = edge_pts.get(&e) {
                    let forward = opd.mesh.edge_vertices(e)[0] == tv[j];
                    c.edge_pts[j] = list.iter().map(|&(s, id)| (if forward { s } else { 1.0 - s }, id)).collect();
                }
            }
        }
    };
    fill_edges(&oa, &mut cuts_a, &a_edge_pts);
    fill_edges(&ob, &mut cuts_b, &b_edge_pts);
    let group_offset = a.max_group_id() + 1 - b.triangle_ids().map(|t| b.tri_group(t)).min().unwrap_or(0);
    let pieces_a = cut_operand(&oa, cuts_a, &pos, 0)?;
    let pieces_b = cut_operand(&ob, cuts_b, &pos, group_offset)?;
    let wa = mm_solid::WindingNumber::new(a);
    let wb = mm_solid::WindingNumber::new(b);
    let in_b = classify(&pieces_a, &seam, &pos, &wb);
    let in_a = classify(&pieces_b, &seam, &pos, &wa);
    let (keep_a_inside, keep_b_inside, flip_b) = match op {
        BooleanOp::Union => (false, false, false),
        BooleanOp::Intersection => (true, true, false),
        BooleanOp::Difference => (false, true, true),
    };
    let mut tris: Vec<[u32; 3]> = Vec::new();
    let mut groups: Vec<i32> = Vec::new();
    for (i, t) in pieces_a.tris.iter().enumerate() {
        if in_b[i] == keep_a_inside {
            tris.push(*t);
            groups.push(pieces_a.groups[i]);
        }
    }
    for (i, t) in pieces_b.tris.iter().enumerate() {
        if in_a[i] == keep_b_inside {
            tris.push(if flip_b { [t[0], t[2], t[1]] } else { *t });
            groups.push(pieces_b.groups[i]);
        }
    }
    // Compact the vertex table.
    let mut remap = vec![u32::MAX; pos.len()];
    let mut out_pos = Vec::new();
    for t in tris.iter_mut() {
        for v in t.iter_mut() {
            if remap[*v as usize] == u32::MAX {
                remap[*v as usize] = out_pos.len() as u32;
                out_pos.push(pos[*v as usize]);
            }
            *v = remap[*v as usize];
        }
    }
    let (mesh, report) = Mesh::from_triangles(&out_pos, &tris, Some(&groups));
    if report.nonmanifold_split > 0 || !mesh.is_closed() {
        return Err(Degenerate);
    }
    let seam_vertices = seam_points.iter().map(|&p| remap[p as usize]).filter(|&v| v != u32::MAX).collect();
    Ok(BooleanResult { mesh, seam_vertices, group_offset, perturbed: shift != DVec3::ZERO })
}

/// Exact boolean of two closed meshes. The result keeps every input triangle that is not
/// crossed by the other mesh; the second mesh's face groups are renumbered after the
/// first's. Fails when an input is open, or when the meshes self-intersect in a way the
/// curve cannot be traced through (the volumetric boolean handles such input).
pub fn mesh_boolean(a: &Mesh, b: &Mesh, op: BooleanOp) -> EditResult<BooleanResult> {
    for (m, which) in [(a, "first"), (b, "second")] {
        if !m.is_closed() {
            return Err(EditError::Failed(format!("the {which} mesh has open boundaries; the precise boolean needs closed meshes")));
        }
    }
    let size = a.bounds().union(&b.bounds()).diagonal().max(1e-300);
    // A direction unlikely to line up with any modelled feature.
    let dir = DVec3::new(0.5773, 0.6181, 0.5331).normalize();
    for scale in [0.0, 1e-9, 1e-8, 1e-7, 1e-6] {
        if let Ok(r) = attempt(a, b, op, dir * scale * size) {
            return Ok(r);
        }
    }
    Err(EditError::Failed("the meshes intersect in a way the precise boolean cannot resolve (self-intersections?); try Fast Approximate".into()))
}
