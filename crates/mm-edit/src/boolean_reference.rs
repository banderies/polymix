//! The boolean as it was before its search structures were shared between attempts (test
//! only): a BVH over the whole larger mesh, rebuilt with the winding numbers in every
//! attempt, and patches found by hashing every triangle edge. The tests check that
//! [`super::mesh_boolean`] gives exactly the same results.

use super::{edge_plane_point, pair, split_triangle, BooleanOp, BooleanResult, Degenerate, Operand, Pair, PointKey, TriCut};
use crate::{EditError, EditResult};
use mm_core::{Aabb, Bvh, DVec3, Mesh};
use rayon::prelude::*;
use smallvec::SmallVec;
use std::collections::{HashMap, HashSet};

fn tri_edges(op: &Operand, t: u32) -> [u32; 3] {
    let tv = op.mesh.triangle(t);
    [0, 1, 2].map(|j| op.mesh.find_edge(tv[j], tv[(j + 1) % 3]).expect("triangle edge"))
}

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
    let (small, large, small_is_a) = if a.triangle_count() <= b.triangle_count() { (&oa, &ob, true) } else { (&ob, &oa, false) };
    let large_bvh = Bvh::build(large.mesh);
    let large_shift = if small_is_a { shift } else { DVec3::ZERO };
    let pad = 1e-9 * (a.bounds().union(&b.bounds()).diagonal() + shift.length());
    let small_tris: Vec<u32> = small.mesh.triangle_ids().collect();
    let found: Vec<Pair> = small_tris
        .par_iter()
        .flat_map_iter(|&ts| {
            let ps = small.tri_pos(ts);
            let es = tri_edges(small, ts);
            let mut bb = Aabb::from_points(ps.map(|p| p - large_shift));
            bb.min -= DVec3::splat(pad);
            bb.max += DVec3::splat(pad);
            let mut out = Vec::new();
            large_bvh.query_aabb(&bb, |tl| {
                let pl = large.tri_pos(tl);
                let el = tri_edges(large, tl);
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
    let mut pos: Vec<DVec3> = oa.pos.clone();
    pos.resize((na + b.max_vid() as u32) as usize, DVec3::ZERO);
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
    let fill_edges = |opd: &Operand, cuts: &mut HashMap<u32, TriCut>, edge_pts: &HashMap<u32, Vec<(f64, u32)>>| {
        for (&t, c) in cuts.iter_mut() {
            let tv = opd.mesh.triangle(t);
            for (j, e) in tri_edges(opd, t).into_iter().enumerate() {
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

pub(super) fn mesh_boolean(a: &Mesh, b: &Mesh, op: BooleanOp) -> EditResult<BooleanResult> {
    for (m, which) in [(a, "first"), (b, "second")] {
        if !m.is_closed() {
            return Err(EditError::Failed(format!("the {which} mesh has open boundaries; the precise boolean needs closed meshes")));
        }
    }
    let size = a.bounds().union(&b.bounds()).diagonal().max(1e-300);
    let dir = DVec3::new(0.5773, 0.6181, 0.5331).normalize();
    for scale in [0.0, 1e-9, 1e-8, 1e-7, 1e-6] {
        if let Ok(r) = attempt(a, b, op, dir * scale * size) {
            return Ok(r);
        }
    }
    Err(EditError::Failed("the meshes intersect in a way the precise boolean cannot resolve (self-intersections?); try Fast Approximate".into()))
}
