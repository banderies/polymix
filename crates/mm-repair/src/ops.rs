//! Individual repair operations used by the Inspector: welding coincident boundary
//! vertices (zipping cracks), removing degenerate triangles, splitting bowtie vertices and
//! removing small floating components.
//!
//! Every operation keeps the mesh valid at all times; edits that would create a
//! non-manifold edge are rolled back instead.

use crate::util::{FxHashMap, LocalIndex};
use glam::DVec3;
use mm_core::mesh::Mesh;
use mm_core::topology::{self, EdgeLoop};
use mm_core::INVALID;
use rayon::prelude::*;
use smallvec::SmallVec;

type SavedTris = SmallVec<[([u32; 3], i32); 12]>;

/// Replace vertex `from` by `to` in all triangles of `from`. Rolls back and returns false
/// if any re-pointed triangle cannot be added (non-manifold edge, duplicate or degenerate).
/// On success `from` is left isolated (not freed).
fn try_repoint_vertex(mesh: &mut Mesh, from: u32, to: u32, tris: &[u32]) -> bool {
    let saved: SavedTris = tris.iter().map(|&t| (mesh.triangle(t), mesh.tri_group(t))).collect();
    for &t in tris {
        let _ = mesh.remove_triangle(t, false);
    }
    let mut added: SmallVec<[u32; 12]> = SmallVec::new();
    for &(tv, g) in &saved {
        let nt = tv.map(|v| if v == from { to } else { v });
        match mesh.append_triangle(nt, g) {
            Ok(t) => added.push(t),
            Err(_) => {
                for &t in &added {
                    let _ = mesh.remove_triangle(t, false);
                }
                for &(tv, g) in &saved {
                    let r = mesh.append_triangle(tv, g);
                    debug_assert!(r.is_ok(), "restoring removed triangles cannot fail");
                }
                return false;
            }
        }
    }
    true
}

/// Result of [`weld_boundary_vertices`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WeldReport {
    /// Vertices merged into a coincident partner.
    pub welded: usize,
    /// Coincident pairs that could not be merged without breaking manifoldness.
    pub rejected: usize,
}

/// Merge boundary vertices that lie within `tolerance` of each other (zips cracks and the
/// duplicated seams typical of STL exports). Merges that would create a non-manifold edge
/// are skipped. Vertices left without triangles are removed.
pub fn weld_boundary_vertices(mesh: &mut Mesh, tolerance: f64) -> WeldReport {
    weld_boundary_vertices_in(mesh, tolerance, &|_| true)
}

/// [`weld_boundary_vertices`] restricted to vertices accepted by `only`.
fn weld_boundary_vertices_in(mesh: &mut Mesh, tolerance: f64, only: &(dyn Fn(u32) -> bool + Sync)) -> WeldReport {
    let mut report = WeldReport::default();
    let bverts: Vec<u32> = {
        let m: &Mesh = mesh;
        let ids: Vec<u32> = m.vertex_ids().collect();
        ids.into_par_iter().filter(|&v| only(v) && m.is_boundary_vertex(v)).collect()
    };
    if bverts.len() < 2 {
        return report;
    }
    let tol = tolerance.max(0.0);
    let cell = if tol > 0.0 { tol } else { 1.0 };
    let key = |p: DVec3| -> (i64, i64, i64) {
        if tol > 0.0 {
            ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64)
        } else {
            (p.x.to_bits() as i64, p.y.to_bits() as i64, p.z.to_bits() as i64)
        }
    };
    let mut grid: FxHashMap<(i64, i64, i64), SmallVec<[u32; 2]>> = FxHashMap::default();
    grid.reserve(bverts.len());
    for (i, &v) in bverts.iter().enumerate() {
        grid.entry(key(mesh.vertex(v))).or_default().push(i as u32);
    }
    // Union-find over boundary vertices.
    let mut parent: Vec<u32> = (0..bverts.len() as u32).collect();
    fn find(p: &mut [u32], mut x: u32) -> u32 {
        while p[x as usize] != x {
            p[x as usize] = p[p[x as usize] as usize];
            x = p[x as usize];
        }
        x
    }
    let tol2 = tol * tol;
    for (i, &v) in bverts.iter().enumerate() {
        let p = mesh.vertex(v);
        let k = key(p);
        let range: &[i64] = if tol > 0.0 { &[-1, 0, 1] } else { &[0] };
        for &dx in range {
            for &dy in range {
                for &dz in range {
                    if let Some(c) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        for &j in c {
                            if j as usize <= i {
                                continue;
                            }
                            if mesh.vertex(bverts[j as usize]).distance_squared(p) <= tol2 {
                                let (a, b) = (find(&mut parent, i as u32), find(&mut parent, j));
                                if a != b {
                                    parent[a.max(b) as usize] = a.min(b);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    let mut clusters: FxHashMap<u32, SmallVec<[u32; 4]>> = FxHashMap::default();
    for i in 0..bverts.len() as u32 {
        let r = find(&mut parent, i);
        if r != i {
            clusters.entry(r).or_default().push(i);
        }
    }
    let mut roots: Vec<u32> = clusters.keys().copied().collect();
    roots.sort_unstable();
    let mut any = false;
    for r in roots {
        let keep = bverts[r as usize];
        for &i in &clusters[&r] {
            let v = bverts[i as usize];
            if !mesh.is_vertex(v) || !mesh.is_vertex(keep) {
                continue;
            }
            let tris = mesh.vertex_triangles(v);
            if try_repoint_vertex(mesh, v, keep, &tris) {
                report.welded += 1;
                any = true;
            } else {
                report.rejected += 1;
            }
        }
    }
    if any {
        mesh.remove_isolated_vertices();
    }
    report
}

/// Result of [`close_cracks`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CloseCracksReport {
    /// Boundary edges split where another boundary vertex lay on them (T-junctions).
    pub splits: usize,
    /// Vertex merges.
    pub weld: WeldReport,
}

/// Close cracks (Meshmixer *Edit > Close Cracks*): open boundaries lying on top of each
/// other within `tolerance` are zipped together. Boundary edges that have another boundary
/// vertex on them are split first (T-junctions, e.g. where a fine and a coarse patch meet),
/// then coincident boundary vertices are welded (see [`weld_boundary_vertices`]).
pub fn close_cracks(mesh: &mut Mesh, tolerance: f64) -> CloseCracksReport {
    close_cracks_in(mesh, tolerance, None)
}

/// [`close_cracks`] restricted to the given vertices (Meshmixer *Weld Boundaries* on a
/// selection): only boundary vertices in `only` are welded or used to split edges, and only
/// boundary edges with both ends in `only` are split.
pub fn close_cracks_in(mesh: &mut Mesh, tolerance: f64, only: Option<&[u32]>) -> CloseCracksReport {
    let mask: Option<Vec<bool>> = only.map(|vs| {
        let mut m = vec![false; mesh.max_vid()];
        for &v in vs {
            if (v as usize) < m.len() {
                m[v as usize] = true;
            }
        }
        m
    });
    // Vertices created by splits are always allowed (they lie on allowed edges).
    let n0 = mesh.max_vid();
    let allowed = |v: u32| mask.as_ref().is_none_or(|m| v as usize >= n0 || m[v as usize]);
    let tol = tolerance.max(0.0);
    // T-junction splits only help where the split vertices can then be welded, so they are
    // made on a copy, kept only if it leaves fewer open edges than plain welding. The first
    // round is planned on the mesh itself, so there is no copy when nothing would be split.
    let mut split: Option<(Mesh, CloseCracksReport)> = None;
    if tol > 0.0 {
        let first = plan_t_junctions(mesh, tol, &allowed);
        if !first.is_empty() {
            let mut copy = mesh.clone();
            let mut report = CloseCracksReport::default();
            let mut planned = Some(first);
            // A few rounds: a split creates new boundary edges that may host further vertices.
            for _ in 0..8 {
                let n = match planned.take() {
                    Some(splits) => apply_t_junctions(&mut copy, splits),
                    None => split_t_junctions(&mut copy, tol, &allowed),
                };
                report.splits += n;
                if n == 0 {
                    break;
                }
            }
            if report.splits > 0 {
                split = Some((copy, report));
            }
        }
    }
    // Plain welding, in place.
    let plain = CloseCracksReport { splits: 0, weld: weld_boundary_vertices_in(mesh, tol, &allowed) };
    if let Some((mut copy, mut report)) = split {
        report.weld = weld_boundary_vertices_in(&mut copy, tol, &allowed);
        if copy.boundary_edge_ids().count() < mesh.boundary_edge_ids().count() {
            *mesh = copy;
            return report;
        }
    }
    plain
}

/// Split boundary edges that have a boundary vertex (not their own) within `tol` of their
/// interior, at that vertex's projection. Returns the number of splits.
fn split_t_junctions(mesh: &mut Mesh, tol: f64, allowed: &dyn Fn(u32) -> bool) -> usize {
    let splits = plan_t_junctions(mesh, tol, allowed);
    apply_t_junctions(mesh, splits)
}

/// The splits [`split_t_junctions`] makes: (boundary edge, split parameter), at most one per
/// edge.
fn plan_t_junctions(mesh: &Mesh, tol: f64, allowed: &dyn Fn(u32) -> bool) -> Vec<(u32, f64)> {
    let bedges: Vec<u32> = mesh.boundary_edge_ids().filter(|&e| mesh.edge_vertices(e).iter().all(|&v| allowed(v))).collect();
    if bedges.is_empty() {
        return Vec::new();
    }
    let mean = bedges.iter().map(|&e| mesh.edge_length(e)).sum::<f64>() / bedges.len() as f64;
    let cell = mean.max(tol * 2.0).max(1e-12);
    let key = |p: DVec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
    let mut grid: FxHashMap<(i64, i64, i64), SmallVec<[u32; 4]>> = FxHashMap::default();
    let mut bverts: Vec<u32> = bedges.iter().flat_map(|&e| mesh.edge_vertices(e)).collect();
    bverts.sort_unstable();
    bverts.dedup();
    for &v in &bverts {
        grid.entry(key(mesh.vertex(v))).or_default().push(v);
    }
    // (edge, split parameter) — at most one split per edge per round.
    let mut splits: Vec<(u32, f64)> = Vec::new();
    for &e in &bedges {
        let [a, b] = mesh.edge_vertices(e);
        let (pa, pb) = (mesh.vertex(a), mesh.vertex(b));
        let d = pb - pa;
        let l2 = d.length_squared();
        if l2 <= tol * tol {
            continue;
        }
        let (lo, hi) = (key(pa.min(pb) - DVec3::splat(tol)), key(pa.max(pb) + DVec3::splat(tol)));
        if (hi.0 - lo.0 + 1) * (hi.1 - lo.1 + 1) * (hi.2 - lo.2 + 1) > 4096 {
            continue;
        }
        let mut best: Option<(f64, f64)> = None;
        for x in lo.0..=hi.0 {
            for y in lo.1..=hi.1 {
                for z in lo.2..=hi.2 {
                    let Some(c) = grid.get(&(x, y, z)) else { continue };
                    for &v in c {
                        if v == a || v == b {
                            continue;
                        }
                        let p = mesh.vertex(v);
                        let t = (p - pa).dot(d) / l2;
                        if t <= 0.0 || t >= 1.0 {
                            continue;
                        }
                        let q = pa + d * t;
                        let dist = q.distance(p);
                        if dist <= tol && p.distance(pa) > tol && p.distance(pb) > tol && best.is_none_or(|(bd, _)| dist < bd) {
                            best = Some((dist, t));
                        }
                    }
                }
            }
        }
        if let Some((_, t)) = best {
            splits.push((e, t));
        }
    }
    splits
}

/// Make the splits [`plan_t_junctions`] planned on `mesh`. Returns the number made.
fn apply_t_junctions(mesh: &mut Mesh, splits: Vec<(u32, f64)>) -> usize {
    let mut n = 0;
    for (e, t) in splits {
        if mesh.is_edge(e) && mesh.is_boundary_edge(e) && mesh.split_edge_at(e, t).is_ok() {
            n += 1;
        }
    }
    n
}

/// True if triangle `t` has (near) zero area relative to its longest edge:
/// `area ≤ eps · max_edge²`, or all vertices coincide.
#[inline]
pub fn is_degenerate_triangle(mesh: &Mesh, t: u32, eps: f64) -> bool {
    let [a, b, c] = mesh.tri_positions(t);
    let l2 = a.distance_squared(b).max(b.distance_squared(c)).max(c.distance_squared(a));
    if l2 <= 0.0 {
        return true;
    }
    0.5 * (b - a).cross(c - a).length() <= eps * l2
}

/// All degenerate triangles (see [`is_degenerate_triangle`]).
pub fn find_degenerate_triangles(mesh: &Mesh, eps: f64) -> Vec<u32> {
    let ids: Vec<u32> = mesh.triangle_ids().collect();
    ids.into_par_iter().filter(|&t| is_degenerate_triangle(mesh, t, eps)).collect()
}

/// Result of [`remove_degenerate_triangles`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DegenerateReport {
    /// Needles fixed by an edge collapse.
    pub collapsed: usize,
    /// Caps fixed by flipping their long edge.
    pub flipped: usize,
    /// Boundary caps removed.
    pub removed: usize,
    /// Degenerate triangles that could not be fixed.
    pub remaining: usize,
}

/// Collapse the short edge of a needle-shaped degenerate triangle.
fn collapse_short_edge(mesh: &mut Mesh, t: u32) -> bool {
    let tv = mesh.triangle(t);
    let mut edges: SmallVec<[(f64, u32, u32); 3]> = (0..3)
        .map(|j| {
            let (a, b) = (tv[j], tv[(j + 1) % 3]);
            (mesh.vertex(a).distance_squared(mesh.vertex(b)), a, b)
        })
        .collect();
    edges.sort_by(|x, y| x.0.total_cmp(&y.0));
    for &(_, a, b) in edges.iter().take(2) {
        let (ba, bb) = (mesh.is_boundary_vertex(a), mesh.is_boundary_vertex(b));
        let (pa, pb) = (mesh.vertex(a), mesh.vertex(b));
        let order: [(u32, u32, DVec3); 2] = if ba && !bb {
            [(a, b, pa), (b, a, pa)]
        } else if bb && !ba {
            [(b, a, pb), (a, b, pb)]
        } else {
            let mid = (pa + pb) * 0.5;
            [(a, b, mid), (b, a, mid)]
        };
        for (keep, remove, p) in order {
            if mesh.collapse_edge(keep, remove).is_ok() {
                mesh.set_vertex(keep, p);
                return true;
            }
        }
    }
    false
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DegenFix {
    Collapsed,
    Flipped,
    Removed,
}

fn fix_degenerate(mesh: &mut Mesh, t: u32) -> Option<DegenFix> {
    let tv = mesh.triangle(t);
    let te = mesh.tri_edges(t);
    let lens: [f64; 3] = std::array::from_fn(|j| mesh.vertex(tv[j]).distance(mesh.vertex(tv[(j + 1) % 3])));
    let jmax = (0..3).max_by(|&a, &b| lens[a].total_cmp(&lens[b])).unwrap();
    let jmin = (0..3).min_by(|&a, &b| lens[a].total_cmp(&lens[b])).unwrap();
    if lens[jmin] < 0.25 * lens[jmax] {
        return collapse_short_edge(mesh, t).then_some(DegenFix::Collapsed);
    }
    // Cap: the vertex opposite the longest edge lies on it.
    let e = te[jmax];
    if !mesh.is_boundary_edge(e) {
        if mesh.flip_edge(e).is_ok() {
            return Some(DegenFix::Flipped);
        }
    } else {
        let apex = tv[(jmax + 2) % 3];
        if !mesh.is_boundary_vertex(apex) {
            let _ = mesh.remove_triangle(t, true);
            return Some(DegenFix::Removed);
        }
    }
    collapse_short_edge(mesh, t).then_some(DegenFix::Collapsed)
}

/// Fix one degenerate triangle (see [`remove_degenerate_triangles`]). Returns false if no
/// valid local edit was found.
pub fn fix_degenerate_triangle(mesh: &mut Mesh, t: u32) -> bool {
    mesh.is_triangle(t) && fix_degenerate(mesh, t).is_some()
}

/// Remove degenerate (zero-area) triangles: needles are fixed by collapsing their short
/// edge, caps by flipping their long edge (or, on the mesh boundary, by removing the cap,
/// which leaves the boundary curve unchanged).
pub fn remove_degenerate_triangles(mesh: &mut Mesh, eps: f64) -> DegenerateReport {
    let mut rep = DegenerateReport::default();
    for _pass in 0..8 {
        let degen = find_degenerate_triangles(mesh, eps);
        if degen.is_empty() {
            break;
        }
        let mut progress = 0usize;
        for t in degen {
            if !mesh.is_triangle(t) || !is_degenerate_triangle(mesh, t, eps) {
                continue;
            }
            match fix_degenerate(mesh, t) {
                Some(DegenFix::Collapsed) => rep.collapsed += 1,
                Some(DegenFix::Flipped) => rep.flipped += 1,
                Some(DegenFix::Removed) => rep.removed += 1,
                None => continue,
            }
            progress += 1;
        }
        if progress == 0 {
            break;
        }
    }
    rep.remaining = find_degenerate_triangles(mesh, eps).len();
    rep
}

/// Triangle fans around `v`: groups of incident triangles connected through edges at `v`.
pub fn vertex_fans(mesh: &Mesh, v: u32) -> Vec<Vec<u32>> {
    let tris = mesh.vertex_triangles(v);
    let mut fan_of: SmallVec<[u32; 12]> = SmallVec::from_elem(u32::MAX, tris.len());
    let mut fans: Vec<Vec<u32>> = Vec::new();
    for s in 0..tris.len() {
        if fan_of[s] != u32::MAX {
            continue;
        }
        let id = fans.len() as u32;
        let mut fan = vec![tris[s]];
        fan_of[s] = id;
        let mut k = 0;
        while k < fan.len() {
            let t = fan[k];
            k += 1;
            for e in mesh.tri_edges(t) {
                if !mesh.edge_vertices(e).contains(&v) {
                    continue;
                }
                let o = mesh.edge_other_triangle(e, t);
                if o == INVALID {
                    continue;
                }
                if let Some(i) = tris.iter().position(|&x| x == o) {
                    if fan_of[i] == u32::MAX {
                        fan_of[i] = id;
                        fan.push(o);
                    }
                }
            }
        }
        fans.push(fan);
    }
    fans
}

/// Split a bowtie (non-manifold) vertex: every triangle fan but the largest gets its own
/// copy of the vertex. Returns the new vertex ids (empty if `v` was not a bowtie).
pub fn split_bowtie_vertex(mesh: &mut Mesh, v: u32) -> Vec<u32> {
    if !mesh.is_vertex(v) {
        return Vec::new();
    }
    let mut fans = vertex_fans(mesh, v);
    if fans.len() < 2 {
        return Vec::new();
    }
    fans.sort_by_key(|f| std::cmp::Reverse(f.len()));
    let p = mesh.vertex(v);
    let mut out = Vec::new();
    for fan in &fans[1..] {
        let nv = mesh.append_vertex(p);
        if try_repoint_vertex(mesh, v, nv, fan) {
            out.push(nv);
        } else {
            // Cannot happen for a valid mesh; drop the unused vertex.
            mesh.remove_isolated_vertices();
        }
    }
    out
}

/// All bowtie vertices of the mesh.
pub fn find_bowtie_vertices(mesh: &Mesh) -> Vec<u32> {
    let ids: Vec<u32> = mesh.vertex_ids().collect();
    ids.into_par_iter().filter(|&v| mesh.is_bowtie_vertex(v)).collect()
}

/// Split every bowtie vertex. Returns (bowtie vertices split, vertices created).
pub fn split_all_bowties(mesh: &mut Mesh) -> (usize, usize) {
    let bow = find_bowtie_vertices(mesh);
    let mut created = 0;
    let mut split = 0;
    for v in bow {
        let n = split_bowtie_vertex(mesh, v).len();
        if n > 0 {
            split += 1;
            created += n;
        }
    }
    (split, created)
}

/// Summary of one edge-connected component.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ComponentInfo {
    /// Triangle ids.
    pub triangles: Vec<u32>,
    /// Surface area.
    pub area: f64,
    /// Area-weighted centroid.
    pub centroid: DVec3,
}

/// Edge-connected components with area and area-weighted centroid, largest area first.
pub fn component_infos(mesh: &Mesh) -> Vec<ComponentInfo> {
    let comps = topology::connected_components(mesh);
    let mut out: Vec<ComponentInfo> = comps
        .into_par_iter()
        .map(|tris| {
            let mut area = 0.0;
            let mut c = DVec3::ZERO;
            for &t in &tris {
                let a = mesh.tri_area(t);
                area += a;
                c += mesh.tri_centroid(t) * a;
            }
            let centroid = if area > 0.0 {
                c / area
            } else {
                tris.iter().map(|&t| mesh.tri_centroid(t)).sum::<DVec3>() / tris.len().max(1) as f64
            };
            ComponentInfo { triangles: tris, area, centroid }
        })
        .collect();
    out.sort_by(|a, b| b.area.total_cmp(&a.area).then(b.triangles.len().cmp(&a.triangles.len())));
    out
}

/// Components that are not the largest and have `area < area_fraction · total_area` or
/// fewer than `min_triangles` triangles.
pub fn small_components(infos: &[ComponentInfo], area_fraction: f64, min_triangles: usize) -> Vec<usize> {
    let total: f64 = infos.iter().map(|c| c.area).sum();
    (1..infos.len())
        .filter(|&i| infos[i].area < area_fraction * total || infos[i].triangles.len() < min_triangles)
        .collect()
}

/// Remove small floating components. Returns (components removed, triangles removed).
pub fn remove_small_components(mesh: &mut Mesh, area_fraction: f64, min_triangles: usize) -> (usize, usize) {
    let infos = component_infos(mesh);
    let small = small_components(&infos, area_fraction, min_triangles);
    let mut ntri = 0;
    for &i in &small {
        ntri += infos[i].triangles.len();
        mesh.remove_triangles(infos[i].triangles.iter().copied());
    }
    (small.len(), ntri)
}

/// Number of edges whose two triangles disagree on orientation.
pub fn count_inconsistent_edges(mesh: &Mesh) -> usize {
    let ids: Vec<u32> = mesh.edge_ids().collect();
    ids.into_par_iter()
        .filter(|&e| {
            let [t0, t1] = mesh.edge_triangles(e);
            if t1 == INVALID {
                return false;
            }
            let [a, b] = mesh.edge_vertices(e);
            let tv0 = mesh.triangle(t0);
            let tv1 = mesh.triangle(t1);
            let j0 = mm_core::mesh::tri_edge_index(tv0, a, b).unwrap();
            let j1 = mm_core::mesh::tri_edge_index(tv1, a, b).unwrap();
            tv0[j0] != tv1[(j1 + 1) % 3]
        })
        .count()
}

/// Number of boundary vertices that have another boundary vertex within `tolerance`
/// (an estimate of crack / duplicated-seam size).
pub fn count_coincident_boundary_vertices(mesh: &Mesh, tolerance: f64) -> usize {
    let ids: Vec<u32> = mesh.vertex_ids().collect();
    let bverts: Vec<u32> = ids.into_par_iter().filter(|&v| mesh.is_boundary_vertex(v)).collect();
    if bverts.len() < 2 || tolerance < 0.0 {
        return 0;
    }
    let cell = tolerance.max(1e-300);
    let key = |p: DVec3| ((p.x / cell).floor() as i64, (p.y / cell).floor() as i64, (p.z / cell).floor() as i64);
    let mut grid: FxHashMap<(i64, i64, i64), SmallVec<[u32; 2]>> = FxHashMap::default();
    for &v in &bverts {
        grid.entry(key(mesh.vertex(v))).or_default().push(v);
    }
    let t2 = tolerance * tolerance;
    let mut marked = LocalIndex::new(mesh.max_vid(), bverts.len());
    let mut count = 0;
    for &v in &bverts {
        let p = mesh.vertex(v);
        let k = key(p);
        let mut hit = false;
        'outer: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(c) = grid.get(&(k.0 + dx, k.1 + dy, k.2 + dz)) {
                        if c.iter().any(|&w| w != v && mesh.vertex(w).distance_squared(p) <= t2) {
                            hit = true;
                            break 'outer;
                        }
                    }
                }
            }
        }
        if hit && marked.get(v).is_none() {
            marked.insert(v, 0);
            count += 1;
        }
    }
    count
}

/// From boundary edge `e` arriving at `q`, rotate around `q` through its triangle fan
/// (crossing edges incident to `q`, ignoring winding) to the next boundary edge.
fn next_boundary_edge_any(mesh: &Mesh, e: u32, q: u32) -> Option<u32> {
    let mut t = mesh.edge_triangles(e)[0];
    let mut cur = e;
    for _ in 0..=mesh.vertex_valence(q) {
        let f = mesh.tri_edges(t).into_iter().find(|&x| x != cur && mesh.edge_vertices(x).contains(&q))?;
        let n = mesh.edge_other_triangle(f, t);
        if n == INVALID {
            return Some(f);
        }
        cur = f;
        t = n;
    }
    None
}

/// All boundary loops, like [`mm_core::topology::boundary_loops`] but independent of the
/// triangle winding, so loops are also found on inconsistently oriented meshes (raw STL
/// soups). Each loop follows the winding of its first edge's triangle; bowtie vertices are
/// handled per fan, so a vertex can appear in several loops or twice in one loop.
pub fn boundary_loops(mesh: &Mesh) -> Vec<EdgeLoop> {
    let mut used = vec![false; mesh.max_eid()];
    let mut loops = Vec::new();
    let bedges: Vec<u32> = mesh.boundary_edge_ids().collect();
    for e0 in bedges {
        if used[e0 as usize] {
            continue;
        }
        let [p, q] = mesh.oriented_edge_vertices(e0);
        let mut lp = EdgeLoop { vertices: vec![p], edges: vec![e0] };
        used[e0 as usize] = true;
        let (mut e, mut cur) = (e0, q);
        let mut closed = false;
        for _ in 0..=mesh.edge_count() {
            if cur == p {
                closed = true;
                break;
            }
            let Some(ne) = next_boundary_edge_any(mesh, e, cur) else { break };
            if used[ne as usize] {
                break;
            }
            used[ne as usize] = true;
            lp.vertices.push(cur);
            lp.edges.push(ne);
            cur = mesh.edge_other_vertex(ne, cur);
            e = ne;
        }
        if closed {
            loops.push(lp);
        }
    }
    loops
}

/// Make the winding consistent (only if needed) and turn every closed or open component
/// so that its signed volume is positive. Returns the number of triangles reversed.
/// Equivalent to [`mm_core::topology::orient_consistently`], but skips the flood fill on
/// already consistent meshes and evaluates component volumes in parallel.
pub fn orient_outward(mesh: &mut Mesh) -> usize {
    if count_inconsistent_edges(mesh) > 0 {
        return topology::orient_consistently(mesh);
    }
    let comps = topology::connected_components(mesh);
    let flip: Vec<&Vec<u32>> = comps
        .par_iter()
        .filter(|c| {
            let vol: f64 = c
                .iter()
                .map(|&t| {
                    let [a, b, cc] = mesh.tri_positions(t);
                    a.dot(b.cross(cc))
                })
                .sum();
            vol < 0.0
        })
        .collect();
    let mut n = 0;
    for c in flip {
        for &t in c {
            mesh.reverse_triangle(t);
        }
        n += c.len();
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// Exactly the same mesh: vertex and triangle slots, positions (bits), triangles, groups.
    fn same_mesh(a: &Mesh, b: &Mesh) -> bool {
        let bits = |p: DVec3| p.to_array().map(f64::to_bits);
        a.max_vid() == b.max_vid()
            && a.max_tid() == b.max_tid()
            && (0..a.max_vid() as u32).all(|v| a.is_vertex(v) == b.is_vertex(v) && (!a.is_vertex(v) || bits(a.vertex(v)) == bits(b.vertex(v))))
            && (0..a.max_tid() as u32).all(|t| a.is_triangle(t) == b.is_triangle(t) && (!a.is_triangle(t) || (a.triangle(t) == b.triangle(t) && a.tri_group(t) == b.tri_group(t))))
    }

    /// [`close_cracks_in`] as it was: plain welding on one copy, T-junction splits on another,
    /// the better copy replacing the mesh.
    fn close_cracks_reference(mesh: &mut Mesh, tolerance: f64, only: Option<&[u32]>) -> CloseCracksReport {
        let mask: Option<Vec<bool>> = only.map(|vs| {
            let mut m = vec![false; mesh.max_vid()];
            for &v in vs {
                if (v as usize) < m.len() {
                    m[v as usize] = true;
                }
            }
            m
        });
        let n0 = mesh.max_vid();
        let allowed = |v: u32| mask.as_ref().is_none_or(|m| v as usize >= n0 || m[v as usize]);
        let tol = tolerance.max(0.0);
        let mut welded = mesh.clone();
        let plain = CloseCracksReport { splits: 0, weld: weld_boundary_vertices_in(&mut welded, tol, &allowed) };
        let mut best = (welded, plain);
        if tol > 0.0 {
            let mut split = mesh.clone();
            let mut report = CloseCracksReport::default();
            for _ in 0..8 {
                let n = split_t_junctions(&mut split, tol, &allowed);
                report.splits += n;
                if n == 0 {
                    break;
                }
            }
            if report.splits > 0 {
                report.weld = weld_boundary_vertices_in(&mut split, tol, &allowed);
                if split.boundary_edge_ids().count() < best.0.boundary_edge_ids().count() {
                    best = (split, report);
                }
            }
        }
        *mesh = best.0;
        best.1
    }

    /// A fine and a coarse grid meeting along x = 0 without sharing vertices (T-junctions).
    fn t_junctions() -> Mesh {
        let mut left = primitives::plane_grid(2.0, 4.0, 2, 4);
        left.transform(|p| p - DVec3::new(1.0, 0.0, 0.0));
        let mut right = primitives::plane_grid(2.0, 4.0, 2, 8);
        right.transform(|p| p + DVec3::new(1.0 + 1e-7, 0.0, 0.0));
        let mut m = left.clone();
        m.append_mesh(&right, 0);
        m
    }

    /// A grid whose triangles right of x = `cut` have their own copies of the vertices (a
    /// duplicated seam), the copies nudged by `nudge`.
    fn seam(cut: f64, nudge: f64) -> Mesh {
        let g = primitives::plane_grid(4.0, 4.0, 24, 24);
        let (pos, tris, _) = g.to_arrays();
        let mut p = pos.clone();
        let mut t = Vec::new();
        for tv in tris {
            let c = (pos[tv[0] as usize] + pos[tv[1] as usize] + pos[tv[2] as usize]) / 3.0;
            if c.x > cut {
                let base = p.len() as u32;
                for (k, &v) in tv.iter().enumerate() {
                    p.push(pos[v as usize] + DVec3::new(nudge * (k as f64 - 1.0), nudge, 0.0));
                }
                t.push([base, base + 1, base + 2]);
            } else {
                t.push(tv);
            }
        }
        Mesh::from_triangles(&p, &t, None).0
    }

    /// A triangle whose tip touches the middle of a strip's edge: splitting there and welding
    /// the tip leaves more open edges than not splitting, so the plain weld is kept.
    fn touching_tip() -> Mesh {
        let mut m = primitives::plane_grid(2.0, 1.0, 1, 1);
        let tip = Mesh::from_triangles(&[DVec3::new(0.0, 0.5, 0.0), DVec3::new(0.2, 1.5, 0.0), DVec3::new(-0.2, 1.5, 0.0)], &[[0, 1, 2]], None).0;
        m.append_mesh(&tip, 0);
        m
    }

    #[test]
    fn close_cracks_matches_the_two_copy_version() {
        // The tip is split for, and that copy turned down.
        assert!(!plan_t_junctions(&touching_tip(), 1e-6, &|_| true).is_empty());
        let cases: Vec<(&str, Mesh, f64)> = vec![
            ("touching tip", touching_tip(), 1e-6),
            ("t-junctions", t_junctions(), 1e-5),
            ("t-junctions, no tolerance", t_junctions(), 0.0),
            ("seam", seam(0.3, 0.0), 1e-6),
            ("nudged seam", seam(-0.7, 1e-4), 1e-3),
            ("seam too far", seam(0.3, 1e-2), 1e-3),
            ("closed sphere", primitives::icosphere(1.0, 2), 1e-6),
        ];
        for (name, m, tol) in cases {
            let half: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).y > 0.0).collect();
            for only in [None, Some(&half[..])] {
                let (mut a, mut b) = (m.clone(), m.clone());
                let ra = close_cracks_in(&mut a, tol, only);
                let rb = close_cracks_reference(&mut b, tol, only);
                assert_eq!(ra, rb, "{name} (only {})", only.is_some());
                assert!(same_mesh(&a, &b), "{name} (only {}): {ra:?}", only.is_some());
                a.check_validity().unwrap();
            }
        }
    }
}
