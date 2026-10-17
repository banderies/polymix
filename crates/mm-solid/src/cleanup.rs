//! Post-processing of extracted surfaces. Every operation here preserves closedness,
//! edge-manifoldness and orientation: collapses go through mm-core's link-condition
//! checks, flips through `flip_edge`'s checks, and both additionally reject any change
//! that would flip or degenerate a triangle.

use glam::DVec3;
use mm_core::mesh::INVALID;
use mm_core::{topology, Bvh, Mesh};
use rayon::prelude::*;

/// Would collapsing `remove` into `keep` (moved to `p`) keep every surviving incident
/// triangle non-degenerate and within ~60° of its old normal?
fn collapse_geometry_ok(m: &Mesh, keep: u32, remove: u32, p: DVec3, scale2: f64) -> bool {
    let ta = m.vertex_triangles(keep);
    let tb = m.vertex_triangles(remove);
    let mut reference = DVec3::ZERO;
    for &t in ta.iter().chain(tb.iter()) {
        reference += m.tri_cross(t);
    }
    for &t in ta.iter().chain(tb.iter()) {
        let tv = m.triangle(t);
        let hk = tv.contains(&keep);
        let hr = tv.contains(&remove);
        if hk && hr {
            continue;
        }
        let q = tv.map(|x| if x == keep || x == remove { p } else { m.vertex(x) });
        let n_new = (q[1] - q[0]).cross(q[2] - q[0]);
        let ln2 = n_new.length_squared();
        if ln2 <= 1e-20 * scale2 * scale2 {
            return false;
        }
        let n_old = m.tri_cross(t);
        let lo2 = n_old.length_squared();
        if lo2 > 1e-12 * scale2 * scale2 {
            // cos(angle) >= 0.5
            let d = n_old.dot(n_new);
            if d <= 0.0 || d * d < 0.25 * lo2 * ln2 {
                return false;
            }
        } else if n_new.dot(reference) <= 0.0 {
            return false;
        }
    }
    true
}

#[inline]
fn edge_len2(m: &Mesh, e: u32) -> f64 {
    let [a, b] = m.edge_vertices(e);
    m.vertex(a).distance_squared(m.vertex(b))
}

/// Collapse edges shorter than `min_len` (shortest first) to their midpoints. Returns the
/// number of collapses.
pub fn collapse_short_edges(m: &mut Mesh, min_len: f64) -> usize {
    if !(min_len > 0.0) {
        return 0;
    }
    let min2 = min_len * min_len;
    let mut total = 0;
    for _pass in 0..6 {
        let ids: Vec<u32> = m.edge_ids().collect();
        let mut cand: Vec<(f64, u32)> = ids
            .par_iter()
            .filter_map(|&e| {
                let l2 = edge_len2(m, e);
                (l2 < min2).then_some((l2, e))
            })
            .collect();
        if cand.is_empty() {
            break;
        }
        cand.par_sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        let mut n = 0;
        for (_, e) in cand {
            if !m.is_edge(e) {
                continue;
            }
            let [a, b] = m.edge_vertices(e);
            let (pa, pb) = (m.vertex(a), m.vertex(b));
            if pa.distance_squared(pb) >= min2 {
                continue;
            }
            let mid = (pa + pb) * 0.5;
            // `collapse_edge` performs the link-condition check itself.
            if collapse_geometry_ok(m, a, b, mid, min2) && m.collapse_edge(a, b).is_ok() {
                m.set_vertex(a, mid);
                n += 1;
            }
        }
        total += n;
        if n == 0 {
            break;
        }
    }
    total
}

#[inline]
fn cot(a: DVec3, b: DVec3) -> f64 {
    // Cotangent of the angle between a and b.
    let c = a.cross(b).length();
    a.dot(b) / c.max(1e-300)
}

/// Delaunay-style edge flips on nearly flat regions: an edge is flipped when the angles
/// opposite it sum to more than π (which increases the minimum angle), provided the quad
/// is nearly planar (the flip barely changes the shape) and both new triangles keep the
/// orientation. Returns the number of flips.
pub fn flip_to_improve(m: &mut Mesh, passes: usize) -> usize {
    let mut work: Vec<u32> = m.edge_ids().collect();
    let mut queued = vec![true; m.max_eid()];
    let limit = passes.max(1) * work.len().max(16);
    let mut total = 0;
    let mut steps = 0;
    while let Some(e) = work.pop() {
        queued[e as usize] = false;
        steps += 1;
        if steps > limit {
            break;
        }
        if !m.is_edge(e) || m.is_boundary_edge(e) {
            continue;
        }
        let [a, b] = m.oriented_edge_vertices(e);
        let [c, d] = m.edge_opposite_vertices(e);
        if c == INVALID || d == INVALID || c == d {
            continue;
        }
        let (pa, pb, pc, pd) = (m.vertex(a), m.vertex(b), m.vertex(c), m.vertex(d));
        // Angles at c (triangle a,b,c) and d (triangle b,a,d).
        if cot(pa - pc, pb - pc) + cot(pa - pd, pb - pd) >= -1e-3 {
            continue;
        }
        let ab = pa.distance(pb);
        let cd = pc.distance(pd);
        let vol6 = (pb - pa).dot((pc - pa).cross(pd - pa)).abs();
        if vol6 > 0.02 * ab * cd * ab.max(cd) {
            continue;
        }
        // After the flip: (c, a, d) and (d, b, c).
        let navg = (pb - pa).cross(pc - pa) + (pa - pb).cross(pd - pb);
        let m0 = (pa - pc).cross(pd - pc);
        let m1 = (pb - pd).cross(pc - pd);
        if m0.dot(navg) <= 0.0 || m1.dot(navg) <= 0.0 {
            continue;
        }
        if m.flip_edge(e).is_ok() {
            total += 1;
            let [t0, t1] = m.edge_triangles(e);
            for t in [t0, t1] {
                for e2 in m.tri_edges(t) {
                    if e2 != e && !queued[e2 as usize] {
                        queued[e2 as usize] = true;
                        work.push(e2);
                    }
                }
            }
        }
    }
    total
}

/// Signed volume of a set of triangles (positive for outward-oriented closed shells).
pub fn shell_volume(m: &Mesh, tris: &[u32]) -> f64 {
    tris.iter()
        .map(|&t| {
            let [a, b, c] = m.tri_positions(t);
            a.dot(b.cross(c))
        })
        .sum::<f64>()
        / 6.0
}

/// Remove connected shells whose enclosed |volume| is below `min_volume`. Returns the
/// number of shells removed.
pub fn remove_small_shells(m: &mut Mesh, min_volume: f64) -> usize {
    if !(min_volume > 0.0) {
        return 0;
    }
    let comps = topology::connected_components(m);
    if comps.len() <= 1 {
        return 0;
    }
    let mut removed = 0;
    // Never remove everything: keep the largest shell regardless.
    let vols: Vec<f64> = comps.iter().map(|c| shell_volume(m, c).abs()).collect();
    let largest = vols.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0);
    for (i, c) in comps.iter().enumerate() {
        if i != largest && vols[i] < min_volume {
            m.remove_triangles(c.iter().copied());
            removed += 1;
        }
    }
    removed
}

/// Number of bowtie (non-manifold) vertices.
pub fn bowtie_count(m: &Mesh) -> usize {
    m.vertex_ids().filter(|&v| m.is_bowtie_vertex(v)).count()
}

/// Is moving `v` to `p` safe (no incident triangle flips or degenerates)?
fn move_ok(m: &Mesh, v: u32, p: DVec3, cos_limit: f64) -> bool {
    for t in m.vertex_triangles(v) {
        let tv = m.triangle(t);
        let q = tv.map(|x| if x == v { p } else { m.vertex(x) });
        let n_new = (q[1] - q[0]).cross(q[2] - q[0]);
        let n_old = m.tri_cross(t);
        let (ln, lo) = (n_new.length(), n_old.length());
        if ln <= 1e-12 * (lo + 1e-300) || n_old.dot(n_new) < cos_limit * lo * ln {
            return false;
        }
    }
    true
}

/// Project vertices of `out` onto the nearest point of `input` (shifted by `offset` along
/// the input normal) when that point is within `max_dist`, lies in the interior of the
/// input surface (not on an open boundary) and the input normal agrees with the output
/// normal (`flip_input` accounts for an inverted input). Moves that would flip a triangle
/// are halved, then skipped. Input triangles flagged in `exclude` (indexed by triangle id,
/// e.g. triangles that were artificially thickened) are never projection targets.
/// Returns the number of vertices moved.
pub fn project_to_input(
    out: &mut Mesh,
    input: &Mesh,
    bvh: &Bvh,
    max_dist: f64,
    offset: f64,
    flip_input: bool,
    exclude: Option<&[bool]>,
) -> usize {
    let normals = out.compute_vertex_normals();
    let ids: Vec<u32> = out.vertex_ids().collect();
    let targets: Vec<Option<DVec3>> = ids
        .par_iter()
        .map(|&v| {
            let p = out.vertex(v);
            let hit = bvh.nearest(input, p, max_dist + offset.abs())?;
            if exclude.is_some_and(|x| x.get(hit.tid as usize).copied().unwrap_or(false)) {
                return None;
            }
            let mut n = input.tri_normal(hit.tid);
            if flip_input {
                n = -n;
            }
            if n.dot(normals[v as usize]) < 0.7 {
                return None;
            }
            // Reject points on open boundary edges of the input (hole rims).
            let tv = input.triangle(hit.tid);
            for j in 0..3 {
                let w = hit.bary[(j + 2) % 3];
                if w < 1e-9 {
                    if let Some(e) = input.find_edge(tv[j], tv[(j + 1) % 3]) {
                        if input.is_boundary_edge(e) {
                            return None;
                        }
                    }
                }
            }
            let target = hit.point + n * offset;
            (target.distance(p) <= max_dist).then_some(target)
        })
        .collect();
    let mut moved = 0;
    for (&v, t) in ids.iter().zip(targets) {
        let Some(t) = t else { continue };
        let p = out.vertex(v);
        for s in [1.0, 0.5] {
            let q = p.lerp(t, s);
            if move_ok(out, v, q, 0.3) {
                out.set_vertex(v, q);
                moved += 1;
                break;
            }
        }
    }
    moved
}
