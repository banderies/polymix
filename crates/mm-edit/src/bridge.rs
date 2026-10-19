//! Bridge (Meshmixer *Edit > Bridge*): connect two boundary spans, or two whole boundary
//! loops, with a strip of triangles. Vertex counts may differ: each band between two chains
//! is triangulated by dynamic programming (minimum total diagonal length over all monotone
//! strip triangulations), and intermediate rows are resampled by arc length.

use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::{DQuat, DVec3, Mesh};

/// Parameters for [`bridge`] and [`join_loops`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BridgeParams {
    /// Rows of triangles across the gap; 0 = automatic (gap / mean edge length).
    pub segments: usize,
    /// [`join_loops`] only: follow a smooth Hermite curve leaving each loop along its
    /// outward direction (otherwise straight lines).
    pub smooth: bool,
}

impl Default for BridgeParams {
    fn default() -> Self {
        Self { segments: 0, smooth: true }
    }
}

/// Output of [`bridge`] / [`join_loops`].
#[derive(Clone, Debug, Default)]
pub struct BridgeResult {
    /// Bridge triangles.
    pub triangles: Vec<u32>,
    /// Vertices of intermediate rows.
    pub new_vertices: Vec<u32>,
    /// Face group of the bridge.
    pub group: i32,
}

/// Triangulate the band between chains `p` (edges used forward, p_i → p_{i+1}) and `q`
/// (edges used backward, q_{j+1} → q_j), from the pair (p_0, q_0) to (p_m, q_k), minimising
/// the total length of the new diagonals. Shared end vertices are handled.
pub(crate) fn strip(mesh: &Mesh, p: &[u32], q: &[u32]) -> Option<Vec<[u32; 3]>> {
    strip_with(mesh, p, q, &|_| 0.0)
}

/// [`strip`] with an extra cost per candidate triangle (e.g. against folding it over).
pub(crate) fn strip_with(mesh: &Mesh, p: &[u32], q: &[u32], extra: &dyn Fn([u32; 3]) -> f64) -> Option<Vec<[u32; 3]>> {
    if p.len() < 2 && q.len() < 2 {
        return None;
    }
    let (m, k) = (p.len() - 1, q.len() - 1);
    let mut pre = Vec::new();
    let mut post = Vec::new();
    let (mut i0, mut j0, mut i1, mut j1) = (0usize, 0usize, m, k);
    if p[0] == q[0] {
        if m < 1 || k < 1 {
            return None;
        }
        pre.push([p[0], p[1], q[1]]);
        i0 = 1;
        j0 = 1;
    }
    if p[m] == q[k] {
        if m < 1 || k < 1 {
            return None;
        }
        post.push([p[m - 1], p[m], q[k - 1]]);
        i1 = m - 1;
        j1 = k - 1;
    }
    if i0 > i1 || j0 > j1 {
        return None;
    }
    let w = j1 - j0 + 1;
    let h = i1 - i0 + 1;
    let idx = |i: usize, j: usize| (i - i0) * w + (j - j0);
    let mut cost = vec![f64::INFINITY; w * h];
    let mut from = vec![0u8; w * h];
    cost[0] = 0.0;
    let pos = |v: u32| mesh.vertex(v);
    let distinct = |a: u32, b: u32, c: u32| a != b && b != c && a != c;
    for i in i0..=i1 {
        for j in j0..=j1 {
            let c = cost[idx(i, j)];
            if !c.is_finite() {
                continue;
            }
            if i < i1 && distinct(p[i], p[i + 1], q[j]) {
                let terminal = i + 1 == i1 && j == j1;
                if terminal || mesh.find_edge(p[i + 1], q[j]).is_none() {
                    let nc = c + pos(p[i + 1]).distance(pos(q[j])) + extra([p[i], p[i + 1], q[j]]);
                    if nc < cost[idx(i + 1, j)] {
                        cost[idx(i + 1, j)] = nc;
                        from[idx(i + 1, j)] = 1;
                    }
                }
            }
            if j < j1 && distinct(p[i], q[j + 1], q[j]) {
                let terminal = i == i1 && j + 1 == j1;
                if terminal || mesh.find_edge(p[i], q[j + 1]).is_none() {
                    let nc = c + pos(p[i]).distance(pos(q[j + 1])) + extra([p[i], q[j + 1], q[j]]);
                    if nc < cost[idx(i, j + 1)] {
                        cost[idx(i, j + 1)] = nc;
                        from[idx(i, j + 1)] = 2;
                    }
                }
            }
        }
    }
    if !cost[idx(i1, j1)].is_finite() {
        return None;
    }
    let mut tris = Vec::new();
    let (mut i, mut j) = (i1, j1);
    while (i, j) != (i0, j0) {
        match from[idx(i, j)] {
            1 => {
                tris.push([p[i - 1], p[i], q[j]]);
                i -= 1;
            }
            2 => {
                tris.push([p[i], q[j], q[j - 1]]);
                j -= 1;
            }
            _ => return None,
        }
    }
    tris.reverse();
    let mut out = pre;
    out.extend(tris);
    out.extend(post);
    Some(out)
}

/// Band between two rows with equal counts (vertex i ↔ i), shorter diagonal per quad.
pub(crate) fn quad_band(mesh: &Mesh, p: &[u32], q: &[u32]) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(2 * p.len());
    for i in 0..p.len() - 1 {
        let da = mesh.vertex(p[i + 1]).distance_squared(mesh.vertex(q[i]));
        let db = mesh.vertex(p[i]).distance_squared(mesh.vertex(q[i + 1]));
        if da <= db {
            out.push([p[i], p[i + 1], q[i]]);
            out.push([p[i + 1], q[i + 1], q[i]]);
        } else {
            out.push([p[i], q[i + 1], q[i]]);
            out.push([p[i], p[i + 1], q[i + 1]]);
        }
    }
    out
}

pub(crate) fn mean_edge(mesh: &Mesh, chain: &[u32], closed: bool) -> f64 {
    let n = chain.len();
    let segs = if closed { n } else { n - 1 };
    let total: f64 = (0..segs).map(|i| mesh.vertex(chain[i]).distance(mesh.vertex(chain[(i + 1) % n]))).sum();
    total / segs.max(1) as f64
}

pub(crate) fn commit(mesh: &mut Mesh, tris: Vec<[u32; 3]>, new_vertices: Vec<u32>, group: i32) -> EditResult<BridgeResult> {
    let mut added = Vec::with_capacity(tris.len());
    let mut failure = None;
    for tv in tris {
        match mesh.append_triangle(tv, group) {
            Ok(t) => added.push(t),
            Err(e) => {
                failure = Some(EditError::Mesh(e));
                break;
            }
        }
    }
    if failure.is_none() && !locally_consistent(mesh, &added) {
        failure = Some(EditError::Failed("bridge would create inconsistent orientation".into()));
    }
    if let Some(e) = failure {
        rollback(mesh, &added);
        if new_vertices.iter().any(|&v| mesh.is_vertex(v)) {
            mesh.remove_isolated_vertices();
        }
        return Err(e);
    }
    Ok(BridgeResult { triangles: added, new_vertices, group })
}

/// Connect two boundary spans (open chains of consecutive boundary vertices, in either
/// order) with a strip of triangles. The spans may lie on the same or different loops and
/// may share an end vertex. Use [`join_loops`] for whole loops.
pub fn bridge(mesh: &mut Mesh, span_a: &[u32], span_b: &[u32], params: &BridgeParams) -> EditResult<BridgeResult> {
    if span_a.first() == span_a.last() || span_b.first() == span_b.last() {
        return Err(EditError::InvalidLoop("closed span: use join_loops for whole loops".into()));
    }
    let a = orient_boundary_span(mesh, span_a)?;
    let b = orient_boundary_span(mesh, span_b)?;
    let ea: Vec<[u32; 2]> = a.windows(2).map(|w| [w[0].min(w[1]), w[0].max(w[1])]).collect();
    if b.windows(2).any(|w| ea.contains(&[w[0].min(w[1]), w[0].max(w[1])])) {
        return Err(EditError::InvalidLoop("spans overlap".into()));
    }
    let p: Vec<u32> = a.iter().rev().copied().collect();
    let q = b;
    let group = new_group(mesh);
    let shared = p[0] == q[0] || p[p.len() - 1] == q[q.len() - 1];
    let gap = 0.5
        * (mesh.vertex(p[0]).distance(mesh.vertex(q[0]))
            + mesh.vertex(p[p.len() - 1]).distance(mesh.vertex(q[q.len() - 1])));
    let h = 0.5 * (mean_edge(mesh, &p, false) + mean_edge(mesh, &q, false));
    let segs = if shared {
        1
    } else if params.segments == 0 {
        ((gap / h.max(1e-300)).round() as usize).clamp(1, 1000)
    } else {
        params.segments
    };
    if segs == 1 {
        let tris = strip(mesh, &p, &q).ok_or_else(|| EditError::Failed("no valid bridge strip".into()))?;
        return commit(mesh, tris, Vec::new(), group);
    }
    let n = p.len().max(q.len());
    let rp = resample_polyline(&positions(mesh, &p), n, false);
    let rq = resample_polyline(&positions(mesh, &q), n, false);
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut new_vertices = Vec::new();
    for r in 1..segs {
        let t = r as f64 / segs as f64;
        let row: Vec<u32> = (0..n).map(|i| mesh.append_vertex(rp[i].lerp(rq[i], t))).collect();
        new_vertices.extend_from_slice(&row);
        rows.push(row);
    }
    let mut tris = Vec::new();
    let first = strip(mesh, &p, &rows[0]);
    let last = strip(mesh, &rows[rows.len() - 1], &q);
    let (Some(first), Some(last)) = (first, last) else {
        mesh.remove_isolated_vertices();
        return Err(EditError::Failed("no valid bridge strip".into()));
    };
    tris.extend(first);
    for w in rows.windows(2) {
        tris.extend(quad_band(mesh, &w[0], &w[1]));
    }
    tris.extend(last);
    commit(mesh, tris, new_vertices, group)
}

/// Two boundary loops oriented for a tube between them.
pub(crate) struct AlignedLoops {
    /// The loops in boundary orientation.
    pub a: Vec<u32>,
    pub b: Vec<u32>,
    /// A against its boundary orientation, and B with it, rotated so its start best matches
    /// A's start (the cyclic shift of the normalised shapes that untwists the tube).
    pub p: Vec<u32>,
    pub q: Vec<u32>,
}

/// `dirs`: the directions the tube leaves A and B in (out of each opening), when known. A's
/// shape is then turned the way the tube turns between them before matching it to B's —
/// for openings facing the same way (a handle) the shapes match equally well at any shift
/// of a circle, and only the turn tells the outer side of the arch from the inner.
pub(crate) fn aligned_loops(mesh: &Mesh, loop_a: &[u32], loop_b: &[u32], dirs: Option<[DVec3; 2]>) -> EditResult<AlignedLoops> {
    let a = orient_boundary_loop(mesh, loop_a)?;
    let b = orient_boundary_loop(mesh, loop_b)?;
    if a.iter().any(|v| b.contains(v)) {
        return Err(EditError::InvalidLoop("loops share vertices".into()));
    }
    // The strip runs along A against its boundary orientation and along B with it.
    let p: Vec<u32> = a.iter().rev().copied().collect();
    let pa = positions(mesh, &p);
    let pb = positions(mesh, &b);
    let (ca, cb) = (mean(&pa), mean(&pb));
    let ra = pa.iter().map(|x| x.distance(ca)).sum::<f64>() / pa.len() as f64;
    let rb = pb.iter().map(|x| x.distance(cb)).sum::<f64>() / pb.len() as f64;
    let (ra, rb) = (ra.max(1e-300), rb.max(1e-300));

    // Align B's start with A's: best cyclic shift of the normalised, resampled shapes.
    const M: usize = 64;
    let turn = dirs.map_or(DQuat::IDENTITY, |[da, db]| tube_turn(da, -db, cb - ca));
    let sa: Vec<DVec3> = resample_polyline(&pa, M, true).into_iter().map(|x| turn * ((x - ca) / ra)).collect();
    let sb = resample_polyline(&pb, M, true);
    let mut best = (0usize, f64::INFINITY);
    for sh in 0..M {
        let cost: f64 = (0..M).map(|i| (sa[i] - (sb[(i + sh) % M] - cb) / rb).length_squared()).sum();
        if cost < best.1 {
            best = (sh, cost);
        }
    }
    let target = sb[best.0];
    let best_j = (0..b.len())
        .min_by(|&x, &y| pb[x].distance_squared(target).partial_cmp(&pb[y].distance_squared(target)).unwrap())
        .unwrap_or(0);
    let mut q = b[best_j..].to_vec();
    q.extend_from_slice(&b[..best_j]);
    Ok(AlignedLoops { a, b, p, q })
}

/// The rotation carrying a tube's cross-section from its start (travelling along `from`) to
/// its end (travelling along `to`): parallel transport along the path start direction →
/// chord → end direction, so an arch turns in the plane of its chord even when the end
/// directions are nearly opposite (where the smallest turn between them could be about any
/// axis). Identity when a direction is unknown (zero).
fn tube_turn(from: DVec3, to: DVec3, chord: DVec3) -> DQuat {
    let (from, to, chord) = (from.normalize_or_zero(), to.normalize_or_zero(), chord.normalize_or_zero());
    if from == DVec3::ZERO || to == DVec3::ZERO {
        return DQuat::IDENTITY;
    }
    if chord == DVec3::ZERO {
        return DQuat::from_rotation_arc(from, to);
    }
    DQuat::from_rotation_arc(chord, to) * DQuat::from_rotation_arc(from, chord)
}

/// Join two whole boundary loops (either orientation) with a tube of triangles, e.g. to
/// reconnect a vessel across a gap. The loops may have different vertex counts.
pub fn join_loops(mesh: &mut Mesh, loop_a: &[u32], loop_b: &[u32], params: &BridgeParams) -> EditResult<BridgeResult> {
    let AlignedLoops { a, b, p, q } = aligned_loops(mesh, loop_a, loop_b, None)?;
    let group = new_group(mesh);
    let gap = mean(&positions(mesh, &p)).distance(mean(&positions(mesh, &q)));
    let h = 0.5 * (mean_edge(mesh, &p, true) + mean_edge(mesh, &q, true));
    let segs = if params.segments == 0 {
        ((gap / h.max(1e-300)).round() as usize).clamp(1, 1000)
    } else {
        params.segments
    };
    let close = |v: &[u32]| {
        let mut c = v.to_vec();
        c.push(v[0]);
        c
    };
    if segs == 1 {
        let tris = strip(mesh, &close(&p), &close(&q)).ok_or_else(|| EditError::Failed("no valid strip".into()))?;
        return commit(mesh, tris, Vec::new(), group);
    }
    let n = p.len().max(q.len());
    let rp = resample_polyline(&positions(mesh, &p), n, true);
    let rq = resample_polyline(&positions(mesh, &q), n, true);
    // Hermite tangents: leave A along its outward direction, arrive at B against B's.
    let (ta, tb) = if params.smooth {
        let da = loop_outward_direction(mesh, &a).unwrap_or(DVec3::ZERO);
        let db = loop_outward_direction(mesh, &b).unwrap_or(DVec3::ZERO);
        (da * gap, -db * gap)
    } else {
        (DVec3::ZERO, DVec3::ZERO)
    };
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut new_vertices = Vec::new();
    for r in 1..segs {
        let t = r as f64 / segs as f64;
        let (t2, t3) = (t * t, t * t * t);
        let (h00, h10, h01, h11) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2);
        let row: Vec<u32> = (0..n)
            .map(|i| {
                let pos = if params.smooth {
                    rp[i] * h00 + ta * h10 + rq[i] * h01 + tb * h11
                } else {
                    rp[i].lerp(rq[i], t)
                };
                mesh.append_vertex(pos)
            })
            .collect();
        new_vertices.extend_from_slice(&row);
        rows.push(row);
    }
    let first = strip(mesh, &close(&p), &close(&rows[0]));
    let last = strip(mesh, &close(&rows[rows.len() - 1]), &close(&q));
    let (Some(first), Some(last)) = (first, last) else {
        mesh.remove_isolated_vertices();
        return Err(EditError::Failed("no valid strip".into()));
    };
    let mut tris = first;
    for w in rows.windows(2) {
        tris.extend(quad_band(mesh, &close(&w[0]), &close(&w[1])));
    }
    tris.extend(last);
    commit(mesh, tris, new_vertices, group)
}
