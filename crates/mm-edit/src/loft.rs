//! Loft: connect two boundary loops with a tube, e.g. to rejoin a vessel across a gap.
//!
//! Each column of the tube (one point of A's loop to the matching point of B's) follows a
//! cubic Hermite curve. At a G1 end the curve leaves along the surrounding wall, taken per
//! boundary vertex (so an obliquely cut vessel blends without a kink), with the end's
//! tension scaling the tangent; at a G0 end the curve is free (zero curvature there), so it
//! only meets the opening. The cross-section blends from A's shape to B's along the way.

use crate::bridge::{aligned_loops, commit, mean_edge, quad_band, strip, AlignedLoops, BridgeResult};
use crate::util::*;
use crate::{EditError, EditResult};
use mm_core::mesh::tri_other_vertex;
use mm_core::{DVec3, Mesh, INVALID};

/// Shape of a loft, like the fill types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum LoftShape {
    /// One band of triangles straight across, no new vertices.
    Minimal,
    /// Straight tube with rings at the requested edge length.
    Straight,
    /// Curved tube following each end's continuity.
    #[default]
    Smooth,
}

/// How a [`LoftShape::Smooth`] tube meets an opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EndContinuity {
    /// Position only: the tube meets the opening, with a crease if the directions differ.
    G0,
    /// Tangent: the tube leaves along the surrounding wall.
    #[default]
    G1,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoftEnd {
    pub continuity: EndContinuity,
    /// Length of the end tangent relative to the gap (G1 only); larger keeps the tube
    /// following the wall's direction for longer.
    pub tension: f64,
}

impl Default for LoftEnd {
    fn default() -> Self {
        Self { continuity: EndContinuity::G1, tension: 1.0 }
    }
}

/// Parameters for [`loft`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoftParams {
    pub shape: LoftShape,
    /// End at the first loop and at the second.
    pub ends: [LoftEnd; 2],
    /// Target edge length of the tube; `None` = the loops' mean edge length.
    pub edge_length: Option<f64>,
    /// Extra rotation of the second loop against the first, in degrees (after automatic
    /// alignment).
    pub twist_deg: f64,
    /// Widens (+) or narrows (−) the middle of the tube: 0.5 = 50 % wider halfway.
    pub bulge: f64,
    /// Face group of the tube; `None` = a new group.
    pub group: Option<i32>,
}

impl Default for LoftParams {
    fn default() -> Self {
        Self { shape: LoftShape::Smooth, ends: [LoftEnd::default(); 2], edge_length: None, twist_deg: 0.0, bulge: 0.0, group: None }
    }
}

/// Connect boundary loops `loop_a` and `loop_b` (vertex lists of the same mesh, either
/// orientation, possibly different vertex counts) with a tube.
pub fn loft(mesh: &mut Mesh, loop_a: &[u32], loop_b: &[u32], params: &LoftParams) -> EditResult<BridgeResult> {
    if params.bulge.is_nan() || params.bulge <= -1.0 || params.ends.iter().any(|e| e.tension.is_nan() || e.tension <= 0.0) {
        return Err(EditError::InvalidParameter("bulge must exceed −100 % and tension must be positive".into()));
    }
    let AlignedLoops { p, q, .. } = aligned_loops(mesh, loop_a, loop_b)?;
    let group = params.group.unwrap_or_else(|| new_group(mesh));
    let close = |v: &[u32]| {
        let mut c = v.to_vec();
        c.push(v[0]);
        c
    };
    if params.shape == LoftShape::Minimal {
        let tris = strip(mesh, &close(&p), &close(&q)).ok_or_else(|| EditError::Failed("no valid strip between the openings".into()))?;
        return commit(mesh, tris, Vec::new(), group);
    }

    let h = params.edge_length.unwrap_or_else(|| 0.5 * (mean_edge(mesh, &p, true) + mean_edge(mesh, &q, true))).max(1e-300);
    let perimeter = |c: &[u32]| mean_edge(mesh, c, true) * c.len() as f64;
    let n = ((perimeter(&p).max(perimeter(&q)) / h).round() as usize).clamp(3, 10_000);
    // A's points start at p[0]; B's are shifted by the twist.
    let twist = params.twist_deg.rem_euclid(360.0) / 360.0;
    let (rp, dp) = resample_with_directions(mesh, &p, n, 0.0);
    let (rq, dq) = resample_with_directions(mesh, &q, n, twist);

    // Column tangents (Hermite), from each end's continuity.
    let smooth = params.shape == LoftShape::Smooth;
    let tangents: Vec<(DVec3, DVec3)> = (0..n)
        .map(|i| {
            let d = rq[i] - rp[i];
            if !smooth {
                return (d, d);
            }
            let len = d.length();
            let end = |e: &LoftEnd, dir: DVec3| (e.continuity == EndContinuity::G1 && dir != DVec3::ZERO).then(|| dir * e.tension * len);
            // B's wall direction points out of B's surface; the tube arrives against it.
            match (end(&params.ends[0], dp[i]), end(&params.ends[1], -dq[i])) {
                (Some(t0), Some(t1)) => (t0, t1),
                // A free end has zero curvature: T = (3d − T_other) / 2.
                (Some(t0), None) => (t0, (3.0 * d - t0) * 0.5),
                (None, Some(t1)) => ((3.0 * d - t1) * 0.5, t1),
                (None, None) => (d, d),
            }
        })
        .collect();
    let hermite = |i: usize, t: f64| {
        let (t2, t3) = (t * t, t * t * t);
        let (h00, h10, h01, h11) = (2.0 * t3 - 3.0 * t2 + 1.0, t3 - 2.0 * t2 + t, -2.0 * t3 + 3.0 * t2, t3 - t2);
        rp[i] * h00 + tangents[i].0 * h10 + rq[i] * h01 + tangents[i].1 * h11
    };
    // Rows along the tube: the length of its middle curve over the edge length.
    let length: f64 = {
        let mid = |t: f64| (0..n).map(|i| hermite(i, t)).sum::<DVec3>() / n as f64;
        (0..32).map(|k| mid(k as f64 / 32.0).distance(mid((k + 1) as f64 / 32.0))).sum()
    };
    let segs = ((length / h).round() as usize).clamp(1, 10_000);
    // B's chain starts at its vertex nearest the first row point, so the last band is untwisted.
    let start = (0..q.len()).min_by(|&x, &y| mesh.vertex(q[x]).distance_squared(rq[0]).total_cmp(&mesh.vertex(q[y]).distance_squared(rq[0]))).unwrap_or(0);
    let q: Vec<u32> = q[start..].iter().chain(&q[..start]).copied().collect();
    if segs == 1 {
        let tris = strip(mesh, &close(&p), &close(&q)).ok_or_else(|| EditError::Failed("no valid strip between the openings".into()))?;
        return commit(mesh, tris, Vec::new(), group);
    }

    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(segs - 1);
    let mut new_vertices = Vec::new();
    for r in 1..segs {
        let t = r as f64 / segs as f64;
        let pts: Vec<DVec3> = (0..n).map(|i| hermite(i, t)).collect();
        let c = mean(&pts);
        let scale = 1.0 + params.bulge * (t * std::f64::consts::PI).sin();
        let row: Vec<u32> = pts.iter().map(|&x| mesh.append_vertex(c + (x - c) * scale)).collect();
        new_vertices.extend_from_slice(&row);
        rows.push(row);
    }
    let first = strip(mesh, &close(&p), &close(&rows[0]));
    let last = strip(mesh, &close(&rows[rows.len() - 1]), &close(&q));
    let (Some(first), Some(last)) = (first, last) else {
        mesh.remove_isolated_vertices();
        return Err(EditError::Failed("no valid strip between the openings".into()));
    };
    let mut tris = first;
    for w in rows.windows(2) {
        tris.extend(quad_band(mesh, &close(&w[0]), &close(&w[1])));
    }
    tris.extend(last);
    commit(mesh, tris, new_vertices, group)
}

/// The direction the surface continues past each vertex of boundary chain `c` (closed):
/// in the plane of the boundary triangles, perpendicular to the boundary edges, away from
/// the surface. Smoothed along the loop so a noisy edge gives a steady direction.
pub(crate) fn wall_directions(mesh: &Mesh, c: &[u32]) -> Vec<DVec3> {
    let n = c.len();
    let edge_out = |a: u32, b: u32| -> DVec3 {
        let Some(e) = mesh.find_edge(a, b) else { return DVec3::ZERO };
        let t = mesh.edge_triangles(e)[0];
        let x = tri_other_vertex(mesh.triangle(t), a, b);
        if x == INVALID {
            return DVec3::ZERO;
        }
        let (pa, pb) = (mesh.vertex(a), mesh.vertex(b));
        let along = (pb - pa).normalize_or_zero();
        let out = (pa + pb) * 0.5 - mesh.vertex(x);
        (out - along * out.dot(along)).normalize_or_zero()
    };
    let edges: Vec<DVec3> = (0..n).map(|i| edge_out(c[i], c[(i + 1) % n])).collect();
    let mut d: Vec<DVec3> = (0..n).map(|i| (edges[(i + n - 1) % n] + edges[i]).normalize_or_zero()).collect();
    for _ in 0..3 {
        d = (0..n).map(|i| (d[(i + n - 1) % n] + 2.0 * d[i] + d[(i + 1) % n]).normalize_or_zero()).collect();
    }
    d
}

/// `n` points evenly spaced by arc length around closed chain `c`, starting `offset`
/// (a fraction of the perimeter) after its first vertex, with the wall direction there.
fn resample_with_directions(mesh: &Mesh, c: &[u32], n: usize, offset: f64) -> (Vec<DVec3>, Vec<DVec3>) {
    let pts = positions(mesh, c);
    let dirs = wall_directions(mesh, c);
    let m = pts.len();
    let mut cum = vec![0.0];
    for i in 0..m {
        cum.push(cum[i] + pts[i].distance(pts[(i + 1) % m]));
    }
    let total = cum[m].max(1e-300);
    let mut out = (Vec::with_capacity(n), Vec::with_capacity(n));
    for i in 0..n {
        let s = (offset + i as f64 / n as f64).rem_euclid(1.0) * total;
        let k = cum.partition_point(|&x| x <= s).clamp(1, m) - 1;
        let l = cum[k + 1] - cum[k];
        let f = if l > 0.0 { ((s - cum[k]) / l).clamp(0.0, 1.0) } else { 0.0 };
        out.0.push(pts[k].lerp(pts[(k + 1) % m], f));
        out.1.push(dirs[k].lerp(dirs[(k + 1) % m], f).normalize_or_zero());
    }
    out
}
