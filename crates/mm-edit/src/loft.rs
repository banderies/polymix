//! Loft: connect two boundary loops with a tube, e.g. to rejoin a vessel across a gap.
//!
//! Each column of the tube (one point of A's loop to the matching point of B's) follows a
//! Hermite curve. At a G1 end the curve leaves along the surrounding wall, taken per
//! boundary vertex (so an obliquely cut vessel blends without a kink), with the end's
//! tension scaling the tangent; a G2 end also continues the vessel's bend (the curvature
//! of the centres of the rings of mesh behind the opening, adjusted per column for its
//! distance from the bend's centre), which makes the column quintic; at a G0 end the curve
//! is free (zero curvature there), so it only meets the opening. The cross-section blends from A's shape to B's along the way, optionally
//! rounded towards a circle in the middle, and the tube can be faired with the surroundings
//! held fixed (order 2 or 3, so G1 / G2 ends stay smooth).

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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, PartialOrd, Ord)]
pub enum EndContinuity {
    /// Position only: the tube meets the opening, with a crease if the directions differ.
    G0,
    /// Tangent: the tube leaves along the surrounding wall.
    #[default]
    G1,
    /// Curvature: the tube also continues the wall's bend.
    G2,
}

impl EndContinuity {
    fn level(self) -> usize {
        self as usize
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoftEnd {
    pub continuity: EndContinuity,
    /// Length of the end tangent relative to the gap (G1 / G2); larger keeps the tube
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
    /// 0..1: how far the middle cross-section is made circular (vessels with ragged
    /// openings).
    pub round: f64,
    /// 0..1: blend towards the tube faired with the surroundings held fixed
    /// ([`LoftShape::Smooth`] only).
    pub fair: f64,
    /// Face group of the tube; `None` = a new group.
    pub group: Option<i32>,
}

impl Default for LoftParams {
    fn default() -> Self {
        Self {
            shape: LoftShape::Smooth,
            ends: [LoftEnd::default(); 2],
            edge_length: None,
            twist_deg: 0.0,
            bulge: 0.0,
            round: 0.0,
            fair: 0.0,
            group: None,
        }
    }
}

/// Weight of the middle-of-the-tube adjustments (bulge, round) along it: 1 halfway, and
/// zero with zero first and second derivatives at the ends, so G1 / G2 ends are kept.
fn middle_weight(t: f64) -> f64 {
    (t * std::f64::consts::PI).sin().powi(3)
}

/// Connect boundary loops `loop_a` and `loop_b` (vertex lists of the same mesh, either
/// orientation, possibly different vertex counts) with a tube.
pub fn loft(mesh: &mut Mesh, loop_a: &[u32], loop_b: &[u32], params: &LoftParams) -> EditResult<BridgeResult> {
    let unit = |x: f64| (0.0..=1.0).contains(&x);
    if params.bulge.is_nan() || params.bulge <= -1.0 || params.ends.iter().any(|e| e.tension.is_nan() || e.tension <= 0.0) || !unit(params.round) || !unit(params.fair) {
        return Err(EditError::InvalidParameter("bulge must exceed −100 %, tension must be positive, round and fair lie in 0..1".into()));
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
    let smooth = params.shape == LoftShape::Smooth;
    let ea = EndData::sample(mesh, &p, n, 0.0, smooth);
    let eb = EndData::sample(mesh, &q, n, twist, smooth);

    let columns: Vec<Column> = (0..n).map(|i| Column::new(&ea, &eb, i, params, smooth)).collect();
    // Rows along the tube: the length of its middle curve over the edge length.
    let mid = |t: f64| columns.iter().map(|c| c.at(t)).sum::<DVec3>() / n as f64;
    let length: f64 = (0..32).map(|k| mid(k as f64 / 32.0).distance(mid((k + 1) as f64 / 32.0))).sum();
    let segs = ((length / h).round() as usize).clamp(1, 10_000);
    // B's chain starts at its vertex nearest the first row point, so the last band is untwisted.
    let start = (0..q.len()).min_by(|&x, &y| mesh.vertex(q[x]).distance_squared(eb.pos[0]).total_cmp(&mesh.vertex(q[y]).distance_squared(eb.pos[0]))).unwrap_or(0);
    let q: Vec<u32> = q[start..].iter().chain(&q[..start]).copied().collect();
    if segs == 1 {
        let tris = strip(mesh, &close(&p), &close(&q)).ok_or_else(|| EditError::Failed("no valid strip between the openings".into()))?;
        return commit(mesh, tris, Vec::new(), group);
    }

    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(segs - 1);
    let mut new_vertices = Vec::new();
    for r in 1..segs {
        let t = r as f64 / segs as f64;
        let pts = shape_row(columns.iter().map(|c| c.at(t)).collect(), (mid(t + 1e-3) - mid(t - 1e-3)).normalize_or_zero(), middle_weight(t), params);
        let row: Vec<u32> = pts.into_iter().map(|x| mesh.append_vertex(x)).collect();
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
    let result = commit(mesh, tris, new_vertices, group)?;
    if smooth && params.fair > 0.0 {
        fair_tube(mesh, &rows, params);
    }
    Ok(result)
}

/// Round and bulge one row of points about its centre, by `w` (the middle weight).
fn shape_row(pts: Vec<DVec3>, axis: DVec3, w: f64, params: &LoftParams) -> Vec<DVec3> {
    let c = mean(&pts);
    let radial = |x: DVec3| {
        let d = x - c;
        d - axis * d.dot(axis)
    };
    let radius = pts.iter().map(|&x| radial(x).length()).sum::<f64>() / pts.len() as f64;
    let round = params.round * w;
    let scale = 1.0 + params.bulge * w;
    pts.into_iter()
        .map(|x| {
            let r = radial(x);
            let circle = x - r + r.normalize_or_zero() * radius;
            c + (x.lerp(circle, round) - c) * scale
        })
        .collect()
}

/// Fair the tube's middle. The order follows the highest end continuity (at least 2, so
/// the tube does not sag like a soap film), and the first `order − 1` rows at each end stay
/// where the Hermite curves put them: they carry each end's own continuity (G0, G1 or G2)
/// into the solve at the tube's row spacing — the surrounding wall's rows are usually much
/// coarser, which leaves a kink when the solve meets the wall directly. Linear fairing
/// shrinks a tube, so each faired row gets its mean radius back (about its new centre). The
/// result is blended with the unfaired tube by `params.fair`.
fn fair_tube(mesh: &mut Mesh, rows: &[Vec<u32>], params: &LoftParams) {
    let levels = params.ends.map(|e| e.continuity.level());
    let order = (1 + levels[0].max(levels[1])).max(2);
    let hold = (order - 1).min(rows.len() / 2);
    let rows = &rows[hold..rows.len() - hold];
    let free: Vec<u32> = rows.iter().flatten().copied().collect();
    if free.is_empty() {
        return;
    }
    let before: Vec<DVec3> = free.iter().map(|&v| mesh.vertex(v)).collect();
    let radius = |mesh: &Mesh, row: &[u32]| {
        let pts = positions(mesh, row);
        let c = mean(&pts);
        (c, pts.iter().map(|p| p.distance(c)).sum::<f64>() / pts.len() as f64)
    };
    let radii: Vec<f64> = rows.iter().map(|r| radius(mesh, r).1).collect();
    let fp = mm_repair::FairParams { order: order as u32, ..Default::default() };
    if mm_repair::fair_region_with(mesh, &free, &fp, None).is_err() {
        return;
    }
    for (row, &r0) in rows.iter().zip(&radii) {
        let (c, r1) = radius(mesh, row);
        if r1 > 1e-300 {
            for &v in row {
                mesh.set_vertex(v, c + (mesh.vertex(v) - c) * (r0 / r1));
            }
        }
    }
    for (&v, b) in free.iter().zip(before) {
        let faired = mesh.vertex(v);
        mesh.set_vertex(v, b.lerp(faired, params.fair));
    }
}

/// Samples around one opening: points evenly spaced along the loop, the wall's direction
/// there and (for G2) the vessel's bend behind it.
struct EndData {
    pos: Vec<DVec3>,
    dir: Vec<DVec3>,
    bend: Option<Bend>,
}

impl EndData {
    fn sample(mesh: &Mesh, c: &[u32], n: usize, offset: f64, with_bend: bool) -> Self {
        let (mut dirs, depths) = wall_directions(mesh, c);
        let bend = if with_bend { vessel_bend(mesh, c) } else { None };
        let pts = positions(mesh, c);
        if let Some(b) = &bend {
            // A direction measured across the last row of triangles is the wall's direction
            // half a row back; turn it on by the bend over that half row.
            for k in 0..dirs.len() {
                dirs[k] = (dirs[k] + b.at(pts[k]) * (0.5 * depths[k])).normalize_or_zero();
            }
        }
        let m = pts.len();
        let mut cum = vec![0.0];
        for i in 0..m {
            cum.push(cum[i] + pts[i].distance(pts[(i + 1) % m]));
        }
        let total = cum[m].max(1e-300);
        let mut out = EndData { pos: Vec::with_capacity(n), dir: Vec::with_capacity(n), bend };
        for i in 0..n {
            let s = (offset + i as f64 / n as f64).rem_euclid(1.0) * total;
            let k = cum.partition_point(|&x| x <= s).clamp(1, m) - 1;
            let l = cum[k + 1] - cum[k];
            let f = if l > 0.0 { ((s - cum[k]) / l).clamp(0.0, 1.0) } else { 0.0 };
            out.pos.push(pts[k].lerp(pts[(k + 1) % m], f));
            out.dir.push(dirs[k].lerp(dirs[(k + 1) % m], f).normalize_or_zero());
        }
        out
    }

    /// Second derivative per unit arc length of the wall's column through sample `i`
    /// continued past the opening: the vessel's bend, tighter on the inside of it.
    fn bend_at(&self, i: usize) -> DVec3 {
        self.bend.as_ref().map_or(DVec3::ZERO, |b| b.at(self.pos[i]))
    }
}

/// A vessel's bend just behind an opening.
struct Bend {
    /// Centre of the opening.
    centre: DVec3,
    /// Towards the centre of the bend.
    normal: DVec3,
    curvature: f64,
}

impl Bend {
    /// Curvature vector of the wall's column through `p`: the bend is tighter on its inside.
    fn at(&self, p: DVec3) -> DVec3 {
        let offset = (p - self.centre).dot(self.normal);
        self.normal * (self.curvature / (1.0 - self.curvature * offset).max(0.2))
    }
}

/// One column of the tube: a quintic Hermite curve (a cubic when neither end is G2).
struct Column {
    p: [DVec3; 2],
    t: [DVec3; 2],
    a: [DVec3; 2],
}

impl Column {
    fn new(ea: &EndData, eb: &EndData, i: usize, params: &LoftParams, smooth: bool) -> Self {
        let (p0, p1) = (ea.pos[i], eb.pos[i]);
        let d = p1 - p0;
        let len = d.length();
        // Free (G0) ends of a cubic have zero curvature: T = (3d − T_other) / 2.
        let (t0, t1) = if !smooth {
            (d, d)
        } else {
            // Tangent lengths: the length of the circular arc leaving A and arriving at B in
            // the end directions (the chord when an end is free), so tension 1 follows an arc.
            let travel = [ea.dir[i], -eb.dir[i]];
            let free = params.ends.iter().any(|e| e.continuity == EndContinuity::G0);
            let turn = travel[0].angle_between(travel[1]);
            let arc = if free || !turn.is_finite() || turn < 1e-6 { len } else { len * (0.5 * turn) / (0.5 * turn).sin() };
            let end = |e: &LoftEnd, dir: DVec3| (e.continuity != EndContinuity::G0 && dir != DVec3::ZERO).then(|| dir * e.tension * arc);
            // B's wall direction points out of B's surface; the tube arrives against it.
            match (end(&params.ends[0], ea.dir[i]), end(&params.ends[1], -eb.dir[i])) {
                (Some(t0), Some(t1)) => (t0, t1),
                (Some(t0), None) => (t0, (3.0 * d - t0) * 0.5),
                (None, Some(t1)) => ((3.0 * d - t1) * 0.5, t1),
                (None, None) => (d, d),
            }
        };
        // Second derivatives: the cubic's, or the wall's bend at a G2 end (scaled from arc
        // length to the curve parameter by the tangent length squared). Arriving at B
        // reverses the parameter, which leaves the second derivative's sign unchanged.
        let mut a = [6.0 * d - 4.0 * t0 - 2.0 * t1, -6.0 * d + 2.0 * t0 + 4.0 * t1];
        if smooth {
            if params.ends[0].continuity == EndContinuity::G2 {
                a[0] = ea.bend_at(i) * t0.length_squared();
            }
            if params.ends[1].continuity == EndContinuity::G2 {
                a[1] = eb.bend_at(i) * t1.length_squared();
            }
        }
        Column { p: [p0, p1], t: [t0, t1], a }
    }

    fn at(&self, t: f64) -> DVec3 {
        let (t2, t3, t4, t5) = (t * t, t.powi(3), t.powi(4), t.powi(5));
        let h0 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
        let h1 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
        let h2 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5;
        let h3 = 0.5 * t3 - t4 + 0.5 * t5;
        let h4 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
        let h5 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
        self.p[0] * h0 + self.t[0] * h1 + self.a[0] * h2 + self.a[1] * h3 + self.t[1] * h4 + self.p[1] * h5
    }
}

/// The direction the surface continues past each vertex of boundary chain `c` (closed):
/// in the plane of the boundary triangles, perpendicular to the boundary edges, away from
/// the surface. Smoothed along the loop so a noisy edge gives a steady direction. Also
/// returns the depth of the boundary triangles behind each vertex (how far back the
/// direction is measured).
pub(crate) fn wall_directions(mesh: &Mesh, c: &[u32]) -> (Vec<DVec3>, Vec<f64>) {
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
        out - along * out.dot(along)
    };
    let edges: Vec<DVec3> = (0..n).map(|i| edge_out(c[i], c[(i + 1) % n])).collect();
    let depths = (0..n).map(|i| 0.5 * (edges[(i + n - 1) % n].length() + edges[i].length())).collect();
    let mut d: Vec<DVec3> = (0..n).map(|i| (edges[(i + n - 1) % n].normalize_or_zero() + edges[i].normalize_or_zero()).normalize_or_zero()).collect();
    for _ in 0..3 {
        d = (0..n).map(|i| (d[(i + n - 1) % n] + 2.0 * d[i] + d[(i + 1) % n]).normalize_or_zero()).collect();
    }
    (d, depths)
}

/// The bend of the vessel behind the opening `c`: the curvature of the path through the
/// centres of the rings of mesh behind it (rings by edge distance from the opening, out to
/// one and a half opening radii), from a least-squares parabola `c(s) = c₀ + s D + ½ s² K`
/// with `s` the arc length along the centres (negative behind the opening). `None` when the
/// rings give no usable direction (e.g. a hole in a flat sheet).
fn vessel_bend(mesh: &Mesh, c: &[u32]) -> Option<Bend> {
    let pts = positions(mesh, c);
    let c0 = mean(&pts);
    let radius = pts.iter().map(|p| p.distance(c0)).sum::<f64>() / pts.len() as f64;
    let mut seen: std::collections::HashSet<u32> = c.iter().copied().collect();
    let mut ring: Vec<u32> = c.to_vec();
    let (mut s, mut prev) = (0.0, c0);
    let mut samples: Vec<(f64, DVec3)> = Vec::new();
    for _ in 0..40 {
        let next: Vec<u32> = ring.iter().flat_map(|&u| mesh.vertex_neighbors(u)).filter(|&v| seen.insert(v)).collect();
        if next.is_empty() {
            break;
        }
        let centre = mean(&positions(mesh, &next));
        s -= centre.distance(prev);
        prev = centre;
        samples.push((s, centre - c0));
        ring = next;
        if -s > 1.5 * radius && samples.len() >= 3 {
            break;
        }
    }
    if samples.len() < 3 {
        return None;
    }
    // Normal equations of the two vector unknowns D, K.
    let (mut a11, mut a12, mut a22) = (0.0, 0.0, 0.0);
    let (mut b1, mut b2) = (DVec3::ZERO, DVec3::ZERO);
    for &(s, d) in &samples {
        let (f1, f2) = (s, 0.5 * s * s);
        a11 += f1 * f1;
        a12 += f1 * f2;
        a22 += f2 * f2;
        b1 += d * f1;
        b2 += d * f2;
    }
    let det = a11 * a22 - a12 * a12;
    if det.abs() <= 1e-300 {
        return None;
    }
    let d = (b1 * a22 - b2 * a12) / det;
    let k = (b2 * a11 - b1 * a12) / det;
    let speed2 = d.length_squared();
    if speed2 <= 1e-12 {
        return None;
    }
    let dn = d / speed2.sqrt();
    let kv = (k - dn * k.dot(dn)) / speed2;
    let curvature = kv.length();
    curvature.is_finite().then(|| Bend { centre: c0, normal: kv.normalize_or_zero(), curvature })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quintic_column_matches_the_cubic_without_g2() {
        // With the cubic's own second derivatives, the quintic form is the cubic Hermite curve.
        let c = Column { p: [DVec3::ZERO, DVec3::new(1.0, 2.0, 0.5)], t: [DVec3::new(0.0, 0.0, 3.0), DVec3::new(1.0, -1.0, 2.0)], a: [DVec3::ZERO; 2] };
        let d = c.p[1] - c.p[0];
        let c = Column { a: [6.0 * d - 4.0 * c.t[0] - 2.0 * c.t[1], -6.0 * d + 2.0 * c.t[0] + 4.0 * c.t[1]], ..c };
        for k in 0..=10 {
            let t = k as f64 / 10.0;
            let (t2, t3) = (t * t, t * t * t);
            let cubic = c.p[0] * (2.0 * t3 - 3.0 * t2 + 1.0) + c.t[0] * (t3 - 2.0 * t2 + t) + c.p[1] * (-2.0 * t3 + 3.0 * t2) + c.t[1] * (t3 - t2);
            assert!(c.at(t).distance(cubic) < 1e-12, "t = {t}");
        }
    }
}
