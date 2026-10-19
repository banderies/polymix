//! Loft: connect two boundary loops with a tube, e.g. to rejoin a vessel across a gap.
//!
//! Each column of the tube (one point of A's loop to the matching point of B's) follows a
//! Hermite curve. At a G1 end the curve leaves along the surrounding wall, taken per
//! boundary vertex (so an obliquely cut vessel blends without a kink), or straight out of
//! an opening in a surface (a handle), with the end's tension scaling the tangent; a G2 end also continues the vessel's bend (the curvature
//! of the centres of the rings of mesh behind the opening, adjusted per column for its
//! distance from the bend's centre), which makes the column quintic; at a G0 end the curve
//! is free (zero curvature there), so it only meets the opening. The cross-section blends from A's shape to B's along the way, optionally
//! rounded towards a circle in the middle, and the tube can be faired with the surroundings
//! held fixed (order 2 or 3, so G1 / G2 ends stay smooth).
//!
//! With *context* the directions and the bend come from the vessel's axis fitted further
//! behind each opening instead of its last row of triangles (which a ragged or lumpy opening
//! skews); with *blend* ragged openings are tidied first and the join is smoothed into the
//! vessels afterwards; *remesh* evens out the triangles of the join.

use crate::bridge::{aligned_loops, commit, mean_edge, quad_band, strip, strip_with, AlignedLoops, BridgeResult};
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

/// Which way a G1 / G2 end leaves its opening.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum EndDirection {
    /// Along the surrounding wall, per boundary vertex: an opening cut across a vessel.
    #[default]
    Wall,
    /// Straight out of the opening, along its normal: an opening in a surface, which the
    /// tube stands up from like a handle. The tube starts straight (no bend at G2).
    Normal,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LoftEnd {
    pub continuity: EndContinuity,
    /// Length of the end tangent relative to the gap (G1 / G2); larger keeps the tube
    /// following the end's direction for longer.
    pub tension: f64,
    pub direction: EndDirection,
}

impl Default for LoftEnd {
    fn default() -> Self {
        Self { continuity: EndContinuity::G1, tension: 1.0, direction: EndDirection::Wall }
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
    /// 0..1: how much of the vessel behind each opening sets the direction a G1 / G2 end
    /// leaves in and its bend ([`EndDirection::Wall`] only): 0 is the edge of the opening
    /// alone (its last row of triangles; the bend from one and a half radii back), more takes
    /// the axis fitted to the vessel further back — its direction from a quarter on, and from
    /// there out to five diameters of vessel at 1. A ragged or lumpy opening then no longer
    /// steers the tube.
    pub context: f64,
    /// 0..1: blend the tube into the vessels. Above 0 each opening is tidied first (narrow
    /// teeth cut off, narrow notches filled, its outline relaxed along the wall), so a ragged
    /// opening does not fold the tube's first band; afterwards the tube and a band of each
    /// vessel behind its opening (a diameter back at 1) are smoothed, fading out across the
    /// band, so the join has no seam and the lumps right at it are evened out.
    pub blend: f64,
    /// Remesh the tube and the blended bands to even triangles of the tube's edge length
    /// (the result's triangles are then all of them).
    pub remesh: bool,
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
            context: 0.0,
            blend: 0.0,
            remesh: false,
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
    if params.bulge.is_nan() || params.bulge <= -1.0 || params.ends.iter().any(|e| e.tension.is_nan() || e.tension <= 0.0) || !unit(params.round) || !unit(params.fair) || !unit(params.context) || !unit(params.blend) {
        return Err(EditError::InvalidParameter("bulge must exceed −100 %, tension must be positive, round, fair, context and blend lie in 0..1".into()));
    }
    let (loop_a, loop_b) = if params.blend > 0.0 && params.shape != LoftShape::Minimal {
        let relax = (5.0 + 25.0 * params.blend).round() as usize;
        (tidy_opening(mesh, loop_a, relax)?, tidy_opening(mesh, loop_b, relax)?)
    } else {
        (loop_a.to_vec(), loop_b.to_vec())
    };
    let (loop_a, loop_b) = (&loop_a[..], &loop_b[..]);
    let smooth = params.shape == LoftShape::Smooth;
    // The vessels' axes behind the openings (with some context, along the wall).
    let axis = |c: &[u32], e: &LoftEnd| (smooth && params.context > 0.0 && e.direction == EndDirection::Wall).then(|| vessel_axis(mesh, c, context_reach(params.context))).flatten();
    let axes = [axis(loop_a, &params.ends[0]), axis(loop_b, &params.ends[1])];
    let dirs = [
        axes[0].as_ref().map_or_else(|| end_axis(mesh, loop_a, params.ends[0].direction), |x| x.dir),
        axes[1].as_ref().map_or_else(|| end_axis(mesh, loop_b, params.ends[1].direction), |x| x.dir),
    ];
    let AlignedLoops { p, q, .. } = aligned_loops(mesh, loop_a, loop_b, Some(dirs))?;
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
    // Points around: the openings' length smoothed (a ragged opening's notches would
    // otherwise crowd the tube with columns) over the edge length.
    let n = ((smoothed_perimeter(&positions(mesh, &p)).max(smoothed_perimeter(&positions(mesh, &q))) / h).round() as usize).clamp(3, 10_000);
    // A's points start at p[0]; B's are shifted by the twist.
    let twist = params.twist_deg.rem_euclid(360.0) / 360.0;
    let weight = context_weight(params.context);
    let mut ea = EndData::sample(mesh, &p, n, 0.0, smooth, params.ends[0].direction, axes[0].as_ref().map(|x| (x, weight)));
    let mut eb = EndData::sample(mesh, &q, n, twist, smooth, params.ends[1].direction, axes[1].as_ref().map(|x| (x, weight)));
    // The columns run between smoothed openings; each opening's own raggedness is added
    // back near it only (see `end_detail`).
    let detail = [ea.smooth_profile(), eb.smooth_profile()];

    let columns: Vec<Column> = (0..n).map(|i| Column::new(&ea, &eb, i, params, smooth)).collect();
    // Rows along the tube: the length of its middle curve over the edge length.
    let mid = |t: f64| columns.iter().map(|c| c.at(t)).sum::<DVec3>() / n as f64;
    let length: f64 = (0..32).map(|k| mid(k as f64 / 32.0).distance(mid((k + 1) as f64 / 32.0))).sum();
    let segs = ((length / h).round() as usize).clamp(1, 10_000);
    // B's chain starts at its vertex nearest the first row point, so the last band is untwisted.
    let b0 = eb.pos[0] + detail[1][0];
    let start = (0..q.len()).min_by(|&x, &y| mesh.vertex(q[x]).distance_squared(b0).total_cmp(&mesh.vertex(q[y]).distance_squared(b0))).unwrap_or(0);
    let q: Vec<u32> = q[start..].iter().chain(&q[..start]).copied().collect();
    if segs == 1 {
        let tris = strip(mesh, &close(&p), &close(&q)).ok_or_else(|| EditError::Failed("no valid strip between the openings".into()))?;
        return commit(mesh, tris, Vec::new(), group);
    }

    let mut rows: Vec<Vec<u32>> = Vec::with_capacity(segs - 1);
    let mut new_vertices = Vec::new();
    for r in 1..segs {
        let t = r as f64 / segs as f64;
        let (wa, wb) = (end_detail(t, segs), end_detail(1.0 - t, segs));
        let pts = columns.iter().enumerate().map(|(i, c)| c.at(t) + detail[0][i] * wa + detail[1][i] * wb).collect();
        let pts = shape_row(pts, (mid(t + 1e-3) - mid(t - 1e-3)).normalize_or_zero(), middle_weight(t), params);
        let row: Vec<u32> = pts.into_iter().map(|x| mesh.append_vertex(x)).collect();
        new_vertices.extend_from_slice(&row);
        rows.push(row);
    }
    // The end bands avoid triangles facing against the surface they join (folded over a
    // ragged opening's tooth or notch) wherever the rows allow.
    let unfolded = |t: [u32; 3]| {
        let pts = t.map(|v| mesh.vertex(v));
        let normal = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        let surface: DVec3 = t.iter().filter(|&&v| !mesh.vertex_triangles(v).is_empty()).map(|&v| mesh.vertex_normal(v)).sum();
        if normal.dot(surface) < 0.0 { 1e3 * h } else { 0.0 }
    };
    let first = strip_with(mesh, &close(&p), &close(&rows[0]), &unfolded);
    let last = strip_with(mesh, &close(&rows[rows.len() - 1]), &close(&q), &unfolded);
    let (Some(first), Some(last)) = (first, last) else {
        mesh.remove_isolated_vertices();
        return Err(EditError::Failed("no valid strip between the openings".into()));
    };
    let mut tris = first;
    for w in rows.windows(2) {
        tris.extend(quad_band(mesh, &close(&w[0]), &close(&w[1])));
    }
    tris.extend(last);
    let mut result = commit(mesh, tris, new_vertices, group)?;
    if smooth && params.fair > 0.0 {
        fair_tube(mesh, &rows, params);
    }
    let band = if params.blend > 0.0 { blend_join(mesh, &result.triangles, [&p, &q], params.blend) } else { Vec::new() };
    if params.remesh {
        remesh_join(mesh, &mut result, band, h);
    }
    Ok(result)
}

/// Smooth the join between a new tube (`tube`) and the vessels at its `openings`: the tube
/// and a band of each vessel behind its opening, `blend` diameters wide, get Taubin λ|μ
/// steps (which smooth without shrinking) at full strength on the tube and its seams, fading
/// to nothing across the band (smootherstep of the distance along the surface). Other open
/// edges stay where they are. Returns the vessels' triangles in the bands.
fn blend_join(mesh: &mut Mesh, tube: &[u32], openings: [&[u32]; 2], blend: f64) -> Vec<u32> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap, HashSet};
    let tube_verts: HashSet<u32> = tube.iter().flat_map(|&t| mesh.triangle(t)).collect();
    let on_openings: HashSet<u32> = openings.iter().flat_map(|c| c.iter().copied()).collect();
    let mut weight: HashMap<u32, f64> = tube_verts.iter().map(|&v| (v, 1.0)).collect();
    for c in openings {
        let pts = positions(mesh, c);
        let c0 = mean(&pts);
        let width = blend * 2.0 * pts.iter().map(|p| p.distance(c0)).sum::<f64>() / pts.len() as f64;
        if width <= 0.0 {
            continue;
        }
        // Along the vessel only: not through the tube.
        let mut dist: HashMap<u32, f64> = c.iter().map(|&v| (v, 0.0)).collect();
        let mut heap: BinaryHeap<Reverse<(u64, u32)>> = c.iter().map(|&v| Reverse((0f64.to_bits(), v))).collect();
        while let Some(Reverse((bits, u))) = heap.pop() {
            let d = f64::from_bits(bits);
            if d > dist[&u] {
                continue;
            }
            let pu = mesh.vertex(u);
            for v in mesh.vertex_neighbors(u) {
                if tube_verts.contains(&v) && !on_openings.contains(&v) {
                    continue;
                }
                let nd = d + pu.distance(mesh.vertex(v));
                if nd < width && dist.get(&v).is_none_or(|&old| nd < old) {
                    dist.insert(v, nd);
                    heap.push(Reverse((nd.to_bits(), v)));
                }
            }
        }
        for (v, d) in dist {
            let x = d / width;
            let w = 1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x);
            let e = weight.entry(v).or_insert(0.0);
            *e = e.max(w);
        }
    }
    let tube_set: HashSet<u32> = tube.iter().copied().collect();
    let band: Vec<u32> = {
        let mut b: Vec<u32> = weight.keys().flat_map(|&v| mesh.vertex_triangles(v)).filter(|t| !tube_set.contains(t)).collect();
        b.sort_unstable();
        b.dedup();
        b
    };
    let movable: Vec<(u32, f64, Vec<u32>)> = weight
        .into_iter()
        .filter(|&(v, w)| w > 0.0 && !mesh.is_boundary_vertex(v))
        .map(|(v, w)| (v, w, mesh.vertex_neighbors(v).collect()))
        .collect();
    let iterations = (10.0 + 30.0 * blend).round() as usize;
    for _ in 0..iterations {
        for step in [0.5, -0.53] {
            let moved: Vec<DVec3> = movable
                .iter()
                .map(|(v, w, nb)| {
                    let p = mesh.vertex(*v);
                    let avg = nb.iter().map(|&u| mesh.vertex(u)).sum::<DVec3>() / nb.len().max(1) as f64;
                    p + (avg - p) * (step * w)
                })
                .collect();
            for ((v, _, _), p) in movable.iter().zip(moved) {
                mesh.set_vertex(*v, p);
            }
        }
    }
    band
}

/// Remesh a tube and the bands around it (`band`) to even triangles of edge length `h`,
/// the region's border held; the tube's own group border stays a curve. `result` then lists
/// every triangle of the remeshed region and the vertices inside it.
fn remesh_join(mesh: &mut Mesh, result: &mut BridgeResult, band: Vec<u32>, h: f64) {
    let mut region = result.triangles.clone();
    region.extend(band);
    let params = mm_remesh::RemeshParams { target_edge_length: h, iterations: 8, region: Some(region), ..mm_remesh::RemeshParams::with_target(h) };
    let Ok(report) = mm_remesh::remesh(mesh, &params, None) else { return };
    let Some(tris) = report.region_triangles else { return };
    let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
    let mut inside: Vec<u32> = tris.iter().flat_map(|&t| mesh.triangle(t)).filter(|&v| mesh.vertex_triangles(v).iter().all(|t| set.contains(t))).collect();
    inside.sort_unstable();
    inside.dedup();
    result.triangles = tris;
    result.new_vertices = inside;
}

/// How much of an opening's raggedness a row at `t` (0 at the opening) keeps, with `segs`
/// rows in all: all of it at the opening, none from the third row on, fading with zero
/// first and second derivatives at both ends so G1 / G2 ends are kept. Carried along the
/// tube, the notches of a ragged opening (the edge of a brush-selected region) fold the
/// rows over themselves where the tube's cross-section narrows.
fn end_detail(t: f64, segs: usize) -> f64 {
    let x = (t / (3.0 / segs as f64).min(0.4)).clamp(0.0, 1.0);
    1.0 - x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
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

/// Samples around one opening: points evenly spaced along the loop, the direction the tube
/// leaves in there and (for G2) the vessel's bend behind it.
struct EndData {
    pos: Vec<DVec3>,
    dir: Vec<DVec3>,
    bend: Option<Bend>,
}

impl EndData {
    /// `axis`: the vessel's fitted axis behind the opening and how much it sets the
    /// directions (0..1; the bend comes from it whenever it is given).
    fn sample(mesh: &Mesh, c: &[u32], n: usize, offset: f64, with_bend: bool, direction: EndDirection, axis: Option<(&Axis, f64)>) -> Self {
        let (mut dirs, depths) = match direction {
            EndDirection::Wall => wall_directions(mesh, c),
            EndDirection::Normal => (vec![opening_normal(mesh, c); c.len()], vec![0.0; c.len()]),
        };
        let bend = match axis {
            _ if !with_bend || direction != EndDirection::Wall => None,
            Some((x, _)) => Some(x.bend()),
            None => vessel_bend(mesh, c),
        };
        let pts = positions(mesh, c);
        if let Some(b) = &bend {
            // A direction measured across the last row of triangles is the wall's direction
            // half a row back; turn it on by the bend over that half row.
            for k in 0..dirs.len() {
                dirs[k] = (dirs[k] + b.at(pts[k]) * (0.5 * depths[k])).normalize_or_zero();
            }
        }
        if let Some((x, w)) = axis {
            for k in 0..dirs.len() {
                dirs[k] = dirs[k].lerp(x.wall_direction(pts[k]), w).normalize_or_zero();
            }
        }
        let m = pts.len();
        let cum = loop_parameter(&pts);
        let mut out = EndData { pos: Vec::with_capacity(n), dir: Vec::with_capacity(n), bend };
        for i in 0..n {
            let s = (offset + i as f64 / n as f64).rem_euclid(1.0);
            let k = cum.partition_point(|&x| x <= s).clamp(1, m) - 1;
            let l = cum[k + 1] - cum[k];
            let f = if l > 0.0 { ((s - cum[k]) / l).clamp(0.0, 1.0) } else { 0.0 };
            out.pos.push(pts[k].lerp(pts[(k + 1) % m], f));
            out.dir.push(dirs[k].lerp(dirs[(k + 1) % m], f).normalize_or_zero());
        }
        out
    }

    /// Smooth the sampled points around the opening, keeping its size and shape (Taubin
    /// λ|μ smoothing, which removes zigzags a few samples long without shrinking the
    /// loop); returns what was removed from each point.
    fn smooth_profile(&mut self) -> Vec<DVec3> {
        let n = self.pos.len();
        let raw = self.pos.clone();
        for k in 0..40 {
            let w = if k % 2 == 0 { 0.5 } else { -0.53 };
            let p = &self.pos;
            self.pos = (0..n).map(|i| p[i] + ((p[(i + n - 1) % n] + p[(i + 1) % n]) * 0.5 - p[i]) * w).collect();
        }
        raw.iter().zip(&self.pos).map(|(r, s)| *r - *s).collect()
    }

    /// Second derivative per unit arc length of the wall's column through sample `i`
    /// continued past the opening: the vessel's bend, tighter on the inside of it.
    fn bend_at(&self, i: usize) -> DVec3 {
        self.bend.as_ref().map_or(DVec3::ZERO, |b| b.at(self.pos[i]))
    }
}

/// Where each point of the closed loop `pts` lies around it: `m + 1` values from 0 at the
/// first point to 1 back at it. Mostly by angle about the loop's centre in its plane, so a
/// ragged opening (the edge of a brush-selected region, with notches) is sampled evenly
/// around rather than evenly along every notch: sampled by length, two ragged openings blend
/// into rows that fold over themselves. Where the angle runs backwards (in a notch) it is
/// held; a tenth of the parameter follows the length, so it always increases. By length
/// alone when the loop does not wind around its centre.
fn loop_parameter(pts: &[DVec3]) -> Vec<f64> {
    let m = pts.len();
    let mut length = vec![0.0];
    for i in 0..m {
        length.push(length[i] + pts[i].distance(pts[(i + 1) % m]));
    }
    let total = length[m].max(1e-300);
    let by_length: Vec<f64> = length.iter().map(|l| l / total).collect();
    let (c, normal) = (mean(pts), area_vector(pts).normalize_or_zero());
    if normal == DVec3::ZERO {
        return by_length;
    }
    let (e1, e2) = (normal.any_orthonormal_vector(), normal.cross(normal.any_orthonormal_vector()));
    let angle = |p: DVec3| (p - c).dot(e2).atan2((p - c).dot(e1));
    let mut turn = vec![0.0];
    for i in 0..m {
        let a = angle(pts[(i + 1) % m]) - angle(pts[i]);
        turn.push(turn[i] + (a + std::f64::consts::PI).rem_euclid(std::f64::consts::TAU) - std::f64::consts::PI);
    }
    if turn[m].abs() < std::f64::consts::PI {
        return by_length;
    }
    let sign = turn[m].signum();
    let mut held = 0.0f64;
    let by_angle: Vec<f64> = turn.iter().map(|t| {
        held = held.max(t * sign);
        held
    }).collect();
    let total = by_angle[m].max(1e-300);
    (0..=m).map(|k| 0.9 * by_angle[k] / total + 0.1 * by_length[k]).collect()
}

/// Tidy a ragged opening (the edge of a brush-deleted region) before lofting from it: a few
/// passes cut off its one-triangle teeth and fill its narrow notches, judged by the surface's
/// angle at each of its vertices (under 75°: a tooth; over 285°: a notch, closed with a
/// triangle across), then `relax` passes straighten its outline along the wall
/// ([`crate::smooth::smooth_boundary`]). A cut or fill that would pinch the surface at a
/// vertex (leaving it two open edges apart) is undone, so the opening stays one loop.
/// Returns the opening's loop afterwards.
fn tidy_opening(mesh: &mut Mesh, c: &[u32], relax: usize) -> EditResult<Vec<u32>> {
    let surface_angle = |mesh: &Mesh, v: u32| -> f64 {
        mesh.vertex_triangles(v)
            .iter()
            .map(|&t| {
                let tv = mesh.triangle(t);
                let k = tv.iter().position(|&x| x == v).unwrap_or(0);
                let (p, a, b) = (mesh.vertex(v), mesh.vertex(tv[(k + 1) % 3]), mesh.vertex(tv[(k + 2) % 3]));
                (a - p).angle_between(b - p)
            })
            .sum()
    };
    let pinched = |mesh: &Mesh, vs: &[u32]| vs.iter().any(|&v| mesh.is_vertex(v) && mm_repair::ops::vertex_fans(mesh, v).len() > 1);
    let mut lp = orient_boundary_loop(mesh, c)?;
    for _ in 0..3 {
        if lp.len() < 8 {
            break;
        }
        let mut changed = false;
        // Teeth: a loop vertex whose only triangle comes to a sharp point.
        for &v in &lp {
            let tris = mesh.vertex_triangles(v);
            if tris.len() != 1 || !mesh.is_vertex(v) || surface_angle(mesh, v) >= 75f64.to_radians() {
                continue;
            }
            let (t, tv) = (tris[0], mesh.triangle(tris[0]));
            let group = mesh.tri_group(t);
            if mesh.remove_triangle(t, false).is_err() {
                continue;
            }
            let others: Vec<u32> = tv.iter().copied().filter(|&x| x != v).collect();
            if pinched(mesh, &others) || others.iter().any(|&x| mesh.vertex_triangles(x).is_empty()) {
                let _ = mesh.append_triangle(tv, group);
            } else {
                mesh.remove_isolated_vertices();
                changed = true;
            }
        }
        let start = lp.iter().copied().find(|&v| mesh.is_vertex(v) && mesh.is_boundary_vertex(v)).ok_or_else(|| EditError::Failed("the opening closed up".into()))?;
        lp = boundary_loop_through(mesh, start)?;
        // Notches: a loop vertex the surface wraps nearly all the way round.
        let n = lp.len();
        let mut used = std::collections::HashSet::new();
        for i in 0..n {
            let (a, b, c) = (lp[(i + n - 1) % n], lp[i], lp[(i + 1) % n]);
            if used.contains(&a) || used.contains(&b) || used.contains(&c) || mesh.find_edge(a, c).is_some() || surface_angle(mesh, b) <= 285f64.to_radians() {
                continue;
            }
            let group = mesh.find_edge(a, b).map_or(0, |e| mesh.tri_group(mesh.edge_triangles(e)[0]));
            if let Ok(t) = mesh.append_triangle([b, a, c], group) {
                if pinched(mesh, &[a, b, c]) {
                    let _ = mesh.remove_triangle(t, false);
                } else {
                    used.extend([a, b, c]);
                    changed = true;
                }
            }
        }
        if !changed {
            break;
        }
        let start = lp.iter().copied().find(|&v| mesh.is_vertex(v) && mesh.is_boundary_vertex(v)).ok_or_else(|| EditError::Failed("the opening closed up".into()))?;
        lp = boundary_loop_through(mesh, start)?;
    }
    crate::smooth::smooth_boundary(mesh, &lp, relax)?;
    Ok(lp)
}

/// The boundary loop through boundary vertex `v`, in boundary orientation.
fn boundary_loop_through(mesh: &Mesh, v: u32) -> EditResult<Vec<u32>> {
    mm_core::topology::boundary_loops(mesh)
        .into_iter()
        .find(|l| l.vertices.contains(&v))
        .map(|l| l.vertices)
        .ok_or_else(|| EditError::InvalidLoop("the opening is no longer a boundary".into()))
}

/// How far back along a vessel [`LoftParams::context`] looks, in opening radii: one and a
/// half at none, ten (five diameters) at all.
fn context_reach(context: f64) -> f64 {
    1.5 + 8.5 * context
}

/// How much of the fitted axis sets the end directions: all of it from a context of a
/// quarter on.
fn context_weight(context: f64) -> f64 {
    (context / 0.25).min(1.0)
}

/// Length of the closed loop `pts` with its zigzags smoothed away (as [`EndData::smooth_profile`]
/// does): a ragged opening counts about as long as its outline.
fn smoothed_perimeter(pts: &[DVec3]) -> f64 {
    let n = pts.len();
    let mut p = pts.to_vec();
    for k in 0..40 {
        let w = if k % 2 == 0 { 0.5 } else { -0.53 };
        p = (0..n).map(|i| p[i] + ((p[(i + n - 1) % n] + p[(i + 1) % n]) * 0.5 - p[i]) * w).collect();
    }
    (0..n).map(|i| p[i].distance(p[(i + 1) % n])).sum()
}

/// A vessel's axis just behind an opening, fitted over a length of it: the axis is
/// `origin + s dir + ½ s² bend` with `s` the distance towards the opening (0 at it), and the
/// wall's radius changes by `taper` per unit of `s`.
struct Axis {
    origin: DVec3,
    /// Out of the opening (along the vessel, away from it).
    dir: DVec3,
    /// Curvature vector, towards the bend's centre.
    bend: DVec3,
    taper: f64,
}

impl Axis {
    /// The direction the wall continues in past wall point `p`: along the axis where `p`
    /// is (an obliquely cut opening reaches along it), tilted by the taper.
    fn wall_direction(&self, p: DVec3) -> DVec3 {
        let s = (p - self.origin).dot(self.dir);
        let t = (self.dir + self.bend * s).normalize_or_zero();
        let q = p - (self.origin + self.dir * s + self.bend * (0.5 * s * s));
        let radial = (q - t * q.dot(t)).normalize_or_zero();
        (t + radial * self.taper).normalize_or_zero()
    }

    fn bend(&self) -> Bend {
        Bend { centre: self.origin, normal: self.bend.normalize_or_zero(), curvature: self.bend.length() }
    }
}

/// The axis of the vessel behind opening `c` (closed), from slices of its wall by distance
/// along the surface from the opening, out to `reach` opening radii: a least-squares
/// parabola through the slices' centres (a line through fewer than five) and a line through
/// their radii. Stops where the wall no longer looks like the same vessel (a slice twice or
/// half the opening's radius: a branch, an aneurysm, the end of the mesh). `None` with fewer
/// than three slices.
fn vessel_axis(mesh: &Mesh, c: &[u32], reach: f64) -> Option<Axis> {
    use std::cmp::Reverse;
    use std::collections::{BinaryHeap, HashMap};
    let pts = positions(mesh, c);
    let c0 = mean(&pts);
    let radius = pts.iter().map(|p| p.distance(c0)).sum::<f64>() / pts.len() as f64;
    if radius <= 0.0 {
        return None;
    }
    let limit = reach * radius;
    // Distance along the surface from the opening (non-negative f64 bits order as the values).
    let mut dist: HashMap<u32, f64> = c.iter().map(|&v| (v, 0.0)).collect();
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = c.iter().map(|&v| Reverse((0f64.to_bits(), v))).collect();
    while let Some(Reverse((bits, u))) = heap.pop() {
        let d = f64::from_bits(bits);
        if d > dist[&u] || d > limit {
            continue;
        }
        let pu = mesh.vertex(u);
        for v in mesh.vertex_neighbors(u) {
            let nd = d + pu.distance(mesh.vertex(v));
            if nd <= limit && dist.get(&v).is_none_or(|&old| nd < old) {
                dist.insert(v, nd);
                heap.push(Reverse((nd.to_bits(), v)));
            }
        }
    }
    // Slices a twelfth of the reach wide, each vertex weighted by its share of area.
    const SLICES: usize = 12;
    let width = limit / SLICES as f64;
    let mut acc = vec![(0.0, DVec3::ZERO, 0.0); SLICES];
    let mut members: Vec<Vec<(DVec3, f64)>> = vec![Vec::new(); SLICES];
    for (&v, &d) in &dist {
        let k = ((d / width) as usize).min(SLICES - 1);
        let a: f64 = mesh.vertex_triangles(v).iter().map(|&t| mesh.tri_area(t)).sum::<f64>() / 3.0;
        let p = mesh.vertex(v);
        acc[k].0 += a;
        acc[k].1 += p * a;
        acc[k].2 += d * a;
        members[k].push((p, a));
    }
    // (position along the axis, centre, the slice's points with their areas)
    type Slice<'a> = (f64, DVec3, &'a [(DVec3, f64)]);
    let mut slices: Vec<Slice> = Vec::new();
    for k in 0..SLICES {
        let (a, sum, ds) = acc[k];
        if a <= 0.0 {
            break;
        }
        let centre = sum / a;
        let r = members[k].iter().map(|(p, w)| p.distance(centre) * w).sum::<f64>() / a;
        if !(0.5..=2.0).contains(&(r / radius)) {
            break;
        }
        slices.push((-ds / a, centre, &members[k]));
    }
    if slices.len() < 3 {
        return None;
    }
    // Weighted least squares for x(s) = X + s D (+ ½ s² K), a 3 × 3 (or 2 × 2) system with
    // vector right-hand sides.
    let quadratic = slices.len() >= 5;
    let m = if quadratic { 3 } else { 2 };
    let mut a = [[0.0f64; 3]; 3];
    let mut b = [DVec3::ZERO; 3];
    for &(s, x, _) in &slices {
        let f = [1.0, s, 0.5 * s * s];
        for i in 0..m {
            for (j, fj) in f.iter().take(m).enumerate() {
                a[i][j] += f[i] * fj;
            }
            b[i] += x * f[i];
        }
    }
    let sol = solve_small(&a, &b, m)?;
    let (origin, d, k) = (sol[0], sol[1], if quadratic { sol[2] } else { DVec3::ZERO });
    let speed2 = d.length_squared();
    if speed2 <= 1e-12 {
        return None;
    }
    let dir = d / speed2.sqrt();
    let bend = (k - dir * k.dot(dir)) / speed2;
    // The wall's radius about the axis per slice, and a line through it.
    let (mut sw, mut ss, mut sr, mut sss, mut ssr) = (0.0, 0.0, 0.0, 0.0, 0.0);
    for &(s, _, pts) in &slices {
        let on_axis = origin + d * s + k * (0.5 * s * s);
        let t = (d + k * s).normalize_or_zero();
        let (mut r, mut w) = (0.0, 0.0);
        for &(p, a) in pts {
            let q = p - on_axis;
            r += (q - t * q.dot(t)).length() * a;
            w += a;
        }
        let r = r / w;
        sw += 1.0;
        ss += s;
        sr += r;
        sss += s * s;
        ssr += s * r;
    }
    let det = sw * sss - ss * ss;
    let taper = if det > 1e-300 { ((sw * ssr - ss * sr) / det).clamp(-0.5, 0.5) } else { 0.0 };
    (dir.is_finite() && bend.is_finite()).then_some(Axis { origin, dir, bend, taper })
}

/// Solve the `m` × `m` (2 or 3) system `a x = b` with vector right-hand sides (Gaussian
/// elimination with partial pivoting); `None` when it is singular.
fn solve_small(a: &[[f64; 3]; 3], b: &[DVec3; 3], m: usize) -> Option<[DVec3; 3]> {
    let (mut a, mut b) = (*a, *b);
    for col in 0..m {
        let piv = (col..m).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in 0..m {
            if row != col {
                let f = a[row][col] / a[col][col];
                let pivot_row = a[col];
                for (x, p) in a[row].iter_mut().zip(pivot_row).take(m) {
                    *x -= f * p;
                }
                b[row] -= b[col] * f;
            }
        }
    }
    let mut x = [DVec3::ZERO; 3];
    for i in 0..m {
        x[i] = b[i] / a[i][i];
    }
    Some(x)
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

/// The overall direction a tube leaves opening `c` in (zero for a hole in a sheet left
/// along the wall, whose directions cancel out).
fn end_axis(mesh: &Mesh, c: &[u32], direction: EndDirection) -> DVec3 {
    match direction {
        EndDirection::Wall => wall_directions(mesh, c).0.iter().sum::<DVec3>().normalize_or_zero(),
        EndDirection::Normal => opening_normal(mesh, c),
    }
}

/// The direction straight out of opening `c` (closed): the mean normal of the triangles
/// around it, for a hole in a surface; where the surface wraps around the opening instead
/// (a vessel cut across, whose normals cancel out), the direction away from the surface
/// behind it.
fn opening_normal(mesh: &Mesh, c: &[u32]) -> DVec3 {
    let n = c.len();
    let (mut sum, mut total) = (DVec3::ZERO, 0.0);
    for i in 0..n {
        if let Some(e) = mesh.find_edge(c[i], c[(i + 1) % n]) {
            let t = mesh.edge_triangles(e)[0];
            let area = mesh.tri_area(t);
            sum += mesh.tri_normal(t) * area;
            total += area;
        }
    }
    if total > 0.0 && sum.length() > 0.5 * total {
        sum.normalize()
    } else {
        loop_outward_direction(mesh, c).unwrap_or(DVec3::ZERO)
    }
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

/// The fitted axis behind an opening for tests elsewhere in the crate: its direction and
/// curvature.
#[cfg(test)]
pub(crate) mod tests_support {
    use super::*;

    pub(crate) fn vessel_axis(mesh: &Mesh, c: &[u32], reach: f64) -> Option<(DVec3, f64)> {
        super::vessel_axis(mesh, c, reach).map(|x| (x.dir, x.bend.length()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loop_parameter_goes_around_by_angle() {
        let circle = |k: usize, n: usize| {
            let a = std::f64::consts::TAU * k as f64 / n as f64;
            DVec3::new(a.cos(), a.sin(), 0.0)
        };
        // A regular polygon: the same as by length.
        let poly: Vec<DVec3> = (0..12).map(|k| circle(k, 12)).collect();
        let u = loop_parameter(&poly);
        assert!(u.iter().enumerate().all(|(k, x)| (x - k as f64 / 12.0).abs() < 1e-12));
        // A notch (in and back out at one angle, as a ragged edge does) takes up a tenth of
        // the loop by length but only its length share of the tenth that follows the length.
        let mut notched: Vec<DVec3> = (0..8).map(|k| circle(k, 8)).collect();
        notched.splice(2..2, [circle(2, 8) * 0.2, circle(2, 8) * 0.6]);
        let u = loop_parameter(&notched);
        assert!(u.windows(2).all(|w| w[1] > w[0]) && (u[notched.len()] - 1.0).abs() < 1e-12, "{u:?}");
        assert!(u[4] - u[2] < 0.08, "the notch is passed quickly ({:.3})", u[4] - u[2]);
    }

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
