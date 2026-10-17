//! Add Tube: a tube between two surface points — straight, a spline leaving each surface
//! along its normal, or a route that keeps clear of the surface — closed at both ends so it
//! can be combined with the mesh by a boolean.

use mm_core::{primitives, Bvh, DVec3, Mesh};

/// How the tube's centreline is found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TubePath {
    /// Straight segment.
    Line,
    /// Hermite spline leaving each end along its surface normal.
    Spline,
    /// Spline relaxed to keep `collision_radius` from the surface; `minimal` starts from the
    /// straight line and pulls harder toward the shortest path.
    AutoRoute { minimal: bool },
}

/// A tube end: surface point and its outward normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TubeEnd {
    pub point: DVec3,
    pub normal: DVec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TubeParams {
    pub path: TubePath,
    /// The tube runs inside the surface (a channel) rather than outside (a handle/bypass).
    pub inside: bool,
    /// Straight tubes only: run along the start's normal instead of toward the end point.
    pub along_normal: bool,
    pub radius: f64,
    /// Radius at the far end (`None` = same as `radius`).
    pub end_radius: Option<f64>,
    /// Spline tangent length as a fraction of the distance between the ends.
    pub tangent: f64,
    pub collision_radius: f64,
    /// 0..1: how strongly a route is pulled shorter each step.
    pub length_scale: f64,
    pub solve_steps: usize,
    /// Samples along the centreline and around the tube.
    pub samples: usize,
    pub slices: usize,
}

impl Default for TubeParams {
    fn default() -> Self {
        Self {
            path: TubePath::Spline,
            inside: false,
            along_normal: false,
            radius: 1.0,
            end_radius: None,
            tangent: 0.5,
            collision_radius: 1.5,
            length_scale: 0.6,
            solve_steps: 100,
            samples: 64,
            slices: 24,
        }
    }
}

fn hermite(p0: DVec3, m0: DVec3, p1: DVec3, m1: DVec3, t: f64) -> DVec3 {
    let (t2, t3) = (t * t, t * t * t);
    p0 * (2.0 * t3 - 3.0 * t2 + 1.0) + m0 * (t3 - 2.0 * t2 + t) + p1 * (-2.0 * t3 + 3.0 * t2) + m1 * (t3 - t2)
}

/// Centreline from `a` to `b` (both ends included).
pub fn tube_path(mesh: &Mesh, bvh: &Bvh, a: TubeEnd, b: TubeEnd, p: &TubeParams) -> Vec<DVec3> {
    let n = p.samples.max(2);
    let side = if p.inside { -1.0 } else { 1.0 };
    let dist = a.point.distance(b.point).max(1e-12);
    let line = |end: DVec3| (0..=n).map(|i| a.point.lerp(end, i as f64 / n as f64)).collect::<Vec<_>>();
    let spline = || {
        let m0 = a.normal.normalize_or_zero() * side * p.tangent * dist * 3.0;
        let m1 = -b.normal.normalize_or_zero() * side * p.tangent * dist * 3.0;
        (0..=n).map(|i| hermite(a.point, m0, b.point, m1, i as f64 / n as f64)).collect::<Vec<_>>()
    };
    match p.path {
        TubePath::Line if p.along_normal => {
            let d = a.normal.normalize_or_zero() * side;
            line(a.point + d * (b.point - a.point).dot(d).abs().max(1e-9))
        }
        TubePath::Line => line(b.point),
        TubePath::Spline => spline(),
        TubePath::AutoRoute { minimal } => {
            let mut pts = if minimal { line(b.point) } else { spline() };
            let clear = p.collision_radius.max(0.0);
            let pull = p.length_scale.clamp(0.0, 1.0) * if minimal { 0.5 } else { 0.25 };
            // Elastic band: keep the ends, push samples out of the clearance zone on the
            // tube's side of the surface, and relax toward the neighbours' midpoint.
            for _ in 0..p.solve_steps {
                let prev = pts.clone();
                for i in 1..n {
                    let mid = (prev[i - 1] + prev[i + 1]) * 0.5;
                    let mut q = prev[i] + (mid - prev[i]) * pull;
                    // Near the ends the route must leave the surface, so the clearance ramps in.
                    let ramp = ((i.min(n - i)) as f64 / (0.15 * n as f64)).min(1.0);
                    if let Some(h) = bvh.nearest(mesh, q, clear * 4.0 + 1e-12) {
                        let nrm = mesh.tri_normal(h.tid);
                        let off = (q - h.point).dot(nrm) * side;
                        let want = clear * ramp;
                        if off < want {
                            q += nrm * side * (want - off);
                        }
                    }
                    pts[i] = q;
                }
            }
            pts
        }
    }
}

/// Closed tube around `path`, radius going linearly from `r0` to `r1`.
pub fn tube_mesh(path: &[DVec3], r0: f64, r1: f64, slices: usize) -> Mesh {
    let n = path.len().max(2);
    let radii: Vec<f64> = (0..n).map(|i| r0 + (r1 - r0) * i as f64 / (n - 1) as f64).collect();
    let mut m = primitives::tube_along(path, &radii, slices.max(3));
    // Cap both ends with a fan at the end point.
    for &end in &[0usize, n - 1] {
        let loops = mm_core::topology::boundary_loops(&m);
        let Some(lp) = loops.into_iter().min_by(|x, y| {
            let c = |l: &mm_core::topology::EdgeLoop| l.positions(&m).iter().sum::<DVec3>() / l.vertices.len() as f64;
            c(x).distance(path[end]).total_cmp(&c(y).distance(path[end]))
        }) else {
            break;
        };
        let centre = m.append_vertex(path[end]);
        let vs = &lp.vertices;
        for i in 0..vs.len() {
            // Boundary loops run opposite to the faces next to them.
            let _ = m.append_triangle([vs[(i + 1) % vs.len()], vs[i], centre], 0);
        }
    }
    if mm_core::measure::volume(&m) < 0.0 {
        m.reverse_orientation();
    }
    m
}

/// The tube for `a` → `b`: path, then a closed tube extended a little past both ends so a
/// boolean with the surface cuts cleanly.
pub fn add_tube(mesh: &Mesh, bvh: &Bvh, a: TubeEnd, b: TubeEnd, p: &TubeParams) -> (Vec<DVec3>, Mesh) {
    let mut path = tube_path(mesh, bvh, a, b, p);
    let r0 = p.radius.max(1e-9);
    let r1 = p.end_radius.filter(|&r| r > 0.0).unwrap_or(r0);
    if path.len() >= 2 {
        let k = path.len();
        let d0 = (path[0] - path[1]).normalize_or_zero();
        let d1 = (path[k - 1] - path[k - 2]).normalize_or_zero();
        path.insert(0, path[0] + d0 * r0);
        path.push(path[path.len() - 1] + d1 * r1);
    }
    let m = tube_mesh(&path, r0, r1, p.slices);
    (path, m)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two surface points 90° apart (+X and +Y extremes).
    fn ends(m: &Mesh) -> (TubeEnd, TubeEnd) {
        let ex = m.vertex_ids().max_by(|&a, &b| m.vertex(a).x.total_cmp(&m.vertex(b).x)).unwrap();
        let ey = m.vertex_ids().max_by(|&a, &b| m.vertex(a).y.total_cmp(&m.vertex(b).y)).unwrap();
        let n = m.compute_vertex_normals();
        (TubeEnd { point: m.vertex(ex), normal: n[ex as usize] }, TubeEnd { point: m.vertex(ey), normal: n[ey as usize] })
    }

    #[test]
    fn tubes_are_closed_and_follow_their_side() {
        let s = primitives::icosphere(2.0, 3);
        let bvh = Bvh::build(&s);
        let (a, b) = ends(&s);
        for path in [TubePath::Line, TubePath::Spline, TubePath::AutoRoute { minimal: false }, TubePath::AutoRoute { minimal: true }] {
            for inside in [false, true] {
                let p = TubeParams { path, inside, radius: 0.2, collision_radius: 0.4, ..Default::default() };
                let (line, tube) = add_tube(&s, &bvh, a, b, &p);
                tube.check_validity().unwrap();
                assert!(tube.is_closed(), "{path:?} inside {inside}: open tube");
                assert!(mm_core::measure::volume(&tube) > 0.0);
                // The middle of the centreline is on the requested side of the sphere.
                let mid = line[line.len() / 2].length();
                match (path, inside) {
                    (TubePath::Line, _) => assert!(mid < 2.0),
                    (_, false) => assert!(mid > 2.0, "{path:?} outside: |mid| {mid}"),
                    (_, true) => assert!(mid < 2.0, "{path:?} inside: |mid| {mid}"),
                }
            }
        }
    }

    #[test]
    fn route_keeps_clear_of_the_surface() {
        let s = primitives::icosphere(2.0, 4);
        let bvh = Bvh::build(&s);
        let (a, b) = ends(&s);
        let p = TubeParams { path: TubePath::AutoRoute { minimal: true }, collision_radius: 0.5, ..Default::default() };
        let line = tube_path(&s, &bvh, a, b, &p);
        let n = line.len();
        for (i, q) in line.iter().enumerate().take(n - n / 5).skip(n / 5) {
            assert!(q.length() > 2.0 + 0.4, "sample {i} at {}", q.length());
        }
    }
}
