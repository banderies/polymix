//! Operations against a *target surface* (Meshmixer's Set as Target): project vertices onto
//! it, attract a selection to it, and rigidly register a mesh to it (ICP).

use mm_core::linalg::closest_rotation;
use mm_core::{Bvh, DMat3, DQuat, DVec3, Mesh};
use std::collections::HashSet;

/// Move each of `verts` to the nearest point of `target`, `offset` along the target's
/// normal there. With `max_angle_deg`, vertices whose normal differs from the target normal
/// by more than that stay put. Returns the number moved.
pub fn project_to_target(mesh: &mut Mesh, verts: &[u32], target: &Mesh, bvh: &Bvh, offset: f64, max_angle_deg: Option<f64>) -> usize {
    let cos = max_angle_deg.map(|a| a.to_radians().cos());
    let normals = cos.map(|_| mesh.compute_vertex_normals());
    let mut moved = 0;
    for &v in verts {
        if !mesh.is_vertex(v) {
            continue;
        }
        let Some(hit) = bvh.nearest(target, mesh.vertex(v), f64::INFINITY) else { continue };
        let n = target.tri_normal(hit.tid);
        if let (Some(c), Some(ns)) = (cos, &normals) {
            if ns[v as usize].dot(n) < c {
                continue;
            }
        }
        mesh.set_vertex(v, hit.point + n * offset);
        moved += 1;
    }
    moved
}

/// Parameters of [`attract_to_target`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AttractParams {
    /// Distance kept from the target, along its normal.
    pub offset: f64,
    /// 0..1: relaxation between projections (evens out the attracted surface).
    pub smooth: f64,
    /// Only attract vertices whose normal is within this angle of the target's.
    pub normal_threshold_deg: f64,
    /// Open-boundary vertices stay put.
    pub preserve_mesh_boundary: bool,
    /// Face-group border vertices stay put.
    pub preserve_group_boundaries: bool,
    pub iterations: usize,
}

impl Default for AttractParams {
    fn default() -> Self {
        Self { offset: 0.0, smooth: 0.5, normal_threshold_deg: 90.0, preserve_mesh_boundary: true, preserve_group_boundaries: false, iterations: 8 }
    }
}

/// Vertices of `selection` that [`attract_to_target`] moves.
pub fn attract_vertices(mesh: &Mesh, selection: &[u32], p: &AttractParams) -> Vec<u32> {
    let mut verts: Vec<u32> = selection.iter().filter(|&&t| mesh.is_triangle(t)).flat_map(|&t| mesh.triangle(t)).collect();
    verts.sort_unstable();
    verts.dedup();
    verts.retain(|&v| {
        let tris = mesh.vertex_triangles(v);
        let on_boundary = mesh.vertex_edges(v).iter().any(|&e| mesh.is_boundary_edge(e));
        let on_group_border = tris.iter().any(|&t| mesh.tri_group(t) != mesh.tri_group(tris[0]));
        !(p.preserve_mesh_boundary && on_boundary) && !(p.preserve_group_boundaries && on_group_border)
    });
    verts
}

/// Pull the selected faces onto `target`: alternate projection with a relaxation step so the
/// result follows the target without crumpling. Returns the vertices moved.
pub fn attract_to_target(mesh: &mut Mesh, selection: &[u32], target: &Mesh, bvh: &Bvh, p: &AttractParams) -> Vec<u32> {
    let verts = attract_vertices(mesh, selection, p);
    let free: HashSet<u32> = verts.iter().copied().collect();
    let iterations = p.iterations.max(1);
    for it in 0..iterations {
        project_to_target(mesh, &verts, target, bvh, p.offset, Some(p.normal_threshold_deg));
        if it + 1 == iterations || p.smooth <= 0.0 {
            break;
        }
        let relaxed: Vec<(u32, DVec3)> = verts
            .iter()
            .map(|&v| {
                let (sum, n) = mesh.vertex_neighbors(v).fold((DVec3::ZERO, 0), |(s, n), u| (s + mesh.vertex(u), n + 1));
                let q = mesh.vertex(v);
                (v, if n > 0 { q + (sum / n as f64 - q) * (0.5 * p.smooth.clamp(0.0, 1.0)) } else { q })
            })
            .collect();
        for (v, q) in relaxed {
            if free.contains(&v) {
                mesh.set_vertex(v, q);
            }
        }
    }
    verts
}

/// Parameters of [`icp_align`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IcpParams {
    /// Allow rotation (otherwise translation only).
    pub rotate: bool,
    pub max_iterations: usize,
    /// Stop once an iteration moves the points by less than this.
    pub tolerance: f64,
}

impl Default for IcpParams {
    fn default() -> Self {
        Self { rotate: true, max_iterations: 100, tolerance: 1e-6 }
    }
}

/// Result of [`icp_align`]: `x ↦ rotation · x + translation` moves the source onto the target.
#[derive(Clone, Copy, Debug)]
pub struct IcpResult {
    pub rotation: DQuat,
    pub translation: DVec3,
    pub iterations: usize,
    /// RMS distance of the (inlier) points to the target at the end.
    pub rms: f64,
}

impl IcpResult {
    pub fn apply(&self, x: DVec3) -> DVec3 {
        self.rotation * x + self.translation
    }
}

/// Rigidly register `points` to the surface `target` (iterative closest point, Kabsch
/// steps). Pairs farther than three times the median distance are ignored each iteration,
/// so parts of the source with no counterpart on the target don't drag the fit.
pub fn icp_align(points: &[DVec3], target: &Mesh, bvh: &Bvh, p: &IcpParams) -> IcpResult {
    let mut r = DQuat::IDENTITY;
    let mut t = DVec3::ZERO;
    let mut rms = 0.0;
    let mut iterations = 0;
    if points.is_empty() {
        return IcpResult { rotation: r, translation: t, iterations, rms };
    }
    for _ in 0..p.max_iterations.max(1) {
        iterations += 1;
        let pairs: Vec<(DVec3, DVec3)> = points
            .iter()
            .filter_map(|&x| {
                let y = r * x + t;
                bvh.nearest(target, y, f64::INFINITY).map(|h| (y, h.point))
            })
            .collect();
        if pairs.is_empty() {
            break;
        }
        let mut d: Vec<f64> = pairs.iter().map(|(a, b)| a.distance(*b)).collect();
        d.sort_by(f64::total_cmp);
        let cut = 3.0 * d[d.len() / 2] + 1e-12;
        let inliers: Vec<(DVec3, DVec3)> = pairs.into_iter().filter(|(a, b)| a.distance(*b) <= cut).collect();
        let n = inliers.len() as f64;
        rms = (inliers.iter().map(|(a, b)| a.distance_squared(*b)).sum::<f64>() / n).sqrt();
        let ca = inliers.iter().map(|x| x.0).sum::<DVec3>() / n;
        let cb = inliers.iter().map(|x| x.1).sum::<DVec3>() / n;
        let step_r = if p.rotate {
            let s = inliers.iter().fold(DMat3::ZERO, |acc, &(a, b)| {
                let (a, b) = (a - ca, b - cb);
                acc + DMat3::from_cols(b * a.x, b * a.y, b * a.z)
            });
            DQuat::from_mat3(&closest_rotation(s)).normalize()
        } else {
            DQuat::IDENTITY
        };
        // y ↦ step_r (y − ca) + cb, composed with the current transform.
        let step_t = cb - step_r * ca;
        r = (step_r * r).normalize();
        t = step_r * t + step_t;
        // Largest move this step, over the bounding points of the cloud.
        let moved = inliers.iter().map(|&(a, _)| (step_r * a + step_t).distance(a)).fold(0.0, f64::max);
        if moved < p.tolerance {
            break;
        }
    }
    IcpResult { rotation: r, translation: t, iterations, rms }
}

