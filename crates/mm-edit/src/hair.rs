//! Grow Hair (Meshmixer's Select > Edit > Grow Hair): strands rooted at random points of the
//! selected faces, grown along the surface normal, bent by gravity according to their
//! stiffness (which varies from root to tip), curled into a helix, optionally kept out of
//! the object, and finally turned into tapered tubes.

use crate::tube_mesh;
use glam::DVec3;
use mm_core::{Bvh, Mesh};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct HairParams {
    /// Strands.
    pub count: usize,
    /// Points along each strand.
    pub points: usize,
    pub length: f64,
    /// Helix radius (0 = straight).
    pub curl_radius: f64,
    /// Length of one helix turn.
    pub curl_pitch: f64,
    /// 0..1 at the root and at the tip: how much a strand resists gravity.
    pub hair_stiffness: f64,
    pub tip_stiffness: f64,
    /// 0..1.
    pub gravity: f64,
    /// Direction gravity pulls.
    #[serde(skip)]
    #[schemars(skip)]
    pub down: DVec3,
    /// Push strands out of the object.
    pub collisions: bool,
    pub base_radius: f64,
    pub tip_radius: f64,
    pub seed: u64,
}

impl Default for HairParams {
    fn default() -> Self {
        Self {
            count: 200,
            points: 12,
            length: 10.0,
            curl_radius: 0.0,
            curl_pitch: 5.0,
            hair_stiffness: 0.8,
            tip_stiffness: 0.2,
            gravity: 0.3,
            down: -DVec3::Y,
            collisions: true,
            base_radius: 0.3,
            tip_radius: 0.1,
            seed: 7,
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// Smoothly interpolated normal at a point of triangle `t` (barycentric `b`).
fn normal_at(m: &Mesh, t: u32, b: DVec3) -> DVec3 {
    let tv = m.triangle(t);
    (m.vertex_normal(tv[0]) * b.x + m.vertex_normal(tv[1]) * b.y + m.vertex_normal(tv[2]) * b.z).normalize_or(m.tri_normal(t))
}

/// Strand polylines rooted on `tris` of `mesh`.
pub fn hair_curves(mesh: &Mesh, bvh: &Bvh, tris: &[u32], p: &HairParams) -> Vec<Vec<DVec3>> {
    let tris: Vec<u32> = tris.iter().copied().filter(|&t| mesh.is_triangle(t) && mesh.tri_area(t) > 0.0).collect();
    if tris.is_empty() || p.count == 0 || p.points < 2 || p.length.is_nan() || p.length <= 0.0 {
        return Vec::new();
    }
    let mut cum = Vec::with_capacity(tris.len());
    let mut total = 0.0;
    for &t in &tris {
        total += mesh.tri_area(t);
        cum.push(total);
    }
    let mut rng = Rng(p.seed.max(1));
    let down = p.down.normalize_or(-DVec3::Y);
    let seg = p.length / (p.points - 1) as f64;
    let clearance = p.base_radius.max(p.tip_radius) * 1.5;
    (0..p.count)
        .map(|_| {
            let t = tris[cum.partition_point(|&c| c < rng.next() * total).min(tris.len() - 1)];
            let (mut s1, mut s2) = (rng.next(), rng.next());
            if s1 + s2 > 1.0 {
                s1 = 1.0 - s1;
                s2 = 1.0 - s2;
            }
            let b = DVec3::new(1.0 - s1 - s2, s1, s2);
            let [a, c, d] = mesh.tri_positions(t);
            let root = a * b.x + c * b.y + d * b.z;
            let n = normal_at(mesh, t, b);
            // Curl frame around the root normal, with a random start angle.
            let (u, v) = mm_core::geom::orthonormal_basis(n);
            let phase = rng.next() * std::f64::consts::TAU;
            let mut dir = n;
            let mut spine = root;
            let mut pts = vec![root];
            for k in 1..p.points {
                let s = k as f64 / (p.points - 1) as f64;
                let stiff = p.hair_stiffness + (p.tip_stiffness - p.hair_stiffness) * s;
                // Gravity turns the strand toward `down` at every step, more where it is soft
                // (at full gravity and no stiffness, up to a full turn over the strand).
                let max_turn = std::f64::consts::TAU / (p.points - 1) as f64;
                let turn = (p.gravity.clamp(0.0, 1.0) * (1.0 - stiff.clamp(0.0, 1.0)) * max_turn).min(dir.angle_between(down));
                if turn > 0.0 {
                    let axis = dir.cross(down);
                    let axis = if axis.length_squared() > 1e-18 { axis.normalize() } else { mm_core::geom::orthonormal_basis(dir).0 };
                    dir = (glam::DQuat::from_axis_angle(axis, turn) * dir).normalize();
                }
                spine += dir * seg;
                let mut q = spine;
                if p.curl_radius > 0.0 && p.curl_pitch > 0.0 {
                    let a = phase + std::f64::consts::TAU * (s * p.length) / p.curl_pitch;
                    // The helix grows in from the root so the strand starts on the surface.
                    q += (u * a.cos() + v * a.sin() - u * phase.cos() - v * phase.sin()) * (p.curl_radius * s.min(1.0));
                }
                if p.collisions {
                    if let Some(h) = bvh.nearest(mesh, q, clearance * 4.0 + p.length) {
                        let nn = normal_at(mesh, h.tid, h.bary);
                        let off = (q - h.point).dot(nn);
                        if off < clearance {
                            q = h.point + nn * clearance;
                            spine = q;
                        }
                    }
                }
                pts.push(q);
            }
            pts
        })
        .collect()
}

/// Tapered tubes along the strands, as one mesh.
pub fn hair_mesh(curves: &[Vec<DVec3>], base_radius: f64, tip_radius: f64, slices: usize) -> Mesh {
    let mut m = Mesh::new();
    let tip = tip_radius.max(base_radius * 0.05).max(1e-9);
    for c in curves.iter().filter(|c| c.len() >= 2) {
        m.append_mesh(&tube_mesh(c, base_radius.max(1e-9), tip, slices), 0);
    }
    m
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn hair_grows_out_of_the_selection_and_droops() {
        let ball = primitives::icosphere(10.0, 3);
        let bvh = Bvh::build(&ball);
        let top: Vec<u32> = ball.triangle_ids().filter(|&t| ball.tri_centroid(t).z > 8.0).collect();
        let p = HairParams { count: 50, length: 6.0, gravity: 0.0, down: -DVec3::Z, ..Default::default() };
        let straight = hair_curves(&ball, &bvh, &top, &p);
        assert_eq!(straight.len(), 50);
        for c in &straight {
            assert_eq!(c.len(), p.points);
            assert!(c[0].z > 7.5, "rooted on the selection");
            // Without gravity or curl the strand runs along the normal: tip ≈ 16 from the centre.
            assert!((c[c.len() - 1].length() - 16.0).abs() < 0.5, "{}", c[c.len() - 1].length());
        }
        // Gravity (toward -z, sideways for strands near the top) lowers the tips.
        let droop = hair_curves(&ball, &bvh, &top, &HairParams { gravity: 1.0, ..p });
        let mean_z = |cs: &[Vec<DVec3>]| cs.iter().map(|c| c[c.len() - 1].z).sum::<f64>() / cs.len() as f64;
        assert!(mean_z(&droop) < mean_z(&straight) - 1.0);
        // Collisions keep every point (past the root) outside the ball's faceted surface,
        // which sags about 0.01 inside the radius.
        let side: Vec<u32> = ball.triangle_ids().filter(|&t| ball.tri_centroid(t).x > 8.0).collect();
        // Gravity pointing back into the ball.
        let soft = HairParams { gravity: 1.0, hair_stiffness: 0.0, tip_stiffness: 0.0, length: 25.0, down: -DVec3::X, ..p };
        let hang = hair_curves(&ball, &bvh, &side, &HairParams { collisions: true, ..soft });
        assert!(hang.iter().flat_map(|c| &c[1..]).all(|q| q.length() > 9.98), "{:?}", hang.iter().flat_map(|c| &c[1..]).map(|q| q.length()).fold(f64::MAX, f64::min));
        let through = hair_curves(&ball, &bvh, &side, &HairParams { collisions: false, ..soft });
        assert!(through.iter().flatten().any(|q| q.length() < 9.0), "without collisions strands sag into the ball");
        // Tubes.
        let m = hair_mesh(&straight, 0.3, 0.1, 6);
        m.check_validity().unwrap();
        assert!(m.is_closed());
        assert_eq!(mm_core::topology::connected_components(&m).len(), 50);
    }

    #[test]
    fn curls_wind_around_the_strand() {
        let plane = primitives::plane_grid(10.0, 10.0, 4, 4);
        let bvh = Bvh::build(&plane);
        let all: Vec<u32> = plane.triangle_ids().collect();
        let p = HairParams { count: 5, points: 60, length: 20.0, curl_radius: 1.0, curl_pitch: 5.0, gravity: 0.0, collisions: false, ..Default::default() };
        for c in hair_curves(&plane, &bvh, &all, &p) {
            // Four turns: the horizontal offset from the root reaches about twice the radius.
            let reach = c.iter().map(|q| (*q - c[0]).truncate().length()).fold(0.0, f64::max);
            assert!(reach > 1.5 && reach < 2.2, "{reach}");
            assert!((c[c.len() - 1].z - 20.0).abs() < 1e-6);
        }
    }
}
