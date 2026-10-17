#![allow(dead_code)]

use mm_core::topology::{self, EdgeLoop};
use mm_core::{DVec3, Mesh};

/// Tiny deterministic PRNG.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
    /// Uniform in [-1, 1).
    pub fn sym(&mut self) -> f64 {
        self.next_u32() as f64 / (1u64 << 30) as f64 - 1.0
    }
}

/// Remove all triangles whose centroid lies within `radius` of `center`.
pub fn punch_hole(m: &mut Mesh, center: DVec3, radius: f64) -> usize {
    let tris: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).distance(center) < radius).collect();
    let n = tris.len();
    m.remove_triangles(tris);
    n
}

pub fn assert_valid(m: &Mesh) {
    if let Err(e) = m.check_validity() {
        panic!("invalid mesh: {e}");
    }
}

pub fn assert_watertight(m: &Mesh) {
    assert_valid(m);
    assert!(m.is_closed(), "mesh has {} boundary loops", topology::boundary_loops(m).len());
    assert!(m.is_consistently_oriented(), "inconsistent orientation");
    assert_eq!(topology::connected_components(m).len(), 1, "expected one component");
    assert!(m.vertex_ids().all(|v| !m.is_bowtie_vertex(v)), "bowtie vertex present");
    assert!(mm_core::measure::volume(m) > 0.0, "inward orientation");
}

pub fn loops(m: &Mesh) -> Vec<EdgeLoop> {
    topology::boundary_loops(m)
}

/// Best-fit-plane distance stats of the loop: returns (plane point, unit normal, max loop deviation).
pub fn loop_plane(m: &Mesh, lp: &EdgeLoop) -> (DVec3, DVec3, f64) {
    let pts = lp.positions(m);
    let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
    let mut n = DVec3::ZERO;
    for i in 0..pts.len() {
        n += (pts[i] - c).cross(pts[(i + 1) % pts.len()] - c);
    }
    let n = n.normalize();
    let dev = pts.iter().map(|p| (*p - c).dot(n).abs()).fold(0.0, f64::max);
    (c, n, dev)
}

/// Max and RMS of `f` over the given vertices.
pub fn max_rms(m: &Mesh, verts: &[u32], f: impl Fn(DVec3) -> f64) -> (f64, f64) {
    let mut mx: f64 = 0.0;
    let mut s = 0.0;
    for &v in verts {
        let d = f(m.vertex(v)).abs();
        mx = mx.max(d);
        s += d * d;
    }
    (mx, (s / verts.len().max(1) as f64).sqrt())
}
