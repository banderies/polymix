#![allow(dead_code)]
//! Shared helpers for the mm-remesh integration tests.

use mm_core::geom::closest_point_on_triangle;
use mm_core::{Bvh, DVec3, Mesh, INVALID};

/// Tiny deterministic PRNG.
pub struct Lcg(pub u64);
impl Lcg {
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 11
    }
    /// Uniform in [0, 1).
    pub fn f(&mut self) -> f64 {
        (self.next_u64() as f64) / ((1u64 << 53) as f64)
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

pub fn assert_valid(m: &Mesh) {
    if let Err(e) = m.check_validity() {
        panic!("invalid mesh: {e}");
    }
    assert!(m.is_consistently_oriented(), "orientation broken");
}

/// Distance from `p` to the surface of `m`.
pub fn dist_to(m: &Mesh, bvh: &Bvh, p: DVec3) -> f64 {
    bvh.nearest(m, p, f64::INFINITY).map(|h| h.dist_sq.sqrt()).unwrap_or(f64::INFINITY)
}

/// Random points on the surface (plus all vertices).
pub fn surface_samples(m: &Mesh, per_tri: usize, rng: &mut Lcg) -> Vec<DVec3> {
    let mut out: Vec<DVec3> = m.vertex_ids().map(|v| m.vertex(v)).collect();
    for t in m.triangle_ids() {
        let [a, b, c] = m.tri_positions(t);
        for _ in 0..per_tri {
            let (mut u, mut v) = (rng.f(), rng.f());
            if u + v > 1.0 {
                u = 1.0 - u;
                v = 1.0 - v;
            }
            out.push(a + (b - a) * u + (c - a) * v);
        }
    }
    out
}

/// One-sided Hausdorff distance from samples of `a` to surface `b`.
pub fn hausdorff_one_sided(a: &Mesh, b: &Mesh, per_tri: usize, seed: u64) -> f64 {
    let bvh = Bvh::build(b);
    let mut rng = Lcg(seed);
    surface_samples(a, per_tri, &mut rng).iter().map(|&p| dist_to(b, &bvh, p)).fold(0.0, f64::max)
}

/// Snapshot of everything about triangles outside `region` (for bit-identity checks).
pub type TriRecord = (u32, [u32; 3], [u32; 3], i32, [[u64; 3]; 3]);
#[derive(Debug, PartialEq)]
pub struct OutsideSnapshot(pub Vec<TriRecord>);

pub fn snapshot_outside(m: &Mesh, region: &[u32]) -> OutsideSnapshot {
    let mut inr = vec![false; m.max_tid()];
    for &t in region {
        inr[t as usize] = true;
    }
    let mut v = Vec::new();
    for t in m.triangle_ids() {
        if inr[t as usize] {
            continue;
        }
        let tv = m.triangle(t);
        let pos = tv.map(|x| {
            let p = m.vertex(x);
            [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()]
        });
        v.push((t, tv, m.tri_edges(t), m.tri_group(t), pos));
    }
    OutsideSnapshot(v)
}

/// Original boundary segments of a mesh.
pub fn boundary_segments(m: &Mesh) -> Vec<(DVec3, DVec3)> {
    m.boundary_edge_ids()
        .map(|e| {
            let [a, b] = m.edge_vertices(e);
            (m.vertex(a), m.vertex(b))
        })
        .collect()
}

pub fn dist_to_segments(p: DVec3, segs: &[(DVec3, DVec3)]) -> f64 {
    segs.iter()
        .map(|&(a, b)| {
            let d = b - a;
            let t = ((p - a).dot(d) / d.length_squared()).clamp(0.0, 1.0);
            (a + d * t).distance(p)
        })
        .fold(f64::INFINITY, f64::min)
}

pub fn boundary_vertices(m: &Mesh) -> Vec<u32> {
    m.vertex_ids().filter(|&v| m.is_boundary_vertex(v)).collect()
}

/// Distance from `p` to triangle `t` of `m`.
pub fn dist_to_tri(m: &Mesh, t: u32, p: DVec3) -> f64 {
    let [a, b, c] = m.tri_positions(t);
    closest_point_on_triangle(p, a, b, c).0.distance(p)
}

/// Triangles adjacent to the triangle list that are not INVALID.
pub fn tri_set(m: &Mesh, tris: &[u32]) -> Vec<bool> {
    let mut s = vec![false; m.max_tid()];
    for &t in tris {
        assert!(m.is_triangle(t), "dead id {t} in returned region");
        assert!(t != INVALID);
        s[t as usize] = true;
    }
    s
}

pub fn min_quality(m: &Mesh) -> f64 {
    m.triangle_ids().map(|t| mm_remesh::triangle_quality(m, t)).fold(1.0, f64::min)
}

pub fn mean_quality(m: &Mesh) -> f64 {
    m.triangle_ids().map(|t| mm_remesh::triangle_quality(m, t)).sum::<f64>() / m.triangle_count() as f64
}
