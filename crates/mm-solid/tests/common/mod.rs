#![allow(dead_code)]
//! Shared helpers for the mm-solid integration tests.

use glam::DVec3;
use mm_core::{measure, topology, Bvh, Mesh};

/// Small deterministic xorshift PRNG.
pub struct Rng(pub u64);
impl Rng {
    pub fn new(seed: u64) -> Self {
        Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    /// Uniform in [0, 1).
    pub fn f(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, a: f64, b: f64) -> f64 {
        a + (b - a) * self.f()
    }
    pub fn vec(&mut self, s: f64) -> DVec3 {
        DVec3::new(self.range(-s, s), self.range(-s, s), self.range(-s, s))
    }
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Assert the mesh is a valid, closed, edge-manifold, consistently oriented surface
/// without bowtie vertices and with positive volume.
pub fn assert_solid(m: &Mesh, what: &str) {
    assert!(m.triangle_count() > 0, "{what}: empty");
    m.check_validity().unwrap_or_else(|e| panic!("{what}: invalid mesh: {e}"));
    assert!(m.is_closed(), "{what}: has {} boundary edges", m.boundary_edge_ids().count());
    assert!(m.is_consistently_oriented(), "{what}: inconsistent orientation");
    let bowties = m.vertex_ids().filter(|&v| m.is_bowtie_vertex(v)).count();
    assert_eq!(bowties, 0, "{what}: {bowties} bowtie vertices");
    assert!(measure::volume(m) > 0.0, "{what}: non-positive volume");
}

/// Assert manifold structure only (closed, oriented, no bowties), volume may be any sign.
pub fn assert_closed_manifold(m: &Mesh, what: &str) {
    m.check_validity().unwrap_or_else(|e| panic!("{what}: invalid mesh: {e}"));
    assert!(m.is_closed(), "{what}: has {} boundary edges", m.boundary_edge_ids().count());
    assert!(m.is_consistently_oriented(), "{what}: inconsistent orientation");
    let bowties = m.vertex_ids().filter(|&v| m.is_bowtie_vertex(v)).count();
    assert_eq!(bowties, 0, "{what}: {bowties} bowtie vertices");
}

/// Every closed component must have positive volume unless it is nested inside another.
pub fn component_count(m: &Mesh) -> usize {
    topology::connected_components(m).len()
}

/// Max and mean distance from the vertices of `out` to the surface of `reference`.
pub fn distance_to(out: &Mesh, reference: &Mesh) -> (f64, f64) {
    let bvh = Bvh::build(reference);
    let mut max: f64 = 0.0;
    let mut sum = 0.0;
    let mut n = 0;
    for v in out.vertex_ids() {
        let d = bvh.nearest(reference, out.vertex(v), f64::INFINITY).unwrap().dist_sq.sqrt();
        max = max.max(d);
        sum += d;
        n += 1;
    }
    (max, sum / n.max(1) as f64)
}

/// Minimum triangle angle in degrees.
pub fn min_angle_deg(m: &Mesh) -> f64 {
    let mut best = 180.0f64;
    for t in m.triangle_ids() {
        let [a, b, c] = m.tri_positions(t);
        for (p, q, r) in [(a, b, c), (b, c, a), (c, a, b)] {
            best = best.min((q - p).angle_between(r - p).to_degrees());
        }
    }
    best
}

/// Rebuild a mesh from a raw soup (as an STL would arrive): every triangle gets its own
/// vertices welded by exact position.
pub fn weld_soup(tris: &[[DVec3; 3]]) -> Mesh {
    let mut map = std::collections::HashMap::new();
    let mut pos = Vec::new();
    let mut idx = Vec::new();
    for t in tris {
        let mut f = [0u32; 3];
        for (k, p) in t.iter().enumerate() {
            let key = (p.x.to_bits(), p.y.to_bits(), p.z.to_bits());
            f[k] = *map.entry(key).or_insert_with(|| {
                pos.push(*p);
                (pos.len() - 1) as u32
            });
        }
        idx.push(f);
    }
    Mesh::from_triangles(&pos, &idx, None).0
}

pub fn triangles_of(m: &Mesh) -> Vec<[DVec3; 3]> {
    m.triangle_ids().map(|t| m.tri_positions(t)).collect()
}
