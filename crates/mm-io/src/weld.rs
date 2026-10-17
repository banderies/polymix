//! Vertex welding: exact (bit-identical coordinates) and tolerance-based (spatial hash).

use glam::DVec3;
use smallvec::SmallVec;
use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash, Hasher};

/// Hasher for keys that pre-mix themselves into a single `write_u64`.
#[derive(Default)]
pub(crate) struct PremixedHasher(u64);

impl Hasher for PremixedHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
    #[inline]
    fn write_u64(&mut self, x: u64) {
        self.0 = x;
    }
    fn write(&mut self, bytes: &[u8]) {
        // Not used by our keys; FNV-1a keeps it correct anyway.
        for &b in bytes {
            self.0 = (self.0 ^ b as u64).wrapping_mul(0x100_0000_01b3);
        }
    }
}

pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<PremixedHasher>>;

#[inline]
fn mix3(a: u64, b: u64, c: u64) -> u64 {
    let mut h = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    h = (h.rotate_left(29) ^ b).wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    h = (h.rotate_left(29) ^ c).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^ (h >> 31)
}

/// Three-component hash key.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct Key3<T>(pub [T; 3]);

impl Hash for Key3<u32> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(mix3(self.0[0] as u64, self.0[1] as u64, self.0[2] as u64));
    }
}
impl Hash for Key3<u64> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(mix3(self.0[0], self.0[1], self.0[2]));
    }
}
impl Hash for Key3<i64> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_u64(mix3(self.0[0] as u64, self.0[1] as u64, self.0[2] as u64));
    }
}

/// Bit pattern of an `f32` with -0.0 folded into +0.0.
#[inline]
pub(crate) fn f32_key(x: f32) -> u32 {
    (x + 0.0).to_bits()
}

/// Bit pattern of an `f64` with -0.0 folded into +0.0.
#[inline]
pub(crate) fn f64_key(x: f64) -> u64 {
    (x + 0.0).to_bits()
}

#[inline]
pub(crate) fn dvec_key(p: DVec3) -> Key3<u64> {
    Key3([f64_key(p.x), f64_key(p.y), f64_key(p.z)])
}

/// Exact weld of `n` items by key. Returns `(remap, first)`: `remap[i]` is the unique id of
/// item `i` (ids assigned in order of first appearance) and `first[u]` is the first item
/// with unique id `u`.
pub(crate) fn weld_exact_by<K: Hash + Eq + Copy>(n: usize, key: impl Fn(usize) -> K) -> (Vec<u32>, Vec<u32>) {
    let mut map: FastMap<K, u32> = FastMap::default();
    // Closed triangle meshes have ~6 corners per vertex.
    map.reserve(n / 5 + 16);
    let mut remap = Vec::with_capacity(n);
    let mut first = Vec::with_capacity(n / 5 + 16);
    for i in 0..n {
        let next = first.len() as u32;
        let id = *map.entry(key(i)).or_insert(next);
        if id == next {
            first.push(i as u32);
        }
        remap.push(id);
    }
    (remap, first)
}

/// Tolerance weld: greedily merge each point into the nearest earlier representative
/// within `tol`. Returns `(remap, first)` like [`weld_exact_by`].
pub(crate) fn weld_tolerance(positions: &[DVec3], tol: f64) -> (Vec<u32>, Vec<u32>) {
    // Cells of size 2*tol: any point within tol lies in the point's own cell or the
    // neighbour toward the nearer face on each axis, so 8 cells cover the search.
    let inv = 1.0 / (2.0 * tol);
    let mut grid: FastMap<Key3<i64>, SmallVec<[u32; 2]>> = FastMap::default();
    grid.reserve(positions.len() / 2 + 16);
    let mut remap = Vec::with_capacity(positions.len());
    let mut first: Vec<u32> = Vec::with_capacity(positions.len());
    let tol2 = tol * tol;
    for (i, &p) in positions.iter().enumerate() {
        let q = p * inv;
        let c = q.floor();
        let f = q - c;
        let base = [c.x as i64, c.y as i64, c.z as i64];
        let off = [
            if f.x < 0.5 { -1 } else { 1 },
            if f.y < 0.5 { -1 } else { 1 },
            if f.z < 0.5 { -1 } else { 1 },
        ];
        let mut best = u32::MAX;
        let mut best_d2 = tol2;
        for k in 0..8 {
            let cell = Key3([
                base[0].saturating_add(if k & 1 != 0 { off[0] } else { 0 }),
                base[1].saturating_add(if k & 2 != 0 { off[1] } else { 0 }),
                base[2].saturating_add(if k & 4 != 0 { off[2] } else { 0 }),
            ]);
            if let Some(list) = grid.get(&cell) {
                for &r in list {
                    let d2 = positions[first[r as usize] as usize].distance_squared(p);
                    if d2 <= best_d2 {
                        best_d2 = d2;
                        best = r;
                    }
                }
            }
        }
        if best == u32::MAX {
            best = first.len() as u32;
            first.push(i as u32);
            grid.entry(Key3(base)).or_default().push(best);
        }
        remap.push(best);
    }
    (remap, first)
}

/// Result of [`weld_vertices`].
#[derive(Clone, Debug, Default)]
pub struct WeldResult {
    /// Welded positions (one per representative, in order of first appearance).
    pub positions: Vec<DVec3>,
    /// Triangles re-indexed into `positions`. Out-of-range input indices become
    /// `u32::MAX`; triangles may become degenerate when their corners merge
    /// ([`mm_core::Mesh::from_triangles`] skips those).
    pub triangles: Vec<[u32; 3]>,
    /// For every input vertex, the index of the output vertex it was merged into.
    pub remap: Vec<u32>,
    /// Number of input vertices that were merged away (`input - output`).
    pub welded: usize,
}

/// Merge vertices of an indexed triangle set.
///
/// With `tol <= 0` (or NaN) only bit-identical positions merge (-0.0 equals 0.0).
/// Otherwise each vertex merges into the closest earlier representative within `tol`
/// (greedy, using a spatial hash), so chains of points spaced just under `tol` do not
/// collapse transitively into one.
pub fn weld_vertices(positions: &[DVec3], tris: &[[u32; 3]], tol: f64) -> WeldResult {
    let (remap, first) = if tol > 0.0 && tol.is_finite() {
        weld_tolerance(positions, tol)
    } else {
        weld_exact_by(positions.len(), |i| dvec_key(positions[i]))
    };
    let out_pos: Vec<DVec3> = first.iter().map(|&i| positions[i as usize]).collect();
    let triangles = tris
        .iter()
        .map(|t| t.map(|v| remap.get(v as usize).copied().unwrap_or(u32::MAX)))
        .collect();
    WeldResult { welded: positions.len() - out_pos.len(), positions: out_pos, triangles, remap }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_weld_folds_negative_zero() {
        let p = [DVec3::new(0.0, 1.0, 2.0), DVec3::new(-0.0, 1.0, 2.0), DVec3::new(0.0, 1.0, 2.0000001)];
        let r = weld_vertices(&p, &[[0, 1, 2]], 0.0);
        assert_eq!(r.positions.len(), 2);
        assert_eq!(r.remap, vec![0, 0, 1]);
        assert_eq!(r.triangles, vec![[0, 0, 1]]);
        assert_eq!(r.welded, 1);
    }

    #[test]
    fn tolerance_weld() {
        let p = [
            DVec3::new(0.0, 0.0, 0.0),
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0009, 0.0, 0.0),
            DVec3::new(1.0, 0.0005, -0.0005),
            DVec3::new(0.5, 0.5, 0.0),
            DVec3::new(-0.0004, 0.0, 0.0),
        ];
        let r = weld_vertices(&p, &[[0, 1, 4], [2, 3, 4], [5, 1, 9]], 1e-3);
        assert_eq!(r.positions.len(), 3);
        assert_eq!(r.remap, vec![0, 1, 0, 1, 2, 0]);
        assert_eq!(r.triangles[1], [0, 1, 2]);
        assert_eq!(r.triangles[2], [0, 1, u32::MAX]);
        // Cell-boundary straddling pairs still merge.
        for k in 0..50 {
            let x = k as f64 * 0.001 - 0.025;
            let p = [DVec3::splat(x), DVec3::splat(x + 0.0004)];
            assert_eq!(weld_vertices(&p, &[], 1e-3).positions.len(), 1, "{x}");
        }
    }
}
