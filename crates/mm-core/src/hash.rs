//! A small fast hasher for integer keys (vertex, triangle and edge ids, packed coordinates).
//!
//! The standard `HashMap` hashes with SipHash under a random seed, which guards against
//! crafted keys but costs several times more than the lookup itself for small integer keys.
//! [`FastMap`] and [`FastSet`] hash with a fixed integer mixer instead: deterministic (no
//! random seed), so the same keys always land in the same buckets. Their iteration order is
//! still arbitrary, so use them for lookups only, never where the order of a walk matters.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

/// Accumulates integer writes and finalizes with the murmur3 64-bit mixer, so that keys
/// differing only in their high bits (packed coordinates) still spread over all buckets.
#[derive(Default, Clone, Copy)]
pub struct FastHasher(u64);

impl Hasher for FastHasher {
    #[inline]
    fn finish(&self) -> u64 {
        let mut z = self.0;
        z ^= z >> 33;
        z = z.wrapping_mul(0xff51_afd7_ed55_8ccd);
        z ^= z >> 33;
        z = z.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        z ^= z >> 33;
        z
    }
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        for &b in bytes {
            self.0 = (self.0.rotate_left(8) ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = self.0.rotate_left(23) ^ i;
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.write_u64(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.write_u64(i as u64);
    }
}

/// Builds [`FastHasher`]s (for `with_capacity_and_hasher` and the like).
pub type FastBuildHasher = BuildHasherDefault<FastHasher>;

/// A `HashMap` with [`FastHasher`]: for lookups by integer keys, not for ordered walks.
pub type FastMap<K, V> = HashMap<K, V, FastBuildHasher>;

/// A `HashSet` with [`FastHasher`]: for membership tests, not for ordered walks.
pub type FastSet<K> = HashSet<K, FastBuildHasher>;

/// An empty [`FastMap`] with room for `n` entries.
pub fn fast_map_with_capacity<K, V>(n: usize) -> FastMap<K, V> {
    FastMap::with_capacity_and_hasher(n, FastBuildHasher::default())
}

/// An empty [`FastSet`] with room for `n` entries.
pub fn fast_set_with_capacity<K>(n: usize) -> FastSet<K> {
    FastSet::with_capacity_and_hasher(n, FastBuildHasher::default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn hash_of<T: Hash>(x: T) -> u64 {
        FastBuildHasher::default().hash_one(x)
    }

    #[test]
    fn maps_and_sets_behave_like_std() {
        let mut m: FastMap<u32, usize> = fast_map_with_capacity(16);
        let mut s: FastSet<(u32, u32)> = FastSet::default();
        let mut reference = std::collections::HashMap::new();
        // Packed coordinates and ids, with collisions in the low bits.
        for i in 0..5000u32 {
            let k = i.wrapping_mul(0x9e37_79b9) ^ (i << 20);
            *m.entry(k).or_default() += 1;
            *reference.entry(k).or_insert(0usize) += 1;
            s.insert((k, i % 7));
        }
        assert_eq!(m.len(), reference.len());
        assert!(reference.iter().all(|(k, v)| m.get(k) == Some(v)));
        assert!(s.contains(&(0, 0)) && !s.contains(&(0, 1)));
        assert_eq!(s.len(), 5000);
    }

    #[test]
    fn deterministic_and_spread() {
        // No random seed: the same key always hashes the same.
        assert_eq!(hash_of(42u32), hash_of(42u32));
        assert_eq!(hash_of((7u32, 9u32)), FastBuildHasher::default().hash_one((7u32, 9u32)));
        assert_ne!(hash_of((7u32, 9u32)), hash_of((9u32, 7u32)));
        // Keys that differ only in their high bits still reach different buckets.
        let buckets: FastSet<u64> = (0..64u64).map(|i| hash_of(i << 40) & 63).collect();
        assert!(buckets.len() > 32, "only {} of 64 buckets used", buckets.len());
    }
}
