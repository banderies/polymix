//! A small fast hasher for integer keys (packed brick coordinates, vertex keys).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// Accumulates integer writes and finalizes with the murmur3 64-bit mixer, so that keys
/// differing only in their high bits (packed coordinates) still spread over all buckets.
#[derive(Default, Clone, Copy)]
pub(crate) struct KeyHasher(u64);

impl Hasher for KeyHasher {
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

pub(crate) type FastMap<K, V> = HashMap<K, V, BuildHasherDefault<KeyHasher>>;
