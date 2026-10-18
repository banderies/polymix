//! A binary mask over a volume's voxel grid (one byte per voxel, 0 or 1, i fastest), and
//! the logical operations between masks.

use rayon::prelude::*;

/// Voxel neighbourhood for connected regions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Connectivity {
    /// Faces only.
    Six,
    /// Faces, edges and corners.
    TwentySix,
}

impl Connectivity {
    pub(crate) fn offsets(self) -> Vec<[i64; 3]> {
        let mut v = Vec::new();
        for dz in -1i64..=1 {
            for dy in -1i64..=1 {
                for dx in -1i64..=1 {
                    let n = dx.abs() + dy.abs() + dz.abs();
                    if n == 0 || (self == Connectivity::Six && n > 1) {
                        continue;
                    }
                    v.push([dx, dy, dz]);
                }
            }
        }
        v
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Mask {
    pub dims: [usize; 3],
    pub data: Vec<u8>,
}

impl Mask {
    pub fn new(dims: [usize; 3]) -> Self {
        Self { dims, data: vec![0; dims[0] * dims[1] * dims[2]] }
    }

    /// Voxels of `labels` equal to `label`.
    pub fn from_labels(dims: [usize; 3], labels: &[u8], label: u8) -> Self {
        Self { dims, data: labels.par_iter().map(|&l| (l == label) as u8).collect() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    pub fn ijk(&self, n: usize) -> [usize; 3] {
        let plane = self.dims[0] * self.dims[1];
        [n % self.dims[0], (n / self.dims[0]) % self.dims[1], n / plane]
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> bool {
        self.data[self.index(i, j, k)] != 0
    }

    pub fn set(&mut self, i: usize, j: usize, k: usize, on: bool) {
        let n = self.index(i, j, k);
        self.data[n] = on as u8;
    }

    /// Voxels set.
    pub fn count(&self) -> usize {
        self.data.par_iter().filter(|&&v| v != 0).count()
    }

    /// Smallest and largest voxel indices set (inclusive), if any.
    pub fn bounds(&self) -> Option<([usize; 3], [usize; 3])> {
        let plane = self.dims[0] * self.dims[1];
        let per_slice: Vec<Option<([usize; 3], [usize; 3])>> = self
            .data
            .par_chunks(plane.max(1))
            .enumerate()
            .map(|(k, s)| {
                let mut b: Option<([usize; 3], [usize; 3])> = None;
                for (n, &v) in s.iter().enumerate() {
                    if v != 0 {
                        let (i, j) = (n % self.dims[0], n / self.dims[0]);
                        let (lo, hi) = b.get_or_insert(([i, j, k], [i, j, k]));
                        *lo = [lo[0].min(i), lo[1].min(j), k];
                        *hi = [hi[0].max(i), hi[1].max(j), k];
                    }
                }
                b
            })
            .collect();
        per_slice.into_iter().flatten().reduce(|(a, b), (c, d)| ([a[0].min(c[0]), a[1].min(c[1]), a[2].min(c[2])], [b[0].max(d[0]), b[1].max(d[1]), b[2].max(d[2])]))
    }

    fn zip(&mut self, other: &Mask, f: impl Fn(u8, u8) -> u8 + Sync) {
        assert_eq!(self.dims, other.dims, "masks of different sizes");
        self.data.par_iter_mut().zip(&other.data).for_each(|(a, &b)| *a = f(*a, b));
    }

    pub fn union(&mut self, other: &Mask) {
        self.zip(other, |a, b| a | b);
    }

    pub fn intersect(&mut self, other: &Mask) {
        self.zip(other, |a, b| a & b);
    }

    pub fn subtract(&mut self, other: &Mask) {
        self.zip(other, |a, b| a & (1 - b));
    }

    pub fn invert(&mut self) {
        self.data.par_iter_mut().for_each(|v| *v = 1 - (*v != 0) as u8);
    }

    /// Write `label` into `labels` where the mask is set (over other labels too when
    /// `overwrite`), and clear voxels of `label` where it is not.
    pub fn write_label(&self, labels: &mut [u8], label: u8, overwrite: bool) {
        labels.par_iter_mut().zip(&self.data).for_each(|(l, &m)| {
            if m != 0 {
                if *l == 0 || overwrite {
                    *l = label;
                }
            } else if *l == label {
                *l = 0;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logic_and_bounds() {
        let mut a = Mask::new([4, 3, 2]);
        assert_eq!(a.bounds(), None);
        a.set(1, 2, 0, true);
        a.set(3, 0, 1, true);
        assert_eq!(a.bounds(), Some(([1, 0, 0], [3, 2, 1])));
        assert_eq!(a.ijk(a.index(3, 0, 1)), [3, 0, 1]);
        let mut b = Mask::new([4, 3, 2]);
        b.set(3, 0, 1, true);
        b.set(0, 0, 0, true);
        let mut u = a.clone();
        u.union(&b);
        assert_eq!(u.count(), 3);
        let mut s = a.clone();
        s.subtract(&b);
        assert_eq!(s.count(), 1);
        let mut i = a.clone();
        i.intersect(&b);
        assert_eq!(i.count(), 1);
        i.invert();
        assert_eq!(i.count(), 23);
        let mut labels = vec![0u8; 24];
        labels[0] = 2;
        labels[5] = 1;
        a.write_label(&mut labels, 1, false);
        assert_eq!(labels.iter().filter(|&&l| l == 1).count(), 2);
        assert_eq!(labels[0], 2);
        assert_eq!(Connectivity::Six.offsets().len(), 6);
        assert_eq!(Connectivity::TwentySix.offsets().len(), 26);
    }
}
