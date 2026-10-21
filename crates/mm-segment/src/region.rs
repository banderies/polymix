//! Boxes of voxels within a grid (i fastest), and moving values between a grid and a box of
//! it a row at a time. Operations whose effect stays near a segment work on its bounding box
//! grown by their reach, so small segments in big volumes stay cheap.

use rayon::prelude::*;

/// A box of voxels within a grid: its first voxel and its size.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Region {
    pub(crate) lo: [usize; 3],
    pub(crate) dims: [usize; 3],
}

impl Region {
    /// The whole grid of `dims`.
    pub(crate) fn all(dims: [usize; 3]) -> Self {
        Self { lo: [0; 3], dims }
    }

    /// The voxels from `lo` to `hi` (inclusive) and `pad` more along each axis, within a grid
    /// of `full`.
    pub(crate) fn around(lo: [usize; 3], hi: [usize; 3], pad: [usize; 3], full: [usize; 3]) -> Self {
        let a = [0, 1, 2].map(|x| lo[x].saturating_sub(pad[x]));
        let b = [0, 1, 2].map(|x| hi[x].saturating_add(pad[x]).min(full[x] - 1));
        Self { lo: a, dims: [0, 1, 2].map(|x| b[x] - a[x] + 1) }
    }

    pub(crate) fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Whether grid voxel `g` is in the region.
    pub(crate) fn contains(&self, g: [usize; 3]) -> bool {
        (0..3).all(|a| g[a] >= self.lo[a] && g[a] - self.lo[a] < self.dims[a])
    }

    /// Grown by `pad` voxels along each axis, within a grid of `grid` voxels.
    pub(crate) fn grown(&self, pad: [usize; 3], grid: [usize; 3]) -> Region {
        let lo = [0, 1, 2].map(|a| self.lo[a].saturating_sub(pad[a]));
        let hi = [0, 1, 2].map(|a| (self.lo[a] + self.dims[a]).saturating_add(pad[a]).min(grid[a]));
        Region { lo, dims: [0, 1, 2].map(|a| hi[a] - lo[a]) }
    }

    /// The smallest region holding both.
    pub(crate) fn union(&self, o: &Region) -> Region {
        let lo = [0, 1, 2].map(|a| self.lo[a].min(o.lo[a]));
        let hi = [0, 1, 2].map(|a| (self.lo[a] + self.dims[a]).max(o.lo[a] + o.dims[a]));
        Region { lo, dims: [0, 1, 2].map(|a| hi[a] - lo[a]) }
    }

    /// Grid (j, k) of the region's row `r` (rows run along i).
    pub(crate) fn row_jk(&self, r: usize) -> (usize, usize) {
        (self.lo[1] + r % self.dims[1], self.lo[2] + r / self.dims[1])
    }

    /// The region's row number of grid row (j, k), if the region has it.
    pub(crate) fn row_at(&self, j: usize, k: usize) -> Option<usize> {
        let (y, z) = (j.wrapping_sub(self.lo[1]), k.wrapping_sub(self.lo[2]));
        // Lazily: out of range, y and z have wrapped round and their row number would overflow.
        (y < self.dims[1] && z < self.dims[2]).then(|| z * self.dims[1] + y)
    }

    /// Grid index of the first voxel of the region's row `r`, in a grid of `full`.
    pub(crate) fn row_start(&self, r: usize, full: [usize; 3]) -> usize {
        let (j, k) = self.row_jk(r);
        (k * full[1] + j) * full[0] + self.lo[0]
    }

    /// The region's index of grid voxel `g` (inside the region).
    pub(crate) fn local(&self, g: [usize; 3]) -> usize {
        ((g[2] - self.lo[2]) * self.dims[1] + g[1] - self.lo[1]) * self.dims[0] + g[0] - self.lo[0]
    }

    /// Grid voxel of the region's voxel `n`.
    pub(crate) fn grid(&self, n: usize) -> [usize; 3] {
        let [nx, ny, _] = self.dims;
        [n % nx + self.lo[0], (n / nx) % ny + self.lo[1], n / (nx * ny) + self.lo[2]]
    }

    /// `f` of each of the region's voxels of `data` (a grid of `full`), as a grid of its own.
    pub(crate) fn gather<T: Copy + Sync, U: Copy + Default + Send>(&self, data: &[T], full: [usize; 3], f: impl Fn(T) -> U + Sync) -> Vec<U> {
        let mut out = vec![U::default(); self.len()];
        if out.is_empty() {
            return out;
        }
        out.par_chunks_mut(self.dims[0]).enumerate().for_each(|(r, row)| {
            let at = self.row_start(r, full);
            for (o, &v) in row.iter_mut().zip(&data[at..at + self.dims[0]]) {
                *o = f(v);
            }
        });
        out
    }

    /// The region's part of each row of `data` (a grid of `full`), in parallel, with the row's
    /// number in the region.
    pub(crate) fn par_rows_mut<'a, T: Send>(&self, data: &'a mut [T], full: [usize; 3]) -> impl ParallelIterator<Item = (usize, &'a mut [T])> + 'a {
        let b = *self;
        data.par_chunks_mut(full[0].max(1)).enumerate().filter_map(move |(r, row)| b.row_at(r % full[1], r / full[1]).map(|q| (q, &mut row[b.lo[0]..b.lo[0] + b.dims[0]])))
    }

    /// The region's part of each row of `data` (a grid of `full`), grouped by the region's j:
    /// group `y` holds the rows of grid j = `lo[1] + y` in order of k, for work along k that
    /// takes a whole group at a time (each row belongs to one group, so groups can go to
    /// different threads).
    pub(crate) fn rows_by_j<'a, T>(&self, data: &'a mut [T], full: [usize; 3]) -> Vec<Vec<&'a mut [T]>> {
        let mut groups: Vec<Vec<&'a mut [T]>> = (0..self.dims[1]).map(|_| Vec::with_capacity(self.dims[2])).collect();
        for (r, row) in data.chunks_mut(full[0].max(1)).enumerate() {
            if let Some(q) = self.row_at(r % full[1], r / full[1]) {
                groups[q % self.dims[1]].push(&mut row[self.lo[0]..self.lo[0] + self.dims[0]]);
            }
        }
        groups
    }

    /// Set the voxels of a grid mask `out` (of `full`) in the region to `f` of the region's
    /// index.
    pub(crate) fn write(&self, out: &mut [u8], full: [usize; 3], f: impl Fn(usize) -> bool + Sync) {
        let dx = self.dims[0];
        self.par_rows_mut(out, full).for_each(|(r, row)| {
            for (i, m) in row.iter_mut().enumerate() {
                *m = f(r * dx + i) as u8;
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_rows_gather_and_groups() {
        let full = [5, 4, 3];
        let data: Vec<u32> = (0..60).collect();
        let b = Region::around([1, 1, 1], [2, 2, 1], [1, 0, 1], full);
        assert_eq!(b, Region { lo: [0, 1, 0], dims: [4, 2, 3] });
        let g = b.gather(&data, full, |v| v);
        for (n, &v) in g.iter().enumerate() {
            let p = b.grid(n);
            assert_eq!(v as usize, (p[2] * 4 + p[1]) * 5 + p[0]);
            assert_eq!(b.local(p), n);
        }
        let mut grid = vec![0u32; 60];
        let groups = b.rows_by_j(&mut grid, full);
        assert_eq!(groups.len(), 2);
        for (y, rows) in groups.into_iter().enumerate() {
            for (z, row) in rows.into_iter().enumerate() {
                row.fill((10 * y + z) as u32 + 1);
            }
        }
        for (n, &v) in grid.iter().enumerate() {
            let (i, j, k) = (n % 5, (n / 5) % 4, n / 20);
            let inside = i < 4 && (1..3).contains(&j);
            assert_eq!(v, if inside { (10 * (j - 1) + k) as u32 + 1 } else { 0 }, "{i} {j} {k}");
        }
        let mut m = vec![0u8; 60];
        b.write(&mut m, full, |q| q % 2 == 0);
        assert_eq!(m.iter().filter(|&&x| x != 0).count(), b.len() / 2);
    }
}
