//! Smoothing and hollowing of segments, as 3D Slicer's Smoothing and Hollow effects do them.
//!
//! - Gaussian smoothing blurs a segment's indicator (1 inside, 0 outside) with a Gaussian and
//!   thresholds it at one half: bumps and pits narrower than about the standard deviation go,
//!   flat faces stay where they were. The blur is separable (one 1D convolution per axis, the
//!   kernel truncated at three standard deviations and normalised) and runs on the segment's
//!   bounding box grown by the kernel's reach, so small segments in big volumes stay cheap.
//!   Past the grid's edges the edge values continue, so a segment cut off by the edge of the
//!   volume is not rounded there.
//! - Joint smoothing smooths several segments of a label map together so that boundaries they
//!   share stay shared: each segment's indicator is blurred, and every voxel takes the class
//!   with the largest blurred share (a multi-label version of the threshold at one half; see
//!   the "Joint smoothing" method of 3D Slicer's Segment Editor, Pinter et al. 2019, "Polymorph
//!   segmentation representation for medical image computing", Comput. Methods Programs
//!   Biomed. 171:19–26). Only one blurred indicator is held at a time, with a running maximum.
//! - Hollowing replaces a segment by a shell along its surface: the voxels within the
//!   thickness of it, on a sub-voxel signed distance (the exact Euclidean distance transforms
//!   of [`crate::ops`], smoothed a little; see [`hollow`]).

use crate::mask::Mask;
use crate::ops::distance_sq;
use rayon::prelude::*;

/// A box of voxels within a grid: its first voxel and its size.
#[derive(Clone, Copy, Debug)]
struct Region {
    lo: [usize; 3],
    dims: [usize; 3],
}

impl Region {
    /// The voxels from `lo` to `hi` (inclusive) and `pad` more along each axis, within a grid
    /// of `full`.
    fn around(lo: [usize; 3], hi: [usize; 3], pad: [usize; 3], full: [usize; 3]) -> Self {
        let a = [0, 1, 2].map(|x| lo[x].saturating_sub(pad[x]));
        let b = [0, 1, 2].map(|x| hi[x].saturating_add(pad[x]).min(full[x] - 1));
        Self { lo: a, dims: [0, 1, 2].map(|x| b[x] - a[x] + 1) }
    }

    fn len(&self) -> usize {
        self.dims[0] * self.dims[1] * self.dims[2]
    }

    /// Grid (j, k) of the region's row `r` (rows run along i).
    fn row_jk(&self, r: usize) -> (usize, usize) {
        (self.lo[1] + r % self.dims[1], self.lo[2] + r / self.dims[1])
    }

    /// The region's row number of grid row (j, k), if the region has it.
    fn row_at(&self, j: usize, k: usize) -> Option<usize> {
        let (y, z) = (j.wrapping_sub(self.lo[1]), k.wrapping_sub(self.lo[2]));
        // Lazily: out of range, y and z have wrapped round and their row number would overflow.
        (y < self.dims[1] && z < self.dims[2]).then(|| z * self.dims[1] + y)
    }

    /// `f` of each of the region's voxels of `data` (a grid of `full`), as a grid of its own.
    fn gather<T: Copy + Sync, U: Copy + Default + Send>(&self, data: &[T], full: [usize; 3], f: impl Fn(T) -> U + Sync) -> Vec<U> {
        let mut out = vec![U::default(); self.len()];
        out.par_chunks_mut(self.dims[0]).enumerate().for_each(|(r, row)| {
            let (j, k) = self.row_jk(r);
            let at = (k * full[1] + j) * full[0] + self.lo[0];
            for (o, &v) in row.iter_mut().zip(&data[at..at + self.dims[0]]) {
                *o = f(v);
            }
        });
        out
    }

    /// The region's part of each row of `data` (a grid of `full`), in parallel, with the row's
    /// number in the region.
    fn par_rows_mut<'a, T: Send>(&self, data: &'a mut [T], full: [usize; 3]) -> impl ParallelIterator<Item = (usize, &'a mut [T])> + 'a {
        let b = *self;
        data.par_chunks_mut(full[0]).enumerate().filter_map(move |(r, row)| b.row_at(r % full[1], r / full[1]).map(|q| (q, &mut row[b.lo[0]..b.lo[0] + b.dims[0]])))
    }
}

/// Normalised weights of a Gaussian with a standard deviation of `s` samples, truncated at
/// three standard deviations (centre at index `len / 2`).
fn kernel(s: f64) -> Vec<f32> {
    let r = (3.0 * s).ceil() as i64;
    let w: Vec<f64> = (-r..=r).map(|x| (-((x * x) as f64) / (2.0 * s * s)).exp()).collect();
    let total: f64 = w.iter().sum();
    w.iter().map(|&v| (v / total) as f32).collect()
}

/// Separable Gaussian blur of `values` (a grid of `dims`, i fastest) with a standard deviation
/// of `sigma` voxels along each axis; the grid's edge values continue past its edges. Below a
/// twentieth of a voxel the kernel is the identity, and the axis is left alone.
pub(crate) fn blur(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let [nx, ny, nz] = dims;
    if values.is_empty() {
        return;
    }
    let mut src = vec![0f32; values.len()];
    for axis in 0..3 {
        if sigma[axis].is_nan() || sigma[axis] < 0.05 || dims[axis] < 2 {
            continue;
        }
        // Past the grid's length a wider Gaussian only flattens further.
        let w = kernel(sigma[axis].min(dims[axis] as f64));
        let r = w.len() / 2;
        src.copy_from_slice(values);
        if axis == 0 {
            values.par_chunks_mut(nx).zip(src.par_chunks(nx)).for_each(|(out, line)| {
                for (x, o) in out.iter_mut().enumerate() {
                    *o = w.iter().enumerate().map(|(t, &wt)| wt * line[(x + t).saturating_sub(r).min(nx - 1)]).sum();
                }
            });
        } else {
            // Whole rows at a time: row (j, k) is the weighted sum of the rows before and after
            // it along the axis, which keeps the memory access sequential.
            let (len, step) = if axis == 1 { (ny, 1) } else { (nz, ny) };
            values.par_chunks_mut(nx).enumerate().for_each(|(l, out)| {
                let p = if axis == 1 { l % ny } else { l / ny };
                let base = l - p * step;
                out.fill(0.0);
                for (t, &wt) in w.iter().enumerate() {
                    let q = base + (p + t).saturating_sub(r).min(len - 1) * step;
                    for (o, &v) in out.iter_mut().zip(&src[q * nx..(q + 1) * nx]) {
                        *o += wt * v;
                    }
                }
            });
        }
    }
}

/// Gaussian smoothing of a mask: its indicator blurred with a Gaussian of `sigma_mm` (per-axis
/// sigma in voxels from the spacing; separable; truncated at 3 sigma), then thresholded at
/// 0.5. A sigma of 0 leaves the mask as it is.
pub fn gaussian(mask: &mut Mask, sigma_mm: f64, spacing: [f64; 3]) {
    if sigma_mm.is_nan() || sigma_mm <= 0.0 {
        return;
    }
    let Some((lo, hi)) = mask.bounds() else { return };
    let sigma = spacing.map(|s| sigma_mm / s);
    // Past the kernel's reach from the segment the blurred indicator is 0: only its box changes.
    let b = Region::around(lo, hi, sigma.map(|s| (3.0 * s).ceil() as usize), mask.dims);
    let mut v = b.gather(&mask.data, mask.dims, |m| if m != 0 { 1f32 } else { 0.0 });
    blur(&mut v, b.dims, sigma);
    let dx = b.dims[0];
    b.par_rows_mut(&mut mask.data, mask.dims).for_each(|(r, row)| {
        for (m, &x) in row.iter_mut().zip(&v[r * dx..(r + 1) * dx]) {
            *m = (x >= 0.5) as u8;
        }
    });
}

type Bounds = ([usize; 3], [usize; 3]);

fn merge(a: Option<Bounds>, b: Option<Bounds>) -> Option<Bounds> {
    match (a, b) {
        (Some((a0, a1)), Some((b0, b1))) => Some(([0, 1, 2].map(|x| a0[x].min(b0[x])), [0, 1, 2].map(|x| a1[x].max(b1[x])))),
        (x, None) | (None, x) => x,
    }
}

/// Joint smoothing (as 3D Slicer's Smoothing effect "Joint smoothing"): the segments listed in
/// `segments` (a subset of the label map's labels) are smoothed together so their shared
/// boundaries stay watertight: each one's indicator is blurred (Gaussian of `sigma_mm`, as
/// [`gaussian`]), and each voxel that held one of them (or was empty, if it is within reach of
/// them) takes the one with the largest blurred value — or stays/becomes 0 when the blurred
/// background is largest. Voxels of other labels are never changed and never taken.
///
/// The background is the blurred share of the empty voxels only (1 − the sum of the segments
/// and of the other labels): where the other labels' blurred share is largest, a voxel keeps
/// what it held, so no gap opens between the smoothed segments and labels left alone. With no
/// other labels nearby this is the plain 1 − sum of the segments.
///
/// Memory: the work covers the involved segments' bounding box grown by twice the kernel's
/// reach (9 bytes a voxel), and each segment is blurred on its own box (8 bytes a voxel).
pub fn joint_smooth(labels: &mut [u8], dims: [usize; 3], spacing: [f64; 3], segments: &[u8], sigma_mm: f64) {
    let [nx, ny, nz] = dims;
    assert_eq!(labels.len(), nx * ny * nz, "label map of the wrong size");
    let mut involved = [false; 256];
    for &s in segments.iter().filter(|&&s| s != 0) {
        involved[s as usize] = true;
    }
    if sigma_mm.is_nan() || sigma_mm <= 0.0 || labels.is_empty() {
        return;
    }
    // Each involved segment's bounding box, in one pass.
    let plane = nx * ny;
    let boxes = (0..nz)
        .into_par_iter()
        .fold(
            || vec![None; 256],
            |mut acc: Vec<Option<Bounds>>, k| {
                for (n, &l) in labels[k * plane..(k + 1) * plane].iter().enumerate() {
                    if involved[l as usize] {
                        let p = [n % nx, n / nx, k];
                        acc[l as usize] = merge(acc[l as usize], Some((p, p)));
                    }
                }
                acc
            },
        )
        .reduce(|| vec![None; 256], |a, b| a.iter().zip(&b).map(|(&x, &y)| merge(x, y)).collect());
    let present: Vec<(u8, Bounds)> = (1..256).filter_map(|l| boxes[l].map(|b| (l as u8, b))).collect();
    let Some((lo, hi)) = present.iter().fold(None, |a, p| merge(a, Some(p.1))) else { return };
    let sigma = spacing.map(|s| sigma_mm / s);
    let reach = sigma.map(|s| (3.0 * s).ceil() as usize);
    // Twice the reach: the other labels' blur is exact wherever a segment's blur reaches.
    let work = Region::around(lo, hi, reach.map(|r| r.saturating_mul(2)), dims);
    let wx = work.dims[0];
    // Per voxel of the work: the largest blurred share so far and whose it is (0: the other
    // labels'), and the sum of all shares but the background's.
    let mut best = vec![0f32; work.len()];
    let mut sum = vec![0f32; work.len()];
    let mut arg = vec![0u8; work.len()];
    let mut others = work.gather(labels, dims, |l| if l != 0 && !involved[l as usize] { 1f32 } else { 0.0 });
    if others.par_iter().any(|&v| v != 0.0) {
        blur(&mut others, work.dims, sigma);
        best.copy_from_slice(&others);
        sum.copy_from_slice(&others);
    }
    drop(others);
    for &(s, (slo, shi)) in &present {
        let b = Region::around(slo, shi, reach, dims);
        let mut v = b.gather(labels, dims, |l| if l == s { 1f32 } else { 0.0 });
        blur(&mut v, b.dims, sigma);
        let off = b.lo[0] - work.lo[0];
        best.par_chunks_mut(wx).zip(sum.par_chunks_mut(wx)).zip(arg.par_chunks_mut(wx)).enumerate().for_each(|(r, ((bst, sm), ag))| {
            let (j, k) = work.row_jk(r);
            let Some(rb) = b.row_at(j, k) else { return };
            for (x, &val) in v[rb * b.dims[0]..(rb + 1) * b.dims[0]].iter().enumerate() {
                let q = off + x;
                sm[q] += val;
                if val > bst[q] {
                    bst[q] = val;
                    ag[q] = s;
                }
            }
        });
    }
    work.par_rows_mut(labels, dims).for_each(|(r, row)| {
        for (x, l) in row.iter_mut().enumerate() {
            let q = r * wx + x;
            if *l != 0 && !involved[*l as usize] {
                continue;
            }
            if 1.0 - sum[q] > best[q] {
                *l = 0;
            } else if arg[q] != 0 {
                *l = arg[q];
            }
        }
    });
}

/// Which side of the segment's surface a hollow shell lies on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HollowSide {
    /// Within the segment: its outer surface stays where it was.
    Inside,
    /// Centred on the segment's surface.
    Medial,
    /// Around the segment: its surface becomes the shell's inner surface.
    Outside,
}

/// Smoothing (standard deviation, voxels) of the signed distance a hollow shell is cut from.
const SHELL_SIGMA: f64 = 0.7;
/// How far (voxels) the nearest voxel centre of the other kind is taken to lie past the
/// surface: half a voxel across faces along the voxel axes, less across oblique ones.
const SHELL_OFFSET: f64 = 0.4;

/// Replace the segment by a shell of `thickness_mm` along its surface: inside it (the shell is
/// the segment minus the segment shrunk by the thickness), centred on its surface (grown by
/// half, minus shrunk by half), or outside it (grown by the thickness, minus the segment).
/// A thickness of 0 or less leaves the mask as it is.
///
/// The shell is the voxels on its side of the surface whose centres lie within the thickness
/// of it, measured on a sub-voxel signed distance: the exact Euclidean distance to the nearest
/// voxel centre of the other kind less 0.4 voxel (zero half-way between inside and outside
/// centres), smoothed with a Gaussian of 0.7 voxel. Margins between voxel centres alone (as
/// [`crate::ops::grow`] and [`crate::ops::shrink`]) make shells about a sixth too thin across
/// oblique surfaces, since within one voxel only the six face neighbours lie; this keeps
/// balls' shells within about 7% of their volume from two voxels' thickness up. The shell
/// also holds every voxel those plain margins would give it, and is at least a voxel thick on
/// the segment's side (Inside, Medial) or outside it (Outside), so it is always closed.
pub fn hollow(mask: &mut Mask, thickness_mm: f64, side: HollowSide, spacing: [f64; 3]) {
    if thickness_mm.is_nan() || thickness_mm <= 0.0 {
        return;
    }
    let Some((lo, hi)) = mask.bounds() else { return };
    let voxel = spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let t = thickness_mm;
    // Reach of the shell outside and inside the surface, and of the plain margins.
    let (t_out, t_in, r_out, r_in) = match side {
        HollowSide::Inside => (0.0, t, 0.0, t.max(voxel)),
        HollowSide::Medial => (0.5 * t, 0.5 * t, 0.5 * t, (0.5 * t).max(voxel)),
        HollowSide::Outside => (t, 0.0, t.max(voxel), 0.0),
    };
    // The box reaches past the shell by the smoothing's reach and a voxel, so the distances
    // and their smoothing in it are those of the whole grid.
    let reach = (3.0 * SHELL_SIGMA).ceil() as usize + 1;
    let b = Region::around(lo, hi, spacing.map(|s| ((r_out / s).ceil() as usize).saturating_add(reach)), mask.dims);
    let seg: Vec<u8> = b.gather(&mask.data, mask.dims, |m| (m != 0) as u8);
    let rest: Vec<u8> = seg.par_iter().map(|&m| 1 - m).collect();
    let (d_in, d_out) = (distance_sq(b.dims, &seg, spacing), distance_sq(b.dims, &rest, spacing));
    drop(rest);
    let off = SHELL_OFFSET * voxel;
    let mut phi: Vec<f32> = (0..b.len()).into_par_iter().map(|q| if seg[q] != 0 { off - d_out[q].sqrt() } else { d_in[q].sqrt() - off } as f32).collect();
    blur(&mut phi, b.dims, [SHELL_SIGMA; 3]);
    let (t_out, t_in) = (t_out as f32, -t_in as f32);
    let (r_out, r_in) = (r_out * r_out + 1e-9, r_in * r_in + 1e-9);
    let shell = |q: usize| if seg[q] != 0 { t_in < 0.0 && (phi[q] > t_in || d_out[q] <= r_in) } else { t_out > 0.0 && (phi[q] <= t_out || d_in[q] <= r_out) };
    let dx = b.dims[0];
    b.par_rows_mut(&mut mask.data, mask.dims).for_each(|(r, row)| {
        for (x, m) in row.iter_mut().enumerate() {
            *m = shell(r * dx + x) as u8;
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    fn ball(n: usize, r: f64) -> Mask {
        let mut m = Mask::new([n; 3]);
        let c = (n as f64 - 1.0) * 0.5;
        for k in 0..n {
            for j in 0..n {
                for i in 0..n {
                    let d2 = (i as f64 - c).powi(2) + (j as f64 - c).powi(2) + (k as f64 - c).powi(2);
                    m.set(i, j, k, d2 <= r * r);
                }
            }
        }
        m
    }

    #[test]
    fn smooth_blur_keeps_mass_and_constants() {
        let dims = [9, 7, 5];
        let mut v = vec![0f32; 315];
        v[(2 * 7 + 3) * 9 + 4] = 1.0;
        blur(&mut v, dims, [1.0, 0.7, 0.5]);
        let total: f32 = v.iter().sum();
        assert!((total - 1.0).abs() < 1e-5, "{total}");
        let mut c = vec![2f32; 315];
        blur(&mut c, dims, [2.0, 2.0, 2.0]);
        assert!(c.iter().all(|&x| (x - 2.0).abs() < 1e-5));
    }

    #[test]
    fn smooth_gaussian_removes_a_spike_and_keeps_the_size() {
        let mut m = Mask::new([40, 40, 40]);
        for k in 10..30 {
            for j in 10..30 {
                for i in 10..30 {
                    m.set(i, j, k, true);
                }
            }
        }
        let n0 = m.count();
        let mut spiky = m.clone();
        spiky.set(30, 20, 20, true);
        let mut same = spiky.clone();
        gaussian(&mut same, 0.0, [1.0; 3]);
        assert_eq!(same, spiky);
        gaussian(&mut spiky, 1.0, [1.0; 3]);
        assert!(!spiky.get(30, 20, 20), "the spike stays");
        assert!(spiky.get(29, 20, 20) && spiky.get(10, 20, 20) && spiky.get(20, 20, 20));
        let n = spiky.count();
        assert!((n as f64 - n0 as f64).abs() < 0.05 * n0 as f64, "{n} vs {n0}");
        // Nothing outside the cube appears.
        let mut outside = spiky.clone();
        outside.subtract(&m);
        assert_eq!(outside.count(), 0);
        // Anisotropic spacing: the same cube on 1 × 1 × 2 mm voxels keeps its faces.
        let mut a = Mask::new([40, 40, 20]);
        for k in 5..15 {
            for j in 10..30 {
                for i in 10..30 {
                    a.set(i, j, k, true);
                }
            }
        }
        gaussian(&mut a, 1.0, [1.0, 1.0, 2.0]);
        assert!(a.get(20, 20, 5) && a.get(20, 20, 14) && !a.get(20, 20, 15));
        // A segment cut off by the grid's edge is not rounded there.
        let mut edge = Mask::new([20, 20, 20]);
        for k in 5..15 {
            for j in 5..15 {
                for i in 0..10 {
                    edge.set(i, j, k, true);
                }
            }
        }
        gaussian(&mut edge, 1.0, [1.0; 3]);
        assert!(edge.get(0, 10, 10) && edge.get(0, 6, 10));
        // Empty stays empty.
        let mut empty = Mask::new([5, 5, 5]);
        gaussian(&mut empty, 1.0, [1.0; 3]);
        assert_eq!(empty.count(), 0);
    }

    #[test]
    fn smooth_joint_keeps_shared_boundaries() {
        let dims = [42, 30, 30];
        let [nx, ny, _] = dims;
        let at = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
        let mut labels = vec![0u8; nx * ny * 30];
        // Box 1 (x 5..=19) and box 2 (x 20..=34) share a face made jagged by 2 × 2 blocks
        // swapped across it; box 3 (x 35..=38), left alone, touches box 2.
        for k in 5..25 {
            for j in 5..25 {
                let swap = (j / 2 + k / 2) % 2 == 0;
                for i in 5..39 {
                    labels[at(i, j, k)] = match i {
                        _ if i >= 35 => 3,
                        19 if !swap => 2,
                        20 if swap => 1,
                        _ if i < 20 => 1,
                        _ => 2,
                    };
                }
            }
        }
        let before = labels.clone();
        let mut same = labels.clone();
        joint_smooth(&mut same, dims, [1.0; 3], &[1, 2], 0.0);
        assert_eq!(same, before);
        // Voxels off the flat face at x = 19.5.
        let jagged = |l: &[u8]| (0..l.len()).filter(|&n| (n % nx < 20 && l[n] == 2) || (n % nx >= 20 && l[n] == 1)).count();
        let j0 = jagged(&before);
        joint_smooth(&mut labels, dims, [1.0; 3], &[1, 2], 1.0);
        // Label 3 is untouched, and nothing became 3.
        for n in 0..labels.len() {
            assert_eq!(labels[n] == 3, before[n] == 3, "{n}");
        }
        // Voxels of boxes 1 and 2 stay in one of them, or are emptied near the outer surface
        // only (within 2 voxels of a voxel that was empty).
        for k in 0..30 {
            for j in 0..ny {
                for i in 0..nx {
                    let n = at(i, j, k);
                    if before[n] == 1 || before[n] == 2 {
                        assert!(labels[n] != 3);
                        if labels[n] == 0 {
                            let near = (k.saturating_sub(2)..=(k + 2).min(29)).any(|c| (j.saturating_sub(2)..=(j + 2).min(ny - 1)).any(|b| (i.saturating_sub(2)..=(i + 2).min(nx - 1)).any(|a| before[at(a, b, c)] == 0)));
                            assert!(near, "emptied inside at {i} {j} {k}");
                        }
                    }
                }
            }
        }
        // No gap between boxes 1 and 2, nor between box 2 and box 3.
        for k in 7..23 {
            for j in 7..23 {
                for i in 15..35 {
                    assert_ne!(labels[at(i, j, k)], 0, "gap at {i} {j} {k}");
                }
            }
        }
        // The shared face is flatter.
        let j1 = jagged(&labels);
        assert!(j1 * 4 < j0, "{j1} vs {j0}");
        // Nothing appears far from the segments.
        assert_eq!(labels[at(1, 1, 1)], 0);
    }

    /// Voxels of `m` with a face neighbour of the other kind (`inner`), or not in `m` with a
    /// face neighbour in it.
    fn boundary_layer(m: &Mask, inner: bool) -> Vec<usize> {
        let [nx, ny, nz] = m.dims;
        (0..m.len())
            .filter(|&q| {
                let [i, j, k] = m.ijk(q);
                let nb = [[i.wrapping_sub(1), j, k], [i + 1, j, k], [i, j.wrapping_sub(1), k], [i, j + 1, k], [i, j, k.wrapping_sub(1)], [i, j, k + 1]];
                let cross = nb.iter().filter(|p| p[0] < nx && p[1] < ny && p[2] < nz).any(|p| m.get(p[0], p[1], p[2]) != (m.data[q] != 0));
                cross && (m.data[q] != 0) == inner
            })
            .collect()
    }

    #[test]
    fn smooth_hollow_shells_match_their_volumes() {
        let (n, r) = (35, 10.0);
        let m = ball(n, r);
        let shell = |a: f64, b: f64| 4.0 / 3.0 * PI * (b.powi(3) - a.powi(3));
        let (inner, outer) = (boundary_layer(&m, true), boundary_layer(&m, false));
        for t in [2.0, 3.0, 4.0] {
            for (side, want) in [(HollowSide::Inside, shell(r - t, r)), (HollowSide::Medial, shell(r - t / 2.0, r + t / 2.0)), (HollowSide::Outside, shell(r, r + t))] {
                let mut h = m.clone();
                hollow(&mut h, t, side, [1.0; 3]);
                let got = h.count() as f64;
                assert!((got - want).abs() < 0.1 * want, "{side:?} {t}: {got} vs {want}");
                let mut inside = h.clone();
                inside.intersect(&m);
                match side {
                    HollowSide::Inside => assert_eq!(inside.count(), h.count(), "the inside shell leaves the segment"),
                    HollowSide::Outside => assert_eq!(inside.count(), 0, "the outside shell overlaps the segment"),
                    HollowSide::Medial => assert!(inside.count() > 0 && inside.count() < h.count()),
                }
                // Closed: it holds the boundary layer on its side; hollow: the centre is empty.
                let layer = if side == HollowSide::Outside { &outer } else { &inner };
                assert!(layer.iter().all(|&q| h.data[q] != 0), "{side:?} {t}: open");
                assert!(!h.get(17, 17, 17));
            }
        }
        // Thinner than a voxel, the shell is still a closed voxel layer.
        for (side, layer) in [(HollowSide::Inside, &inner), (HollowSide::Medial, &inner), (HollowSide::Outside, &outer)] {
            let mut h = m.clone();
            hollow(&mut h, 0.3, side, [1.0; 3]);
            assert!(layer.iter().all(|&q| h.data[q] != 0), "{side:?}: open");
            assert!(h.count() < 2 * layer.len());
        }
        let mut same = m.clone();
        hollow(&mut same, 0.0, HollowSide::Outside, [1.0; 3]);
        assert_eq!(same, m);
        // Anisotropic voxels: a ball of 6 mm on 0.5 × 0.5 × 1 mm voxels, a 1.5 mm outside shell.
        let mut a = Mask::new([40, 40, 20]);
        for k in 0..20 {
            for j in 0..40 {
                for i in 0..40 {
                    let p = [i as f64 * 0.5 - 9.75, j as f64 * 0.5 - 9.75, k as f64 - 9.5];
                    a.set(i, j, k, p.iter().map(|x| x * x).sum::<f64>() <= 36.0);
                }
            }
        }
        let a0 = a.clone();
        hollow(&mut a, 1.5, HollowSide::Outside, [0.5, 0.5, 1.0]);
        let (got, want) = (a.count() as f64 * 0.25, shell(6.0, 7.5));
        assert!((got - want).abs() < 0.1 * want, "anisotropic: {got} vs {want}");
        a.intersect(&a0);
        assert_eq!(a.count(), 0);
        // Next to the grid's edge, the outside shell is cut by it.
        let mut e = Mask::new([10, 10, 10]);
        e.set(0, 5, 5, true);
        hollow(&mut e, 1.0, HollowSide::Outside, [1.0; 3]);
        assert!(e.get(1, 5, 5) && e.get(0, 4, 5) && e.get(0, 6, 5) && e.get(0, 5, 4) && e.get(0, 5, 6) && !e.get(0, 5, 5));
        assert!(e.count() <= 13);
    }
}
