//! Thick-slab projections: every voxel becomes the maximum (MIP), minimum (MinIP) or mean of
//! the voxels within half the slab thickness of it, across the slices of a plane (axial,
//! coronal or sagittal). The grid is unchanged, so the result scrolls like the original: a MIP
//! slab follows vessels from slice to slice (TOF, CTA), a MinIP slab veins (SWI).

use crate::image::{check, Progress};
use anyhow::Result;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::{plane_axes, Plane};
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SlabMode {
    Max,
    Min,
    Mean,
}

impl SlabMode {
    pub const ALL: [SlabMode; 3] = [SlabMode::Max, SlabMode::Min, SlabMode::Mean];

    pub fn label(self) -> &'static str {
        match self {
            SlabMode::Max => "Maximum (MIP)",
            SlabMode::Min => "Minimum (MinIP)",
            SlabMode::Mean => "Mean",
        }
    }
}

/// Project slabs `thickness_mm` thick across the slices of `plane`.
///
/// Every output voxel folds the voxels of its slab in slice order, starting from −∞ (MIP),
/// +∞ (MinIP) or −0 (the mean's sum, as `Iterator::sum` starts), the mean then divided by the
/// count. Across planes of the grid the slices fold whole planes, across rows whole rows, so
/// the voxels side by side fold together.
pub fn slab(vol: &LoadedVolume, plane: Plane, thickness_mm: f64, mode: SlabMode, progress: Progress) -> Result<Vec<f32>> {
    let axis = plane_axes(&vol.geometry, plane).slice;
    // (A negative thickness makes a slab of one slice.)
    let half = (((thickness_mm * 0.5) / vol.geometry.spacing[axis].max(1e-9)).round() as i64).max(0);
    let d = vol.dims;
    let len = d[axis];
    // The slab of slice k: slices a..=b.
    let window = |k: usize| ((k as i64 - half).max(0) as usize, (k as i64 + half).min(len as i64 - 1) as usize);
    let start = match mode {
        SlabMode::Max => f32::NEG_INFINITY,
        SlabMode::Min => f32::INFINITY,
        SlabMode::Mean => -0.0,
    };
    // Fold `src` into `acc` voxel by voxel.
    let fold = |acc: &mut [f32], src: &[f32]| match mode {
        SlabMode::Max => acc.iter_mut().zip(src).for_each(|(a, &x)| *a = a.max(x)),
        SlabMode::Min => acc.iter_mut().zip(src).for_each(|(a, &x)| *a = a.min(x)),
        SlabMode::Mean => acc.iter_mut().zip(src).for_each(|(a, &x)| *a += x),
    };
    let finish = |acc: &mut [f32], (a, b): (usize, usize)| {
        if mode == SlabMode::Mean {
            let count = (b - a + 1) as f32;
            acc.iter_mut().for_each(|x| *x /= count);
        }
    };
    let plane_len = d[0] * d[1];
    let row_of = |j: usize, k: usize| &vol.values[(k * d[1] + j) * d[0]..][..d[0]];
    let done = std::sync::atomic::AtomicUsize::new(0);
    let stop = std::sync::atomic::AtomicBool::new(false);
    let mut out = vec![0.0f32; vol.values.len()];
    out.par_chunks_mut(plane_len.max(1)).enumerate().for_each(|(k, o)| {
        if stop.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        match axis {
            2 => {
                let w = window(k);
                o.fill(start);
                for kk in w.0..=w.1 {
                    fold(o, &vol.values[kk * plane_len..][..plane_len]);
                }
                finish(o, w);
            }
            1 => {
                for (j, row) in o.chunks_mut(d[0]).enumerate() {
                    let w = window(j);
                    row.fill(start);
                    for jj in w.0..=w.1 {
                        fold(row, row_of(jj, k));
                    }
                    finish(row, w);
                }
            }
            _ => {
                // Along the row: the voxels i fold their slab's offsets t in turn, where
                // i + t is in the row (the same terms in the same order as one by one).
                let reach = half.min(len as i64 - 1);
                for (j, row) in o.chunks_mut(d[0]).enumerate() {
                    let src = row_of(j, k);
                    row.fill(start);
                    for t in -reach..=reach {
                        let (lo, hi) = ((-t).max(0) as usize, (len as i64 - t).min(len as i64) as usize);
                        fold(&mut row[lo..hi], &src[(lo as i64 + t) as usize..(hi as i64 + t) as usize]);
                    }
                    for (i, x) in row.iter_mut().enumerate() {
                        finish(std::slice::from_mut(x), window(i));
                    }
                }
            }
        }
        let n = done.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
        if !progress(n as f32 / d[2] as f32) {
            stop.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    });
    if stop.load(std::sync::atomic::Ordering::Relaxed) {
        anyhow::bail!("cancelled");
    }
    check(progress, 1.0)?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::image::no_progress;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;

    #[test]
    fn slabs_along_the_slices() {
        // Axial slices along k, 2 mm apart; a bright dot in slice 5.
        let dims = [3, 3, 10];
        let mut values = vec![1.0f32; 90];
        values[5 * 9 + 4] = 9.0;
        let v = LoadedVolume::from_values("s", dims, values, Geometry::from_axes([3, 3, 10, 1], [1.0, 1.0, 2.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
        // 8 mm slab: ±2 slices.
        let m = slab(&v, Plane::Axial, 8.0, SlabMode::Max, no_progress()).unwrap();
        let at = |k: usize| m[k * 9 + 4];
        assert_eq!((at(2), at(3), at(7), at(8)), (1.0, 9.0, 9.0, 1.0));
        let mean = slab(&v, Plane::Axial, 8.0, SlabMode::Mean, no_progress()).unwrap();
        assert!((mean[5 * 9 + 4] - (9.0 + 4.0) / 5.0).abs() < 1e-6);
        let min = slab(&v, Plane::Axial, 8.0, SlabMode::Min, no_progress()).unwrap();
        assert!(min.iter().all(|&x| x == 1.0));
        // Across another plane the dot spreads along j instead.
        let c = slab(&v, Plane::Coronal, 2.0, SlabMode::Max, no_progress()).unwrap();
        assert_eq!((c[5 * 9 + 1], c[5 * 9 + 7]), (9.0, 9.0));
        assert!(slab(&v, Plane::Axial, 8.0, SlabMode::Max, &|_| false).is_err());
    }

    /// The projection written plainly: every voxel folds its slab's voxels in slice order.
    fn reference(vol: &LoadedVolume, plane: Plane, thickness_mm: f64, mode: SlabMode) -> Vec<f32> {
        let axis = plane_axes(&vol.geometry, plane).slice;
        let half = ((thickness_mm * 0.5) / vol.geometry.spacing[axis].max(1e-9)).round() as i64;
        let d = vol.dims;
        let (stride, len) = ([1, d[0], d[0] * d[1]][axis], d[axis]);
        (0..vol.values.len())
            .map(|q| {
                let k = ((q / stride) % len) as i64;
                let s = q - k as usize * stride;
                let line: Vec<f32> = ((k - half).max(0)..=(k + half).min(len as i64 - 1)).map(|t| vol.values[s + t as usize * stride]).collect();
                match mode {
                    SlabMode::Max => line.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                    SlabMode::Min => line.iter().copied().fold(f32::INFINITY, f32::min),
                    SlabMode::Mean => line.iter().sum::<f32>() / line.len() as f32,
                }
            })
            .collect()
    }

    #[test]
    fn matches_the_plain_projection_bit_for_bit() {
        let dims = [13, 11, 9];
        let mut seed = 17u64;
        let values = (0..dims.iter().product::<usize>())
            .map(|q| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                match q % 23 {
                    0 => -0.0,
                    1 => 0.0,
                    2 => f32::NAN,
                    3 => f32::NEG_INFINITY,
                    _ => ((seed >> 40) as f32 / 65536.0 - 128.0) * 0.37,
                }
            })
            .collect();
        // An oblique-ish grid so the planes do not follow the voxel axes in order.
        let axes = [DVec3::Y, DVec3::Z, DVec3::X];
        let v = LoadedVolume::from_values("s", dims, values, Geometry::from_axes([13, 11, 9, 1], [0.7, 1.1, 1.6], axes, DVec3::ZERO));
        for plane in [Plane::Axial, Plane::Coronal, Plane::Sagittal] {
            for mode in SlabMode::ALL {
                for thickness in [0.1, 2.0, 5.5, 40.0] {
                    let a = slab(&v, plane, thickness, mode, no_progress()).unwrap();
                    let b = reference(&v, plane, thickness, mode);
                    let same = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits() || (x.is_nan() && y.is_nan()));
                    assert!(same, "{plane:?} {mode:?} {thickness} mm");
                }
            }
        }
    }
}
