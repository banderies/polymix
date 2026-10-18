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
pub fn slab(vol: &LoadedVolume, plane: Plane, thickness_mm: f64, mode: SlabMode, progress: Progress) -> Result<Vec<f32>> {
    let axis = plane_axes(&vol.geometry, plane).slice;
    let half = ((thickness_mm * 0.5) / vol.geometry.spacing[axis].max(1e-9)).round() as i64;
    let d = vol.dims;
    let stride = [1, d[0], d[0] * d[1]][axis];
    let len = d[axis];
    // Every line along the axis.
    let starts: Vec<usize> = (0..vol.values.len()).filter(|&n| (n / stride).is_multiple_of(len)).collect();
    let chunks = starts.len().div_ceil(64).max(1);
    let lines: Vec<(usize, Vec<f32>)> = starts
        .par_chunks(chunks)
        .enumerate()
        .flat_map_iter(|(c, block)| {
            let _ = check(progress, c as f32 / 64.0);
            block.iter().map(|&s| {
                let line: Vec<f32> = (0..len).map(|k| vol.values[s + k * stride]).collect();
                let out = (0..len as i64)
                    .map(|k| {
                        let (a, b) = ((k - half).max(0) as usize, (k + half).min(len as i64 - 1) as usize);
                        let w = &line[a..=b];
                        match mode {
                            SlabMode::Max => w.iter().copied().fold(f32::NEG_INFINITY, f32::max),
                            SlabMode::Min => w.iter().copied().fold(f32::INFINITY, f32::min),
                            SlabMode::Mean => w.iter().sum::<f32>() / w.len() as f32,
                        }
                    })
                    .collect();
                (s, out)
            }).collect::<Vec<_>>()
        })
        .collect();
    check(progress, 1.0)?;
    let mut out = vec![0.0f32; vol.values.len()];
    for (s, line) in lines {
        for (k, v) in line.into_iter().enumerate() {
            out[s + k * stride] = v;
        }
    }
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
    }
}
