//! 4D images (time points, echoes, components): pick one 3D volume, or combine all of them
//! voxel by voxel (mean, maximum, minimum), e.g. averaging repeated acquisitions or a maximum
//! over time of a 4D angiogram.

use crate::image::{check, Progress};
use anyhow::{bail, Result};
use mm_volume::load::{load_volume_at, volume_count, LoadedVolume};
use mm_volume::model::Volume;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Combine {
    Mean,
    Max,
    Min,
}

impl Combine {
    pub const ALL: [Combine; 3] = [Combine::Mean, Combine::Max, Combine::Min];

    pub fn label(self) -> &'static str {
        match self {
            Combine::Mean => "Mean",
            Combine::Max => "Maximum",
            Combine::Min => "Minimum",
        }
    }
}

/// Combine every 3D volume of `v`, voxel by voxel.
pub fn combine(v: &Volume, how: Combine, max_voxels: usize, progress: Progress) -> Result<LoadedVolume> {
    let n = volume_count(v);
    if n < 2 {
        bail!("this image has a single volume");
    }
    let mut acc: Option<LoadedVolume> = None;
    for t in 0..n {
        let lo = t as f32 / n as f32;
        let vol = load_volume_at(v, t, max_voxels, &|f| progress(lo + f / n as f32))?;
        match &mut acc {
            None => acc = Some(vol),
            Some(a) => {
                for (x, y) in a.values.iter_mut().zip(&vol.values) {
                    *x = match how {
                        Combine::Mean => *x + y,
                        Combine::Max => x.max(*y),
                        Combine::Min => x.min(*y),
                    };
                }
            }
        }
        check(progress, (t + 1) as f32 / n as f32)?;
    }
    let a = acc.unwrap();
    let values = if how == Combine::Mean { a.values.iter().map(|x| x / n as f32).collect() } else { a.values.clone() };
    Ok(crate::image::with_values(&a, &a.name, values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use mm_volume::model::{DataType, Format, Kind, Voxels};

    #[test]
    fn combines_time_points() {
        let dir = std::env::temp_dir().join(format!("mmi-time-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("t.raw");
        // Three time points of 2 voxels: (1, 10), (3, 20), (2, 0).
        std::fs::write(&p, [1.0f32, 10.0, 3.0, 20.0, 2.0, 0.0].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let v = Volume {
            format: Format::Nifti1,
            kind: Kind::TimeSeries,
            name: "t".into(),
            files: vec![p.clone()],
            bytes: 24,
            geometry: Some(Geometry::from_axes([2, 1, 1, 3], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO)),
            data_type: Some(DataType::F32),
            fields: vec![],
            warnings: vec![],
            voxels: Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::F32, big_endian: false, slope: 1.0, intercept: 0.0 },
            window: None,
            photometric: None,
            patient: None,
        };
        let any = &|_| true;
        assert_eq!(combine(&v, Combine::Mean, 100, any).unwrap().values, vec![2.0, 10.0]);
        assert_eq!(combine(&v, Combine::Max, 100, any).unwrap().values, vec![3.0, 20.0]);
        assert_eq!(combine(&v, Combine::Min, 100, any).unwrap().values, vec![1.0, 0.0]);
        assert_eq!(combine(&v, Combine::Max, 100, any).unwrap().geometry.dims[3], 1);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
