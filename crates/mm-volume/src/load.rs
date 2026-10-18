//! A whole 3D volume in memory, for viewing and segmenting: grey values in voxel order
//! (i fastest, then j, then k) with the grid's placement in patient space. Raw formats are
//! read in one pass; DICOM files and 2D images are decoded in parallel, each file opened
//! once (an Enhanced multi-frame file is decoded whole).

use crate::geometry::Geometry;
use crate::model::{Volume, Voxels};
use crate::slice::load_raw;
use anyhow::{bail, Context, Result};
use rayon::prelude::*;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

#[derive(Clone, Debug)]
pub struct LoadedVolume {
    pub name: String,
    pub dims: [usize; 3],
    pub values: Vec<f32>,
    /// Voxel → RAS millimetres (first 3D volume of a 4D set).
    pub geometry: Geometry,
    /// Smallest and largest finite value.
    pub range: (f32, f32),
    /// Window (centre, width) from the 1st to the 99th percentile.
    pub auto_window: (f64, f64),
    /// The window stored in the file, if any.
    pub file_window: Option<(f64, f64)>,
    /// Show inverted (DICOM MONOCHROME1).
    pub invert: bool,
}

impl LoadedVolume {
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[self.index(i, j, k)]
    }

    pub fn voxel_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// A volume from values in memory (tests, results of processing).
    pub fn from_values(name: &str, dims: [usize; 3], values: Vec<f32>, geometry: Geometry) -> Self {
        let range = finite_range(&values);
        let auto_window = sampled_window(&values);
        Self { name: name.into(), dims, values, geometry, range, auto_window, file_window: None, invert: false }
    }
}

fn finite_range(v: &[f32]) -> (f32, f32) {
    let (lo, hi) = v.par_iter().copied().filter(|x| x.is_finite()).fold(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x))).reduce(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), (c, d)| (a.min(c), b.max(d)));
    if lo <= hi { (lo, hi) } else { (0.0, 0.0) }
}

/// Percentile window from at most a million evenly spread values.
fn sampled_window(v: &[f32]) -> (f64, f64) {
    let step = (v.len() / 1_000_000).max(1);
    let sample: Vec<f32> = v.iter().step_by(step).copied().collect();
    crate::slice::auto_window(&sample)
}

/// Load the first 3D volume of `v` (at most `max_voxels` voxels). `progress` gets the
/// fraction done now and then; returning `false` cancels.
pub fn load_volume(v: &Volume, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<LoadedVolume> {
    let g = v.geometry.clone().context("no geometry")?;
    let dims = [g.dims[0], g.dims[1], g.dims[2]];
    let n = dims.iter().product::<usize>();
    if n == 0 {
        bail!("empty volume");
    }
    if n > max_voxels {
        bail!("too large to load ({} million voxels; the limit is {} million)", n / 1_000_000, max_voxels / 1_000_000);
    }
    let values = match &v.voxels {
        Voxels::Raw { .. } => {
            progress(0.1);
            load_raw(v, max_voxels)?.values
        }
        Voxels::Dicom { slices } => load_planes(slices, dims, progress, true)?,
        Voxels::Images { slices } => load_planes(slices, dims, progress, false)?,
        Voxels::Unsupported(why) => bail!("pixels not readable yet: {why}"),
    };
    if values.len() != n {
        bail!("read {} values, expected {n}", values.len());
    }
    let mut out = LoadedVolume::from_values(&v.name, dims, values, g);
    out.file_window = v.window;
    out.invert = v.photometric.as_deref() == Some("MONOCHROME1");
    Ok(out)
}

/// Decode slices stored file by file (or frame by frame) into one array.
/// Decoded planes of one file: (slice index, values).
type Planes = Vec<(usize, Vec<f32>)>;

fn load_planes(slices: &[(PathBuf, u32)], dims: [usize; 3], progress: &(dyn Fn(f32) -> bool + Sync), dicom: bool) -> Result<Vec<f32>> {
    let plane = dims[0] * dims[1];
    // Consecutive slices from the same file are decoded together.
    let mut groups: Vec<(PathBuf, Vec<(usize, u32)>)> = Vec::new();
    for (k, (p, f)) in slices.iter().enumerate().take(dims[2]) {
        match groups.last_mut() {
            Some((q, list)) if q == p => list.push((k, *f)),
            _ => groups.push((p.clone(), vec![(k, *f)])),
        }
    }
    let done = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let total = dims[2].max(1);
    let decoded: Vec<Result<Planes>> = groups
        .par_iter()
        .map(|(path, frames)| {
            if cancelled.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            let planes = if dicom { dicom_frames(path, frames)? } else { frames.iter().map(|&(k, page)| Ok((k, crate::stack::read_plane(path, page)?.2))).collect::<Result<Vec<_>>>()? };
            let d = done.fetch_add(frames.len(), Ordering::Relaxed) + frames.len();
            if !progress(d as f32 / total as f32) {
                cancelled.store(true, Ordering::Relaxed);
            }
            Ok(planes)
        })
        .collect();
    if cancelled.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    let mut values = vec![0.0f32; plane * dims[2]];
    for r in decoded {
        for (k, p) in r? {
            if p.len() != plane {
                bail!("slice {} has {} pixels, expected {plane}", k + 1, p.len());
            }
            values[k * plane..(k + 1) * plane].copy_from_slice(&p);
        }
    }
    Ok(values)
}

/// Decode frames of one DICOM file to grey values (modality LUT applied).
fn dicom_frames(path: &std::path::Path, frames: &[(usize, u32)]) -> Result<Planes> {
    use dicom_pixeldata::PixelDecoder;
    let obj = dicom_object::open_file(path).with_context(|| format!("cannot read {}", path.display()))?;
    if crate::dicom::is_jpegls(obj.meta().transfer_syntax()) {
        return frames.iter().map(|&(k, f)| Ok((k, crate::dicom::jpegls_frame(&obj, f)?.2))).collect();
    }
    let grey = |raw: Vec<f32>, n: usize| -> Vec<f32> {
        let samples = (raw.len() / n.max(1)).max(1);
        if samples >= 3 { raw.chunks(samples).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect() } else { raw }
    };
    if frames.len() == 1 {
        let (k, f) = frames[0];
        let px = obj.decode_pixel_data_frame(f).map_err(|e| anyhow::anyhow!("cannot decode {}: {e}", path.display()))?;
        let n = px.columns() as usize * px.rows() as usize;
        let raw: Vec<f32> = px.to_vec_frame(0).map_err(|e| anyhow::anyhow!("cannot convert the pixels: {e}"))?;
        return Ok(vec![(k, grey(raw, n))]);
    }
    let px = obj.decode_pixel_data().map_err(|e| anyhow::anyhow!("cannot decode {}: {e}", path.display()))?;
    let n = px.columns() as usize * px.rows() as usize;
    frames
        .iter()
        .map(|&(k, f)| {
            let raw: Vec<f32> = px.to_vec_frame(f).map_err(|e| anyhow::anyhow!("cannot convert frame {f}: {e}"))?;
            Ok((k, grey(raw, n)))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DataType, Format, Kind};
    use glam::DVec3;

    #[test]
    fn raw_and_stack_volumes_load_whole() {
        let dir = std::env::temp_dir().join(format!("mmv-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("v.raw");
        std::fs::write(&p, (0..24i16).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let g = Geometry::from_axes([4, 3, 2, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let mut v = Volume {
            format: Format::Nifti1,
            kind: Kind::Image,
            name: "v".into(),
            files: vec![p.clone()],
            bytes: 48,
            geometry: Some(g.clone()),
            data_type: Some(DataType::I16),
            fields: vec![],
            warnings: vec![],
            voxels: Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 },
            window: None,
            photometric: None,
            patient: None,
        };
        let l = load_volume(&v, 1000, &|_| true).unwrap();
        assert_eq!(l.get(3, 2, 1), 23.0);
        assert_eq!(l.range, (0.0, 23.0));
        assert!(load_volume(&v, 10, &|_| true).is_err());
        // Two PNG slices.
        let mut slices = Vec::new();
        for k in 0..2u8 {
            let f = dir.join(format!("s{k}.png"));
            image::GrayImage::from_fn(4, 3, |x, y| image::Luma([k * 12 + (y * 4 + x) as u8])).save(&f).unwrap();
            slices.push((f, 0));
        }
        v.voxels = Voxels::Images { slices };
        let l = load_volume(&v, 1000, &|_| true).unwrap();
        assert_eq!(l.values, (0..24).map(|x| x as f32).collect::<Vec<_>>());
        assert!(load_volume(&v, 1000, &|_| false).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
