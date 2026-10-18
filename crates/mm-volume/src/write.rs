//! Writing volumes as NIfTI-1 (`.nii`, or gzip-compressed `.nii.gz`): one 3D grid of uint8,
//! int16 or float32 values with its placement (sform and qform both set from the grid), so
//! derived images and label maps open in Polymix, 3D Slicer, ITK-SNAP, FSL and nibabel.

use crate::geometry::Geometry;
use anyhow::{bail, Context, Result};
use glam::{DMat3, DVec3};
use std::io::Write;
use std::path::Path;

/// Voxel values to write, in voxel order (i fastest).
#[derive(Clone, Copy, Debug)]
pub enum Data<'a> {
    U8(&'a [u8]),
    I16(&'a [i16]),
    F32(&'a [f32]),
}

impl Data<'_> {
    fn len(&self) -> usize {
        match self {
            Data::U8(v) => v.len(),
            Data::I16(v) => v.len(),
            Data::F32(v) => v.len(),
        }
    }

    /// NIfTI datatype code and bits per voxel.
    fn code(&self) -> (i16, i16) {
        match self {
            Data::U8(_) => (2, 8),
            Data::I16(_) => (4, 16),
            Data::F32(_) => (16, 32),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Data::U8(v) => v.to_vec(),
            Data::I16(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Data::F32(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        }
    }
}

/// What the values mean (NIfTI intent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    None,
    /// Integer labels (segments).
    Label,
}

/// Quaternion (b, c, d) and qfac of a grid's rotation, as NIfTI stores it (a left-handed grid
/// flips its third axis and sets qfac = -1).
fn quaternion(dirs: [DVec3; 3]) -> ([f64; 3], f64) {
    let mut r = DMat3::from_cols(dirs[0], dirs[1], dirs[2]);
    let qfac = if r.determinant() < 0.0 { -1.0 } else { 1.0 };
    if qfac < 0.0 {
        r.z_axis = -r.z_axis;
    }
    let m = |i: usize, j: usize| r.col(j)[i];
    let trace = m(0, 0) + m(1, 1) + m(2, 2);
    let (a, b, c, d) = if trace > 0.0 {
        let a = 0.5 * (1.0 + trace).sqrt();
        (a, 0.25 * (m(2, 1) - m(1, 2)) / a, 0.25 * (m(0, 2) - m(2, 0)) / a, 0.25 * (m(1, 0) - m(0, 1)) / a)
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let b = 0.5 * (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt();
        (0.25 * (m(2, 1) - m(1, 2)) / b, b, 0.25 * (m(0, 1) + m(1, 0)) / b, 0.25 * (m(0, 2) + m(2, 0)) / b)
    } else if m(1, 1) > m(2, 2) {
        let c = 0.5 * (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt();
        (0.25 * (m(0, 2) - m(2, 0)) / c, 0.25 * (m(0, 1) + m(1, 0)) / c, c, 0.25 * (m(1, 2) + m(2, 1)) / c)
    } else {
        let d = 0.5 * (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt();
        (0.25 * (m(1, 0) - m(0, 1)) / d, 0.25 * (m(0, 2) + m(2, 0)) / d, 0.25 * (m(1, 2) + m(2, 1)) / d, d)
    };
    // NIfTI keeps a ≥ 0.
    let s = if a < 0.0 { -1.0 } else { 1.0 };
    ([b * s, c * s, d * s], qfac)
}

/// Write `data` (on the grid `geometry`, `dims` voxels) as NIfTI-1, gzip-compressed when the
/// name ends in `.gz`. The file appears complete or not at all.
pub fn nifti(path: &Path, data: Data, dims: [usize; 3], geometry: &Geometry, description: &str, intent: Intent) -> Result<()> {
    if data.len() != dims.iter().product::<usize>() {
        bail!("{} values for a {}×{}×{} grid", data.len(), dims[0], dims[1], dims[2]);
    }
    if dims.iter().any(|&d| d > i16::MAX as usize) {
        bail!("too many voxels along an axis for NIfTI-1");
    }
    let mut h = vec![0u8; 352];
    let put_i16 = |h: &mut [u8], at: usize, v: i16| h[at..at + 2].copy_from_slice(&v.to_le_bytes());
    let put_f32 = |h: &mut [u8], at: usize, v: f64| h[at..at + 4].copy_from_slice(&(v as f32).to_le_bytes());
    h[0..4].copy_from_slice(&348i32.to_le_bytes());
    h[38] = b'r';
    put_i16(&mut h, 40, 3);
    for (k, &d) in dims.iter().enumerate() {
        put_i16(&mut h, 42 + 2 * k, d as i16);
    }
    for k in 3..7 {
        put_i16(&mut h, 42 + 2 * k, 1);
    }
    if intent == Intent::Label {
        put_i16(&mut h, 68, 1002);
    }
    let (code, bits) = data.code();
    put_i16(&mut h, 70, code);
    put_i16(&mut h, 72, bits);
    let ([b, c, d], qfac) = quaternion(geometry.directions());
    put_f32(&mut h, 76, qfac);
    for k in 0..3 {
        put_f32(&mut h, 80 + 4 * k, geometry.spacing[k]);
    }
    put_f32(&mut h, 108, 352.0);
    put_f32(&mut h, 112, 1.0);
    h[123] = 2; // millimetres
    let desc = description.as_bytes();
    h[148..148 + desc.len().min(79)].copy_from_slice(&desc[..desc.len().min(79)]);
    put_i16(&mut h, 252, 1); // qform: scanner
    put_i16(&mut h, 254, 1); // sform: scanner
    put_f32(&mut h, 256, b);
    put_f32(&mut h, 260, c);
    put_f32(&mut h, 264, d);
    let a = &geometry.affine;
    for (k, row) in a.iter().take(3).enumerate() {
        put_f32(&mut h, 268 + 4 * k, row[3]);
    }
    for (row, at) in a.iter().zip([280usize, 296, 312]) {
        for (k, &x) in row.iter().enumerate() {
            put_f32(&mut h, at + 4 * k, x);
        }
    }
    h[344..348].copy_from_slice(b"n+1\0");
    h.extend(data.bytes());

    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let tmp = path.with_extension("partial");
    {
        let f = std::io::BufWriter::new(std::fs::File::create(&tmp).with_context(|| format!("cannot write {}", path.display()))?);
        if name.ends_with(".gz") {
            let mut e = flate2::write::GzEncoder::new(f, flate2::Compression::fast());
            e.write_all(&h)?;
            e.finish()?.flush()?;
        } else {
            let mut f = f;
            f.write_all(&h)?;
            f.flush()?;
        }
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DataType, Kind};
    use crate::slice::load_raw;

    #[test]
    fn float_round_trip() {
        let dir = std::env::temp_dir().join(format!("mmv-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let axes = [DVec3::new(0.8, 0.6, 0.0), DVec3::new(-0.6, 0.8, 0.0), DVec3::new(0.0, 0.0, -1.0)];
        let g = Geometry::from_axes([4, 3, 2, 1], [0.5, 0.75, 2.0], axes, DVec3::new(10.0, -20.0, 30.0));
        let values: Vec<f32> = (0..24).map(|v| v as f32 * 0.25 - 1.0).collect();
        let p = dir.join("n4.nii.gz");
        nifti(&p, Data::F32(&values), [4, 3, 2], &g, "test", Intent::None).unwrap();
        let head = crate::nifti::read_head(&p, true, 352).unwrap();
        let mut h = crate::nifti::parse(&head).unwrap();
        let v = crate::nifti::volume(&p, true, &h).unwrap();
        assert_eq!(v.kind, Kind::Image);
        assert_eq!(v.data_type, Some(DataType::F32));
        assert_eq!(load_raw(&v, 100).unwrap().values, values);
        for (r, e) in v.geometry.as_ref().unwrap().affine.iter().flatten().zip(g.affine.iter().flatten()) {
            assert!((r - e).abs() < 1e-4);
        }
        // The qform alone gives the same grid.
        h.sform_code = 0;
        let q = Geometry::from_affine([4, 3, 2, 1], h.affine().0);
        for (r, e) in q.affine.iter().flatten().zip(g.affine.iter().flatten()) {
            assert!((r - e).abs() < 1e-4, "qform {:?} vs {:?}", q.affine, g.affine);
        }
        assert!(nifti(&p, Data::U8(&[1, 2]), [4, 3, 2], &g, "", Intent::Label).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
