//! FreeSurfer MGH (`.mgh`) and gzip-compressed MGZ (`.mgz`): a 284-byte big-endian header
//! (size, type, voxel size, direction cosines and the RAS position of the volume's centre),
//! then the voxels.

use crate::geometry::Geometry;
use crate::model::{DataType, Field, Format, Kind, Volume, Voxels};
use anyhow::{bail, Result};
use glam::{DMat3, DVec3};
use std::path::Path;

/// Header bytes before the voxels.
pub const HEADER_BYTES: usize = 284;

#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub dims: [usize; 4],
    pub type_code: i32,
    /// Whether the header holds a real orientation (else it is the FreeSurfer default).
    pub good_ras: bool,
    pub spacing: [f64; 3],
    /// Unit RAS direction of each voxel axis.
    pub axes: [DVec3; 3],
    /// RAS position of the centre of the volume.
    pub centre: DVec3,
}

fn i32_at(b: &[u8], at: usize) -> i32 {
    i32::from_be_bytes(b[at..at + 4].try_into().unwrap())
}

fn f32_at(b: &[u8], at: usize) -> f64 {
    f32::from_be_bytes(b[at..at + 4].try_into().unwrap()) as f64
}

/// Whether `b` (the first bytes, decompressed) looks like an MGH header.
pub fn sniff(b: &[u8]) -> bool {
    b.len() >= 30 && i32_at(b, 0) == 1 && (1..=4).all(|k| (1..1 << 20).contains(&i32_at(b, 4 * k))) && matches!(i32_at(b, 20), 0 | 1 | 3 | 4)
}

pub fn parse(b: &[u8]) -> Result<Header> {
    if b.len() < HEADER_BYTES || !sniff(b) {
        bail!("not an MGH header");
    }
    let dims = [i32_at(b, 4) as usize, i32_at(b, 8) as usize, i32_at(b, 12) as usize, i32_at(b, 16) as usize];
    let good_ras = i16::from_be_bytes([b[28], b[29]]) > 0;
    let (spacing, axes, centre) = if good_ras {
        let v = |at: usize| DVec3::new(f32_at(b, at), f32_at(b, at + 4), f32_at(b, at + 8));
        let s = v(30);
        (s.to_array(), [v(42).normalize_or_zero(), v(54).normalize_or_zero(), v(66).normalize_or_zero()], v(78))
    } else {
        // FreeSurfer's default for a header without orientation: coronal LIA.
        ([1.0; 3], [DVec3::NEG_X, DVec3::NEG_Z, DVec3::Y], DVec3::ZERO)
    };
    Ok(Header { dims, type_code: i32_at(b, 20), good_ras, spacing, axes, centre })
}

impl Header {
    pub fn data_type(&self) -> Option<DataType> {
        Some(match self.type_code {
            0 => DataType::U8,
            1 => DataType::I32,
            3 => DataType::F32,
            4 => DataType::I16,
            _ => return None,
        })
    }

    pub fn geometry(&self) -> Geometry {
        // The centre voxel (dims / 2) sits at `centre`.
        let m = DMat3::from_cols(self.axes[0] * self.spacing[0], self.axes[1] * self.spacing[1], self.axes[2] * self.spacing[2]);
        let half = DVec3::new(self.dims[0] as f64, self.dims[1] as f64, self.dims[2] as f64) * 0.5;
        Geometry::from_axes(self.dims, self.spacing, self.axes, self.centre - m * half)
    }
}

/// Build the volume for an `.mgh` or `.mgz` file (`gzip` for the latter).
pub fn volume(path: &Path, gzip: bool, h: &Header) -> Result<Volume> {
    let geometry = h.geometry();
    let dtype = h.data_type();
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let lower = name.to_lowercase();
    // FreeSurfer's brainmask and norm volumes are images; aseg, aparc, wmparc… are labels.
    let labels = ["aseg", "aparc", "wmparc", "parc", "label", "seg", "filled", "ribbon"];
    let kind = if h.dims[3] > 1 {
        Kind::TimeSeries
    } else if dtype.is_some_and(|t| t.is_integer()) && labels.iter().any(|w| lower.contains(w)) {
        Kind::LabelMap
    } else {
        Kind::Image
    };
    let voxels = match dtype {
        Some(t) => {
            let (file_offset, skip) = if gzip { (0, HEADER_BYTES as u64) } else { (HEADER_BYTES as u64, 0) };
            Voxels::Raw { file: path.to_path_buf(), file_offset, gzip, skip, interleave: 1, dtype: t, big_endian: true, slope: 1.0, intercept: 0.0 }
        }
        None => Voxels::Unsupported(format!("MGH type {}", h.type_code)),
    };
    let mut warnings = Vec::new();
    if !h.good_ras {
        warnings.push("No orientation in the header (FreeSurfer's default assumed)".into());
    }
    let fields = vec![
        Field::new("File", "Format", if gzip { "MGZ (FreeSurfer, gzip)" } else { "MGH (FreeSurfer)" }),
        Field::new("Geometry", "Dimensions", h.dims.iter().filter(|&&d| d > 1).map(|d| d.to_string()).collect::<Vec<_>>().join(" × ")),
        Field::new("Geometry", "Voxel size", format!("{:.4} × {:.4} × {:.4} mm", h.spacing[0], h.spacing[1], h.spacing[2])),
        Field::new("Geometry", "Orientation", geometry.orientation()),
        Field::new("Pixel", "Data type", dtype.map(|t| t.label().to_string()).unwrap_or_else(|| format!("type {}", h.type_code))),
    ];
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    Ok(Volume { format: Format::Mgh, kind, name, files: vec![path.to_path_buf()], bytes, geometry: Some(geometry), data_type: dtype, fields, warnings, voxels, window: None, photometric: None, patient: None })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice::load_raw;

    /// A conformed-style 4 × 3 × 2 uint8 volume, LIA, 1 mm.
    fn mgh_bytes() -> Vec<u8> {
        let mut b = vec![0u8; HEADER_BYTES];
        for (k, v) in [1i32, 4, 3, 2, 1, 0, 0].iter().enumerate() {
            b[4 * k..4 * k + 4].copy_from_slice(&v.to_be_bytes());
        }
        b[28..30].copy_from_slice(&1i16.to_be_bytes());
        let floats = [1.0f32, 1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0, 5.0, 6.0, 7.0];
        for (k, f) in floats.iter().enumerate() {
            b[30 + 4 * k..34 + 4 * k].copy_from_slice(&f.to_be_bytes());
        }
        b.extend(0..24u8);
        b
    }

    #[test]
    fn mgh_and_mgz() {
        let b = mgh_bytes();
        let h = parse(&b).unwrap();
        assert_eq!(h.dims, [4, 3, 2, 1]);
        let g = h.geometry();
        assert_eq!(g.orientation(), "LIA");
        // The centre voxel (2, 1.5, 1) is at c_ras.
        assert!((g.world(DVec3::new(2.0, 1.5, 1.0)) - DVec3::new(5.0, 6.0, 7.0)).length() < 1e-9);
        let dir = std::env::temp_dir().join(format!("mmv-mgh-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("T1.mgh");
        std::fs::write(&p, &b).unwrap();
        let v = volume(&p, false, &h).unwrap();
        assert_eq!(load_raw(&v, 100).unwrap().values[23], 23.0);
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(&b).unwrap();
        let p = dir.join("aseg.mgz");
        std::fs::write(&p, e.finish().unwrap()).unwrap();
        let v = volume(&p, true, &h).unwrap();
        assert_eq!(v.kind, Kind::LabelMap);
        assert_eq!(load_raw(&v, 100).unwrap().values[5], 5.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
