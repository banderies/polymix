//! MetaImage (ITK): a text header of `Key = Value` lines, then the voxels in the same file
//! (`.mha`, `ElementDataFile = LOCAL`) or in a separate one (`.mhd` + `.raw` / `.zraw`).
//! Positions and directions are in ITK's LPS patient coordinates; `TransformMatrix` lists the
//! direction of each voxel axis in turn.

use crate::geometry::{lps_to_ras, Geometry};
use crate::model::{DataType, Field, Format, Kind, Volume, Voxels};
use anyhow::{bail, Context, Result};
use glam::DVec3;
use std::path::{Path, PathBuf};

/// The header's keys in file order, and where the voxels start (after `ElementDataFile`).
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub pairs: Vec<(String, String)>,
    pub data_offset: u64,
}

impl Header {
    pub fn get(&self, key: &str) -> Option<&str> {
        self.pairs.iter().find(|(k, _)| k.eq_ignore_ascii_case(key)).map(|(_, v)| v.as_str())
    }

    fn numbers(&self, keys: &[&str]) -> Vec<f64> {
        keys.iter().find_map(|k| self.get(k)).map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default()
    }

    fn flag(&self, keys: &[&str]) -> bool {
        keys.iter().find_map(|k| self.get(k)).is_some_and(|v| v.eq_ignore_ascii_case("true") || v == "1")
    }
}

/// Whether a file starts like a MetaImage header.
pub fn sniff(head: &[u8]) -> bool {
    let text = String::from_utf8_lossy(&head[..head.len().min(1024)]);
    let first = text.lines().next().unwrap_or("").trim_start();
    ["ObjectType", "NDims", "Comment", "ObjectSubType", "TransformType"].iter().any(|k| first.starts_with(k)) && text.contains("NDims")
}

/// Parse the header: lines up to and including `ElementDataFile`.
pub fn parse(b: &[u8]) -> Result<Header> {
    let mut pairs = Vec::new();
    let mut at = 0usize;
    while at < b.len() {
        let end = b[at..].iter().position(|&c| c == b'\n').map(|p| at + p + 1).unwrap_or(b.len());
        let line = String::from_utf8_lossy(&b[at..end]);
        at = end;
        let Some((k, v)) = line.split_once('=') else { continue };
        let (k, v) = (k.trim().to_string(), v.trim().to_string());
        let last = k.eq_ignore_ascii_case("ElementDataFile");
        pairs.push((k, v));
        if last {
            return Ok(Header { pairs, data_offset: at as u64 });
        }
        if at > 1 << 20 {
            break;
        }
    }
    bail!("no ElementDataFile in the MetaImage header")
}

fn data_type(t: &str) -> Option<DataType> {
    Some(match t.to_ascii_uppercase().as_str() {
        "MET_UCHAR" => DataType::U8,
        "MET_CHAR" => DataType::I8,
        "MET_USHORT" => DataType::U16,
        "MET_SHORT" => DataType::I16,
        "MET_UINT" | "MET_ULONG" => DataType::U32,
        "MET_INT" | "MET_LONG" => DataType::I32,
        "MET_ULONG_LONG" => DataType::U64,
        "MET_LONG_LONG" => DataType::I64,
        "MET_FLOAT" => DataType::F32,
        "MET_DOUBLE" => DataType::F64,
        _ => return None,
    })
}

/// Build the volume for a MetaImage file (`.mha` or `.mhd`).
pub fn volume(path: &Path, h: &Header) -> Result<Volume> {
    let n = h.get("NDims").and_then(|s| s.trim().parse::<usize>().ok()).context("no NDims")?;
    if !(2..=4).contains(&n) {
        bail!("{n}-dimensional MetaImage files are not supported");
    }
    let sizes: Vec<usize> = h.get("DimSize").context("no DimSize")?.split_whitespace().filter_map(|x| x.parse().ok()).collect();
    if sizes.len() < n || sizes[..n].contains(&0) {
        bail!("bad DimSize");
    }
    let channels = h.get("ElementNumberOfChannels").and_then(|s| s.trim().parse::<usize>().ok()).unwrap_or(1).max(1);
    let mut dims = [1usize; 4];
    dims[..n].copy_from_slice(&sizes[..n]);
    let spacing_in = h.numbers(&["ElementSpacing", "ElementSize"]);
    let spacing = [0, 1, 2].map(|k| spacing_in.get(k).copied().filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0));
    let origin = h.numbers(&["Offset", "Origin", "Position"]);
    let origin = DVec3::new(origin.first().copied().unwrap_or(0.0), origin.get(1).copied().unwrap_or(0.0), origin.get(2).copied().unwrap_or(0.0));
    let m = h.numbers(&["TransformMatrix", "Rotation", "Orientation"]);
    let mut axes = [DVec3::X, DVec3::Y, DVec3::Z];
    if m.len() == n * n {
        for (a, axis) in axes.iter_mut().enumerate().take(n.min(3)) {
            let d = DVec3::new(m[a * n], m[a * n + 1], if n >= 3 { m[a * n + 2] } else { 0.0 });
            if d.length() > 1e-9 {
                *axis = d.normalize();
            }
        }
        if n == 2 {
            axes[2] = axes[0].cross(axes[1]).normalize_or_zero();
        }
    }
    let geometry = Geometry::from_axes(dims, spacing, axes.map(lps_to_ras), lps_to_ras(origin));

    let dtype = h.get("ElementType").and_then(data_type);
    let big = h.flag(&["ElementByteOrderMSB", "BinaryDataByteOrderMSB"]);
    let compressed = h.flag(&["CompressedData"]);
    let data = h.get("ElementDataFile").unwrap_or("");
    let local = data.eq_ignore_ascii_case("LOCAL");
    let mut warnings = Vec::new();
    let (data_file, files): (Option<PathBuf>, Vec<PathBuf>) = if local {
        (Some(path.to_path_buf()), vec![path.to_path_buf()])
    } else if data.starts_with("LIST") || data.contains('%') || data.split_whitespace().count() > 1 {
        (None, vec![path.to_path_buf()])
    } else {
        let p = Path::new(data);
        let p = if p.is_absolute() { p.to_path_buf() } else { path.with_file_name(data) };
        if !p.exists() {
            warnings.push("The data file is missing".into());
        }
        (Some(p.clone()), vec![path.to_path_buf(), p])
    };
    let voxels = match (dtype, &data_file) {
        (Some(t), Some(f)) => {
            let voxel_bytes = dims.iter().product::<usize>() as u64 * channels as u64 * t.size() as u64;
            // HeaderSize -1: the data are the last bytes of the file.
            let header_size: i64 = h.get("HeaderSize").and_then(|s| s.trim().parse().ok()).unwrap_or(0);
            let file_offset = if local {
                h.data_offset
            } else if header_size == -1 && !compressed {
                std::fs::metadata(f).map(|m| m.len().saturating_sub(voxel_bytes)).unwrap_or(0)
            } else {
                header_size.max(0) as u64
            };
            Voxels::Raw { file: f.clone(), file_offset, gzip: compressed, skip: 0, interleave: channels, dtype: t, big_endian: big, slope: 1.0, intercept: 0.0 }
        }
        (None, _) => Voxels::Unsupported(format!("element type “{}”", h.get("ElementType").unwrap_or(""))),
        (_, None) => Voxels::Unsupported(format!("data in several files ({data})")),
    };
    if let Voxels::Unsupported(why) = &voxels {
        warnings.push(format!("Pixels not readable yet: {why}"));
    }
    if m.is_empty() {
        warnings.push("No TransformMatrix: orientation assumed".into());
    }
    let name = path.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default();
    let lower = name.to_lowercase();
    let kind = if channels > 1 && !h.get("ElementType").is_some_and(|t| t.eq_ignore_ascii_case("MET_UCHAR")) {
        Kind::VectorField
    } else if dims[3] > 1 {
        Kind::TimeSeries
    } else if dtype.is_some_and(|t| t.is_integer()) && crate::nifti::LABEL_HINTS.iter().any(|w| lower.contains(w)) {
        Kind::LabelMap
    } else {
        Kind::Image
    };
    let mut fields = vec![
        Field::new("File", "Format", format!("MetaImage{}{}", if local { "" } else { " (separate data file)" }, if compressed { ", zlib" } else { "" })),
        Field::new("Geometry", "Dimensions", sizes[..n].iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" × ")),
        Field::new("Geometry", "Voxel size", format!("{:.4} × {:.4} × {:.4} mm", spacing[0], spacing[1], spacing[2])),
        Field::new("Geometry", "Orientation", geometry.orientation()),
        Field::new("Pixel", "Data type", h.get("ElementType").unwrap_or("?")),
    ];
    if channels > 1 {
        fields.push(Field::new("Pixel", "Channels", channels.to_string()));
    }
    for (k, label) in [("Modality", "Modality"), ("AnatomicalOrientation", "Anatomical orientation"), ("Comment", "Comment")] {
        if let Some(v) = h.get(k) {
            fields.push(Field::new("File", label, v));
        }
    }
    let bytes = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    Ok(Volume { format: Format::MetaImage, kind, name, files, bytes, geometry: Some(geometry), data_type: dtype, fields, warnings, voxels, window: None, photometric: None, patient: None })
}

/// Read and parse the header of a MetaImage file.
pub fn read(path: &Path) -> Result<Header> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?.take(64 << 10).read_to_end(&mut buf)?;
    parse(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::slice::load_raw;

    #[test]
    fn local_and_detached_compressed() {
        let dir = std::env::temp_dir().join(format!("mmv-mha-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let values: Vec<i16> = (0..24).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        // .mha with the data after the header; directions axial LPS, origin in LPS.
        let head = "ObjectType = Image\nNDims = 3\nBinaryData = True\nBinaryDataByteOrderMSB = False\nCompressedData = False\nTransformMatrix = 1 0 0 0 1 0 0 0 1\nOffset = -10 -20 30\nElementSpacing = 0.5 0.5 2\nDimSize = 4 3 2\nElementType = MET_SHORT\nElementDataFile = LOCAL\n";
        let p = dir.join("ct.mha");
        std::fs::write(&p, [head.as_bytes(), &bytes].concat()).unwrap();
        assert!(sniff(&std::fs::read(&p).unwrap()));
        let v = volume(&p, &read(&p).unwrap()).unwrap();
        let g = v.geometry.as_ref().unwrap();
        assert_eq!(g.dims, [4, 3, 2, 1]);
        assert_eq!(g.orientation(), "LPS");
        assert_eq!(g.world(DVec3::ZERO), DVec3::new(10.0, 20.0, 30.0));
        assert_eq!(load_raw(&v, 1000).unwrap().values, values.iter().map(|&x| x as f32).collect::<Vec<_>>());
        // .mhd + zlib-compressed .zraw.
        use std::io::Write;
        let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(&bytes).unwrap();
        std::fs::write(dir.join("ct.zraw"), e.finish().unwrap()).unwrap();
        let p = dir.join("ct.mhd");
        std::fs::write(&p, head.replace("CompressedData = False", "CompressedData = True").replace("LOCAL", "ct.zraw")).unwrap();
        let v = volume(&p, &read(&p).unwrap()).unwrap();
        assert_eq!(v.files.len(), 2);
        assert_eq!(load_raw(&v, 1000).unwrap().values[23], 23.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
