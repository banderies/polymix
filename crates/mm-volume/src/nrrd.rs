//! NRRD (`.nrrd`, and `.nhdr` with a separate data file), as 3D Slicer writes it, including
//! Slicer segmentations (`.seg.nrrd`, with named segments).

use crate::geometry::Geometry;
use crate::model::{DataType, Field, Format, Kind, Volume, Voxels};
use anyhow::{bail, Context, Result};
use glam::DVec3;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// The parsed header.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Header {
    pub version: String,
    /// Field name (lower case) → value.
    pub fields: BTreeMap<String, String>,
    /// `key:=value` pairs (e.g. Slicer's `Segment0_Name`).
    pub keys: BTreeMap<String, String>,
    /// Byte offset of the data in an attached file (just after the blank line).
    pub data_offset: u64,
}

/// Whether `b` starts like a NRRD header.
pub fn sniff(b: &[u8]) -> bool {
    b.starts_with(b"NRRD000")
}

/// Parse a header from the start of the file (`b` must hold the whole header: NRRD headers
/// are a few KB; Slicer segmentations with many segments can reach tens of KB).
pub fn parse(b: &[u8]) -> Result<Header> {
    if !sniff(b) {
        bail!("not a NRRD file");
    }
    let mut h = Header::default();
    let mut pos = 0usize;
    let mut first = true;
    loop {
        let Some(nl) = b[pos..].iter().position(|&c| c == b'\n') else {
            // Header without data (a detached header file ending without a blank line).
            if pos < b.len() {
                parse_line(&mut h, &String::from_utf8_lossy(&b[pos..]), first);
            }
            h.data_offset = b.len() as u64;
            break;
        };
        let line = String::from_utf8_lossy(&b[pos..pos + nl]).trim_end_matches('\r').to_string();
        pos += nl + 1;
        if line.is_empty() {
            h.data_offset = pos as u64;
            break;
        }
        parse_line(&mut h, &line, first);
        first = false;
    }
    if !h.fields.contains_key("sizes") || !h.fields.contains_key("type") {
        bail!("NRRD header without sizes or type");
    }
    Ok(h)
}

fn parse_line(h: &mut Header, line: &str, first: bool) {
    if first {
        h.version = line.trim().to_string();
        return;
    }
    if line.starts_with('#') {
        return;
    }
    if let Some((k, v)) = line.split_once(":=") {
        h.keys.insert(k.trim().to_string(), v.trim().to_string());
    } else if let Some((k, v)) = line.split_once(": ") {
        h.fields.insert(k.trim().to_lowercase(), v.trim().to_string());
    }
}

impl Header {
    fn get(&self, k: &str) -> Option<&str> {
        self.fields.get(k).map(|s| s.as_str())
    }

    pub fn sizes(&self) -> Vec<usize> {
        self.get("sizes").map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default()
    }

    pub fn data_type(&self) -> Option<DataType> {
        Some(match self.get("type")?.to_lowercase().as_str() {
            "signed char" | "int8" | "int8_t" => DataType::I8,
            "uchar" | "unsigned char" | "uint8" | "uint8_t" => DataType::U8,
            "short" | "short int" | "signed short" | "signed short int" | "int16" | "int16_t" => DataType::I16,
            "ushort" | "unsigned short" | "unsigned short int" | "uint16" | "uint16_t" => DataType::U16,
            "int" | "signed int" | "int32" | "int32_t" => DataType::I32,
            "uint" | "unsigned int" | "uint32" | "uint32_t" => DataType::U32,
            "longlong" | "long long" | "long long int" | "signed long long" | "signed long long int" | "int64" | "int64_t" => DataType::I64,
            "ulonglong" | "unsigned long long" | "unsigned long long int" | "uint64" | "uint64_t" => DataType::U64,
            "float" => DataType::F32,
            "double" => DataType::F64,
            _ => return None,
        })
    }

    /// Axis directions (in the file's space) per axis; `None` for non-spatial axes.
    fn space_directions(&self) -> Vec<Option<DVec3>> {
        let Some(s) = self.get("space directions") else { return Vec::new() };
        let mut out = Vec::new();
        let mut rest = s;
        loop {
            rest = rest.trim_start();
            if rest.is_empty() {
                break;
            }
            if let Some(r) = rest.strip_prefix("none") {
                out.push(None);
                rest = r;
            } else if rest.starts_with('(') {
                let end = rest.find(')').unwrap_or(rest.len() - 1);
                out.push(vec3(&rest[1..end]));
                rest = &rest[(end + 1).min(rest.len())..];
            } else {
                break;
            }
        }
        out
    }

    /// Whether the file's space is left-posterior-superior (DICOM-like), so x and y flip
    /// into RAS.
    fn space_is_lps(&self) -> Option<bool> {
        let s = self.get("space")?.to_lowercase();
        Some(match s.as_str() {
            "left-posterior-superior" | "lps" | "left-posterior-superior-time" | "lpst" => true,
            // A generic right-handed space is taken as RAS (what readers such as ITK do).
            "right-anterior-superior" | "ras" | "right-anterior-superior-time" | "rast" | "3d-right-handed" | "3d-right-handed-time" => false,
            "left-anterior-superior" | "las" => {
                // Rare; treat as LPS with y kept, handled by the caller as unknown.
                return None;
            }
            _ => return None,
        })
    }

    pub fn segment_names(&self) -> Vec<String> {
        let mut segs: Vec<(usize, String)> = self
            .keys
            .iter()
            .filter_map(|(k, v)| {
                let n = k.strip_prefix("Segment")?.strip_suffix("_Name")?.parse().ok()?;
                Some((n, v.clone()))
            })
            .collect();
        segs.sort();
        segs.into_iter().map(|(_, v)| v).collect()
    }
}

fn vec3(s: &str) -> Option<DVec3> {
    let v: Vec<f64> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    (v.len() == 3).then(|| DVec3::new(v[0], v[1], v[2]))
}

/// The data file of a header (the file itself when attached).
pub fn data_file(path: &Path, h: &Header) -> Result<Option<PathBuf>> {
    match h.get("data file").or_else(|| h.get("datafile")) {
        None => Ok(None),
        Some(f) if f.starts_with("LIST") || f.contains('%') => bail!("multi-file NRRD data ({f}) is not supported yet"),
        Some(f) => {
            let p = Path::new(f);
            Ok(Some(if p.is_absolute() { p.to_path_buf() } else { path.with_file_name(f) }))
        }
    }
}

/// Build the volume for a NRRD file (`.nrrd` or `.nhdr`).
pub fn volume(path: &Path, h: &Header) -> Result<Volume> {
    let sizes = h.sizes();
    if sizes.is_empty() || sizes.contains(&0) {
        bail!("bad sizes");
    }
    let dtype = h.data_type();
    let detached = data_file(path, h)?;
    let file = detached.clone().unwrap_or_else(|| path.to_path_buf());
    let encoding = h.get("encoding").unwrap_or("raw").to_lowercase();
    let gzip = matches!(encoding.as_str(), "gzip" | "gz");
    let big = h.get("endian").is_some_and(|e| e.eq_ignore_ascii_case("big"));
    let byte_skip: i64 = h.get("byte skip").and_then(|s| s.trim().parse().ok()).unwrap_or(0);
    let mut warnings = Vec::new();

    // Spatial axes have a direction; the others (lists, vectors, time) do not.
    let dirs = h.space_directions();
    let spatial: Vec<usize> = (0..sizes.len()).filter(|&a| dirs.get(a).is_some_and(|d| d.is_some())).collect();
    let lps = h.space_is_lps();
    let flip = |v: DVec3| if lps == Some(true) { DVec3::new(-v.x, -v.y, v.z) } else { v };
    let (dims, geometry) = if spatial.len() >= 2 {
        let mut d = [1usize; 4];
        let mut axes = [DVec3::X, DVec3::Y, DVec3::Z];
        let mut spacing = [1.0; 3];
        for (slot, &a) in spatial.iter().take(3).enumerate() {
            let v = flip(dirs[a].unwrap());
            d[slot] = sizes[a];
            spacing[slot] = v.length().max(f64::MIN_POSITIVE);
            axes[slot] = v / spacing[slot];
        }
        if spatial.len() == 2 {
            axes[2] = axes[0].cross(axes[1]).normalize_or_zero();
        }
        d[3] = (0..sizes.len()).filter(|a| !spatial.contains(a)).map(|a| sizes[a]).product::<usize>().max(1);
        let origin = h.get("space origin").and_then(|s| vec3(s.trim().trim_start_matches('(').trim_end_matches(')'))).map(flip).unwrap_or(DVec3::ZERO);
        if lps.is_none() {
            warnings.push(format!("Unknown space “{}”: orientation may be wrong", h.get("space").unwrap_or("none")));
        }
        (d, Some(Geometry::from_axes(d, spacing, axes, origin)))
    } else {
        // No space information: spacings only (or nothing).
        let sp: Vec<f64> = h.get("spacings").map(|s| s.split_whitespace().filter_map(|x| x.parse().ok()).collect()).unwrap_or_default();
        let mut d = [1usize; 4];
        for (k, &s) in sizes.iter().take(3).enumerate() {
            d[k] = s;
        }
        d[3] = sizes.iter().skip(3).product::<usize>().max(1);
        let spacing = [0, 1, 2].map(|k| sp.get(k).copied().filter(|v| v.is_finite() && *v > 0.0).unwrap_or(1.0));
        warnings.push("No space directions: orientation unknown".into());
        (d, Some(Geometry::from_axes(d, spacing, [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO)))
    };

    let segments = h.segment_names();
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let lower = name.to_lowercase();
    let kinds = h.get("kinds").unwrap_or("").to_lowercase();
    let colour = kinds.contains("rgb-color") || kinds.contains("rgba-color");
    if colour {
        warnings.push("Colour image (RGB)".into());
    }
    let kind = if !segments.is_empty() || lower.ends_with(".seg.nrrd") {
        Kind::Segmentation
    } else if colour {
        Kind::Image
    } else if kinds.contains("vector") || kinds.contains("covariant") || kinds.contains("2-vector") || kinds.contains("3-vector") {
        Kind::VectorField
    } else if dims[3] > 1 {
        Kind::TimeSeries
    } else if dtype.is_some_and(|t| t.is_integer()) && ["mask", "label", "seg"].iter().any(|w| lower.contains(w)) {
        Kind::LabelMap
    } else {
        Kind::Image
    };
    let voxels = match (dtype, encoding.as_str()) {
        (Some(t), "raw" | "gzip" | "gz") if byte_skip >= 0 => {
            // The data follow the text header (attached) or start the data file; byte skip
            // counts decompressed bytes for gzip data.
            let start = if detached.is_some() { 0 } else { h.data_offset };
            let (file_offset, skip) = if gzip { (start, byte_skip as u64) } else { (start + byte_skip as u64, 0) };
            // Non-spatial axes before the first spatial one are interleaved per voxel.
            let interleave = (0..sizes.len()).take_while(|a| !spatial.contains(a)).map(|a| sizes[a]).product::<usize>().max(1);
            let interleave = if spatial.is_empty() { 1 } else { interleave };
            Voxels::Raw { file: file.clone(), file_offset, gzip, skip, interleave, dtype: t, big_endian: big, slope: 1.0, intercept: 0.0 }
        }
        (None, _) => Voxels::Unsupported(format!("type “{}”", h.get("type").unwrap_or(""))),
        (_, e) => Voxels::Unsupported(format!("{e} encoding")),
    };
    if let Voxels::Unsupported(why) = &voxels {
        warnings.push(format!("Pixels not readable yet: {why}"));
    }
    let g = geometry.as_ref().unwrap();
    let mut fields = vec![
        Field::new("File", "Format", format!("NRRD ({}){}", h.version.trim_start_matches("NRRD"), if detached.is_some() { ", separate data file" } else { "" })),
        Field::new("File", "Encoding", encoding.clone()),
        Field::new("Geometry", "Sizes", sizes.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(" × ")),
        Field::new("Geometry", "Voxel size", format!("{:.4} × {:.4} × {:.4} mm", g.spacing[0], g.spacing[1], g.spacing[2])),
        Field::new("Geometry", "Orientation", format!("{} (space: {})", g.orientation(), h.get("space").unwrap_or("none"))),
        Field::new("Pixel", "Data type", h.get("type").unwrap_or("?")),
    ];
    if let Some(k) = h.get("kinds") {
        fields.push(Field::new("Geometry", "Kinds", k));
    }
    if !segments.is_empty() {
        fields.push(Field::new("Segmentation", "Segments", format!("{}: {}", segments.len(), segments.join(", "))));
    }
    if let Some(c) = h.get("content") {
        fields.push(Field::new("File", "Content", c));
    }
    let mut files = vec![path.to_path_buf()];
    if let Some(d) = &detached {
        if !d.exists() {
            warnings.push("The data file is missing".into());
        }
        files.push(d.clone());
    }
    let bytes = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    Ok(Volume { format: Format::Nrrd, kind, name, files, bytes, geometry, data_type: dtype, fields, warnings, voxels, window: None, photometric: None, patient: None })
}

/// Read and parse the header of a NRRD file (up to `max` bytes).
pub fn read(path: &Path, max: usize) -> Result<Header> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?.take(max as u64).read_to_end(&mut buf)?;
    parse(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SLICER: &str = "NRRD0004\n# Complete NRRD file format specification at:\ntype: short\ndimension: 3\nspace: left-posterior-superior\nsizes: 256 256 130\nspace directions: (0.9375,0,0) (0,0.9375,0) (0,0,1.2)\nkinds: domain domain domain\nendian: little\nencoding: gzip\nspace origin: (-120,-110,-60)\n\n";

    #[test]
    fn slicer_volume() {
        let h = parse(SLICER.as_bytes()).unwrap();
        assert_eq!(h.sizes(), vec![256, 256, 130]);
        assert_eq!(h.data_offset as usize, SLICER.len());
        let dir = std::env::temp_dir().join(format!("mmv-nrrd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("Brain Ax T1.nrrd");
        std::fs::write(&p, SLICER).unwrap();
        let v = volume(&p, &h).unwrap();
        assert_eq!(v.kind, Kind::Image);
        let g = v.geometry.unwrap();
        assert_eq!(g.dims, [256, 256, 130, 1]);
        assert_eq!(g.orientation(), "LPS");
        // LPS origin (-120, -110, -60) is RAS (120, 110, -60).
        assert!((g.world(DVec3::ZERO) - DVec3::new(120.0, 110.0, -60.0)).length() < 1e-9);
        assert!(matches!(v.voxels, Voxels::Raw { gzip: true, dtype: DataType::I16, file_offset, skip: 0, interleave: 1, .. } if file_offset as usize == SLICER.len()));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn segmentation_and_multivolume() {
        let seg = "NRRD0004\ntype: unsigned char\ndimension: 4\nspace: right-anterior-superior\nsizes: 2 64 64 40\nspace directions: none (1,0,0) (0,1,0) (0,0,2)\nkinds: list domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\nSegment0_Name:=Brain\nSegment1_Name:=Tumour\nSegment0_LabelValue:=1\n\n";
        let h = parse(seg.as_bytes()).unwrap();
        assert_eq!(h.segment_names(), vec!["Brain", "Tumour"]);
        let dir = std::env::temp_dir().join(format!("mmv-seg-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("mask.seg.nrrd");
        std::fs::write(&p, seg).unwrap();
        let v = volume(&p, &h).unwrap();
        assert_eq!(v.kind, Kind::Segmentation);
        let g = v.geometry.unwrap();
        assert_eq!(g.dims, [64, 64, 40, 2], "the list axis becomes the 4th dimension");
        assert!(matches!(v.voxels, Voxels::Raw { interleave: 2, .. }), "and comes first in the data");
        assert_eq!(g.orientation(), "RAS");
        // Detached header + raw.gz data file.
        let nhdr = "NRRD0004\ntype: short\ndimension: 4\nspace: left-posterior-superior\nsizes: 128 128 30 5\nspace directions: (1.5,0,0) (0,1.5,0) (0,0,4) none\nkinds: domain domain domain list\nencoding: gzip\ndata file: run.raw.gz\n";
        let p = dir.join("ASL.nhdr");
        std::fs::write(&p, nhdr).unwrap();
        std::fs::write(dir.join("run.raw.gz"), b"").unwrap();
        let h = parse(nhdr.as_bytes()).unwrap();
        let v = volume(&p, &h).unwrap();
        assert_eq!(v.kind, Kind::TimeSeries);
        assert_eq!(v.geometry.unwrap().dims, [128, 128, 30, 5]);
        assert_eq!(v.files.len(), 2);
        assert!(matches!(&v.voxels, Voxels::Raw { file, file_offset: 0, interleave: 1, .. } if file.ends_with("run.raw.gz")));
        let _ = std::fs::remove_dir_all(&dir);
        assert!(parse(b"P5 not nrrd").is_err());
    }
}
