//! NIfTI-1, NIfTI-2 and Analyze 7.5 headers: one parser for the 348- and 540-byte layouts in
//! either byte order, single files (`.nii`, `.nii.gz`) and header / image pairs
//! (`.hdr` + `.img`).

use crate::geometry::Geometry;
use crate::model::{DataType, Field, Format, Kind, PatientKey, Volume, Voxels};
use anyhow::{bail, Context, Result};
use glam::{DMat4, DVec3, DVec4};
use std::io::Read;
use std::path::{Path, PathBuf};

/// Which header layout a file uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flavour {
    /// `n+1`: header and data in one file.
    Nifti1,
    /// `ni1`: header in `.hdr`, data in `.img`.
    Nifti1Pair,
    Nifti2,
    Nifti2Pair,
    /// No magic: the Analyze 7.5 layout NIfTI-1 grew from.
    Analyze,
}

/// The fields of a NIfTI-1 / NIfTI-2 / Analyze header that matter here.
#[derive(Clone, Debug, PartialEq)]
pub struct Header {
    pub flavour: Flavour,
    pub big_endian: bool,
    pub dim: [i64; 8],
    pub datatype: i16,
    pub bitpix: i16,
    pub pixdim: [f64; 8],
    pub vox_offset: f64,
    pub scl_slope: f64,
    pub scl_inter: f64,
    pub cal_min: f64,
    pub cal_max: f64,
    pub qform_code: i32,
    pub sform_code: i32,
    pub quatern: [f64; 3],
    pub qoffset: [f64; 3],
    pub srow: [[f64; 4]; 3],
    pub intent_code: i32,
    pub intent_name: String,
    pub descrip: String,
    pub aux_file: String,
    pub xyzt_units: u8,
    /// Analyze only: the SPM origin voxel (1-based), when set.
    pub analyze_origin: Option<[i16; 3]>,
    /// Analyze only: the patient ID field.
    pub analyze_patient_id: String,
    /// NIfTI extensions follow the header.
    pub has_extensions: bool,
}

/// Bytes read from the start of a file: enough for either header and the extension flag.
pub const HEADER_BYTES: usize = 544;

/// The magic of a NIfTI-1 / NIfTI-2 header at the start of `b`, or `None`.
pub fn sniff(b: &[u8]) -> Option<Flavour> {
    if b.len() >= 348 {
        match &b[344..348] {
            b"n+1\0" => return Some(Flavour::Nifti1),
            b"ni1\0" => return Some(Flavour::Nifti1Pair),
            _ => {}
        }
    }
    if b.len() >= 12 {
        match &b[4..12] {
            b"n+2\0\r\n\x1a\n" => return Some(Flavour::Nifti2),
            b"ni2\0\r\n\x1a\n" => return Some(Flavour::Nifti2Pair),
            _ => {}
        }
    }
    None
}

/// Whether `b` looks like an Analyze 7.5 header (348 bytes, no NIfTI magic, sane dims).
pub fn is_analyze(b: &[u8]) -> bool {
    if b.len() < 348 || sniff(b).is_some() {
        return false;
    }
    for big in [false, true] {
        let r = Bytes { b, big };
        if r.i32(0) == 348 {
            let n = r.i16(40);
            return (1..=7).contains(&n) && (1..=n as usize).all(|k| r.i16(40 + 2 * k) > 0);
        }
    }
    false
}

struct Bytes<'a> {
    b: &'a [u8],
    big: bool,
}

impl Bytes<'_> {
    fn arr<const N: usize>(&self, at: usize) -> [u8; N] {
        let mut a = [0u8; N];
        a.copy_from_slice(&self.b[at..at + N]);
        if self.big {
            a.reverse();
        }
        a
    }
    fn i16(&self, at: usize) -> i16 {
        i16::from_le_bytes(self.arr(at))
    }
    fn i32(&self, at: usize) -> i32 {
        i32::from_le_bytes(self.arr(at))
    }
    fn i64(&self, at: usize) -> i64 {
        i64::from_le_bytes(self.arr(at))
    }
    fn f32(&self, at: usize) -> f64 {
        f32::from_le_bytes(self.arr(at)) as f64
    }
    fn f64(&self, at: usize) -> f64 {
        f64::from_le_bytes(self.arr(at))
    }
    fn text(&self, at: usize, len: usize) -> String {
        let s = &self.b[at..at + len];
        let end = s.iter().position(|&c| c == 0).unwrap_or(len);
        String::from_utf8_lossy(&s[..end]).trim().to_string()
    }
}

/// Parse a header from the first bytes of a file (at least 348, or 540 for NIfTI-2).
pub fn parse(b: &[u8]) -> Result<Header> {
    let flavour = match sniff(b) {
        Some(f) => f,
        None if is_analyze(b) => Flavour::Analyze,
        None => bail!("not a NIfTI or Analyze header"),
    };
    let two = matches!(flavour, Flavour::Nifti2 | Flavour::Nifti2Pair);
    let size: usize = if two { 540 } else { 348 };
    if b.len() < size {
        bail!("header cut short ({} of {size} bytes)", b.len());
    }
    let want = size as i32;
    let big = i32::from_le_bytes(b[0..4].try_into().unwrap()) != want;
    let r = Bytes { b, big };
    if r.i32(0) != want {
        bail!("bad header size field");
    }
    let mut h = Header {
        flavour,
        big_endian: big,
        dim: [0; 8],
        datatype: 0,
        bitpix: 0,
        pixdim: [0.0; 8],
        vox_offset: 0.0,
        scl_slope: 0.0,
        scl_inter: 0.0,
        cal_min: 0.0,
        cal_max: 0.0,
        qform_code: 0,
        sform_code: 0,
        quatern: [0.0; 3],
        qoffset: [0.0; 3],
        srow: [[0.0; 4]; 3],
        intent_code: 0,
        intent_name: String::new(),
        descrip: String::new(),
        aux_file: String::new(),
        xyzt_units: 0,
        analyze_origin: None,
        analyze_patient_id: String::new(),
        has_extensions: false,
    };
    if two {
        h.datatype = r.i16(12);
        h.bitpix = r.i16(14);
        for k in 0..8 {
            h.dim[k] = r.i64(16 + 8 * k);
            h.pixdim[k] = r.f64(104 + 8 * k);
        }
        h.vox_offset = r.i64(168) as f64;
        h.scl_slope = r.f64(176);
        h.scl_inter = r.f64(184);
        h.cal_max = r.f64(192);
        h.cal_min = r.f64(200);
        h.descrip = r.text(240, 80);
        h.aux_file = r.text(320, 24);
        h.qform_code = r.i32(344);
        h.sform_code = r.i32(348);
        h.quatern = [r.f64(352), r.f64(360), r.f64(368)];
        h.qoffset = [r.f64(376), r.f64(384), r.f64(392)];
        for (row, at) in [400usize, 432, 464].into_iter().enumerate() {
            for c in 0..4 {
                h.srow[row][c] = r.f64(at + 8 * c);
            }
        }
        h.xyzt_units = r.i32(500) as u8;
        h.intent_code = r.i32(504);
        h.intent_name = r.text(508, 16);
        h.has_extensions = b.len() > 540 && b[540] != 0;
    } else {
        for k in 0..8 {
            h.dim[k] = r.i16(40 + 2 * k) as i64;
            h.pixdim[k] = r.f32(76 + 4 * k);
        }
        h.datatype = r.i16(70);
        h.bitpix = r.i16(72);
        h.vox_offset = r.f32(108);
        h.descrip = r.text(148, 80);
        h.aux_file = r.text(228, 24);
        if flavour == Flavour::Analyze {
            // Analyze keeps no orientation; SPM stores an origin voxel in `originator`.
            let o = [r.i16(253), r.i16(255), r.i16(257)];
            if o.iter().any(|&v| v != 0) {
                h.analyze_origin = Some(o);
            }
            h.analyze_patient_id = r.text(283, 10);
            h.scl_slope = 1.0;
        } else {
            h.intent_code = r.i16(68) as i32;
            h.scl_slope = r.f32(112);
            h.scl_inter = r.f32(116);
            h.xyzt_units = b[123];
            h.cal_max = r.f32(124);
            h.cal_min = r.f32(128);
            h.qform_code = r.i16(252) as i32;
            h.sform_code = r.i16(254) as i32;
            h.quatern = [r.f32(256), r.f32(260), r.f32(264)];
            h.qoffset = [r.f32(268), r.f32(272), r.f32(276)];
            for (row, at) in [280usize, 296, 312].into_iter().enumerate() {
                for c in 0..4 {
                    h.srow[row][c] = r.f32(at + 4 * c);
                }
            }
            h.intent_name = r.text(328, 16);
            h.has_extensions = b.len() > 348 && b[348] != 0;
        }
    }
    let n = h.dim[0];
    if !(1..=7).contains(&n) || (1..=n as usize).any(|k| h.dim[k] <= 0) {
        bail!("bad dimensions {:?}", &h.dim[..(n.clamp(0, 7) as usize + 1)]);
    }
    Ok(h)
}

impl Header {
    /// Voxel data type, if supported.
    pub fn data_type(&self) -> Option<DataType> {
        Some(match self.datatype {
            2 => DataType::U8,
            4 => DataType::I16,
            8 => DataType::I32,
            16 => DataType::F32,
            32 => DataType::Complex64,
            64 => DataType::F64,
            128 => DataType::Rgb24,
            256 => DataType::I8,
            512 => DataType::U16,
            768 => DataType::U32,
            1024 => DataType::I64,
            1280 => DataType::U64,
            1792 => DataType::Complex128,
            2304 => DataType::Rgba32,
            _ => return None,
        })
    }

    /// Millimetres per spatial unit of `pixdim` and the affines.
    fn unit_mm(&self) -> f64 {
        match self.xyzt_units & 0x07 {
            1 => 1000.0,
            3 => 0.001,
            _ => 1.0,
        }
    }

    /// Voxel → RAS (mm) transform and which header fields it came from: the sform if set,
    /// else the qform, else the voxel sizes alone (as the NIfTI-1 standard orders them).
    pub fn affine(&self) -> (DMat4, &'static str) {
        let u = self.unit_mm();
        let dx = self.pixdim[1].abs().max(f64::MIN_POSITIVE);
        let dy = self.pixdim[2].abs().max(f64::MIN_POSITIVE);
        let dz = if self.dim[0] >= 3 { self.pixdim[3].abs().max(f64::MIN_POSITIVE) } else { 1.0 };
        if self.sform_code > 0 {
            let s = &self.srow;
            let m = DMat4::from_cols(
                DVec4::new(s[0][0], s[1][0], s[2][0], 0.0) * u,
                DVec4::new(s[0][1], s[1][1], s[2][1], 0.0) * u,
                DVec4::new(s[0][2], s[1][2], s[2][2], 0.0) * u,
                DVec4::new(s[0][3] * u, s[1][3] * u, s[2][3] * u, 1.0),
            );
            return (m, "sform");
        }
        if self.qform_code > 0 {
            let [b, c, d] = self.quatern;
            let a = (1.0 - (b * b + c * c + d * d)).max(0.0).sqrt();
            let rot = [
                [a * a + b * b - c * c - d * d, 2.0 * (b * c - a * d), 2.0 * (b * d + a * c)],
                [2.0 * (b * c + a * d), a * a + c * c - b * b - d * d, 2.0 * (c * d - a * b)],
                [2.0 * (b * d - a * c), 2.0 * (c * d + a * b), a * a + d * d - c * c - b * b],
            ];
            let qfac = if self.pixdim[0] < 0.0 { -1.0 } else { 1.0 };
            let sizes = [dx, dy, dz * qfac];
            let col = |j: usize| DVec4::new(rot[0][j], rot[1][j], rot[2][j], 0.0) * sizes[j] * u;
            let m = DMat4::from_cols(col(0), col(1), col(2), DVec4::new(self.qoffset[0] * u, self.qoffset[1] * u, self.qoffset[2] * u, 1.0));
            return (m, "qform");
        }
        // No orientation: voxel sizes only (Analyze uses its SPM origin if set).
        let origin = match self.analyze_origin {
            Some(o) => DVec3::new(-(o[0] as f64 - 1.0) * dx, -(o[1] as f64 - 1.0) * dy, -(o[2] as f64 - 1.0) * dz),
            None => DVec3::ZERO,
        };
        let m = DMat4::from_cols(DVec4::new(dx * u, 0.0, 0.0, 0.0), DVec4::new(0.0, dy * u, 0.0, 0.0), DVec4::new(0.0, 0.0, dz * u, 0.0), (origin * u).extend(1.0));
        (m, "voxel sizes only")
    }

    /// Size of the 3D grid and of everything beyond it (volumes × components).
    pub fn dims4(&self) -> [usize; 4] {
        let n = self.dim[0] as usize;
        let d = |k: usize| if k <= n { self.dim[k].max(1) as usize } else { 1 };
        [d(1), d(2), d(3), (4..=n).map(d).product::<usize>().max(1)]
    }
}

/// Space and time unit names.
fn units(code: u8) -> String {
    let space = match code & 0x07 {
        1 => "m",
        2 => "mm",
        3 => "µm",
        _ => "unknown",
    };
    let time = match code & 0x38 {
        8 => "s",
        16 => "ms",
        24 => "µs",
        _ => "",
    };
    if time.is_empty() { space.to_string() } else { format!("{space}, {time}") }
}

fn form_name(code: i32) -> &'static str {
    match code {
        0 => "unknown (0)",
        1 => "scanner (1)",
        2 => "aligned (2)",
        3 => "Talairach (3)",
        4 => "MNI 152 (4)",
        5 => "template (5)",
        _ => "other",
    }
}

fn intent_name(code: i32) -> &'static str {
    match code {
        0 => "none",
        1002 => "label",
        1003 => "neuroname",
        1004 => "general matrix",
        1005 => "symmetric matrix",
        1006 => "displacement vector",
        1007 => "vector",
        1008 => "point set",
        1009 => "triangle",
        1010 => "quaternion",
        1011 => "dimensionless",
        2001 => "time series",
        2002 => "node index",
        2003 => "RGB vector",
        2004 => "RGBA vector",
        2005 => "shape",
        _ if (2..=24).contains(&code) => "statistic",
        _ => "other",
    }
}

/// Read the first bytes of `path`, inflating it first when `gzip`.
pub fn read_head(path: &Path, gzip: bool, len: usize) -> Result<Vec<u8>> {
    let f = std::fs::File::open(path).with_context(|| format!("cannot open {}", path.display()))?;
    let mut buf = Vec::with_capacity(len);
    if gzip {
        flate2::read::MultiGzDecoder::new(f).take(len as u64).read_to_end(&mut buf)?;
    } else {
        f.take(len as u64).read_to_end(&mut buf)?;
    }
    Ok(buf)
}

/// `name.nii.gz` → `name`; `name.hdr` → `name`.
pub fn stem(path: &Path) -> String {
    let n = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let lower = n.to_lowercase();
    for ext in [".nii.gz", ".hdr.gz", ".img.gz", ".nii", ".hdr", ".img"] {
        if lower.ends_with(ext) {
            return n[..n.len() - ext.len()].to_string();
        }
    }
    n
}

/// The image file of a header / image pair (`.hdr` → `.img`, keeping `.gz`).
pub fn pair_image(hdr: &Path) -> Option<PathBuf> {
    let name = hdr.file_name()?.to_string_lossy().to_string();
    let lower = name.to_lowercase();
    let (base, gz) = match (lower.strip_suffix(".hdr.gz"), lower.strip_suffix(".hdr")) {
        (Some(b), _) => (&name[..b.len()], true),
        (None, Some(b)) => (&name[..b.len()], false),
        (None, None) => return None,
    };
    for ext in if gz { ["img.gz", "IMG.gz"] } else { ["img", "IMG"] } {
        let p = hdr.with_file_name(format!("{base}.{ext}"));
        if p.exists() {
            return Some(p);
        }
    }
    None
}

/// Identifiers in a dcm2niix / BIDS JSON sidecar next to `path`, when it has them.
pub fn sidecar_patient(path: &Path) -> Option<PatientKey> {
    let json = path.with_file_name(format!("{}.json", stem(path)));
    let text = std::fs::read_to_string(json).ok()?;
    let v: serde_json::Value = serde_json::from_str(&text).ok()?;
    let get = |k: &str| v.get(k).and_then(|x| x.as_str()).map(|s| s.trim().to_string()).unwrap_or_default();
    let key = PatientKey { id: get("PatientID"), name: get("PatientName") };
    (!key.id.is_empty() || !key.name.is_empty()).then_some(key)
}

/// Names that mark integer images as label maps.
const LABEL_HINTS: [&str; 11] = ["mask", "label", "seg", "aseg", "aparc", "parc", "atlas", "hemi", "lesion", "roi", "dseg"];

/// Build the volume for a NIfTI / Analyze file. `path` is the `.nii(.gz)` or `.hdr(.gz)`.
pub fn volume(path: &Path, gzip: bool, h: &Header) -> Result<Volume> {
    let pair = matches!(h.flavour, Flavour::Nifti1Pair | Flavour::Nifti2Pair | Flavour::Analyze);
    let data_file = if pair { pair_image(path).context("the .img file of this header is missing")? } else { path.to_path_buf() };
    let data_gzip = if pair { data_file.extension().is_some_and(|e| e.eq_ignore_ascii_case("gz")) } else { gzip };
    let dims = h.dims4();
    let (m, from) = h.affine();
    let geometry = Geometry::from_affine(dims, m);
    let dtype = h.data_type();
    let format = match h.flavour {
        Flavour::Nifti1 | Flavour::Nifti1Pair => Format::Nifti1,
        Flavour::Nifti2 | Flavour::Nifti2Pair => Format::Nifti2,
        Flavour::Analyze => Format::Analyze,
    };
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let lower = name.to_lowercase();
    let n = h.dim[0] as usize;
    let field_name = ["map", "warp", "def", "disp", "field"].iter().any(|w| lower.contains(w));
    let kind = if h.intent_code == 1002 || (dtype.is_some_and(|t| t.is_integer()) && LABEL_HINTS.iter().any(|w| lower.contains(w))) {
        Kind::LabelMap
    } else if matches!(h.intent_code, 1006 | 1007) || (n >= 5 && h.dim[5] > 1) || (n == 4 && h.dim[4] == 3 && field_name) {
        Kind::VectorField
    } else if lower.contains("jacobian") {
        Kind::Derived
    } else if n >= 4 && h.dim[4] > 1 {
        Kind::TimeSeries
    } else {
        Kind::Image
    };
    let mut warnings = Vec::new();
    if h.flavour == Flavour::Analyze {
        warnings.push("Analyze 7.5 stores no orientation: left and right are ambiguous".into());
    } else if h.qform_code == 0 && h.sform_code == 0 {
        warnings.push("No orientation (qform and sform are both unset)".into());
    }
    if dtype.is_none() {
        warnings.push(format!("Unsupported data type code {}", h.datatype));
    }
    let slope = if h.scl_slope == 0.0 || !h.scl_slope.is_finite() { 1.0 } else { h.scl_slope };
    let intercept = if h.scl_inter.is_finite() && h.scl_slope != 0.0 { h.scl_inter } else { 0.0 };
    let voxels = match dtype {
        Some(t) => Voxels::Raw { file: data_file.clone(), offset: if pair { h.vox_offset.max(0.0) as u64 } else { h.vox_offset.max(if format == Format::Nifti2 { 544.0 } else { 352.0 }) as u64 }, gzip: data_gzip, dtype: t, big_endian: h.big_endian, slope, intercept },
        None => Voxels::Unsupported(format!("data type code {}", h.datatype)),
    };
    let dimstr = (1..=n).map(|k| h.dim[k].to_string()).collect::<Vec<_>>().join(" × ");
    let mut fields = vec![
        Field::new("File", "Format", format!("{}{}", format.label(), if pair { " (header + image pair)" } else if gzip { " (gzip)" } else { "" })),
        Field::new("File", "Byte order", if h.big_endian { "big-endian" } else { "little-endian" }),
        Field::new("Geometry", "Dimensions", dimstr),
        Field::new("Geometry", "Voxel size", format!("{:.4} × {:.4} × {:.4} {}", h.pixdim[1], h.pixdim[2], h.pixdim[3], units(h.xyzt_units))),
        Field::new("Geometry", "Orientation", format!("{} (from {from})", geometry.orientation())),
        Field::new("Pixel", "Data type", dtype.map(|t| t.label().to_string()).unwrap_or_else(|| format!("code {}", h.datatype))),
    ];
    if h.flavour != Flavour::Analyze {
        fields.push(Field::new("Geometry", "qform", form_name(h.qform_code)));
        fields.push(Field::new("Geometry", "sform", form_name(h.sform_code)));
        if h.scl_slope != 0.0 && (h.scl_slope != 1.0 || h.scl_inter != 0.0) {
            fields.push(Field::new("Pixel", "Scaling", format!("{} · v + {}", h.scl_slope, h.scl_inter)));
        }
        if h.intent_code != 0 {
            fields.push(Field::new("Pixel", "Intent", format!("{} ({}){}", intent_name(h.intent_code), h.intent_code, if h.intent_name.is_empty() { String::new() } else { format!(" “{}”", h.intent_name) })));
        }
        if n >= 4 && h.pixdim[4] > 0.0 && h.dim[4] > 1 {
            fields.push(Field::new("Geometry", "Time step", format!("{}", h.pixdim[4])));
        }
    }
    if h.cal_max > h.cal_min {
        fields.push(Field::new("Pixel", "Display range", format!("{} – {}", h.cal_min, h.cal_max)));
    }
    if !h.descrip.is_empty() {
        fields.push(Field::new("File", "Description", h.descrip.clone()));
    }
    if !h.aux_file.is_empty() {
        fields.push(Field::new("File", "Aux file", h.aux_file.clone()));
    }
    let mut patient = sidecar_patient(path);
    if patient.is_none() && !h.analyze_patient_id.is_empty() {
        patient = Some(PatientKey { id: h.analyze_patient_id.clone(), name: String::new() });
    }
    if let Some(p) = &patient {
        fields.push(Field::new("Patient", "Patient ID", p.id.clone()));
        if !p.name.is_empty() {
            fields.push(Field::new("Patient", "Patient name", p.name.clone()));
        }
    }
    let mut files = vec![path.to_path_buf()];
    if pair {
        files.push(data_file.clone());
    }
    let bytes = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    let window = (h.cal_max > h.cal_min).then_some(((h.cal_max + h.cal_min) * 0.5, h.cal_max - h.cal_min));
    Ok(Volume { format, kind, name, files, bytes, geometry: Some(geometry), data_type: dtype, fields, warnings, voxels, window, photometric: None, patient })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A NIfTI-1 header for a `nx × ny × nz` int16 volume with the given qform / sform.
    pub(crate) fn nifti1(dims: [i16; 3], pixdim: [f32; 3], magic: &[u8; 4], big: bool) -> Vec<u8> {
        let mut b = vec![0u8; 352];
        let put16 = |b: &mut Vec<u8>, at: usize, v: i16| b[at..at + 2].copy_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
        let put32 = |b: &mut Vec<u8>, at: usize, v: i32| b[at..at + 4].copy_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
        let putf = |b: &mut Vec<u8>, at: usize, v: f32| b[at..at + 4].copy_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
        put32(&mut b, 0, 348);
        put16(&mut b, 40, 3);
        for (k, d) in dims.iter().enumerate() {
            put16(&mut b, 42 + 2 * k, *d);
        }
        put16(&mut b, 70, 4);
        put16(&mut b, 72, 16);
        putf(&mut b, 76, 1.0);
        for (k, p) in pixdim.iter().enumerate() {
            putf(&mut b, 80 + 4 * k, *p);
        }
        putf(&mut b, 108, 352.0);
        b[123] = 2;
        b[344..348].copy_from_slice(magic);
        b
    }

    #[test]
    fn nifti1_qform_and_sform() {
        let mut b = nifti1([64, 48, 32], [0.5, 0.5, 2.0], b"n+1\0", false);
        let h = parse(&b).unwrap();
        assert_eq!(h.flavour, Flavour::Nifti1);
        assert_eq!(h.dims4(), [64, 48, 32, 1]);
        assert_eq!(h.affine().1, "voxel sizes only");
        // qform: 180° about z (b, c, d = 0, 0, 1) → x and y flip: LPS.
        b[252..254].copy_from_slice(&1i16.to_le_bytes());
        b[264..268].copy_from_slice(&1.0f32.to_le_bytes());
        b[268..272].copy_from_slice(&10.0f32.to_le_bytes());
        let h = parse(&b).unwrap();
        let (m, from) = h.affine();
        assert_eq!(from, "qform");
        let g = Geometry::from_affine(h.dims4(), m);
        assert_eq!(g.orientation(), "LPS");
        assert!((g.world(DVec3::new(2.0, 0.0, 0.0)) - DVec3::new(9.0, 0.0, 0.0)).length() < 1e-6);
        // sform wins over qform.
        b[254..256].copy_from_slice(&2i16.to_le_bytes());
        for (row, at) in [280usize, 296, 312].into_iter().enumerate() {
            let vals: [f32; 4] = match row {
                0 => [0.0, 0.0, 2.0, -5.0],
                1 => [0.5, 0.0, 0.0, 0.0],
                _ => [0.0, 0.5, 0.0, 7.0],
            };
            for (c, v) in vals.iter().enumerate() {
                b[at + 4 * c..at + 4 * c + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        let h = parse(&b).unwrap();
        let (m, from) = h.affine();
        assert_eq!(from, "sform");
        assert_eq!(Geometry::from_affine(h.dims4(), m).orientation(), "ASR");
    }

    #[test]
    fn big_endian_pairs_and_analyze() {
        let b = nifti1([10, 11, 12], [1.0, 1.0, 1.5], b"ni1\0", true);
        let h = parse(&b).unwrap();
        assert!(h.big_endian);
        assert_eq!(h.flavour, Flavour::Nifti1Pair);
        assert_eq!(h.dims4(), [10, 11, 12, 1]);
        assert!((h.pixdim[3] - 1.5).abs() < 1e-6);
        let mut a = nifti1([10, 11, 12], [1.0, 1.0, 1.5], b"\0\0\0\0", false);
        assert!(is_analyze(&a));
        a[253..255].copy_from_slice(&6i16.to_le_bytes());
        let h = parse(&a).unwrap();
        assert_eq!(h.flavour, Flavour::Analyze);
        assert_eq!(h.analyze_origin, Some([6, 0, 0]));
        let (m, _) = h.affine();
        assert!((m.transform_point3(DVec3::new(5.0, 0.0, 0.0)).x).abs() < 1e-9, "the SPM origin voxel sits at 0");
        // Random bytes are neither.
        assert!(parse(&[7u8; 600]).is_err());
    }

    #[test]
    fn nifti2_header() {
        let mut b = vec![0u8; 544];
        b[0..4].copy_from_slice(&540i32.to_le_bytes());
        b[4..12].copy_from_slice(b"n+2\0\r\n\x1a\n");
        b[12..14].copy_from_slice(&16i16.to_le_bytes());
        for (k, d) in [4i64, 100, 120, 90, 5].iter().enumerate() {
            b[16 + 8 * k..24 + 8 * k].copy_from_slice(&d.to_le_bytes());
        }
        for (k, p) in [1.0f64, 1.2, 1.2, 1.5, 2.0].iter().enumerate() {
            b[104 + 8 * k..112 + 8 * k].copy_from_slice(&p.to_le_bytes());
        }
        b[168..176].copy_from_slice(&544i64.to_le_bytes());
        let h = parse(&b).unwrap();
        assert_eq!(h.flavour, Flavour::Nifti2);
        assert_eq!(h.dims4(), [100, 120, 90, 5]);
        assert_eq!(h.data_type(), Some(DataType::F32));
        assert!((h.affine().0.z_axis.z - 1.5).abs() < 1e-12);
    }

    #[test]
    fn matches_the_nifti_crate() {
        // Cross-check the qform maths against nifti-rs on an oblique header.
        let mut b = nifti1([20, 30, 40], [0.8, 0.9, 1.7], b"n+1\0", false);
        b[252..254].copy_from_slice(&1i16.to_le_bytes());
        b[76..80].copy_from_slice(&(-1.0f32).to_le_bytes());
        for (k, v) in [0.1f32, -0.2, 0.3, 4.0, -5.0, 6.0].iter().enumerate() {
            b[256 + 4 * k..260 + 4 * k].copy_from_slice(&v.to_le_bytes());
        }
        let ours = parse(&b).unwrap().affine().0;
        let theirs = nifti::NiftiHeader::from_reader(std::io::Cursor::new(&b[..348])).unwrap().qform_affine::<f64>();
        for r in 0..3 {
            for c in 0..4 {
                let a = ours.col(c)[r];
                let t = theirs[(r, c)];
                assert!((a - t).abs() < 1e-5, "affine[{r}][{c}]: {a} vs {t}");
            }
        }
    }
}
