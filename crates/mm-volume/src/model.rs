//! What a folder scan finds: volumes (one 3D or 4D image each), the DICOM patient → study →
//! series tree they belong to, and what was skipped or could not be read.

use crate::geometry::Geometry;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::PathBuf;

/// File format of a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Format {
    Dicom,
    Nifti1,
    Nifti2,
    Analyze,
    Nrrd,
    MetaImage,
    Mgh,
    /// A folder of 2D images (JPEG, PNG, BMP, TIFF).
    ImageStack,
    /// A multi-page TIFF file.
    Tiff,
}

impl Format {
    pub fn label(self) -> &'static str {
        match self {
            Format::Dicom => "DICOM",
            Format::Nifti1 => "NIfTI-1",
            Format::Nifti2 => "NIfTI-2",
            Format::Analyze => "Analyze 7.5",
            Format::Nrrd => "NRRD",
            Format::MetaImage => "MetaImage",
            Format::Mgh => "MGH / MGZ",
            Format::ImageStack => "Image stack",
            Format::Tiff => "Multi-page TIFF",
        }
    }
}

/// What a volume holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Kind {
    /// An intensity image (MR, CT, …).
    Image,
    /// Integer labels (masks, parcellations, atlases).
    LabelMap,
    /// A segmentation with named segments (e.g. Slicer `.seg.nrrd`).
    Segmentation,
    /// Vectors per voxel (deformation / displacement fields).
    VectorField,
    /// Several volumes over time or another dimension (4D).
    TimeSeries,
    /// Scout / localizer images.
    Localizer,
    /// Computed from other images (reformats, projections, maps).
    Derived,
}

impl Kind {
    pub fn label(self) -> &'static str {
        match self {
            Kind::Image => "Image",
            Kind::LabelMap => "Label map",
            Kind::Segmentation => "Segmentation",
            Kind::VectorField => "Vector field",
            Kind::TimeSeries => "Time series",
            Kind::Localizer => "Localizer",
            Kind::Derived => "Derived",
        }
    }
}

/// Voxel data type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DataType {
    U8,
    I8,
    U16,
    I16,
    U32,
    I32,
    U64,
    I64,
    F32,
    F64,
    Rgb24,
    Rgba32,
    Complex64,
    Complex128,
}

impl DataType {
    /// Bytes per voxel.
    pub fn size(self) -> usize {
        match self {
            DataType::U8 | DataType::I8 => 1,
            DataType::U16 | DataType::I16 => 2,
            DataType::U32 | DataType::I32 | DataType::F32 => 4,
            DataType::U64 | DataType::I64 | DataType::F64 | DataType::Complex64 => 8,
            DataType::Rgb24 => 3,
            DataType::Rgba32 => 4,
            DataType::Complex128 => 16,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            DataType::U8 => "uint8",
            DataType::I8 => "int8",
            DataType::U16 => "uint16",
            DataType::I16 => "int16",
            DataType::U32 => "uint32",
            DataType::I32 => "int32",
            DataType::U64 => "uint64",
            DataType::I64 => "int64",
            DataType::F32 => "float32",
            DataType::F64 => "float64",
            DataType::Rgb24 => "RGB 24-bit",
            DataType::Rgba32 => "RGBA 32-bit",
            DataType::Complex64 => "complex64",
            DataType::Complex128 => "complex128",
        }
    }

    pub fn is_integer(self) -> bool {
        matches!(self, DataType::U8 | DataType::I8 | DataType::U16 | DataType::I16 | DataType::U32 | DataType::I32 | DataType::U64 | DataType::I64)
    }
}

/// Where a volume's voxels are, so slices can be read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Voxels {
    /// DICOM: the file and frame of every slice, in slice order; a 4D set lists its volumes one
    /// after the other (`geometry.dims[2]` slices each).
    Dicom { slices: Vec<(PathBuf, u32)> },
    /// A plain array of `dtype` (x fastest) in `file` (NIfTI, Analyze, NRRD, MetaImage, MGH):
    /// the data start `file_offset` bytes into the file, then (after decompression, if `gzip`:
    /// gzip, or zlib as in MetaImage, told apart by the first bytes) another
    /// `skip` bytes. `interleave` > 1 when a component axis comes first (e.g. RGB, or the
    /// layers of a segmentation): component 0 is every `interleave`-th value. Values are
    /// scaled by `slope · v + intercept`.
    Raw { file: PathBuf, file_offset: u64, gzip: bool, skip: u64, interleave: usize, dtype: DataType, big_endian: bool, slope: f64, intercept: f64 },
    /// 2D images (or TIFF pages), one per slice, in slice order.
    Images { slices: Vec<(PathBuf, u32)> },
    /// Recognised but not readable yet (e.g. bzip2-compressed NRRD).
    Unsupported(String),
}

/// One header field, for display.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Field {
    pub group: String,
    pub name: String,
    pub value: String,
}

impl Field {
    pub fn new(group: &str, name: &str, value: impl Into<String>) -> Self {
        Self { group: group.into(), name: name.into(), value: value.into() }
    }
}

/// Who a scan belongs to, as identifiers in the files.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct PatientKey {
    pub id: String,
    pub name: String,
}

/// One 3D (or 4D) image.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Volume {
    pub format: Format,
    pub kind: Kind,
    /// Display name: the file name, or "Series 8 · T2 AX" for DICOM.
    pub name: String,
    /// Every file it is made of.
    pub files: Vec<PathBuf>,
    /// Size on disk.
    pub bytes: u64,
    pub geometry: Option<Geometry>,
    pub data_type: Option<DataType>,
    /// Header fields worth showing, grouped.
    pub fields: Vec<Field>,
    /// Problems worth knowing about (missing slices, unknown orientation, …).
    pub warnings: Vec<String>,
    pub voxels: Voxels,
    /// Display window (centre, width) stored in the file.
    pub window: Option<(f64, f64)>,
    /// DICOM photometric interpretation (MONOCHROME1 shows inverted; RGB is colour).
    pub photometric: Option<String>,
    /// Identifiers carried by a non-DICOM file (sidecar, header), for exact matching.
    pub patient: Option<PatientKey>,
}

/// A DICOM object that is not part of an image volume (report, PDF, screenshot, …).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DicomObject {
    pub path: PathBuf,
    /// What it is ("Secondary capture", "Structured report", …).
    pub what: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Series {
    pub uid: String,
    pub number: Option<i64>,
    pub modality: String,
    pub description: String,
    pub date: String,
    pub time: String,
    /// DICOM files (instances) in the series.
    pub instances: usize,
    /// Volumes the series was split into (indices into [`Catalog::volumes`]).
    pub volumes: Vec<usize>,
    pub objects: Vec<DicomObject>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Study {
    pub uid: String,
    pub date: String,
    pub time: String,
    pub description: String,
    pub accession: String,
    pub series: Vec<Series>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Patient {
    pub key: PatientKey,
    pub name: String,
    pub id: String,
    pub birth_date: String,
    pub sex: String,
    pub studies: Vec<Study>,
    /// Non-DICOM volumes whose identifiers match exactly (indices into [`Catalog::volumes`]).
    pub other_volumes: Vec<usize>,
}

/// A file that looked like imaging data but could not be read.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Problem {
    pub path: PathBuf,
    pub message: String,
}

/// A file in a volumetric format Polymix recognises but cannot read yet.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Recognised {
    pub path: PathBuf,
    pub format: String,
}

/// What a scan passed over.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Skipped {
    /// Files that are not imaging data, counted by extension ("" = none).
    pub by_extension: BTreeMap<String, usize>,
    /// Folders left out (virtual environments, version control, …).
    pub folders: Vec<PathBuf>,
    /// Cloud files not downloaded to this computer (reading them would start a download).
    pub not_downloaded: Vec<PathBuf>,
    /// Folders or files the system would not let Polymix read.
    pub no_permission: Vec<PathBuf>,
}

impl Skipped {
    pub fn file_count(&self) -> usize {
        self.by_extension.values().sum()
    }
}

/// Everything a scan found.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Catalog {
    pub roots: Vec<PathBuf>,
    pub volumes: Vec<Volume>,
    /// DICOM patients, sorted by name.
    pub patients: Vec<Patient>,
    /// Non-DICOM volumes not matched to a patient (indices into `volumes`), in path order.
    pub files: Vec<usize>,
    pub recognised: Vec<Recognised>,
    pub problems: Vec<Problem>,
    pub dicomdirs: Vec<PathBuf>,
    pub skipped: Skipped,
    /// Files looked at.
    pub files_seen: usize,
}
