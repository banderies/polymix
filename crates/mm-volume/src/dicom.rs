//! DICOM headers: one compact [`Instance`] per file, read up to (not including) the pixel
//! data, with the per-frame geometry of Enhanced multi-frame files.

use crate::model::Field;
use anyhow::{Context, Result};
use dicom_core::Tag;
use dicom_dictionary_std::tags;
use dicom_object::{InMemDicomObject, OpenFileOptions};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Geometry and content of one frame of an Enhanced multi-frame file.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct FrameInfo {
    pub position: Option<[f64; 3]>,
    pub orientation: Option<[f64; 6]>,
    pub pixel_spacing: Option<[f64; 2]>,
    pub slice_thickness: Option<f64>,
    pub stack_id: String,
    pub in_stack_position: Option<i64>,
    pub temporal_index: Option<i64>,
    pub echo_time: Option<f64>,
}

/// What one DICOM file says about itself (no pixels).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Instance {
    pub path: PathBuf,
    pub bytes: u64,
    pub sop_class: String,
    pub sop_uid: String,
    pub transfer_syntax: String,
    pub patient_name: String,
    pub patient_id: String,
    pub birth_date: String,
    pub sex: String,
    pub age: String,
    pub study_uid: String,
    pub study_date: String,
    pub study_time: String,
    pub study_description: String,
    pub accession: String,
    pub study_id: String,
    pub series_uid: String,
    pub series_number: Option<i64>,
    pub series_description: String,
    pub series_date: String,
    pub series_time: String,
    pub modality: String,
    pub image_type: Vec<String>,
    pub instance_number: Option<i64>,
    pub acquisition_number: Option<i64>,
    pub echo_number: Option<i64>,
    pub temporal_position: Option<i64>,
    pub diffusion_b: Option<f64>,
    pub rows: u32,
    pub columns: u32,
    pub frames: u32,
    pub pixel_spacing: Option<[f64; 2]>,
    pub slice_thickness: Option<f64>,
    pub spacing_between_slices: Option<f64>,
    pub position: Option<[f64; 3]>,
    pub orientation: Option<[f64; 6]>,
    /// Enhanced multi-frame files: one entry per frame.
    pub frame_info: Vec<FrameInfo>,
    pub bits_allocated: u16,
    pub bits_stored: u16,
    pub pixel_representation: u16,
    pub samples_per_pixel: u16,
    pub photometric: String,
    pub rescale: Option<(f64, f64)>,
    pub window: Option<(f64, f64)>,
    pub frame_of_reference: String,
    pub gantry_tilt: Option<f64>,
    pub burned_in_annotation: bool,
    /// Equipment and acquisition fields, for display.
    pub fields: Vec<Field>,
}

fn text(obj: &InMemDicomObject, tag: Tag) -> String {
    obj.element_opt(tag).ok().flatten().and_then(|e| e.to_str().ok()).map(|s| s.trim_matches(|c: char| c == '\0' || c.is_whitespace()).to_string()).unwrap_or_default()
}

fn multi(obj: &InMemDicomObject, tag: Tag) -> Vec<String> {
    obj.element_opt(tag).ok().flatten().and_then(|e| e.to_multi_str().ok()).map(|v| v.iter().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect()).unwrap_or_default()
}

fn float(obj: &InMemDicomObject, tag: Tag) -> Option<f64> {
    obj.element_opt(tag).ok().flatten().and_then(|e| e.to_float64().ok()).filter(|v| v.is_finite())
}

fn floats(obj: &InMemDicomObject, tag: Tag) -> Vec<f64> {
    obj.element_opt(tag).ok().flatten().and_then(|e| e.to_multi_float64().ok()).unwrap_or_default()
}

fn int(obj: &InMemDicomObject, tag: Tag) -> Option<i64> {
    obj.element_opt(tag).ok().flatten().and_then(|e| e.to_int::<i64>().ok())
}

fn arr<const N: usize>(v: Vec<f64>) -> Option<[f64; N]> {
    (v.len() >= N && v.iter().take(N).all(|x| x.is_finite())).then(|| std::array::from_fn(|i| v[i]))
}

/// The first item of a sequence.
fn item(obj: &InMemDicomObject, tag: Tag) -> Option<&InMemDicomObject> {
    obj.element_opt(tag).ok().flatten()?.items()?.first()
}

/// A field from a functional-group macro (e.g. Plane Position) of a per-frame item, else of
/// the shared item.
fn group<'a>(frame: Option<&'a InMemDicomObject>, shared: Option<&'a InMemDicomObject>, seq: Tag) -> Option<&'a InMemDicomObject> {
    frame.and_then(|f| item(f, seq)).or_else(|| shared.and_then(|s| item(s, seq)))
}

/// Read the header of a DICOM file (Part 10, with or without the preamble).
pub fn read(path: &Path) -> Result<Instance> {
    let obj = OpenFileOptions::new().read_until(tags::PIXEL_DATA).open_file(path).with_context(|| "not readable as DICOM".to_string())?;
    let ts = obj.meta().transfer_syntax().trim_end_matches('\0').to_string();
    let bytes = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
    let mut i = from_object(&obj, path, bytes, ts);
    // Some objects (DICOMDIR) carry their SOP class only in the file header.
    if i.sop_class.is_empty() {
        i.sop_class = obj.meta().media_storage_sop_class_uid().trim_end_matches('\0').to_string();
    }
    Ok(i)
}

/// Read a DICOM data set without the Part 10 file header (old or exported files), guessing
/// its encoding. Reads at most `max` bytes.
pub fn read_raw(path: &Path, max: u64) -> Result<Instance> {
    use std::io::Read;
    let mut buf = Vec::new();
    std::fs::File::open(path)?.take(max).read_to_end(&mut buf)?;
    let explicit = buf.len() > 6 && buf[4].is_ascii_uppercase() && buf[5].is_ascii_uppercase();
    let ts = if explicit { dicom_transfer_syntax_registry::entries::EXPLICIT_VR_LITTLE_ENDIAN.erased() } else { dicom_transfer_syntax_registry::entries::IMPLICIT_VR_LITTLE_ENDIAN.erased() };
    let obj = InMemDicomObject::read_dataset_with_ts(std::io::Cursor::new(&buf), &ts).context("not readable as a raw DICOM data set")?;
    Ok(from_object(&obj, path, buf.len() as u64, ts.uid().to_string()))
}

fn from_object(obj: &InMemDicomObject, path: &Path, bytes: u64, transfer_syntax: String) -> Instance {
    let mut i = Instance {
        path: path.to_path_buf(),
        bytes,
        transfer_syntax,
        sop_class: text(obj, tags::SOP_CLASS_UID),
        sop_uid: text(obj, tags::SOP_INSTANCE_UID),
        patient_name: text(obj, tags::PATIENT_NAME),
        patient_id: text(obj, tags::PATIENT_ID),
        birth_date: text(obj, tags::PATIENT_BIRTH_DATE),
        sex: text(obj, tags::PATIENT_SEX),
        age: text(obj, tags::PATIENT_AGE),
        study_uid: text(obj, tags::STUDY_INSTANCE_UID),
        study_date: text(obj, tags::STUDY_DATE),
        study_time: text(obj, tags::STUDY_TIME),
        study_description: text(obj, tags::STUDY_DESCRIPTION),
        accession: text(obj, tags::ACCESSION_NUMBER),
        study_id: text(obj, tags::STUDY_ID),
        series_uid: text(obj, tags::SERIES_INSTANCE_UID),
        series_number: int(obj, tags::SERIES_NUMBER),
        series_description: text(obj, tags::SERIES_DESCRIPTION),
        series_date: text(obj, tags::SERIES_DATE),
        series_time: text(obj, tags::SERIES_TIME),
        modality: text(obj, tags::MODALITY),
        image_type: multi(obj, tags::IMAGE_TYPE).into_iter().map(|s| s.to_uppercase()).collect(),
        instance_number: int(obj, tags::INSTANCE_NUMBER),
        acquisition_number: int(obj, tags::ACQUISITION_NUMBER),
        echo_number: int(obj, tags::ECHO_NUMBERS),
        temporal_position: int(obj, tags::TEMPORAL_POSITION_IDENTIFIER),
        diffusion_b: float(obj, tags::DIFFUSION_B_VALUE),
        rows: int(obj, tags::ROWS).unwrap_or(0).max(0) as u32,
        columns: int(obj, tags::COLUMNS).unwrap_or(0).max(0) as u32,
        frames: int(obj, tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1) as u32,
        pixel_spacing: arr(floats(obj, tags::PIXEL_SPACING)),
        slice_thickness: float(obj, tags::SLICE_THICKNESS),
        spacing_between_slices: float(obj, tags::SPACING_BETWEEN_SLICES),
        position: arr(floats(obj, tags::IMAGE_POSITION_PATIENT)),
        orientation: arr(floats(obj, tags::IMAGE_ORIENTATION_PATIENT)),
        frame_info: Vec::new(),
        bits_allocated: int(obj, tags::BITS_ALLOCATED).unwrap_or(0) as u16,
        bits_stored: int(obj, tags::BITS_STORED).unwrap_or(0) as u16,
        pixel_representation: int(obj, tags::PIXEL_REPRESENTATION).unwrap_or(0) as u16,
        samples_per_pixel: int(obj, tags::SAMPLES_PER_PIXEL).unwrap_or(1) as u16,
        photometric: text(obj, tags::PHOTOMETRIC_INTERPRETATION).to_uppercase(),
        rescale: float(obj, tags::RESCALE_SLOPE).map(|s| (s, float(obj, tags::RESCALE_INTERCEPT).unwrap_or(0.0))),
        window: float(obj, tags::WINDOW_CENTER).zip(float(obj, tags::WINDOW_WIDTH)),
        frame_of_reference: text(obj, tags::FRAME_OF_REFERENCE_UID),
        gantry_tilt: float(obj, tags::GANTRY_DETECTOR_TILT),
        burned_in_annotation: text(obj, tags::BURNED_IN_ANNOTATION).eq_ignore_ascii_case("YES"),
        fields: Vec::new(),
    };
    // Enhanced multi-frame: geometry per frame from the functional groups.
    let per_frame = obj.element_opt(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE).ok().flatten().and_then(|e| e.items());
    if let Some(items) = per_frame.filter(|f| !f.is_empty()) {
        let shared = item(obj, tags::SHARED_FUNCTIONAL_GROUPS_SEQUENCE);
        for f in items {
            let f = Some(f);
            let pos = group(f, shared, tags::PLANE_POSITION_SEQUENCE).and_then(|g| arr(floats(g, tags::IMAGE_POSITION_PATIENT)));
            let ori = group(f, shared, tags::PLANE_ORIENTATION_SEQUENCE).and_then(|g| arr(floats(g, tags::IMAGE_ORIENTATION_PATIENT)));
            let measures = group(f, shared, tags::PIXEL_MEASURES_SEQUENCE);
            let content = group(f, shared, tags::FRAME_CONTENT_SEQUENCE);
            i.frame_info.push(FrameInfo {
                position: pos,
                orientation: ori,
                pixel_spacing: measures.and_then(|g| arr(floats(g, tags::PIXEL_SPACING))),
                slice_thickness: measures.and_then(|g| float(g, tags::SLICE_THICKNESS)),
                stack_id: content.map(|g| text(g, tags::STACK_ID)).unwrap_or_default(),
                in_stack_position: content.and_then(|g| int(g, tags::IN_STACK_POSITION_NUMBER)),
                temporal_index: content.and_then(|g| int(g, tags::TEMPORAL_POSITION_INDEX)),
                echo_time: group(f, shared, tags::MR_ECHO_SEQUENCE).and_then(|g| float(g, tags::EFFECTIVE_ECHO_TIME)),
            });
        }
        // Shared values stand in for the classic attributes.
        let first = i.frame_info.first().cloned().unwrap_or_default();
        i.pixel_spacing = i.pixel_spacing.or(first.pixel_spacing);
        i.slice_thickness = i.slice_thickness.or(first.slice_thickness);
        i.orientation = i.orientation.or(first.orientation);
        if i.rescale.is_none() {
            i.rescale = group(items.first(), shared, tags::PIXEL_VALUE_TRANSFORMATION_SEQUENCE).and_then(|g| float(g, tags::RESCALE_SLOPE).map(|s| (s, float(g, tags::RESCALE_INTERCEPT).unwrap_or(0.0))));
        }
        if i.window.is_none() {
            i.window = group(items.first(), shared, tags::FRAME_VOILUT_SEQUENCE).and_then(|g| float(g, tags::WINDOW_CENTER).zip(float(g, tags::WINDOW_WIDTH)));
        }
    }
    // Equipment and acquisition, for the details panel.
    let mut push = |group: &str, name: &str, tag: Tag| {
        let v = text(obj, tag);
        if !v.is_empty() {
            i.fields.push(Field::new(group, name, v));
        }
    };
    push("Series", "Protocol", tags::PROTOCOL_NAME);
    push("Series", "Body part", tags::BODY_PART_EXAMINED);
    push("Series", "Patient position", tags::PATIENT_POSITION);
    push("Equipment", "Manufacturer", tags::MANUFACTURER);
    push("Equipment", "Model", tags::MANUFACTURER_MODEL_NAME);
    push("Equipment", "Institution", tags::INSTITUTION_NAME);
    push("Equipment", "Software", tags::SOFTWARE_VERSIONS);
    push("Equipment", "Field strength (T)", tags::MAGNETIC_FIELD_STRENGTH);
    push("Acquisition", "Sequence", tags::SEQUENCE_NAME);
    push("Acquisition", "Scanning sequence", tags::SCANNING_SEQUENCE);
    push("Acquisition", "Repetition time (ms)", tags::REPETITION_TIME);
    push("Acquisition", "Echo time (ms)", tags::ECHO_TIME);
    push("Acquisition", "Flip angle (°)", tags::FLIP_ANGLE);
    push("Acquisition", "kVp", tags::KVP);
    push("Acquisition", "Tube current (mA)", tags::X_RAY_TUBE_CURRENT);
    push("Acquisition", "Kernel", tags::CONVOLUTION_KERNEL);
    push("Acquisition", "Contrast agent", tags::CONTRAST_BOLUS_AGENT);
    push("Acquisition", "Slice thickness (mm)", tags::SLICE_THICKNESS);
    push("Acquisition", "Spacing between slices (mm)", tags::SPACING_BETWEEN_SLICES);
    i
}

/// Readable name of a transfer syntax, and whether Polymix can decode its pixels.
pub fn transfer_syntax_name(uid: &str) -> (&'static str, bool) {
    match uid {
        "1.2.840.10008.1.2" => ("Implicit VR Little Endian", true),
        "1.2.840.10008.1.2.1" => ("Explicit VR Little Endian", true),
        "1.2.840.10008.1.2.2" => ("Explicit VR Big Endian", true),
        "1.2.840.10008.1.2.1.99" => ("Deflated Explicit VR Little Endian", true),
        "1.2.840.10008.1.2.4.50" => ("JPEG Baseline", true),
        "1.2.840.10008.1.2.4.51" => ("JPEG Extended", true),
        "1.2.840.10008.1.2.4.57" => ("JPEG Lossless", true),
        "1.2.840.10008.1.2.4.70" => ("JPEG Lossless SV1", true),
        "1.2.840.10008.1.2.4.80" => ("JPEG-LS Lossless", true),
        "1.2.840.10008.1.2.4.81" => ("JPEG-LS Near-Lossless", true),
        "1.2.840.10008.1.2.4.90" => ("JPEG 2000 Lossless", true),
        "1.2.840.10008.1.2.4.91" => ("JPEG 2000", true),
        "1.2.840.10008.1.2.4.201" | "1.2.840.10008.1.2.4.202" | "1.2.840.10008.1.2.4.203" => ("High-Throughput JPEG 2000", false),
        "1.2.840.10008.1.2.5" => ("RLE Lossless", true),
        _ => ("other", false),
    }
}

/// Whether a transfer syntax is JPEG-LS (decoded by [`crate::jpegls`], not dicom-rs).
pub fn is_jpegls(ts: &str) -> bool {
    matches!(ts.trim_end_matches(['\0', ' ']), "1.2.840.10008.1.2.4.80" | "1.2.840.10008.1.2.4.81")
}

/// The bytes of frame `frame` in encapsulated pixel data: by the basic offset table, one
/// fragment per frame, or a new frame at each fragment starting a JPEG stream.
fn frame_bytes(fragments: &[Vec<u8>], offsets: &[u32], frames: usize, frame: usize) -> Result<Vec<u8>> {
    if frames <= 1 {
        return Ok(fragments.concat());
    }
    if fragments.len() == frames {
        return fragments.get(frame).cloned().context("frame out of range");
    }
    let mut starts = Vec::new();
    if offsets.len() == frames {
        let mut at = 0u64;
        let mut first = vec![usize::MAX; frames];
        for (i, f) in fragments.iter().enumerate() {
            if let Some(k) = offsets.iter().rposition(|&o| o as u64 <= at) {
                if first[k] == usize::MAX {
                    first[k] = i;
                }
            }
            at += f.len() as u64 + 8;
        }
        starts = first;
    } else {
        for (i, f) in fragments.iter().enumerate() {
            if f.starts_with(&[0xff, 0xd8]) {
                starts.push(i);
            }
        }
    }
    let s = *starts.get(frame).filter(|&&s| s != usize::MAX).context("cannot find the frame in the pixel data")?;
    let e = starts.iter().skip(frame + 1).copied().find(|&e| e != usize::MAX).unwrap_or(fragments.len());
    Ok(fragments[s..e].concat())
}

/// Grey values (modality LUT applied) of frame `frame` of a JPEG-LS compressed object, with
/// its columns and rows.
pub fn jpegls_frame(obj: &dicom_object::DefaultDicomObject, frame: u32) -> Result<(usize, usize, Vec<f32>)> {
    use dicom_core::value::Value;
    let px = obj.element(tags::PIXEL_DATA).context("no pixel data")?;
    let Value::PixelSequence(seq) = px.value() else { anyhow::bail!("JPEG-LS pixel data are not encapsulated") };
    let int = |tag: Tag| obj.element_opt(tag).ok().flatten().and_then(|e| e.to_int::<i64>().ok());
    let float = |tag: Tag| obj.element_opt(tag).ok().flatten().and_then(|e| e.to_float64().ok());
    let frames = int(tags::NUMBER_OF_FRAMES).unwrap_or(1).max(1) as usize;
    let bytes = frame_bytes(seq.fragments(), seq.offset_table(), frames, frame as usize)?;
    let img = crate::jpegls::decode(&bytes)?;
    let bits = int(tags::BITS_STORED).unwrap_or(img.bits as i64).clamp(1, 16) as u32;
    let signed = int(tags::PIXEL_REPRESENTATION) == Some(1);
    let slope = float(tags::RESCALE_SLOPE).filter(|s| *s != 0.0).unwrap_or(1.0);
    let intercept = float(tags::RESCALE_INTERCEPT).unwrap_or(0.0);
    let value = |s: u16| -> f32 {
        let mut v = s as i32;
        if signed {
            let shift = 32 - bits;
            v = (v << shift) >> shift;
        }
        (v as f64 * slope + intercept) as f32
    };
    let nc = img.components;
    let grey = if nc >= 3 {
        img.samples.chunks(nc).map(|p| 0.299 * value(p[0]) + 0.587 * value(p[1]) + 0.114 * value(p[2])).collect()
    } else {
        img.samples.iter().step_by(nc).map(|&s| value(s)).collect()
    };
    Ok((img.width, img.height, grey))
}

/// What a non-image (or not-a-volume) DICOM object is, by SOP class; `None` for images.
pub fn non_image_kind(sop_class: &str, rows: u32) -> Option<&'static str> {
    let c = sop_class;
    Some(match c {
        _ if c == "1.2.840.10008.1.3.10" => "DICOMDIR",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.7") => "Secondary capture",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.88.59") => "Key object selection",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.88") => "Structured report",
        _ if c == "1.2.840.10008.5.1.4.1.1.104.1" => "Encapsulated PDF",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.104") => "Encapsulated document",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.11") => "Presentation state",
        _ if c == "1.2.840.10008.5.1.4.1.1.481.3" => "RT structure set",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.481") => "Radiotherapy object",
        _ if c == "1.2.840.10008.5.1.4.1.1.66.4" => "DICOM segmentation (not readable yet)",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.66") => "Raw data or registration",
        _ if c.starts_with("1.2.840.10008.5.1.4.1.1.9") => "Waveform",
        _ if rows == 0 => "Object without an image",
        _ => return None,
    })
}

/// Whether the first bytes of a file look like a DICOM data set without the Part 10 file
/// header: at least two elements of group 0008 with ascending element numbers (implicit
/// or explicit VR, little endian).
pub fn sniff_raw(b: &[u8]) -> bool {
    let u16le = |at: usize| b.get(at..at + 2).map(|x| u16::from_le_bytes([x[0], x[1]]));
    let u32le = |at: usize| b.get(at..at + 4).map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]));
    let (Some(g0), Some(e0)) = (u16le(0), u16le(2)) else { return false };
    if g0 != 0x0008 {
        return false;
    }
    let explicit = b.len() > 6 && b[4].is_ascii_uppercase() && b[5].is_ascii_uppercase();
    let next = if explicit {
        let vr = &b[4..6];
        if matches!(vr, b"OB" | b"OW" | b"OF" | b"SQ" | b"UT" | b"UN") {
            return false;
        }
        u16le(6).map(|l| 8 + l as usize)
    } else {
        u32le(4).filter(|&l| l < 1024).map(|l| 8 + l as usize)
    };
    let Some(n) = next else { return false };
    matches!((u16le(n), u16le(n + 2)), (Some(0x0008), Some(e1)) if e1 > e0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_core::value::{PixelFragmentSequence, Value};
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::uids;
    use dicom_object::FileMetaTableBuilder;

    #[test]
    fn jpegls_frames_with_rescale() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/jpegls");
        let jls = std::fs::read(dir.join("grey12.jls")).unwrap();
        let raw: Vec<u16> = std::fs::read(dir.join("grey12.u16")).unwrap().chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let mut o = InMemDicomObject::new_empty();
        o.put(DataElement::new(tags::SOP_CLASS_UID, VR::UI, PrimitiveValue::from("1.2.840.10008.5.1.4.1.1.2")));
        o.put(DataElement::new(tags::SOP_INSTANCE_UID, VR::UI, PrimitiveValue::from("1.2.3.4")));
        o.put(DataElement::new(tags::NUMBER_OF_FRAMES, VR::IS, PrimitiveValue::from("2")));
        o.put(DataElement::new(tags::RESCALE_SLOPE, VR::DS, PrimitiveValue::from("2")));
        o.put(DataElement::new(tags::RESCALE_INTERCEPT, VR::DS, PrimitiveValue::from("-100")));
        for (tag, v) in [(tags::ROWS, 30u16), (tags::COLUMNS, 40), (tags::BITS_ALLOCATED, 16), (tags::BITS_STORED, 12), (tags::HIGH_BIT, 11), (tags::PIXEL_REPRESENTATION, 0), (tags::SAMPLES_PER_PIXEL, 1)] {
            o.put(DataElement::new(tag, VR::US, PrimitiveValue::from(v)));
        }
        // Two frames, the second split over two fragments, found through the offset table.
        // (Fragments have even lengths: DICOM pads odd ones with a zero byte.)
        let (a, b) = jls.split_at(jls.len() / 4 * 2);
        let offsets = vec![0u32, jls.len().next_multiple_of(2) as u32 + 8];
        let seq = PixelFragmentSequence::new(offsets, vec![jls.clone(), a.to_vec(), b.to_vec()]);
        o.put(DataElement::new(tags::PIXEL_DATA, VR::OB, Value::PixelSequence(seq)));
        let file = o.with_meta(FileMetaTableBuilder::new().transfer_syntax(uids::JPEGLS_LOSSLESS).media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.2").media_storage_sop_instance_uid("1.2.3.4")).unwrap();
        let p = std::env::temp_dir().join(format!("mmv-jls-{}.dcm", std::process::id()));
        file.write_to_file(&p).unwrap();
        let obj = dicom_object::open_file(&p).unwrap();
        assert!(is_jpegls(obj.meta().transfer_syntax()));
        for f in 0..2 {
            let (w, h, v) = jpegls_frame(&obj, f).unwrap();
            assert_eq!((w, h), (40, 30));
            assert!(v.iter().zip(&raw).all(|(&x, &r)| x == r as f32 * 2.0 - 100.0), "frame {f}");
        }
        let _ = std::fs::remove_file(&p);
    }
}
