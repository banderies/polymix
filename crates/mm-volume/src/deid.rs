//! De-identified copies of images, for sharing data: DICOM with the identifying attributes
//! removed or replaced (after DICOM PS3.15's Basic Application Level Confidentiality Profile:
//! names, IDs, birth date, addresses, physicians, institution, accession; private attributes
//! dropped; UIDs replaced consistently so series and studies still group; dates optionally
//! shifted), and NIfTI / Analyze / NRRD / MetaImage with their free-text fields and NIfTI
//! extensions (which can hold a whole DICOM header) removed. Sidecar JSON files are not copied.
//!
//! Faces are not removed (no defacing), and text burned into pixels (screenshots, secondary
//! captures) cannot be removed: those objects are skipped unless asked for.

use crate::model::{Format, Volume};
use anyhow::{bail, Context, Result};
use dicom_core::{DataElement, PrimitiveValue, Tag, VR};
use dicom_dictionary_std::tags;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct DeidOptions {
    /// Replaces the patient's name and ID (e.g. "Subject 01").
    pub label: String,
    /// Days added to every date (0 keeps dates; dates are needed to tell visits apart).
    pub shift_days: i64,
    /// Keep private (vendor) attributes; they can hold names, but some tools need them.
    pub keep_private: bool,
    /// Export secondary captures and other screenshots (text may be burned into them).
    pub include_screenshots: bool,
    /// Mixed into the replacement UIDs; the same salt gives the same UIDs on a later export.
    pub salt: String,
}

impl Default for DeidOptions {
    fn default() -> Self {
        Self { label: "Anonymous".into(), shift_days: 0, keep_private: false, include_screenshots: false, salt: "polymix".into() }
    }
}

/// Attributes removed outright.
const REMOVE: [Tag; 38] = [
    tags::PATIENT_BIRTH_DATE,
    tags::PATIENT_BIRTH_TIME,
    tags::PATIENT_BIRTH_NAME,
    tags::PATIENT_ADDRESS,
    tags::PATIENT_TELEPHONE_NUMBERS,
    tags::PATIENT_MOTHER_BIRTH_NAME,
    Tag(0x0010, 0x1000), // Other Patient IDs (retired, still in old files)
    tags::OTHER_PATIENT_NAMES,
    tags::OTHER_PATIENT_I_DS_SEQUENCE,
    Tag(0x0010, 0x1090), // Medical Record Locator (retired)
    tags::MILITARY_RANK,
    tags::BRANCH_OF_SERVICE,
    tags::OCCUPATION,
    tags::ADDITIONAL_PATIENT_HISTORY,
    tags::PATIENT_COMMENTS,
    tags::ISSUER_OF_PATIENT_ID,
    tags::REFERRING_PHYSICIAN_ADDRESS,
    tags::REFERRING_PHYSICIAN_TELEPHONE_NUMBERS,
    tags::PERFORMING_PHYSICIAN_NAME,
    tags::OPERATORS_NAME,
    tags::PHYSICIANS_OF_RECORD,
    tags::NAME_OF_PHYSICIANS_READING_STUDY,
    tags::REQUESTING_PHYSICIAN,
    tags::INSTITUTION_NAME,
    tags::INSTITUTION_ADDRESS,
    tags::INSTITUTIONAL_DEPARTMENT_NAME,
    tags::STATION_NAME,
    tags::DEVICE_SERIAL_NUMBER,
    tags::REQUEST_ATTRIBUTES_SEQUENCE,
    tags::REFERENCED_PATIENT_SEQUENCE,
    tags::REFERENCED_STUDY_SEQUENCE,
    tags::CONTENT_CREATOR_NAME,
    tags::IMAGE_COMMENTS,
    Tag(0x0032, 0x4000), // Study Comments (retired)
    tags::PERFORMED_PROCEDURE_STEP_DESCRIPTION,
    tags::SCHEDULED_PROCEDURE_STEP_DESCRIPTION,
    tags::REQUESTED_PROCEDURE_DESCRIPTION,
    tags::DERIVATION_DESCRIPTION,
];

/// Attributes kept but emptied (required to be present).
const EMPTY: [Tag; 3] = [tags::ACCESSION_NUMBER, tags::REFERRING_PHYSICIAN_NAME, tags::STUDY_ID];

/// UIDs replaced (consistently).
const UIDS: [Tag; 4] = [tags::STUDY_INSTANCE_UID, tags::SERIES_INSTANCE_UID, tags::SOP_INSTANCE_UID, tags::FRAME_OF_REFERENCE_UID];

/// Dates shifted.
const DATES: [Tag; 6] = [tags::STUDY_DATE, tags::SERIES_DATE, tags::ACQUISITION_DATE, tags::CONTENT_DATE, tags::INSTANCE_CREATION_DATE, tags::PERFORMED_PROCEDURE_STEP_START_DATE];

/// A new UID for `uid`: `2.25.` and a 128-bit hash of the salt and the old UID, as decimal.
pub fn replacement_uid(uid: &str, salt: &str) -> String {
    // FNV-1a over two lanes gives 128 well-mixed bits without another dependency.
    let mut h = [0xcbf29ce484222325u64, 0x84222325cbf29ce4u64];
    for (lane, seed) in h.iter_mut().zip([0u8, 0x5a]) {
        for b in salt.bytes().chain([0xff, seed]).chain(uid.trim_end_matches('\0').bytes()) {
            *lane ^= b as u64;
            *lane = lane.wrapping_mul(0x100000001b3);
        }
    }
    let v = ((h[0] as u128) << 64) | h[1] as u128;
    format!("2.25.{v}")
}

/// `YYYYMMDD` moved by `days`.
fn shift_date(d: &str, days: i64) -> Option<String> {
    let d = d.trim();
    if d.len() != 8 {
        return None;
    }
    let (y, m, day): (i64, i64, i64) = (d[0..4].parse().ok()?, d[4..6].parse().ok()?, d[6..8].parse().ok()?);
    // Days from the civil date (Howard Hinnant's algorithm), shifted, and back.
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let z = era * 146097 + doe + days;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let dd = doy - (153 * mp + 2) / 5 + 1;
    let mm = if mp < 10 { mp + 3 } else { mp - 9 };
    let yy = yoe + era * 400 + if mm <= 2 { 1 } else { 0 };
    Some(format!("{yy:04}{mm:02}{dd:02}"))
}

/// Whether a DICOM object is a screenshot-like object (text may be in its pixels).
fn is_screenshot(sop_class: &str) -> bool {
    sop_class.starts_with("1.2.840.10008.5.1.4.1.1.7") || sop_class.starts_with("1.2.840.10008.5.1.4.1.1.104")
}

/// De-identify one DICOM file into `out`. Returns `false` when it was skipped (screenshot).
pub fn dicom_file(path: &Path, out: &Path, o: &DeidOptions) -> Result<bool> {
    let mut obj = dicom_object::open_file(path).with_context(|| format!("cannot read {}", path.display()))?;
    let sop_class = obj.meta().media_storage_sop_class_uid.trim_end_matches('\0').to_string();
    if is_screenshot(&sop_class) && !o.include_screenshots {
        return Ok(false);
    }
    for t in REMOVE {
        obj.remove_element(t);
    }
    for t in EMPTY {
        if let Some(vr) = obj.get(t).map(|e| e.vr()) {
            obj.put(DataElement::new(t, vr, PrimitiveValue::Empty));
        }
    }
    obj.put(DataElement::new(tags::PATIENT_NAME, VR::PN, PrimitiveValue::from(o.label.as_str())));
    obj.put(DataElement::new(tags::PATIENT_ID, VR::LO, PrimitiveValue::from(o.label.as_str())));
    obj.put(DataElement::new(tags::PATIENT_IDENTITY_REMOVED, VR::CS, PrimitiveValue::from("YES")));
    obj.put(DataElement::new(tags::DEIDENTIFICATION_METHOD, VR::LO, PrimitiveValue::from("Polymix: DICOM PS3.15 basic profile (subset)")));
    for t in UIDS {
        let old = obj.get(t).and_then(|e| e.to_str().ok().map(|s| s.to_string()));
        if let Some(old) = old {
            obj.put(DataElement::new(t, VR::UI, PrimitiveValue::from(replacement_uid(&old, &o.salt).as_str())));
        }
    }
    if o.shift_days != 0 {
        for t in DATES {
            let old = obj.get(t).and_then(|e| e.to_str().ok().map(|s| s.to_string()));
            if let Some(new) = old.as_deref().and_then(|d| shift_date(d, o.shift_days)) {
                obj.put(DataElement::new(t, VR::DA, PrimitiveValue::from(new.as_str())));
            }
        }
    }
    if !o.keep_private {
        let private: Vec<Tag> = obj.tags().filter(|t| t.group() % 2 == 1).collect();
        for t in private {
            obj.remove_element(t);
        }
    }
    let sop = obj.get(tags::SOP_INSTANCE_UID).and_then(|e| e.to_str().ok().map(|s| s.to_string())).unwrap_or_default();
    let meta = obj.meta_mut();
    meta.media_storage_sop_instance_uid = sop;
    meta.update_information_group_length();
    if let Some(dir) = out.parent() {
        std::fs::create_dir_all(dir)?;
    }
    obj.write_to_file(out).with_context(|| format!("cannot write {}", out.display()))?;
    Ok(true)
}

/// Read a file, inflating it when gzip-compressed.
fn read_maybe_gzip(path: &Path) -> Result<(Vec<u8>, bool)> {
    let raw = std::fs::read(path).with_context(|| format!("cannot read {}", path.display()))?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut out = Vec::new();
        flate2::read::MultiGzDecoder::new(&raw[..]).read_to_end(&mut out)?;
        Ok((out, true))
    } else {
        Ok((raw, false))
    }
}

fn write_maybe_gzip(path: &Path, bytes: &[u8], gzip: bool) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir)?;
    }
    if gzip {
        let mut e = flate2::write::GzEncoder::new(std::fs::File::create(path)?, flate2::Compression::fast());
        e.write_all(bytes)?;
        e.finish()?;
    } else {
        std::fs::write(path, bytes)?;
    }
    Ok(())
}

/// A NIfTI-1 / NIfTI-2 / Analyze header with its free text cleared and, for single files,
/// its extensions dropped (the data move up to follow the header).
fn clean_nifti(bytes: &[u8], single_file: bool) -> Result<Vec<u8>> {
    let n2 = bytes.len() >= 4 && (i32::from_le_bytes(bytes[0..4].try_into().unwrap()) == 540 || i32::from_be_bytes(bytes[0..4].try_into().unwrap()) == 540);
    let big = !n2 && i32::from_be_bytes(bytes[0..4].try_into().unwrap()) == 348;
    let mut b = bytes.to_vec();
    let (hdr, descrip, aux, offset_at) = if n2 { (540usize, 240usize, 320usize, 168usize) } else { (348, 148, 228, 108) };
    if b.len() < hdr {
        bail!("truncated header");
    }
    b[descrip..descrip + 80].fill(0);
    b[aux..aux + 24].fill(0);
    if !n2 {
        // Analyze 7.5 keeps a patient ID at 283 (NIfTI-1 uses those bytes for the qform, so
        // only for Analyze, which has no magic), plus originator / scan number fields.
        let magic = &b[344..348];
        if magic != b"n+1\0" && magic != b"ni1\0" {
            b[253..293].fill(0);
        }
    }
    if !single_file {
        return Ok(b[..hdr].to_vec());
    }
    let vox_offset = if n2 {
        let v: [u8; 8] = b[offset_at..offset_at + 8].try_into().unwrap();
        if big { i64::from_be_bytes(v) } else { i64::from_le_bytes(v) }.max(0) as usize
    } else {
        let v: [u8; 4] = b[offset_at..offset_at + 4].try_into().unwrap();
        (if big { f32::from_be_bytes(v) } else { f32::from_le_bytes(v) }).max(0.0) as usize
    };
    let new_offset = hdr + 4;
    let data = b.get(vox_offset.max(hdr)..).unwrap_or(&[]).to_vec();
    let mut out = b[..hdr].to_vec();
    out.extend([0u8; 4]); // no extensions
    if n2 {
        let v = new_offset as i64;
        out[offset_at..offset_at + 8].copy_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
    } else {
        let v = new_offset as f32;
        out[offset_at..offset_at + 4].copy_from_slice(&if big { v.to_be_bytes() } else { v.to_le_bytes() });
    }
    out.extend(data);
    Ok(out)
}

/// A text header (NRRD, MetaImage) without comments and free-text fields.
fn clean_text_header(bytes: &[u8], header_end: usize, drop: &dyn Fn(&str) -> bool) -> Vec<u8> {
    let head = String::from_utf8_lossy(&bytes[..header_end]);
    let mut out: Vec<u8> = head.split_inclusive('\n').filter(|l| !drop(l.trim_start())).flat_map(|l| l.bytes()).collect();
    out.extend_from_slice(&bytes[header_end..]);
    out
}

/// The name for a de-identified copy of `original` (its extension, possibly `.nii.gz`).
fn renamed(out_dir: &Path, name: &str, original: &Path) -> PathBuf {
    let file = original.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let ext = [".nii.gz", ".hdr.gz", ".img.gz", ".raw.gz", ".nii", ".hdr", ".img", ".nrrd", ".nhdr", ".raw", ".zraw", ".mha", ".mhd", ".mgh", ".mgz", ".tiff", ".tif", ".png", ".jpg", ".jpeg", ".bmp"]
        .into_iter()
        .find(|e| file.ends_with(e))
        .map(|e| e.to_string())
        .unwrap_or_else(|| original.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default());
    out_dir.join(format!("{name}{ext}"))
}

/// A text header with the line starting `key` pointing at `file` instead.
fn repoint(header: Vec<u8>, key: &str, sep: &str, file: &str) -> Vec<u8> {
    let text = String::from_utf8_lossy(&header).to_string();
    let mut out = String::new();
    for line in text.split_inclusive('\n') {
        if line.trim_start().to_lowercase().starts_with(&key.to_lowercase()) {
            out.push_str(&format!("{key}{sep}{file}\n"));
        } else {
            out.push_str(line);
        }
    }
    out.into_bytes()
}

/// De-identified copies of volume `v`'s files under `out_dir`, named after `name` (the
/// original file names can hold a patient's name). DICOM series go in a folder of their own.
/// Returns the files written and how many were skipped (screenshots).
pub fn export(v: &Volume, out_dir: &Path, name: &str, o: &DeidOptions, progress: &dyn Fn(f32) -> bool) -> Result<(Vec<PathBuf>, usize)> {
    let mut written = Vec::new();
    let mut skipped = 0;
    let name: String = name.chars().map(|c| if c.is_alphanumeric() || " -_.".contains(c) { c } else { '_' }).collect::<String>().trim().to_string();
    let name = if name.is_empty() { "image".to_string() } else { name };
    std::fs::create_dir_all(out_dir)?;
    match v.format {
        Format::Dicom => {
            let dir = out_dir.join(&name);
            for (k, f) in v.files.iter().enumerate() {
                let out = dir.join(format!("IM{:05}.dcm", k + 1));
                if dicom_file(f, &out, o)? {
                    written.push(out);
                } else {
                    skipped += 1;
                }
                if !progress((k + 1) as f32 / v.files.len() as f32) {
                    bail!("cancelled");
                }
            }
        }
        Format::Nifti1 | Format::Nifti2 | Format::Analyze => {
            let single = v.files.len() == 1;
            let (bytes, gz) = read_maybe_gzip(&v.files[0])?;
            let out = renamed(out_dir, &name, &v.files[0]);
            write_maybe_gzip(&out, &clean_nifti(&bytes, single)?, gz)?;
            written.push(out);
            if let Some(img) = v.files.get(1) {
                let out = renamed(out_dir, &name, img);
                std::fs::copy(img, &out)?;
                written.push(out);
            }
        }
        Format::Nrrd | Format::MetaImage => {
            let bytes = std::fs::read(&v.files[0])?;
            let nrrd = v.format == Format::Nrrd;
            let end = if nrrd {
                bytes.windows(2).position(|w| w == b"\n\n").map(|p| p + 2).unwrap_or(bytes.len())
            } else {
                crate::metaimage::parse(&bytes).map(|h| h.data_offset as usize).unwrap_or(bytes.len())
            };
            // Comments and free text go; NRRD segment keys (names, colours) stay.
            let drop = |l: &str| {
                if nrrd {
                    l.starts_with('#') || l.starts_with("content:") || (l.contains(":=") && !l.starts_with("Segment"))
                } else {
                    l.starts_with("Comment") || l.starts_with("Name")
                }
            };
            let mut cleaned = clean_text_header(&bytes, end, &drop);
            if let Some(data) = v.files.get(1) {
                let data_out = renamed(out_dir, &name, data);
                let data_name = data_out.file_name().unwrap_or_default().to_string_lossy().to_string();
                let header_len = cleaned.len() - (bytes.len() - end);
                let body = cleaned.split_off(header_len);
                cleaned = if nrrd { repoint(cleaned, "data file", ": ", &data_name) } else { repoint(cleaned, "ElementDataFile", " = ", &data_name) };
                cleaned.extend(body);
                std::fs::copy(data, &data_out)?;
                written.push(data_out);
            }
            let out = renamed(out_dir, &name, &v.files[0]);
            std::fs::write(&out, cleaned)?;
            written.insert(0, out);
        }
        Format::Mgh | Format::Tiff | Format::ImageStack => {
            // No identifying header fields; files renamed.
            let many = v.files.len() > 1;
            let dir = if many { out_dir.join(&name) } else { out_dir.to_path_buf() };
            std::fs::create_dir_all(&dir)?;
            for (k, f) in v.files.iter().enumerate() {
                let stem = if many { format!("{name}_{:04}", k + 1) } else { name.clone() };
                let out = renamed(&dir, &stem, f);
                std::fs::copy(f, &out)?;
                written.push(out);
            }
        }
    }
    Ok((written, skipped))
}

#[cfg(test)]
mod tests {
    use super::*;
    use dicom_dictionary_std::uids;
    use dicom_object::{FileMetaTableBuilder, InMemDicomObject};

    #[test]
    fn dates_shift_across_months_and_leap_years() {
        assert_eq!(shift_date("20240228", 1).as_deref(), Some("20240229"));
        assert_eq!(shift_date("20240229", 1).as_deref(), Some("20240301"));
        assert_eq!(shift_date("20231231", 1).as_deref(), Some("20240101"));
        assert_eq!(shift_date("20240101", -366).as_deref(), Some("20221231"));
        assert_eq!(shift_date("bad", 3), None);
        let a = replacement_uid("1.2.3", "s");
        assert!(a.starts_with("2.25.") && a.len() <= 64);
        assert_eq!(a, replacement_uid("1.2.3\0", "s"));
        assert_ne!(a, replacement_uid("1.2.3", "t"));
    }

    #[test]
    fn dicom_identifiers_go() {
        let dir = std::env::temp_dir().join(format!("mmv-deid-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut o = InMemDicomObject::new_empty();
        let mut put = |t, vr, v: &str| {
            o.put(DataElement::new(t, vr, PrimitiveValue::from(v)));
        };
        put(tags::PATIENT_NAME, VR::PN, "Doe^Jane");
        put(tags::PATIENT_ID, VR::LO, "MRN123");
        put(tags::PATIENT_BIRTH_DATE, VR::DA, "19800101");
        put(tags::PATIENT_SEX, VR::CS, "F");
        put(tags::INSTITUTION_NAME, VR::LO, "General Hospital");
        put(tags::ACCESSION_NUMBER, VR::SH, "ACC9");
        put(tags::STUDY_DATE, VR::DA, "20240102");
        put(tags::STUDY_INSTANCE_UID, VR::UI, "1.2.3");
        put(tags::SERIES_INSTANCE_UID, VR::UI, "1.2.3.4");
        put(tags::SOP_INSTANCE_UID, VR::UI, "1.2.3.4.5");
        put(tags::SOP_CLASS_UID, VR::UI, "1.2.840.10008.5.1.4.1.1.4");
        put(Tag(0x0029, 0x1010), VR::LO, "vendor stuff Doe");
        let file = o.with_meta(FileMetaTableBuilder::new().transfer_syntax(uids::EXPLICIT_VR_LITTLE_ENDIAN).media_storage_sop_class_uid("1.2.840.10008.5.1.4.1.1.4").media_storage_sop_instance_uid("1.2.3.4.5")).unwrap();
        let src = dir.join("in.dcm");
        file.write_to_file(&src).unwrap();
        let out = dir.join("out/a.dcm");
        let opts = DeidOptions { label: "Subject 01".into(), shift_days: 10, ..Default::default() };
        assert!(dicom_file(&src, &out, &opts).unwrap());
        let r = dicom_object::open_file(&out).unwrap();
        let s = |t| r.element_opt(t).ok().flatten().map(|e| e.to_str().unwrap().trim().to_string());
        assert_eq!(s(tags::PATIENT_NAME).as_deref(), Some("Subject 01"));
        assert_eq!(s(tags::PATIENT_ID).as_deref(), Some("Subject 01"));
        assert_eq!(s(tags::PATIENT_BIRTH_DATE), None);
        assert_eq!(s(tags::INSTITUTION_NAME), None);
        assert_eq!(s(tags::ACCESSION_NUMBER).as_deref(), Some(""));
        assert_eq!(s(tags::PATIENT_SEX).as_deref(), Some("F"));
        assert_eq!(s(tags::STUDY_DATE).as_deref(), Some("20240112"));
        assert_eq!(s(tags::STUDY_INSTANCE_UID).unwrap(), replacement_uid("1.2.3", "polymix"));
        assert!(r.element_opt(Tag(0x0029, 0x1010)).ok().flatten().is_none());
        assert_eq!(r.meta().media_storage_sop_instance_uid.trim_end_matches('\0'), replacement_uid("1.2.3.4.5", "polymix"));
        // The copy reads back through the scanner as an ordinary instance.
        let i = crate::dicom::read(&out).unwrap();
        assert_eq!(i.patient_name.trim(), "Subject 01");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn copies_are_named_after_the_label() {
        let dir = std::env::temp_dir().join(format!("mmv-deid-names-{}", std::process::id()));
        let src = dir.join("in");
        std::fs::create_dir_all(&src).unwrap();
        let hdr = src.join("Doe_Jane_scan.nhdr");
        std::fs::write(&hdr, "NRRD0004\n# patient Doe Jane\ntype: short\ndimension: 3\nsizes: 2 2 2\nencoding: raw\nendian: little\ndata file: Doe_Jane_scan.raw\n\n").unwrap();
        std::fs::write(src.join("Doe_Jane_scan.raw"), [0u8; 16]).unwrap();
        let h = crate::nrrd::read(&hdr, 1 << 16).unwrap();
        let v = crate::nrrd::volume(&hdr, &h).unwrap();
        let out = dir.join("out");
        let (files, _) = export(&v, &out, "Subject 01 T1", &DeidOptions::default(), &|_| true).unwrap();
        let names: Vec<String> = files.iter().map(|f| f.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert_eq!(names, ["Subject 01 T1.nhdr", "Subject 01 T1.raw"]);
        let text = std::fs::read_to_string(&files[0]).unwrap();
        assert!(!text.contains("Doe") && text.contains("data file: Subject 01 T1.raw"), "{text}");
        let h2 = crate::nrrd::read(&files[0], 1 << 16).unwrap();
        assert!(crate::nrrd::volume(&files[0], &h2).unwrap().warnings.iter().all(|w| !w.contains("missing")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn nifti_free_text_and_extensions_go() {
        let mut b = vec![0u8; 352];
        b[0..4].copy_from_slice(&348i32.to_le_bytes());
        b[40..42].copy_from_slice(&3i16.to_le_bytes());
        for k in 0..3 {
            b[42 + 2 * k..44 + 2 * k].copy_from_slice(&2i16.to_le_bytes());
        }
        b[70..72].copy_from_slice(&2i16.to_le_bytes());
        b[72..74].copy_from_slice(&8i16.to_le_bytes());
        b[108..112].copy_from_slice(&368.0f32.to_le_bytes());
        b[148..160].copy_from_slice(b"Doe Jane MRN");
        b[344..348].copy_from_slice(b"n+1\0");
        b[348] = 1; // an extension follows
        b.extend([0u8; 16]); // 16 bytes of extension
        b.extend(1..=8u8); // the data
        let out = clean_nifti(&b, true).unwrap();
        assert_eq!(out.len(), 352 + 8);
        assert!(out[148..228].iter().all(|&x| x == 0));
        assert_eq!(f32::from_le_bytes(out[108..112].try_into().unwrap()), 352.0);
        assert_eq!(&out[352..], &[1, 2, 3, 4, 5, 6, 7, 8]);
        let h = crate::nifti::parse(&out).unwrap();
        assert!(h.descrip.is_empty());
    }
}
