//! DICOM files into volumes and the Patient → Study → Series tree, the way 3D Slicer's DICOM
//! browser organises them.
//!
//! The rules follow Slicer (`DICOMScalarVolumePlugin.py`, `DICOMUtils.py`; BSD-style licence)
//! and CTK (`ctkDICOMDatabase.cpp`; Apache 2.0), cross-checked with dcm2niix's `isSameSet`:
//! - a **patient** is a (Patient ID, Patient Name) pair; without either, the study stands in;
//! - a **study** is a Study Instance UID (else date, time and accession number);
//! - a **series** is a Series Instance UID, split into volumes by orientation, image size,
//!   image type, echo and stack; frames at the same position become a 4th dimension;
//! - slices are ordered by their position along the slice normal (Image Position · normal),
//!   and the spacing comes from those positions, never from Slice Thickness;
//! - a cine clip (a multi-frame file whose frames step through time) and an ultrasound image
//!   not placed in the patient are images of their own (2D, or 2D over time), never slices of
//!   a volume, sized by the ultrasound regions' calibration when there is no Pixel Spacing.

use crate::dicom::{non_image_kind, transfer_syntax_name, us_pixel_spacing, Instance};
use crate::geometry::{lps_to_ras, Geometry};
use crate::model::{DataType, DicomObject, Field, Format, Kind, Patient, PatientKey, Series, Study, Volume, Voxels};
use glam::DVec3;
use std::collections::BTreeMap;

/// One image frame: a classic single-frame file, or a frame of a multi-frame file.
#[derive(Clone, Debug)]
struct Frame<'a> {
    inst: &'a Instance,
    frame: u32,
    position: Option<DVec3>,
    orientation: Option<[f64; 6]>,
    spacing: Option<[f64; 2]>,
    stack: String,
    temporal: Option<i64>,
    echo: String,
}

/// Organise DICOM instances: volumes are appended to `volumes`; returns the patients.
pub fn organize(instances: &[Instance], volumes: &mut Vec<Volume>) -> Vec<Patient> {
    let mut patients: BTreeMap<PatientKey, Vec<&Instance>> = BTreeMap::new();
    for i in instances {
        let mut key = PatientKey { id: i.patient_id.clone(), name: i.patient_name.clone() };
        if key.id.is_empty() && key.name.is_empty() {
            key.id = format!("(no ID) {}", i.study_uid);
        }
        patients.entry(key).or_default().push(i);
    }
    let mut out: Vec<Patient> = patients
        .into_iter()
        .map(|(key, insts)| {
            let first = insts[0];
            let mut studies: BTreeMap<String, Vec<&Instance>> = BTreeMap::new();
            for i in &insts {
                let k = if i.study_uid.is_empty() { format!("{} {} {}", i.study_date, i.study_time, i.accession) } else { i.study_uid.clone() };
                studies.entry(k).or_default().push(i);
            }
            let mut studies: Vec<Study> = studies.into_iter().map(|(uid, s)| study(uid, &s, volumes)).collect();
            studies.sort_by(|a, b| (&a.date, &a.time, &a.description).cmp(&(&b.date, &b.time, &b.description)));
            Patient {
                name: if first.patient_name.is_empty() { "(no name)".into() } else { first.patient_name.replace('^', " ").trim().to_string() },
                id: first.patient_id.clone(),
                birth_date: first.birth_date.clone(),
                sex: first.sex.clone(),
                key,
                studies,
                other_volumes: Vec::new(),
            }
        })
        .collect();
    out.sort_by(|a, b| (a.name.to_lowercase(), &a.id).cmp(&(b.name.to_lowercase(), &b.id)));
    out
}

fn study(uid: String, insts: &[&Instance], volumes: &mut Vec<Volume>) -> Study {
    let first = insts[0];
    let mut series: BTreeMap<String, Vec<&Instance>> = BTreeMap::new();
    for i in insts {
        let k = if i.series_uid.is_empty() { format!("{:?} {}", i.series_number, i.series_description) } else { i.series_uid.clone() };
        series.entry(k).or_default().push(i);
    }
    let mut series: Vec<Series> = series.into_iter().map(|(uid, s)| one_series(uid, &s, volumes)).collect();
    series.sort_by(|a, b| (a.number.unwrap_or(i64::MAX), &a.description).cmp(&(b.number.unwrap_or(i64::MAX), &b.description)));
    Study {
        uid,
        date: first.study_date.clone(),
        time: first.study_time.clone(),
        description: first.study_description.clone(),
        accession: first.accession.clone(),
        series,
    }
}

fn one_series(uid: String, insts: &[&Instance], volumes: &mut Vec<Volume>) -> Series {
    let first = insts[0];
    let mut objects = Vec::new();
    let mut frames: Vec<Frame> = Vec::new();
    let mut alone: Vec<&Instance> = Vec::new();
    for &i in insts {
        if let Some(what) = non_image_kind(&i.sop_class, i.rows) {
            objects.push(DicomObject { path: i.path.clone(), what: what.to_string() });
            continue;
        }
        if stands_alone(i) {
            alone.push(i);
            continue;
        }
        if !i.frame_info.is_empty() {
            for (k, f) in i.frame_info.iter().enumerate() {
                frames.push(Frame {
                    inst: i,
                    frame: k as u32,
                    position: f.position.map(DVec3::from_array),
                    orientation: f.orientation.or(i.orientation),
                    spacing: f.pixel_spacing.or(i.pixel_spacing),
                    stack: f.stack_id.clone(),
                    temporal: f.temporal_index,
                    echo: f.echo_time.map(|t| format!("{t:.2}")).unwrap_or_default(),
                });
            }
        } else {
            // Classic files (multi-frame ones without per-frame geometry keep their frames
            // in one stack at the file's position).
            for k in 0..i.frames {
                frames.push(Frame {
                    inst: i,
                    frame: k,
                    position: if k == 0 { i.position.map(DVec3::from_array) } else { None },
                    orientation: i.orientation,
                    spacing: i.pixel_spacing.or_else(|| us_pixel_spacing(&i.us_regions).map(|(s, _)| s)),
                    stack: if i.frames > 1 { i.sop_uid.clone() } else { String::new() },
                    temporal: i.temporal_position,
                    echo: i.echo_number.map(|e| e.to_string()).unwrap_or_default(),
                });
            }
        }
    }
    // Split into groups that can form one volume.
    let mut groups: BTreeMap<String, Vec<Frame>> = BTreeMap::new();
    for f in frames {
        let ori = f.orientation.map(|o| o.iter().map(|v| format!("{:.3}", v + 0.0)).collect::<Vec<_>>().join(",")).unwrap_or_else(|| "unknown".into());
        let key = format!("{ori}|{}x{}|{}|{}|{}|{:?}", f.inst.columns, f.inst.rows, f.inst.image_type.join("\\"), f.echo, f.stack, f.inst.diffusion_b.map(|b| b.round() as i64));
        groups.entry(key).or_default().push(f);
    }
    // A three-plane localizer: several orientations with a few images each.
    let orientations: std::collections::HashSet<String> = groups.keys().map(|k| k.split('|').next().unwrap_or("").to_string()).collect();
    let small_groups = groups.values().all(|g| g.len() <= 3);
    let localizer_series = orientations.len() > 1 && small_groups;
    let n_groups = groups.len();
    let mut vols = Vec::new();
    let mut echoes: Vec<String> = groups.values().map(|g| g[0].echo.clone()).collect();
    echoes.sort();
    echoes.dedup();
    for (k, (_, g)) in groups.into_iter().enumerate() {
        let suffix = if n_groups > 1 {
            let e = &g[0].echo;
            if echoes.len() > 1 && !e.is_empty() {
                format!(" · echo {}", if g[0].inst.frame_info.is_empty() { e.clone() } else { format!("TE {e} ms") })
            } else {
                format!(" · part {} of {n_groups}", k + 1)
            }
        } else {
            String::new()
        };
        for v in build_volumes(g, &suffix, localizer_series) {
            vols.push(volumes.len());
            volumes.push(v);
        }
    }
    // Stand-alone images and clips, in instance order, each named by its instance number when
    // those tell them apart.
    alone.sort_by(|a, b| (a.instance_number.unwrap_or(i64::MAX), a.acquisition_number, &a.path).cmp(&(b.instance_number.unwrap_or(i64::MAX), b.acquisition_number, &b.path)));
    let mut numbers: Vec<i64> = alone.iter().filter_map(|i| i.instance_number).collect();
    numbers.dedup();
    let by_number = numbers.len() == alone.len();
    let named = alone.len() > 1 || n_groups > 0;
    for (k, i) in alone.iter().enumerate() {
        let suffix = if named {
            let what = if i.frames > 1 { "clip" } else { "image" };
            format!(" · {what} {}", if by_number { i.instance_number.unwrap_or(0) } else { k as i64 + 1 })
        } else {
            String::new()
        };
        vols.push(volumes.len());
        volumes.push(stand_alone_volume(i, &suffix));
    }
    Series {
        uid,
        number: first.series_number,
        modality: first.modality.clone(),
        description: first.series_description.clone(),
        date: first.series_date.clone(),
        time: first.series_time.clone(),
        instances: insts.len(),
        volumes: vols,
        objects,
    }
}

/// Words (in descriptions) that mark localizers.
const LOCALIZER_WORDS: [&str; 9] = ["localizer", "localiser", "scout", "survey", "topogram", "3-plane", "3plane", "tri-pilot", "loc"];
/// Image types and description words that mark derived images.
const DERIVED_TYPES: [&str; 14] = ["MPR", "MIP", "MNIP", "PROJECTION IMAGE", "ADC", "TRACEW", "FA", "COLFA", "TRACE", "CBF", "CBV", "MTT", "TTP", "SUBTRACTION"];
/// Whole words, so that "MPR" (reformat) does not match "MPRAGE".
const DERIVED_WORDS: [&str; 9] = ["mpr", "mip", "minip", "mnip", "adc", "tracew", "relcbf", "perfusion", "subtraction"];

fn kind_of(inst: &Instance, localizer_series: bool, frames_per_position: usize) -> Kind {
    let it = &inst.image_type;
    let desc = inst.series_description.to_lowercase();
    let words: Vec<&str> = desc.split(|c: char| !c.is_alphanumeric()).collect();
    if localizer_series || it.iter().any(|t| t == "LOCALIZER") || LOCALIZER_WORDS.iter().any(|w| if w.contains('-') || w.contains("plane") { desc.contains(w) } else { words.contains(w) }) {
        Kind::Localizer
    } else if (it.first().is_some_and(|t| t == "DERIVED") && it.get(1).is_some_and(|t| t == "SECONDARY")) || it.iter().any(|t| DERIVED_TYPES.contains(&t.as_str())) || DERIVED_WORDS.iter().any(|w| words.contains(w)) {
        Kind::Derived
    } else if frames_per_position > 1 {
        Kind::TimeSeries
    } else {
        Kind::Image
    }
}

/// One group of compatible frames → one volume (or several, when acquisitions repeat
/// positions unevenly).
fn build_volumes(mut frames: Vec<Frame>, suffix: &str, localizer_series: bool) -> Vec<Volume> {
    let first = frames[0].inst;
    let ori = frames.iter().find_map(|f| f.orientation);
    let (row, col) = match ori {
        Some(o) => (DVec3::new(o[0], o[1], o[2]).normalize_or_zero(), DVec3::new(o[3], o[4], o[5]).normalize_or_zero()),
        None => (DVec3::X, DVec3::Y),
    };
    let normal = row.cross(col).normalize_or_zero();
    let has_positions = frames.iter().all(|f| f.position.is_some());
    let along = |f: &Frame| f.position.map(|p| p.dot(normal)).unwrap_or(0.0);
    frames.sort_by(|a, b| {
        let key = |f: &Frame| (f.temporal.unwrap_or(0), f.inst.acquisition_number.unwrap_or(0), f.inst.instance_number.unwrap_or(0), f.frame);
        if has_positions { along(a).total_cmp(&along(b)).then(key(a).cmp(&key(b))) } else { key(a).cmp(&key(b)) }
    });
    // Positions (within 0.01 mm) and how many frames sit at each.
    let mut positions: Vec<(f64, Vec<usize>)> = Vec::new();
    if has_positions {
        for (k, f) in frames.iter().enumerate() {
            let d = along(f);
            match positions.last_mut() {
                Some((p, idx)) if (d - *p).abs() < 0.01 => idx.push(k),
                _ => positions.push((d, vec![k])),
            }
        }
    } else {
        positions = frames.iter().enumerate().map(|(k, _)| (k as f64, vec![k])).collect();
    }
    let counts: Vec<usize> = positions.iter().map(|(_, i)| i.len()).collect();
    let per_position = counts[0];
    let even = counts.iter().all(|&c| c == per_position);
    let mut warnings = Vec::new();
    if !even {
        // Uneven repeats: separate acquisitions when that explains them.
        let mut acq: BTreeMap<i64, Vec<Frame>> = BTreeMap::new();
        for f in &frames {
            acq.entry(f.inst.acquisition_number.unwrap_or(0)).or_default().push(f.clone());
        }
        if acq.len() > 1 {
            let n = acq.len();
            return acq.into_iter().enumerate().flat_map(|(k, (a, fr))| build_volumes(fr, &format!("{suffix} · acquisition {a} ({} of {n})", k + 1), localizer_series)).collect();
        }
        warnings.push("Repeated slice positions with different counts: showing the first image at each position".to_string());
    }
    let slices: Vec<&Frame> = positions.iter().map(|(_, idx)| &frames[idx[0]]).collect();
    let n = slices.len();
    let t = if even { per_position } else { 1 };

    // Spacing from the positions.
    let gaps: Vec<f64> = positions.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let mut sorted = gaps.clone();
    sorted.sort_by(f64::total_cmp);
    let median = sorted.get(sorted.len() / 2).copied();
    let slice_spacing = median.filter(|m| *m > 1e-6).or(first.spacing_between_slices).or(first.slice_thickness).unwrap_or(1.0);
    if has_positions {
        if let Some(m) = median.filter(|m| *m > 1e-6) {
            let missing: usize = gaps.iter().filter(|g| **g > 1.5 * m).map(|g| ((g / m).round() as usize).saturating_sub(1)).sum();
            if missing > 0 {
                warnings.push(format!("{missing} missing slice(s): gaps up to {:.2} mm against {:.2} mm", sorted.last().unwrap(), m));
            } else if let (Some(lo), Some(hi)) = (sorted.first(), sorted.last()) {
                if hi - lo > (0.01 * m).max(0.01) {
                    warnings.push(format!("Uneven slice spacing: {lo:.3} – {hi:.3} mm"));
                }
            }
        }
        if n >= 2 {
            let span = slices[n - 1].position.unwrap() - slices[0].position.unwrap();
            let angle = span.normalize_or_zero().dot(normal).clamp(-1.0, 1.0).acos().to_degrees();
            if angle > 0.1 {
                warnings.push(format!("Slices are sheared (gantry tilt) by {angle:.1}°"));
            }
        }
    } else if n > 1 {
        warnings.push("No slice positions: slices are ordered by instance number".into());
    }
    if first.image_type.iter().any(|t| t == "MOSAIC") {
        warnings.push("Siemens mosaic: each image tiles many slices (not unpacked yet)".into());
    }
    let (ts_name, viewable) = transfer_syntax_name(&first.transfer_syntax);
    if !viewable {
        warnings.push(format!("Pixels compressed as {ts_name}: not viewable yet"));
    }

    let ps = slices[0].spacing.unwrap_or([1.0, 1.0]);
    if slices[0].spacing.is_none() {
        warnings.push("No pixel spacing: assuming 1 mm".into());
    }
    let dims = [first.columns as usize, first.rows as usize, n, t];
    let origin = slices[0].position.map(lps_to_ras).unwrap_or(DVec3::ZERO);
    let geometry = Geometry::from_axes(dims, [ps[1], ps[0], slice_spacing], [lps_to_ras(row), lps_to_ras(col), lps_to_ras(normal)], origin);
    if ori.is_none() {
        warnings.push("No image orientation: orientation unknown".into());
    }

    let kind = kind_of(first, localizer_series, t);
    let name = format!("{}{suffix}", series_name(first));
    let mut files: Vec<std::path::PathBuf> = frames.iter().map(|f| f.inst.path.clone()).collect();
    files.dedup();
    let mut sizes: BTreeMap<&std::path::Path, u64> = BTreeMap::new();
    for f in &frames {
        sizes.insert(&f.inst.path, f.inst.bytes);
    }
    let bytes = sizes.values().sum();

    let mut fields = header_fields(first);
    fields.extend([
        Field::new("Geometry", "Dimensions", if t > 1 { format!("{} × {} × {} × {t}", dims[0], dims[1], n) } else { format!("{} × {} × {}", dims[0], dims[1], n) }),
        Field::new("Geometry", "Voxel size", format!("{:.4} × {:.4} × {:.4} mm", ps[1], ps[0], slice_spacing)),
        Field::new("Geometry", "Orientation", {
            let ob = geometry.obliquity_deg();
            if ob > 0.5 { format!("{} (oblique {ob:.1}°)", geometry.orientation()) } else { geometry.orientation() }
        }),
        Field::new("Geometry", "Slices", format!("{n} from {} file(s){}", files.len(), if t > 1 { format!(", {t} per position") } else { String::new() })),
    ]);
    fields.extend(pixel_fields(first));
    fields.retain(|f| !f.value.is_empty());
    Vec::from([Volume {
        format: Format::Dicom,
        kind,
        name,
        files,
        bytes,
        geometry: Some(geometry),
        data_type: data_type_of(first),
        fields,
        warnings,
        // Every volume of a 4D set, one after the other (time point 0 first).
        voxels: Voxels::Dicom {
            slices: (0..t).flat_map(|v| positions.iter().map(move |(_, idx)| idx[v.min(idx.len() - 1)])).map(|k| (frames[k].inst.path.clone(), frames[k].frame)).collect(),
        },
        window: first.window,
        photometric: Some(first.photometric.clone()),
        patient: None,
        frame_time_ms: None,
    }])
}

/// Whether a file is an image (or a clip) of its own rather than a slice of a volume: a cine
/// clip, or an ultrasound image not placed in the patient. Enhanced multi-frame files (with
/// per-frame geometry) are volumes.
fn stands_alone(i: &Instance) -> bool {
    i.frame_info.is_empty() && (i.frames_in_time || (is_ultrasound(i) && i.position.is_none()))
}

fn is_ultrasound(i: &Instance) -> bool {
    // Ultrasound Image (and its retired forms), Ultrasound Multi-frame Image (and retired).
    i.modality.eq_ignore_ascii_case("US") || matches!(i.sop_class.as_str(), "1.2.840.10008.5.1.4.1.1.6" | "1.2.840.10008.5.1.4.1.1.6.1" | "1.2.840.10008.5.1.4.1.1.3" | "1.2.840.10008.5.1.4.1.1.3.1")
}

/// "Series 8 · T2 AX" (the modality when there is no description).
fn series_name(first: &Instance) -> String {
    let number = first.series_number.map(|n| format!("Series {n}")).unwrap_or_else(|| "Series".into());
    let desc = if first.series_description.is_empty() { first.modality.clone() } else { first.series_description.clone() };
    format!("{number} · {desc}")
}

fn data_type_of(first: &Instance) -> Option<DataType> {
    match (first.bits_allocated, first.pixel_representation, first.samples_per_pixel) {
        (_, _, 3) => Some(DataType::Rgb24),
        (8, 0, _) => Some(DataType::U8),
        (8, _, _) => Some(DataType::I8),
        (16, 0, _) => Some(DataType::U16),
        (16, _, _) => Some(DataType::I16),
        (32, 0, _) => Some(DataType::U32),
        (32, _, _) => Some(DataType::I32),
        _ => None,
    }
}

/// Patient, study and series fields, then the file's equipment and acquisition fields.
fn header_fields(first: &Instance) -> Vec<Field> {
    let mut fields = vec![
        Field::new("Patient", "Name", first.patient_name.replace('^', " ").trim().to_string()),
        Field::new("Patient", "ID", first.patient_id.clone()),
        Field::new("Patient", "Birth date", date_time(&first.birth_date, "")),
        Field::new("Patient", "Sex", first.sex.clone()),
        Field::new("Patient", "Age", first.age.clone()),
        Field::new("Study", "Date", date_time(&first.study_date, &first.study_time)),
        Field::new("Study", "Description", first.study_description.clone()),
        Field::new("Study", "Accession", first.accession.clone()),
        Field::new("Study", "Study ID", first.study_id.clone()),
        Field::new("Series", "Number", first.series_number.map(|n| n.to_string()).unwrap_or_default()),
        Field::new("Series", "Description", first.series_description.clone()),
        Field::new("Series", "Modality", first.modality.clone()),
        Field::new("Series", "Date", date_time(&first.series_date, &first.series_time)),
        Field::new("Series", "Image type", first.image_type.join(" \\ ")),
    ];
    fields.extend(first.fields.iter().cloned());
    fields
}

/// How the pixels are stored.
fn pixel_fields(first: &Instance) -> Vec<Field> {
    let (ts_name, viewable) = transfer_syntax_name(&first.transfer_syntax);
    let mut fields = vec![
        Field::new("Pixel", "Bits", format!("{} stored of {}{}", first.bits_stored, first.bits_allocated, if first.pixel_representation == 1 { ", signed" } else { "" })),
        Field::new("Pixel", "Photometric", first.photometric.clone()),
        Field::new("Pixel", "Transfer syntax", format!("{ts_name}{}", if viewable { "" } else { " (not viewable yet)" })),
        Field::new("Pixel", "SOP class", sop_class_name(&first.sop_class).to_string()),
    ];
    if let Some((s, i)) = first.rescale {
        fields.push(Field::new("Pixel", "Rescale", format!("{s} · v + {i}")));
    }
    if let Some((c, w)) = first.window {
        fields.push(Field::new("Pixel", "Window", format!("centre {c}, width {w}")));
    }
    fields
}

/// What an ultrasound region's layout is.
fn spatial_format_name(f: u16) -> &'static str {
    match f {
        1 => "2D",
        2 => "M-mode",
        3 => "Spectral",
        4 => "Waveform",
        5 => "Graphics",
        _ => "Region",
    }
}

/// One file as an image of its own: a 2D image, or a 2D image over time (a cine clip, its
/// frames the time points).
fn stand_alone_volume(i: &Instance, suffix: &str) -> Volume {
    let frames = i.frames.max(1) as usize;
    let mut warnings = Vec::new();
    // Pixel Spacing, else the calibration of the ultrasound regions.
    let us = us_pixel_spacing(&i.us_regions);
    let (ps, calibration) = match (i.pixel_spacing, us) {
        (Some(ps), _) => (ps, "Pixel Spacing".to_string()),
        (None, Some((ps, differ))) => {
            if differ {
                warnings.push(format!("Ultrasound regions are calibrated differently: sizes follow the largest image region ({:.4} mm per pixel)", ps[1]));
            }
            (ps, "Ultrasound region calibration".to_string())
        }
        (None, None) => {
            warnings.push(if i.us_regions.is_empty() { "No pixel spacing: assuming 1 mm".to_string() } else { "No calibrated ultrasound region: assuming 1 mm pixels".to_string() });
            ([1.0, 1.0], String::new())
        }
    };
    let (ts_name, viewable) = transfer_syntax_name(&i.transfer_syntax);
    if !viewable {
        warnings.push(format!("Pixels compressed as {ts_name}: not viewable yet"));
    }
    let (row, col) = match i.orientation {
        Some(o) => (DVec3::new(o[0], o[1], o[2]).normalize_or_zero(), DVec3::new(o[3], o[4], o[5]).normalize_or_zero()),
        None => (DVec3::X, DVec3::Y),
    };
    let normal = row.cross(col).normalize_or_zero();
    let dims = [i.columns as usize, i.rows as usize, 1, frames];
    // One slice: as thick as a pixel is wide, so the voxels are as square as the pixels.
    let geometry = Geometry::from_axes(dims, [ps[1], ps[0], ps[1]], [lps_to_ras(row), lps_to_ras(col), lps_to_ras(normal)], i.position.map(|p| lps_to_ras(DVec3::from_array(p))).unwrap_or(DVec3::ZERO));
    let kind = if frames > 1 { Kind::Cine } else { kind_of(i, false, 1) };

    let mut fields = header_fields(i);
    fields.push(Field::new("Geometry", "Dimensions", if frames > 1 { format!("{} × {} × {frames} frames", dims[0], dims[1]) } else { format!("{} × {}", dims[0], dims[1]) }));
    fields.push(Field::new("Geometry", "Pixel size", format!("{:.4} × {:.4} mm", ps[1], ps[0])));
    if !calibration.is_empty() {
        fields.push(Field::new("Geometry", "Pixel size from", calibration));
    }
    fields.push(Field::new("Geometry", "Orientation", if i.orientation.is_some() { geometry.orientation() } else { "None: not placed in the patient".to_string() }));
    if frames > 1 {
        if let Some(ms) = i.frame_time_ms {
            fields.push(Field::new("Geometry", "Frame rate", format!("{:.1} frames/s ({ms:.1} ms per frame)", 1000.0 / ms)));
            fields.push(Field::new("Geometry", "Duration", format!("{:.2} s", ms * (frames - 1) as f64 / 1000.0)));
        } else {
            fields.push(Field::new("Geometry", "Frame rate", "not in the file".to_string()));
        }
    }
    for (k, r) in i.us_regions.iter().enumerate().take(8) {
        let [x0, y0, x1, y1] = r.bounds;
        let size = match r.mm_per_pixel() {
            Some([a, d]) => format!(", {:.1} × {:.1} mm at {a:.4} mm per pixel", (x1.saturating_sub(x0) + 1) as f64 * a, (y1.saturating_sub(y0) + 1) as f64 * d),
            None => String::new(),
        };
        fields.push(Field::new("Ultrasound", &format!("Region {}", k + 1), format!("{} {}: columns {x0}–{x1}, rows {y0}–{y1}{size}", spatial_format_name(r.spatial_format), r.what())));
    }
    fields.extend(pixel_fields(i));
    fields.retain(|f| !f.value.is_empty());
    Volume {
        format: Format::Dicom,
        kind,
        name: format!("{}{suffix}", series_name(i)),
        files: vec![i.path.clone()],
        bytes: i.bytes,
        geometry: Some(geometry),
        data_type: data_type_of(i),
        fields,
        warnings,
        voxels: Voxels::Dicom { slices: (0..frames as u32).map(|k| (i.path.clone(), k)).collect() },
        window: i.window,
        photometric: Some(i.photometric.clone()),
        patient: None,
        frame_time_ms: if frames > 1 { i.frame_time_ms } else { None },
    }
}

/// DICOM date and time (`20250902`, `111146.2`) as `2025-09-02 11:11`.
pub fn date_time(d: &str, t: &str) -> String {
    let date = if d.len() == 8 && d.bytes().all(|b| b.is_ascii_digit()) { format!("{}-{}-{}", &d[..4], &d[4..6], &d[6..]) } else { d.to_string() };
    let time = if t.len() >= 4 && t.bytes().take(4).all(|b| b.is_ascii_digit()) { format!("{}:{}", &t[..2], &t[2..4]) } else { String::new() };
    format!("{date} {time}").trim().to_string()
}

/// Readable name of an image SOP class.
pub fn sop_class_name(uid: &str) -> &str {
    match uid {
        "1.2.840.10008.5.1.4.1.1.4" => "MR Image",
        "1.2.840.10008.5.1.4.1.1.4.1" => "Enhanced MR Image",
        "1.2.840.10008.5.1.4.1.1.4.3" => "Enhanced MR Color Image",
        "1.2.840.10008.5.1.4.1.1.4.4" => "Legacy Converted Enhanced MR Image",
        "1.2.840.10008.5.1.4.1.1.2" => "CT Image",
        "1.2.840.10008.5.1.4.1.1.2.1" => "Enhanced CT Image",
        "1.2.840.10008.5.1.4.1.1.2.2" => "Legacy Converted Enhanced CT Image",
        "1.2.840.10008.5.1.4.1.1.128" => "PET Image",
        "1.2.840.10008.5.1.4.1.1.130" => "Enhanced PET Image",
        "1.2.840.10008.5.1.4.1.1.20" => "Nuclear Medicine Image",
        "1.2.840.10008.5.1.4.1.1.6.1" => "Ultrasound Image",
        "1.2.840.10008.5.1.4.1.1.6.2" => "Enhanced Ultrasound Volume",
        "1.2.840.10008.5.1.4.1.1.3.1" => "Ultrasound Multi-frame Image",
        "1.2.840.10008.5.1.4.1.1.12.1" => "X-Ray Angiographic Image",
        "1.2.840.10008.5.1.4.1.1.1" => "Computed Radiography Image",
        "1.2.840.10008.5.1.4.1.1.1.1" => "Digital X-Ray Image",
        _ => uid,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn axial(series: &str, z: f64, number: i64) -> Instance {
        Instance {
            path: format!("/data/{series}/{number}.dcm").into(),
            bytes: 1000,
            sop_class: "1.2.840.10008.5.1.4.1.1.4".into(),
            sop_uid: format!("{series}.{number}"),
            transfer_syntax: "1.2.840.10008.1.2.1".into(),
            patient_name: "Doe^Jane".into(),
            patient_id: "P1".into(),
            study_uid: "S1".into(),
            study_date: "20240101".into(),
            series_uid: series.into(),
            series_number: Some(3),
            series_description: "T1 AX".into(),
            modality: "MR".into(),
            image_type: vec!["ORIGINAL".into(), "PRIMARY".into(), "M".into()],
            instance_number: Some(number),
            rows: 64,
            columns: 48,
            frames: 1,
            pixel_spacing: Some([0.5, 0.6]),
            position: Some([-10.0, -20.0, z]),
            orientation: Some([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            bits_allocated: 16,
            bits_stored: 12,
            photometric: "MONOCHROME2".into(),
            ..Default::default()
        }
    }

    #[test]
    fn time_series_keep_every_volume() {
        // 3 slices × 2 time points: repeated positions become a 4D volume with all frames.
        let mut list = Vec::new();
        for t in 0..2 {
            for k in 0..3 {
                let mut i = axial("T", k as f64 * 2.0, (t * 3 + k + 1) as i64);
                i.temporal_position = Some(t as i64 + 1);
                list.push(i);
            }
        }
        let mut vols = Vec::new();
        organize(&list, &mut vols);
        assert_eq!(vols.len(), 1);
        assert_eq!(vols[0].geometry.as_ref().unwrap().dims, [48, 64, 3, 2]);
        let Voxels::Dicom { slices } = &vols[0].voxels else { panic!() };
        let names: Vec<String> = slices.iter().map(|s| s.0.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert_eq!(names, ["1.dcm", "2.dcm", "3.dcm", "4.dcm", "5.dcm", "6.dcm"]);
        assert_eq!(crate::load::volume_count(&vols[0]), 2);
        assert_eq!(crate::slice::slice_count(&vols[0]), 3);
    }

    #[test]
    fn mprage_is_not_a_reformat() {
        let mut i = axial("A", 0.0, 1);
        i.series_description = "Brain Ax 3D T1_MPRAGE".into();
        assert_eq!(kind_of(&i, false, 1), Kind::Image);
        i.series_description = "Brain Ax 3D T1_MPRAGE_MPR_Brain Cor 3D T1_GRE".into();
        assert_eq!(kind_of(&i, false, 1), Kind::Derived);
        i.series_description = "AX DWI b1000 3mm_ADC".into();
        assert_eq!(kind_of(&i, false, 1), Kind::Derived);
        i.series_description = "Survey-32 channel".into();
        assert_eq!(kind_of(&i, false, 1), Kind::Localizer);
        i.series_description = "Blocked loc test".into();
        assert_eq!(kind_of(&i, false, 1), Kind::Localizer);
        i.series_description = "velocity block".into();
        assert_eq!(kind_of(&i, false, 1), Kind::Image);
    }

    #[test]
    fn a_plain_axial_series() {
        // Files in shuffled order; slices sorted by position.
        let mut insts: Vec<Instance> = (0..10).map(|k| axial("A", 2.0 * k as f64, 10 - k)).collect();
        insts.swap(0, 7);
        let mut vols = Vec::new();
        let patients = organize(&insts, &mut vols);
        assert_eq!(patients.len(), 1);
        assert_eq!(patients[0].name, "Doe Jane");
        let series = &patients[0].studies[0].series[0];
        assert_eq!(series.volumes, vec![0]);
        let v = &vols[0];
        assert_eq!(v.kind, Kind::Image);
        assert!(v.warnings.is_empty(), "{:?}", v.warnings);
        let g = v.geometry.as_ref().unwrap();
        assert_eq!(g.dims, [48, 64, 10, 1]);
        assert_eq!(g.spacing, [0.6, 0.5, 2.0]);
        // DICOM's LPS axes are LPS in RAS terms; voxel (0, 0, 0) is the lowest slice.
        assert_eq!(g.orientation(), "LPS");
        assert!((g.world(DVec3::ZERO) - DVec3::new(10.0, 20.0, 0.0)).length() < 1e-9);
        let Voxels::Dicom { slices } = &v.voxels else { panic!() };
        assert!(slices[0].0.ends_with("10.dcm") && slices[9].0.ends_with("1.dcm"));
    }

    #[test]
    fn missing_slices_tilt_and_echoes() {
        let mut insts: Vec<Instance> = (0..10).filter(|&k| k != 4 && k != 5).map(|k| axial("A", 2.0 * k as f64, k)).collect();
        let mut vols = Vec::new();
        organize(&insts, &mut vols);
        assert!(vols[0].warnings.iter().any(|w| w.starts_with("2 missing slice")), "{:?}", vols[0].warnings);
        // A sheared stack: positions drift in x.
        insts = (0..6).map(|k| {
            let mut i = axial("B", 2.0 * k as f64, k);
            i.position = Some([-10.0 + 0.3 * k as f64, -20.0, 2.0 * k as f64]);
            i
        }).collect();
        vols.clear();
        organize(&insts, &mut vols);
        assert!(vols[0].warnings.iter().any(|w| w.contains("gantry tilt")), "{:?}", vols[0].warnings);
        // Two echoes interleaved in one series: two volumes.
        insts = (0..12).map(|k| {
            let mut i = axial("C", 2.0 * (k / 2) as f64, k);
            i.echo_number = Some(1 + k % 2);
            i
        }).collect();
        vols.clear();
        let p = organize(&insts, &mut vols);
        assert_eq!(p[0].studies[0].series[0].volumes.len(), 2);
        assert!(vols.iter().all(|v| v.geometry.as_ref().unwrap().dims[2] == 6));
        assert!(vols[1].name.ends_with("echo 2"), "{}", vols[1].name);
    }

    #[test]
    fn repeats_become_4d_and_localizers_are_found() {
        // Five time points at the same 8 positions.
        let insts: Vec<Instance> = (0..40).map(|k| {
            let mut i = axial("D", 3.0 * (k % 8) as f64, k);
            i.temporal_position = Some(k / 8);
            i
        }).collect();
        let mut vols = Vec::new();
        organize(&insts, &mut vols);
        assert_eq!(vols.len(), 1);
        assert_eq!(vols[0].geometry.as_ref().unwrap().dims, [48, 64, 8, 5]);
        assert_eq!(vols[0].kind, Kind::TimeSeries);
        // A three-plane localizer: three orientations, a few images each.
        let mut insts = Vec::new();
        for (k, ori) in [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]].iter().enumerate() {
            for j in 0..3 {
                let mut i = axial("E", j as f64 * 5.0, (3 * k + j) as i64);
                i.orientation = Some(*ori);
                i.series_description = "AAHead_Scout".into();
                insts.push(i);
            }
        }
        vols.clear();
        let p = organize(&insts, &mut vols);
        assert_eq!(p[0].studies[0].series[0].volumes.len(), 3);
        assert!(vols.iter().all(|v| v.kind == Kind::Localizer));
    }

    #[test]
    fn patients_studies_series_and_other_objects() {
        let mut insts: Vec<Instance> = (0..4).map(|k| axial("A", k as f64, k)).collect();
        // Same patient, second study with two series; a secondary capture in one series.
        for (k, (series, number)) in [("B", 1), ("B", 2), ("C", 1)].into_iter().enumerate() {
            let mut i = axial(series, number as f64, number);
            i.study_uid = "S2".into();
            i.study_date = "20240301".into();
            i.series_number = Some(if series == "B" { 5 } else { 2 });
            if k == 2 {
                i.sop_class = "1.2.840.10008.5.1.4.1.1.7".into();
            }
            insts.push(i);
        }
        // Another patient with a non-ASCII name and no ID.
        let mut other = axial("Z", 0.0, 1);
        other.patient_id.clear();
        other.patient_name = "Müller^Zoë".into();
        other.study_uid = "S9".into();
        insts.push(other);
        let mut vols = Vec::new();
        let p = organize(&insts, &mut vols);
        assert_eq!(p.len(), 2);
        let doe = p.iter().find(|p| p.id == "P1").unwrap();
        assert_eq!(doe.studies.len(), 2);
        assert_eq!(doe.studies[0].date, "20240101", "studies in date order");
        let s2 = &doe.studies[1];
        assert_eq!(s2.series.iter().map(|s| s.number.unwrap()).collect::<Vec<_>>(), vec![2, 5], "series in number order");
        assert_eq!(s2.series[0].objects.len(), 1);
        assert_eq!(s2.series[0].objects[0].what, "Secondary capture");
        assert!(s2.series[0].volumes.is_empty());
        assert!(p.iter().any(|p| p.name == "Müller Zoë"));
    }

    /// A 2D ultrasound region calibrated in cm (`delta` cm per pixel), covering `bounds`.
    fn region(data_type: u16, bounds: [u32; 4], delta: f64) -> crate::dicom::UsRegion {
        crate::dicom::UsRegion { spatial_format: 1, data_type, bounds, units: [3, 3], delta: [delta, delta] }
    }

    /// An ultrasound image as a GE scanner stores it: no position, orientation or Pixel
    /// Spacing; calibrated by its regions.
    fn ultrasound(number: i64) -> Instance {
        let mut i = axial("US1", 0.0, number);
        i.sop_class = "1.2.840.10008.5.1.4.1.1.6.1".into();
        i.modality = "US".into();
        i.series_description.clear();
        i.series_number = Some(1);
        i.image_type = vec!["ORIGINAL".into(), "PRIMARY".into(), "SMALL PARTS".into()];
        i.position = None;
        i.orientation = None;
        i.pixel_spacing = None;
        i.bits_allocated = 8;
        i.bits_stored = 8;
        i.us_regions = vec![region(1, [2, 5, 40, 60], 0.007)];
        i
    }

    #[test]
    fn ultrasound_stills_are_images_of_their_own() {
        // B-mode and colour Doppler stills of one series, files out of order.
        let mut insts: Vec<Instance> = [3, 1, 2].into_iter().map(ultrasound).collect();
        insts[2].samples_per_pixel = 3;
        insts[2].photometric = "RGB".into();
        insts[2].us_regions[0].data_type = 2;
        let mut vols = Vec::new();
        let p = organize(&insts, &mut vols);
        assert_eq!(p[0].studies[0].series[0].volumes, vec![0, 1, 2]);
        let names: Vec<&str> = vols.iter().map(|v| v.name.as_str()).collect();
        assert_eq!(names, ["Series 1 · US · image 1", "Series 1 · US · image 2", "Series 1 · US · image 3"]);
        for v in &vols {
            assert!(v.warnings.is_empty(), "{}: {:?}", v.name, v.warnings);
            assert_eq!(v.kind, Kind::Image);
            let g = v.geometry.as_ref().unwrap();
            assert_eq!(g.dims, [48, 64, 1, 1]);
            assert!((g.spacing[0] - 0.07).abs() < 1e-12 && (g.spacing[1] - 0.07).abs() < 1e-12, "{:?}", g.spacing);
            assert_eq!(v.files.len(), 1);
            assert!(v.fields.iter().any(|f| f.name == "Pixel size from" && f.value == "Ultrasound region calibration"));
        }
        assert_eq!(vols[1].data_type, Some(DataType::Rgb24));
        assert!(vols[1].fields.iter().any(|f| f.group == "Ultrasound" && f.value.starts_with("2D colour flow")), "{:?}", vols[1].fields);
        // One image alone keeps the series' name; one without calibration says so.
        let mut lone = ultrasound(7);
        lone.us_regions.clear();
        vols.clear();
        organize(&[lone], &mut vols);
        assert_eq!(vols[0].name, "Series 1 · US");
        assert_eq!(vols[0].warnings, ["No pixel spacing: assuming 1 mm"]);
        // Regions calibrated differently: the largest image region wins, with a warning.
        let mut dual = ultrasound(8);
        dual.us_regions.push(region(1, [0, 0, 10, 10], 0.01));
        dual.us_regions.push(crate::dicom::UsRegion { spatial_format: 3, data_type: 3, bounds: [0, 0, 400, 100], units: [4, 7], delta: [0.002, 1.5] });
        vols.clear();
        organize(&[dual], &mut vols);
        assert!((vols[0].geometry.as_ref().unwrap().spacing[0] - 0.07).abs() < 1e-12);
        assert!(vols[0].warnings[0].starts_with("Ultrasound regions are calibrated differently"), "{:?}", vols[0].warnings);
    }

    #[test]
    fn cine_clips_are_2d_over_time() {
        let mut clip = ultrasound(4);
        clip.sop_class = "1.2.840.10008.5.1.4.1.1.3.1".into();
        clip.frames = 10;
        clip.frames_in_time = true;
        clip.frame_time_ms = Some(20.0);
        // An X-ray angiography run: a cine too, with a position.
        let mut xa = axial("XA", 0.0, 1);
        xa.modality = "XA".into();
        xa.frames = 6;
        xa.frames_in_time = true;
        // A classic multi-frame MR file without time: still a stack of slices.
        let mut mr = axial("MR", 0.0, 1);
        mr.frames = 4;
        let mut vols = Vec::new();
        let p = organize(&[clip, ultrasound(5), xa, mr], &mut vols);
        let by_name = |n: &str| vols.iter().find(|v| v.name.contains(n)).unwrap_or_else(|| panic!("{n}: {:?}", vols.iter().map(|v| &v.name).collect::<Vec<_>>()));
        let c = by_name("clip 4");
        assert_eq!(c.kind, Kind::Cine);
        assert_eq!(c.geometry.as_ref().unwrap().dims, [48, 64, 1, 10]);
        assert_eq!(c.frame_time_ms, Some(20.0));
        assert!(c.warnings.is_empty(), "{:?}", c.warnings);
        assert!(c.fields.iter().any(|f| f.name == "Frame rate" && f.value.starts_with("50.0 frames/s")));
        let Voxels::Dicom { slices } = &c.voxels else { panic!() };
        assert_eq!(slices.iter().map(|s| s.1).collect::<Vec<_>>(), (0..10).collect::<Vec<u32>>());
        assert_eq!((crate::load::volume_count(c), crate::slice::slice_count(c)), (10, 1));
        assert_eq!(by_name("image 5").kind, Kind::Image);
        let x = vols.iter().find(|v| v.name.starts_with("Series 3 · T1 AX") && v.kind == Kind::Cine).expect("XA cine");
        assert_eq!(x.geometry.as_ref().unwrap().dims, [48, 64, 1, 6]);
        let m = vols.iter().find(|v| v.geometry.as_ref().unwrap().dims[2] == 4).expect("MR stack");
        assert_eq!(m.kind, Kind::Image);
        assert_eq!(p.len(), 1);
    }

    /// A native ultrasound cine file: `frames` RGB frames of 4 × 3 pixels, stored by colour
    /// plane, frame `f`'s pixel `i` being (10 f + i, 100 + i, 200 − f).
    fn write_cine(path: &std::path::Path, frames: usize) {
        use dicom_core::value::DataSetSequence;
        use dicom_core::{DataElement, PrimitiveValue, VR};
        use dicom_dictionary_std::tags;
        use dicom_object::{FileMetaTableBuilder, InMemDicomObject};
        let class = "1.2.840.10008.5.1.4.1.1.3.1";
        let mut o = InMemDicomObject::new_empty();
        let s = |o: &mut InMemDicomObject, tag, vr, v: &str| o.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
        s(&mut o, tags::SOP_CLASS_UID, VR::UI, class);
        s(&mut o, tags::SOP_INSTANCE_UID, VR::UI, "1.2.3.88");
        s(&mut o, tags::STUDY_INSTANCE_UID, VR::UI, "1.2.3.80");
        s(&mut o, tags::SERIES_INSTANCE_UID, VR::UI, "1.2.3.81");
        s(&mut o, tags::MODALITY, VR::CS, "US");
        s(&mut o, tags::INSTANCE_NUMBER, VR::IS, "3");
        s(&mut o, tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "RGB");
        s(&mut o, tags::NUMBER_OF_FRAMES, VR::IS, &frames.to_string());
        s(&mut o, tags::FRAME_TIME, VR::DS, "33.3");
        o.put(DataElement::new(tags::FRAME_INCREMENT_POINTER, VR::AT, PrimitiveValue::Tags(vec![tags::FRAME_TIME].into())));
        for (tag, v) in [(tags::ROWS, 3u16), (tags::COLUMNS, 4), (tags::BITS_ALLOCATED, 8), (tags::BITS_STORED, 8), (tags::HIGH_BIT, 7), (tags::PIXEL_REPRESENTATION, 0), (tags::SAMPLES_PER_PIXEL, 3), (tags::PLANAR_CONFIGURATION, 1)] {
            o.put(DataElement::new(tag, VR::US, PrimitiveValue::from(v)));
        }
        let mut tissue = InMemDicomObject::new_empty();
        tissue.put(DataElement::new(tags::REGION_SPATIAL_FORMAT, VR::US, PrimitiveValue::from(1u16)));
        tissue.put(DataElement::new(tags::REGION_DATA_TYPE, VR::US, PrimitiveValue::from(1u16)));
        for (tag, v) in [(tags::REGION_LOCATION_MIN_X0, 0u32), (tags::REGION_LOCATION_MIN_Y0, 0), (tags::REGION_LOCATION_MAX_X1, 3), (tags::REGION_LOCATION_MAX_Y1, 2)] {
            tissue.put(DataElement::new(tag, VR::UL, PrimitiveValue::from(v)));
        }
        for tag in [tags::PHYSICAL_UNITS_X_DIRECTION, tags::PHYSICAL_UNITS_Y_DIRECTION] {
            tissue.put(DataElement::new(tag, VR::US, PrimitiveValue::from(3u16)));
        }
        for tag in [tags::PHYSICAL_DELTA_X, tags::PHYSICAL_DELTA_Y] {
            tissue.put(DataElement::new(tag, VR::FD, PrimitiveValue::from(0.025f64)));
        }
        o.put(DataElement::new(tags::SEQUENCE_OF_ULTRASOUND_REGIONS, VR::SQ, DataSetSequence::from(vec![tissue])));
        let mut pixels = Vec::new();
        for f in 0..frames {
            for c in 0..3 {
                pixels.extend((0..12).map(|i| [10 * f + i, 100 + i, 200 - f][c] as u8));
            }
        }
        o.put(DataElement::new(tags::PIXEL_DATA, VR::OB, PrimitiveValue::from(pixels)));
        let file = o.with_meta(FileMetaTableBuilder::new().transfer_syntax("1.2.840.10008.1.2.1").media_storage_sop_class_uid(class).media_storage_sop_instance_uid("1.2.3.88")).unwrap();
        file.write_to_file(path).unwrap();
    }

    #[test]
    fn ultrasound_cine_file_reads_plays_and_loads() {
        let dir = std::env::temp_dir().join(format!("mmv-cine-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cine.dcm");
        write_cine(&path, 5);
        let inst = crate::dicom::read(&path).unwrap();
        assert_eq!((inst.frames, inst.frames_in_time, inst.frame_time_ms), (5, true, Some(33.3)));
        assert_eq!(inst.us_regions.len(), 1);
        assert_eq!(crate::dicom::us_pixel_spacing(&inst.us_regions), Some(([0.25, 0.25], false)));
        let mut vols = Vec::new();
        organize(&[inst], &mut vols);
        let v = &vols[0];
        assert_eq!(v.kind, Kind::Cine);
        assert_eq!(v.geometry.as_ref().unwrap().dims, [4, 3, 1, 5]);
        assert_eq!(v.geometry.as_ref().unwrap().spacing, [0.25, 0.25, 0.25]);
        // Frame 3 in colour (stored by colour plane), and its luminance.
        let s = crate::slice::file_slice(v, 3).unwrap();
        let rgb = s.rgb.as_ref().unwrap();
        assert_eq!(rgb.len(), 12);
        for (i, p) in rgb.iter().enumerate() {
            assert_eq!(*p, [30 + i as u8, 100 + i as u8, 197], "pixel {i}");
        }
        assert!((s.values[5] - (0.299 * 35.0 + 0.587 * 105.0 + 0.114 * 197.0)).abs() < 1e-3);
        assert_eq!(s.mm_per_pixel, [0.25, 0.25]);
        // The full loader (the Segmentation workspace) reads frame 3 the same.
        let l = crate::load::load_volume_at(v, 3, usize::MAX, &|_| true).unwrap();
        assert_eq!(l.dims, [4, 3, 1]);
        assert!(l.values.iter().zip(&s.values).all(|(a, b)| (a - b).abs() < 1e-3), "{:?} / {:?}", l.values, s.values);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn enhanced_multiframe_becomes_one_volume() {
        let mut e = axial("F", 0.0, 1);
        e.sop_class = "1.2.840.10008.5.1.4.1.1.4.1".into();
        e.frames = 20;
        e.position = None;
        e.frame_info = (0..20)
            .map(|k| crate::dicom::FrameInfo { position: Some([0.0, 0.0, 1.5 * (19 - k) as f64]), orientation: e.orientation, pixel_spacing: Some([0.8, 0.8]), stack_id: "1".into(), in_stack_position: Some(k + 1), ..Default::default() })
            .collect();
        let mut vols = Vec::new();
        organize(&[e], &mut vols);
        assert_eq!(vols.len(), 1);
        let g = vols[0].geometry.as_ref().unwrap();
        assert_eq!(g.dims, [48, 64, 20, 1]);
        assert!((g.spacing[2] - 1.5).abs() < 1e-9);
        let Voxels::Dicom { slices } = &vols[0].voxels else { panic!() };
        assert_eq!(slices[0].1, 19, "the lowest slice is the last frame");
    }
}
