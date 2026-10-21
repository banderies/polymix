//! Find imaging data under folders: walk them (skipping tool folders such as virtual
//! environments and version control), recognise files by their first bytes rather than
//! their names (DICOM files often have none), read the headers in parallel, and organise
//! the result into a [`Catalog`].
//!
//! A file that fails to read (or makes a reader panic) becomes a [`Problem`] entry; the scan
//! carries on. A [`FileCache`] keyed by size and modification time makes rescans quick.

use crate::dicom::{self, Instance};
use crate::model::{Catalog, Problem, Recognised, Volume};
use crate::{metaimage, mgh, nifti, nrrd, organize, stack};
use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

/// A folder with at least this many 2D images of one type is reported as an image stack.
const STACK_MIN_IMAGES: usize = 16;

/// What a scan leaves out.
#[derive(Clone, Debug, PartialEq)]
pub struct ScanOptions {
    /// Folder names to skip (case-insensitive), e.g. virtual environments.
    pub skip_folders: Vec<String>,
    /// Skip folders whose name starts with a dot.
    pub skip_hidden: bool,
    /// Threads reading headers.
    pub threads: usize,
}

impl Default for ScanOptions {
    fn default() -> Self {
        Self {
            skip_folders: ["node_modules", "__pycache__", "venv", "env", "site-packages", "dist-packages", "$recycle.bin"].iter().map(|s| s.to_string()).collect(),
            skip_hidden: true,
            threads: std::thread::available_parallelism().map(|n| n.get().clamp(2, 8)).unwrap_or(4),
        }
    }
}

/// What one file turned out to be.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Found {
    Dicom(Box<Instance>),
    DicomDir,
    /// A NIfTI, Analyze, NRRD, MetaImage, MGH or multi-page TIFF volume.
    Volume(Box<Volume>),
    /// A volumetric format Polymix knows but cannot read yet.
    Recognised(String),
    NotImaging,
    Problem(String),
}

/// Scan progress, for the caller to show.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Progress {
    /// Walking the folders: files found so far.
    Listing { files: usize },
    /// Reading headers.
    Reading { done: usize, total: usize },
    /// Building the patient / study / series tree.
    Organising,
}

/// Results per file from earlier scans, reused while a file's size and modification time
/// are unchanged.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct FileCache {
    version: u32,
    entries: HashMap<PathBuf, CachedFile>,
    /// Changed since it was loaded or last saved by [`FileCache::save_if_changed`].
    #[serde(skip)]
    dirty: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct CachedFile {
    size: u64,
    modified: i128,
    found: Found,
}

/// Bump when what is stored per file changes.
const CACHE_VERSION: u32 = 3;

impl FileCache {
    /// Load a cache file (empty when missing, unreadable or from another version).
    pub fn load(path: &Path) -> Self {
        let cache: Option<FileCache> = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok());
        cache.filter(|c| c.version == CACHE_VERSION).unwrap_or_else(|| FileCache { version: CACHE_VERSION, entries: HashMap::new(), dirty: false })
    }

    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let tmp = path.with_extension("tmp");
        std::fs::write(&tmp, serde_json::to_vec(self)?)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Save only when a scan changed the cache since it was loaded or last saved this way. A
    /// change stays to be saved until a save succeeds. Returns whether the file was written.
    pub fn save_if_changed(&mut self, path: &Path) -> Result<bool> {
        if !self.dirty {
            return Ok(false);
        }
        self.save(path)?;
        self.dirty = false;
        Ok(true)
    }

    /// Whether a scan changed the cache since it was loaded or last saved with
    /// [`FileCache::save_if_changed`].
    pub fn is_changed(&self) -> bool {
        self.dirty
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// A file found by the walk.
struct Listed {
    path: PathBuf,
    size: u64,
    modified: i128,
}

fn modified(meta: &std::fs::Metadata) -> i128 {
    meta.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_nanos() as i128).unwrap_or(0)
}

/// macOS cloud files (iCloud, Google Drive, …) not downloaded yet: reading them would start
/// a download.
#[cfg(target_os = "macos")]
fn not_downloaded(meta: &std::fs::Metadata) -> bool {
    use std::os::macos::fs::MetadataExt;
    const SF_DATALESS: u32 = 0x4000_0000;
    meta.st_flags() & SF_DATALESS != 0
}

#[cfg(not(target_os = "macos"))]
fn not_downloaded(_meta: &std::fs::Metadata) -> bool {
    false
}

fn extension(path: &Path) -> String {
    path.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
}

/// Scan `roots` (folders or files). `progress` is called now and then; returning `false`
/// cancels the scan (the result is then an error). `cache` is read and updated.
pub fn scan(roots: &[PathBuf], opts: &ScanOptions, cache: Option<&mut FileCache>, progress: &(dyn Fn(Progress) -> bool + Sync)) -> Result<Catalog> {
    let mut catalog = Catalog { roots: roots.to_vec(), ..Default::default() };
    // 1. List the files.
    let mut listed = Vec::new();
    for root in roots {
        // Folders skipped by name, for the summary, noted as the walk passes them.
        let mut skipped = Vec::new();
        let walker = walkdir::WalkDir::new(root).follow_links(false).into_iter().filter_entry(|e| {
            if e.depth() == 0 || !e.file_type().is_dir() {
                return true;
            }
            let name = e.file_name().to_string_lossy().to_lowercase();
            let skip = (opts.skip_hidden && name.starts_with('.')) || opts.skip_folders.iter().any(|s| s.eq_ignore_ascii_case(&name)) || [".app", ".framework", ".bundle"].iter().any(|x| name.ends_with(x));
            if skip {
                skipped.push(e.path().to_path_buf());
            }
            !skip
        });
        for entry in walker {
            let entry = match entry {
                Ok(e) => e,
                Err(e) => {
                    let path = e.path().map(|p| p.to_path_buf()).unwrap_or_else(|| root.clone());
                    if e.io_error().is_some_and(|io| io.kind() == std::io::ErrorKind::PermissionDenied) {
                        catalog.skipped.no_permission.push(path);
                    } else {
                        catalog.problems.push(Problem { path, message: e.to_string() });
                    }
                    continue;
                }
            };
            if entry.file_type().is_dir() {
                continue;
            }
            if !entry.file_type().is_file() {
                continue; // symlinks, sockets, …
            }
            let Ok(meta) = entry.metadata() else { continue };
            if not_downloaded(&meta) {
                catalog.skipped.not_downloaded.push(entry.path().to_path_buf());
                continue;
            }
            listed.push(Listed { path: entry.into_path(), size: meta.len(), modified: modified(&meta) });
            if listed.len() % 500 == 0 && !progress(Progress::Listing { files: listed.len() }) {
                bail!("cancelled");
            }
        }
        catalog.skipped.folders.append(&mut skipped);
    }
    catalog.files_seen = listed.len();

    // 2. Read what the cache does not already know.
    let mut found: Vec<Option<Found>> = listed
        .iter()
        .map(|l| cache.as_ref().and_then(|c| c.entries.get(&l.path)).filter(|c| c.size == l.size && c.modified == l.modified).map(|c| c.found.clone()))
        .collect();
    let todo: Vec<usize> = (0..listed.len()).filter(|&k| found[k].is_none()).collect();
    let total = listed.len();
    let done = AtomicUsize::new(total - todo.len());
    let cancelled = AtomicBool::new(!progress(Progress::Reading { done: done.load(Ordering::Relaxed), total }));
    let pool = rayon::ThreadPoolBuilder::new().num_threads(opts.threads.max(1)).thread_name(|i| format!("polymix-scan-{i}")).build().context("cannot start the scan threads")?;
    let results: Vec<(usize, Found)> = pool.install(|| {
        use rayon::prelude::*;
        todo.par_iter()
            .map(|&k| {
                if cancelled.load(Ordering::Relaxed) {
                    return (k, Found::Problem("cancelled".into()));
                }
                let l = &listed[k];
                let f = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| classify(&l.path, l.size))).unwrap_or_else(|p| {
                    let msg = p.downcast_ref::<String>().cloned().or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string())).unwrap_or_default();
                    Found::Problem(format!("reader failed: {msg}"))
                });
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if n.is_multiple_of(64) && !progress(Progress::Reading { done: n, total }) {
                    cancelled.store(true, Ordering::Relaxed);
                }
                (k, f)
            })
            .collect()
    });
    if cancelled.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    if let Some(c) = cache {
        // Forget files no longer under these roots and keep the rest; the entries of files
        // read just now are added or replaced (those still valid are already there).
        let here: HashSet<&Path> = listed.iter().map(|l| l.path.as_path()).collect();
        let before = c.entries.len();
        c.entries.retain(|p, _| !roots.iter().any(|r| p.starts_with(r)) || here.contains(p.as_path()));
        let changed = c.version != CACHE_VERSION || c.entries.len() != before || !results.is_empty();
        c.version = CACHE_VERSION;
        for (k, f) in &results {
            let l = &listed[*k];
            c.entries.insert(l.path.clone(), CachedFile { size: l.size, modified: l.modified, found: f.clone() });
        }
        c.dirty |= changed;
    }
    for (k, f) in results {
        found[k] = Some(f);
    }
    if !progress(Progress::Organising) {
        bail!("cancelled");
    }

    // 3. Organise.
    let mut instances = Vec::new();
    let mut not_imaging = Vec::new();
    for (l, f) in listed.iter().zip(found) {
        match f.unwrap() {
            Found::Dicom(i) => instances.push(*i),
            Found::DicomDir => catalog.dicomdirs.push(l.path.clone()),
            Found::Volume(v) => catalog.volumes.push(*v),
            Found::Recognised(format) => catalog.recognised.push(Recognised { path: l.path.clone(), format }),
            Found::NotImaging => not_imaging.push(&l.path),
            Found::Problem(message) => catalog.problems.push(Problem { path: l.path.clone(), message }),
        }
    }
    // Data files of header / image pairs and detached NRRD headers are part of a volume.
    let claimed: HashSet<PathBuf> = catalog.volumes.iter().flat_map(|v| v.files.iter().skip(1).cloned()).collect();
    // Folders of many 2D images (a slice stack exported as JPEG, PNG, TIFF…) are volumes.
    let mut stacks: std::collections::BTreeMap<(PathBuf, String), Vec<PathBuf>> = Default::default();
    for p in &not_imaging {
        let ext = extension(p);
        if matches!(ext.as_str(), "jpg" | "jpeg" | "png" | "tif" | "tiff" | "bmp") {
            if let Some(dir) = p.parent() {
                stacks.entry((dir.to_path_buf(), ext)).or_default().push(p.to_path_buf());
            }
        }
    }
    stacks.retain(|_, files| files.len() >= STACK_MIN_IMAGES);
    for ((dir, ext), files) in &stacks {
        let n = files.len();
        match std::panic::catch_unwind(|| stack::stack_volume(dir, files.clone())) {
            Ok(Ok(v)) => catalog.volumes.push(v),
            _ => catalog.recognised.push(Recognised { path: dir.clone(), format: format!("Image stack ({n} {} files, not readable)", ext.to_uppercase()) }),
        }
    }
    for p in not_imaging {
        let ext = extension(p);
        if claimed.contains(p.as_path()) || p.parent().is_some_and(|d| stacks.contains_key(&(d.to_path_buf(), ext.clone()))) {
            continue;
        }
        *catalog.skipped.by_extension.entry(ext).or_default() += 1;
    }
    let others = catalog.volumes.len();
    catalog.patients = organize::organize(&instances, &mut catalog.volumes);
    // Non-DICOM volumes join a DICOM patient only when their identifiers match exactly.
    let mut files: Vec<usize> = Vec::new();
    for k in 0..others {
        let matched = catalog.volumes[k].patient.as_ref().and_then(|key| catalog.patients.iter_mut().find(|p| p.key == *key));
        match matched {
            Some(p) => p.other_volumes.push(k),
            None => files.push(k),
        }
    }
    files.sort_by(|&a, &b| catalog.volumes[a].files[0].cmp(&catalog.volumes[b].files[0]));
    catalog.files = files;
    catalog.recognised.sort_by(|a, b| a.path.cmp(&b.path));
    catalog.problems.sort_by(|a, b| a.path.cmp(&b.path));
    Ok(catalog)
}

/// Recognise one file by its first bytes and read its header.
pub fn classify(path: &Path, size: u64) -> Found {
    if size == 0 {
        return Found::NotImaging;
    }
    let mut head = Vec::with_capacity(4096);
    if let Err(e) = std::fs::File::open(path).and_then(|f| f.take(4096).read_to_end(&mut head)) {
        return if e.kind() == std::io::ErrorKind::PermissionDenied { Found::Problem("no permission to read".into()) } else { Found::Problem(e.to_string()) };
    }
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let ext = extension(path);
    let problem = |e: anyhow::Error| Found::Problem(format!("{e:#}"));

    // DICOM (Part 10, with or without the 128-byte preamble).
    if (head.len() >= 132 && &head[128..132] == b"DICM") || head.starts_with(b"DICM") {
        return match dicom::read_with_size(path, size) {
            Ok(i) if i.sop_class == "1.2.840.10008.1.3.10" => Found::DicomDir,
            Ok(i) => Found::Dicom(Box::new(i)),
            Err(e) => problem(e),
        };
    }
    // gzip: NIfTI inside, or something else.
    if head.starts_with(&[0x1f, 0x8b]) {
        if ext == "mgz" {
            return match nifti::read_head(path, true, mgh::HEADER_BYTES).and_then(|b| mgh::parse(&b)).and_then(|h| mgh::volume(path, true, &h)) {
                Ok(v) => Found::Volume(Box::new(v)),
                Err(e) => problem(e),
            };
        }
        let inner = match nifti::read_head(path, true, nifti::HEADER_BYTES) {
            Ok(b) => b,
            Err(e) => return if name.ends_with(".nii.gz") { problem(e) } else { Found::NotImaging },
        };
        if nifti::sniff(&inner).is_some() || (name.ends_with(".hdr.gz") && nifti::is_analyze(&inner)) {
            return match nifti::parse(&inner).and_then(|h| nifti::volume(path, true, &h)) {
                Ok(v) => Found::Volume(Box::new(v)),
                Err(e) => problem(e),
            };
        }
        return if name.ends_with(".nii.gz") { Found::Problem("not a NIfTI file".into()) } else { Found::NotImaging };
    }
    // NIfTI (single file or header of a pair) and Analyze headers.
    if let Some(flavour) = nifti::sniff(&head) {
        let is_header_of_pair = matches!(flavour, nifti::Flavour::Nifti1Pair | nifti::Flavour::Nifti2Pair);
        if !is_header_of_pair || ext == "hdr" {
            return match nifti::parse(&head).and_then(|h| nifti::volume(path, false, &h)) {
                Ok(v) => Found::Volume(Box::new(v)),
                Err(e) => problem(e),
            };
        }
    }
    if ext == "hdr" && nifti::is_analyze(&head) {
        return match nifti::pair_image(path) {
            Some(_) => match nifti::parse(&head).and_then(|h| nifti::volume(path, false, &h)) {
                Ok(v) => Found::Volume(Box::new(v)),
                Err(e) => problem(e),
            },
            None => Found::NotImaging,
        };
    }
    // NRRD.
    if nrrd::sniff(&head) {
        return match nrrd::read(path, 1 << 20).and_then(|h| nrrd::volume(path, &h)) {
            Ok(v) => Found::Volume(Box::new(v)),
            Err(e) => problem(e),
        };
    }
    // FreeSurfer MGH, ITK MetaImage, multi-page TIFF.
    if ext == "mgh" && mgh::sniff(&head) {
        return match mgh::parse(&head).and_then(|h| mgh::volume(path, false, &h)) {
            Ok(v) => Found::Volume(Box::new(v)),
            Err(e) => problem(e),
        };
    }
    if matches!(ext.as_str(), "mha" | "mhd") && metaimage::sniff(&head) {
        return match metaimage::read(path).and_then(|h| metaimage::volume(path, &h)) {
            Ok(v) => Found::Volume(Box::new(v)),
            Err(e) => problem(e),
        };
    }
    if matches!(ext.as_str(), "tif" | "tiff") && (head.starts_with(b"II") || head.starts_with(b"MM")) {
        // Single images join a folder's image stack; unreadable ones are not imaging data.
        return match stack::tiff_volume(path) {
            Ok(Some(v)) => Found::Volume(Box::new(v)),
            Ok(None) | Err(_) => Found::NotImaging,
        };
    }
    // Volumetric formats Polymix knows but cannot read yet.
    let text_start = |s: &str| head.starts_with(s.as_bytes());
    let recognised = match ext.as_str() {
        "mnc" if head.starts_with(b"\x89HDF\r\n\x1a\n") => Some("MINC 2"),
        "mnc" if head.starts_with(b"CDF\x01") || head.starts_with(b"CDF\x02") => Some("MINC 1"),
        "ims" if head.starts_with(b"\x89HDF\r\n\x1a\n") => Some("Imaris"),
        "dream3d" => Some("DREAM3D"),
        "mrc" | "map" | "rec" | "st" if head.len() >= 212 && &head[208..212] == b"MAP " => Some("MRC / CCP4"),
        _ if text_start("SIMPLE  =") => Some("FITS"),
        _ if text_start("# vtk DataFile") => Some("VTK legacy"),
        _ if text_start("!INTERFILE") => Some("Interfile"),
        _ if text_start("MATRIX7") => Some("ECAT 7"),
        _ => None,
    };
    if let Some(r) = recognised {
        return Found::Recognised(r.into());
    }
    // A DICOM data set without the Part 10 file header.
    if dicom::sniff_raw(&head) {
        return match dicom::read_raw(path, 64 << 20) {
            Ok(i) => Found::Dicom(Box::new(i)),
            Err(e) => problem(e),
        };
    }
    Found::NotImaging
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Format, Kind};
    use dicom_core::{DataElement, PrimitiveValue, VR};
    use dicom_dictionary_std::{tags, uids};
    use dicom_object::{FileMetaTableBuilder, InMemDicomObject};

    /// Write a small MR slice (4 × 3 pixels) as a Part 10 file.
    fn write_dicom(path: &Path, patient: (&str, &str), study: &str, series: &str, number: i64, z: f64, sop_class: &str) {
        let mut o = InMemDicomObject::new_empty();
        let mut s = |tag, vr, v: &str| {
            o.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
        };
        s(tags::PATIENT_NAME, VR::PN, patient.0);
        s(tags::PATIENT_ID, VR::LO, patient.1);
        s(tags::STUDY_INSTANCE_UID, VR::UI, study);
        s(tags::STUDY_DATE, VR::DA, "20240102");
        s(tags::SERIES_INSTANCE_UID, VR::UI, series);
        s(tags::SOP_CLASS_UID, VR::UI, sop_class);
        let uid = format!("{series}.{number}");
        s(tags::SOP_INSTANCE_UID, VR::UI, &uid);
        s(tags::MODALITY, VR::CS, "MR");
        s(tags::SERIES_DESCRIPTION, VR::LO, "T1 AX");
        s(tags::SERIES_NUMBER, VR::IS, "3");
        s(tags::INSTANCE_NUMBER, VR::IS, &number.to_string());
        s(tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "MONOCHROME2");
        let mut multi = |tag, vr, v: Vec<String>| {
            o.put(DataElement::new(tag, vr, PrimitiveValue::Strs(v.into_iter().collect())));
        };
        multi(tags::IMAGE_POSITION_PATIENT, VR::DS, vec!["0".into(), "0".into(), format!("{z}")]);
        multi(tags::IMAGE_ORIENTATION_PATIENT, VR::DS, ["1", "0", "0", "0", "1", "0"].map(String::from).to_vec());
        multi(tags::PIXEL_SPACING, VR::DS, vec!["0.5".into(), "0.5".into()]);
        multi(tags::IMAGE_TYPE, VR::CS, vec!["ORIGINAL".into(), "PRIMARY".into()]);
        for (tag, v) in [(tags::ROWS, 3u16), (tags::COLUMNS, 4), (tags::BITS_ALLOCATED, 16), (tags::BITS_STORED, 12), (tags::HIGH_BIT, 11), (tags::PIXEL_REPRESENTATION, 0), (tags::SAMPLES_PER_PIXEL, 1)] {
            o.put(DataElement::new(tag, VR::US, PrimitiveValue::from(v)));
        }
        o.put(DataElement::new(tags::PIXEL_DATA, VR::OW, PrimitiveValue::U16((0..12u16).map(|v| v * 100 + number as u16).collect())));
        let file = o.with_meta(FileMetaTableBuilder::new().transfer_syntax(uids::EXPLICIT_VR_LITTLE_ENDIAN).media_storage_sop_class_uid(sop_class).media_storage_sop_instance_uid(uid)).unwrap();
        file.write_to_file(path).unwrap();
    }

    /// A NIfTI-1 int16 volume (header + data).
    fn nifti_bytes(dims: [i16; 3], datatype: i16) -> Vec<u8> {
        let mut b = vec![0u8; 352];
        b[0..4].copy_from_slice(&348i32.to_le_bytes());
        b[40..42].copy_from_slice(&3i16.to_le_bytes());
        for (k, d) in dims.iter().enumerate() {
            b[42 + 2 * k..44 + 2 * k].copy_from_slice(&d.to_le_bytes());
        }
        b[70..72].copy_from_slice(&datatype.to_le_bytes());
        b[72..74].copy_from_slice(&16i16.to_le_bytes());
        for k in 0..4 {
            b[76 + 4 * k..80 + 4 * k].copy_from_slice(&1.0f32.to_le_bytes());
        }
        b[108..112].copy_from_slice(&352.0f32.to_le_bytes());
        b[344..348].copy_from_slice(b"n+1\0");
        let n = dims.iter().map(|&d| d as usize).product::<usize>();
        b.extend((0..n).flat_map(|v| (v as i16).to_le_bytes()));
        b
    }

    fn gzip(b: &[u8]) -> Vec<u8> {
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(b).unwrap();
        e.finish().unwrap()
    }

    /// A messy folder: DICOM without extensions, NIfTI / Analyze / NRRD, a venv and .git,
    /// an exported JPEG stack, junk and broken files.
    fn messy_tree() -> PathBuf {
        let root = std::env::temp_dir().join(format!("mmv-scan-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
        let mk = |p: &str| {
            let p = root.join(p);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            p
        };
        let mr = "1.2.840.10008.5.1.4.1.1.4";
        for k in 0..5 {
            write_dicom(&mk(&format!("CD/DICOM/A/IM_000{k}")), ("Doe^Jane", "P1"), "1.2.3.1", "1.2.3.1.1", k, 2.0 * (4 - k) as f64, mr);
        }
        for k in 0..3 {
            write_dicom(&mk(&format!("CD/DICOM/B/{k}/DICOM")), ("Doe^Jane", "P1"), "1.2.3.1", "1.2.3.1.2", k, k as f64, mr);
        }
        write_dicom(&mk("CD/DICOM/B/sc"), ("Doe^Jane", "P1"), "1.2.3.1", "1.2.3.1.2", 9, 0.0, "1.2.840.10008.5.1.4.1.1.7");
        std::fs::write(mk("niftis/brain.nii"), nifti_bytes([4, 3, 2], 4)).unwrap();
        std::fs::write(mk("niftis/brain_mask.nii.gz"), gzip(&nifti_bytes([4, 3, 2], 4))).unwrap();
        let mut analyze = nifti_bytes([4, 3, 2], 4);
        analyze[344..348].copy_from_slice(b"\0\0\0\0");
        std::fs::write(mk("niftis/old.hdr"), &analyze[..348]).unwrap();
        std::fs::write(mk("niftis/old.img"), &analyze[352..]).unwrap();
        std::fs::write(mk("slicer/vol.nrrd"), b"NRRD0004\ntype: short\ndimension: 3\nspace: left-posterior-superior\nsizes: 4 3 2\nspace directions: (1,0,0) (0,1,0) (0,0,1)\nencoding: raw\nendian: little\n\n").unwrap();
        // Noise.
        write_dicom(&mk("scripts/venv/lib/site-packages/pydicom/test.dcm"), ("Test^Data", "T"), "9", "9.1", 1, 0.0, mr);
        std::fs::write(mk("scripts/.git/objects/ab"), b"blob").unwrap();
        std::fs::write(mk("scripts/run.py"), b"print('hi')").unwrap();
        std::fs::write(mk("CD/AUTORUN.INF"), b"[autorun]").unwrap();
        std::fs::write(mk("CD/empty.dcm"), b"").unwrap();
        let mut garbage = vec![0u8; 400];
        garbage[128..132].copy_from_slice(b"DICM");
        std::fs::write(mk("CD/garbage.dcm"), garbage).unwrap();
        let mut broken = nifti_bytes([4, 3, 2], 4);
        broken[40..42].copy_from_slice(&0i16.to_le_bytes());
        std::fs::write(mk("niftis/broken.nii"), broken).unwrap();
        std::fs::write(mk("niftis/fake.nii.gz"), gzip(b"hello")).unwrap();
        for k in 0..20u8 {
            image::GrayImage::from_pixel(8, 6, image::Luma([k])).save(mk(&format!("export/INV2/slice{k}.png"))).unwrap();
        }
        for k in 0..16 {
            std::fs::write(mk(&format!("export/broken/slice{k:03}.jpg")), b"\xff\xd8\xff").unwrap();
        }
        root
    }

    #[test]
    fn scans_a_messy_folder() {
        let root = messy_tree();
        let c = scan(std::slice::from_ref(&root), &ScanOptions::default(), None, &|_| true).unwrap();
        // DICOM: one patient, one study, two series (the second with a secondary capture).
        assert_eq!(c.patients.len(), 1, "{:?}", c.patients.iter().map(|p| &p.name).collect::<Vec<_>>());
        let p = &c.patients[0];
        assert_eq!(p.name, "Doe Jane");
        assert_eq!(p.studies.len(), 1);
        let series = &p.studies[0].series;
        assert_eq!(series.len(), 2);
        assert_eq!(series.iter().map(|s| s.instances).sum::<usize>(), 9);
        let a = &c.volumes[series.iter().find(|s| s.instances == 5).unwrap().volumes[0]];
        assert_eq!(a.geometry.as_ref().unwrap().dims, [4, 3, 5, 1]);
        assert!(series.iter().any(|s| s.objects.iter().any(|o| o.what == "Secondary capture")));
        // Other files.
        let others: Vec<(&str, Format, Kind)> = c.files.iter().map(|&k| (c.volumes[k].name.as_str(), c.volumes[k].format, c.volumes[k].kind)).collect();
        assert!(others.contains(&("brain.nii", Format::Nifti1, Kind::Image)), "{others:?}");
        assert!(others.contains(&("brain_mask.nii.gz", Format::Nifti1, Kind::LabelMap)), "{others:?}");
        assert!(others.contains(&("old.hdr", Format::Analyze, Kind::Image)), "{others:?}");
        assert!(others.contains(&("vol.nrrd", Format::Nrrd, Kind::Image)), "{others:?}");
        assert!(others.contains(&("INV2", Format::ImageStack, Kind::Image)), "{others:?}");
        assert_eq!(others.len(), 5);
        // Noise: the venv and .git were not opened; junk is counted, broken files reported.
        assert!(c.skipped.folders.iter().any(|f| f.ends_with("venv")));
        assert!(c.skipped.folders.iter().any(|f| f.ends_with(".git")));
        assert!(!c.volumes.iter().any(|v| v.files[0].to_string_lossy().contains("site-packages")));
        let problems: Vec<String> = c.problems.iter().map(|p| p.path.file_name().unwrap().to_string_lossy().to_string()).collect();
        assert!(problems.contains(&"garbage.dcm".to_string()) && problems.contains(&"broken.nii".to_string()) && problems.contains(&"fake.nii.gz".to_string()), "{problems:?}");
        assert_eq!(c.skipped.by_extension.get("py"), Some(&1));
        assert!(!c.skipped.by_extension.contains_key("img"), "the Analyze data file belongs to its header");
        let stack = c.volumes.iter().find(|v| v.format == Format::ImageStack).unwrap();
        assert_eq!(stack.geometry.as_ref().unwrap().dims, [8, 6, 20, 1]);
        assert!(c.recognised.iter().any(|r| r.format == "Image stack (16 JPG files, not readable)"), "{:?}", c.recognised);
        assert!(!c.skipped.by_extension.contains_key("jpg") && !c.skipped.by_extension.contains_key("png"));

        // A cached rescan gives the same catalog.
        let mut cache = FileCache::default();
        let first = scan(std::slice::from_ref(&root), &ScanOptions::default(), Some(&mut cache), &|_| true).unwrap();
        assert!(cache.len() >= 20);
        let path = root.join("cache.json");
        cache.save(&path).unwrap();
        let mut loaded = FileCache::load(&path);
        std::fs::remove_file(&path).unwrap();
        let reads = AtomicUsize::new(0);
        let second = scan(std::slice::from_ref(&root), &ScanOptions::default(), Some(&mut loaded), &|p| {
            if let Progress::Reading { done, total } = p {
                reads.store(total - done, Ordering::Relaxed);
            }
            true
        })
        .unwrap();
        assert_eq!(reads.load(Ordering::Relaxed), 0, "nothing left to read");
        assert_eq!(first, second);
        // Cancelling stops the scan.
        assert!(scan(std::slice::from_ref(&root), &ScanOptions::default(), None, &|p| !matches!(p, Progress::Reading { .. })).is_err());
        let _ = std::fs::remove_dir_all(&root);
    }

    /// The folders the second walk found before (it stopped at the first error).
    fn skipped_by_a_second_walk(root: &Path, opts: &ScanOptions) -> Vec<PathBuf> {
        let mut out = Vec::new();
        let mut it = walkdir::WalkDir::new(root).follow_links(false).into_iter().filter_entry(|e| e.file_type().is_dir());
        while let Some(Ok(e)) = it.next() {
            if e.depth() == 0 {
                continue;
            }
            let name = e.file_name().to_string_lossy().to_lowercase();
            let skip = (opts.skip_hidden && name.starts_with('.')) || opts.skip_folders.iter().any(|s| s.eq_ignore_ascii_case(&name)) || [".app", ".framework", ".bundle"].iter().any(|x| name.ends_with(x));
            if skip {
                out.push(e.path().to_path_buf());
                it.skip_current_dir();
            }
        }
        out
    }

    #[test]
    fn the_cache_changes_only_when_files_do() {
        let root = messy_tree();
        for extra in ["a/.hidden/x", "a/b/node_modules/y", "a/b/c/Thing.app/z", "z/venv/w"] {
            let p = root.join(extra);
            std::fs::create_dir_all(p.parent().unwrap()).unwrap();
            std::fs::write(p, b"x").unwrap();
        }
        let opts = ScanOptions::default();
        let saved = std::env::temp_dir().join(format!("mmv-cache-{}.json", std::process::id()));
        let mut cache = FileCache::default();
        let c = scan(std::slice::from_ref(&root), &opts, Some(&mut cache), &|_| true).unwrap();
        assert_eq!(c.skipped.folders, skipped_by_a_second_walk(&root, &opts));
        assert_eq!(c.skipped.folders.len(), 6);
        assert!(cache.is_changed());
        assert!(cache.save_if_changed(&saved).unwrap());
        assert!(!cache.is_changed() && !cache.save_if_changed(&saved).unwrap());
        // Loaded and rescanned with nothing changed: nothing to save.
        let mut loaded = FileCache::load(&saved);
        assert!(!loaded.is_changed());
        let again = scan(std::slice::from_ref(&root), &opts, Some(&mut loaded), &|_| true).unwrap();
        assert_eq!(again, c);
        assert!(!loaded.is_changed());
        // A new file, a changed one and a removed one each make a change to save; the entries
        // end as a fresh scan's would.
        let rescan = |cache: &mut FileCache| {
            scan(std::slice::from_ref(&root), &opts, Some(cache), &|_| true).unwrap();
            let changed = cache.is_changed();
            cache.save_if_changed(&saved).unwrap();
            let mut fresh = FileCache::default();
            scan(std::slice::from_ref(&root), &opts, Some(&mut fresh), &|_| true).unwrap();
            let paths = |c: &FileCache| {
                let mut p: Vec<(PathBuf, u64)> = c.entries.iter().map(|(p, e)| (p.clone(), e.size)).collect();
                p.sort();
                p
            };
            assert_eq!(paths(cache), paths(&fresh));
            changed
        };
        std::fs::write(root.join("niftis/notes.txt"), b"new").unwrap();
        assert!(rescan(&mut loaded));
        std::fs::write(root.join("scripts/run.py"), b"print('changed')").unwrap();
        assert!(rescan(&mut loaded));
        std::fs::remove_file(root.join("niftis/notes.txt")).unwrap();
        assert!(rescan(&mut loaded));
        assert!(!rescan(&mut loaded));
        let _ = std::fs::remove_file(&saved);
        let _ = std::fs::remove_dir_all(&root);
    }

    #[test]
    fn raw_datasets_are_recognised() {
        // (0008,0016) UI and (0008,0018) UI, implicit VR: group, element, 4-byte length.
        let mut b = Vec::new();
        for (e, v) in [(0x0016u16, "1.2.3\0"), (0x0018, "4.5.6\0")] {
            b.extend(0x0008u16.to_le_bytes());
            b.extend(e.to_le_bytes());
            b.extend((v.len() as u32).to_le_bytes());
            b.extend(v.as_bytes());
        }
        assert!(dicom::sniff_raw(&b));
        assert!(!dicom::sniff_raw(b"\x08\x00\x16\x00garbage that is not dicom"));
        assert!(!dicom::sniff_raw(&[0x93, b'N', b'U', b'M', b'P', b'Y']));
    }
}
