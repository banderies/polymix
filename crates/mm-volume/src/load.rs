//! A whole 3D volume in memory, for viewing and segmenting: grey values in voxel order
//! (i fastest, then j, then k) with the grid's placement in patient space. Raw formats are
//! read a few megabytes at a time; DICOM files and 2D images are decoded in parallel straight
//! into place, each file opened once (the frames of an Enhanced multi-frame file and the pages
//! of a multi-page TIFF are decoded in parallel too). [`shared`] lets the parts of the app that
//! show the same volume share one load of it.

use crate::geometry::Geometry;
use crate::model::{Volume, Voxels};
use anyhow::{anyhow, bail, Context, Result};
use dicom_object::DefaultDicomObject;
use dicom_pixeldata::{DecodedPixelData, PixelDecoder};
use rayon::prelude::*;
use std::collections::HashMap;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex, MutexGuard, Weak};
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct LoadedVolume {
    pub name: String,
    pub dims: [usize; 3],
    pub values: Vec<f32>,
    /// Voxel → RAS millimetres (first 3D volume of a 4D set).
    pub geometry: Geometry,
    /// Smallest and largest finite value.
    pub range: (f32, f32),
    /// Window (centre, width) from the 1st to the 99th percentile.
    pub auto_window: (f64, f64),
    /// The window stored in the file, if any.
    pub file_window: Option<(f64, f64)>,
    /// Show inverted (DICOM MONOCHROME1).
    pub invert: bool,
}

impl LoadedVolume {
    pub fn index(&self, i: usize, j: usize, k: usize) -> usize {
        (k * self.dims[1] + j) * self.dims[0] + i
    }

    pub fn get(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[self.index(i, j, k)]
    }

    pub fn voxel_count(&self) -> usize {
        self.dims.iter().product()
    }

    /// A volume from values in memory (tests, results of processing).
    pub fn from_values(name: &str, dims: [usize; 3], values: Vec<f32>, geometry: Geometry) -> Self {
        let range = finite_range(&values);
        let auto_window = sampled_window(&values);
        Self { name: name.into(), dims, values, geometry, range, auto_window, file_window: None, invert: false }
    }
}

fn finite_range(v: &[f32]) -> (f32, f32) {
    let (lo, hi) = v.par_iter().copied().filter(|x| x.is_finite()).fold(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), x| (a.min(x), b.max(x))).reduce(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), (c, d)| (a.min(c), b.max(d)));
    if lo <= hi { (lo, hi) } else { (0.0, 0.0) }
}

/// Percentile window from at most a million evenly spread values.
fn sampled_window(v: &[f32]) -> (f64, f64) {
    let step = (v.len() / 1_000_000).max(1);
    let sample: Vec<f32> = v.iter().step_by(step).copied().collect();
    crate::slice::auto_window(&sample)
}

/// Load the first 3D volume of `v` (at most `max_voxels` voxels). `progress` gets the
/// fraction done now and then; returning `false` cancels.
pub fn load_volume(v: &Volume, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<LoadedVolume> {
    load_volume_at(v, 0, max_voxels, progress)
}

/// How many 3D volumes `v` holds (time points, echoes, components).
pub fn volume_count(v: &Volume) -> usize {
    let t = v.geometry.as_ref().map(|g| g.dims[3]).unwrap_or(1).max(1);
    match &v.voxels {
        Voxels::Dicom { slices } => {
            let n = v.geometry.as_ref().map(|g| g.dims[2]).unwrap_or(slices.len()).max(1);
            (slices.len() / n).clamp(1, t)
        }
        Voxels::Images { .. } | Voxels::Unsupported(_) => 1,
        Voxels::Raw { .. } => t,
    }
}

/// The checks before reading anything: the volume exists, has a grid, is not empty and fits.
fn checked(v: &Volume, t: usize, max_voxels: usize) -> Result<(Geometry, [usize; 3])> {
    if t >= volume_count(v) {
        bail!("there is no volume {} (of {})", t + 1, volume_count(v));
    }
    let g = v.geometry.clone().context("no geometry")?;
    let dims = [g.dims[0], g.dims[1], g.dims[2]];
    let n = dims.iter().product::<usize>();
    if n == 0 {
        bail!("empty volume");
    }
    if n > max_voxels {
        bail!("too large to load ({} million voxels; the limit is {} million)", n / 1_000_000, max_voxels / 1_000_000);
    }
    Ok((g, dims))
}

/// Load 3D volume `t` (0-based) of a 4D `v`.
pub fn load_volume_at(v: &Volume, t: usize, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<LoadedVolume> {
    load_at(v, t, max_voxels, progress, None)
}

/// Load the 3D volumes `ts` of `v` one after another, handing each to `sink` as it is ready
/// (each the same as [`load_volume_at`] gives). A multi-frame DICOM file that several of them
/// read is opened once for those, and pixel data that cannot be decoded frame by frame (such
/// as uncompressed data) decoded once, rather than once per volume; it is let go after the
/// last volume that reads it. `progress` gets the fraction of all of them done; returning
/// `false` cancels. Stops at the first error, from `sink` too.
pub fn load_volumes_at(v: &Volume, ts: &[usize], max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync), sink: &mut dyn FnMut(usize, LoadedVolume) -> Result<()>) -> Result<()> {
    load_volumes_with(v, ts, max_voxels, progress, sink, &OpenFiles::new(v, ts))
}

fn load_volumes_with(v: &Volume, ts: &[usize], max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync), sink: &mut dyn FnMut(usize, LoadedVolume) -> Result<()>, open: &OpenFiles) -> Result<()> {
    let n = ts.len().max(1) as f32;
    for (i, &t) in ts.iter().enumerate() {
        let vol = load_at(v, t, max_voxels, &|f| progress((i as f32 + f) / n), Some(open))?;
        // Files no later volume reads are closed before the caller works on this one.
        open.done_with(i);
        sink(t, vol)?;
    }
    Ok(())
}

fn load_at(v: &Volume, t: usize, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync), open: Option<&OpenFiles>) -> Result<LoadedVolume> {
    let (g, dims) = checked(v, t, max_voxels)?;
    let n = dims.iter().product::<usize>();
    let values = match &v.voxels {
        Voxels::Raw { .. } => {
            if !progress(0.1) {
                bail!("cancelled");
            }
            crate::slice::load_raw_into(v, t, max_voxels, progress)?.values
        }
        Voxels::Dicom { slices } => load_planes(&slices[(t * dims[2]).min(slices.len())..], dims, progress, true, open)?,
        Voxels::Images { slices } => load_planes(slices, dims, progress, false, None)?,
        Voxels::Unsupported(why) => bail!("pixels not readable yet: {why}"),
    };
    if values.len() != n {
        bail!("read {} values, expected {n}", values.len());
    }
    let mut g = g;
    g.dims[3] = 1;
    let mut out = LoadedVolume::from_values(&v.name, dims, values, g);
    out.file_window = v.window;
    out.invert = v.photometric.as_deref() == Some("MONOCHROME1");
    Ok(out)
}

/// What the files decoded for one load share: the plane size, the cancel flag, and the count
/// of planes done (which reports progress).
struct Planes<'a> {
    plane: usize,
    cancelled: &'a AtomicBool,
    done: &'a (dyn Fn(usize) + Sync),
}

/// Decode slices stored file by file (or frame by frame) into one array.
fn load_planes(slices: &[(PathBuf, u32)], dims: [usize; 3], progress: &(dyn Fn(f32) -> bool + Sync), dicom: bool, open: Option<&OpenFiles>) -> Result<Vec<f32>> {
    let plane = dims[0] * dims[1];
    // Consecutive slices from the same file are decoded together.
    let mut groups: Vec<(&Path, Vec<(usize, u32)>)> = Vec::new();
    for (k, (p, f)) in slices.iter().enumerate().take(dims[2]) {
        match groups.last_mut() {
            Some((q, list)) if *q == p.as_path() => list.push((k, *f)),
            _ => groups.push((p.as_path(), vec![(k, *f)])),
        }
    }
    // The groups cover the slices in order, so each decodes straight into its own part of the
    // values (slices past the end of the list stay zero).
    let mut values = vec![0.0f32; plane * dims[2]];
    let mut parts = Vec::with_capacity(groups.len());
    let mut rest = values.as_mut_slice();
    for (path, frames) in &groups {
        let (part, after) = rest.split_at_mut(frames.len() * plane);
        parts.push((*path, frames.as_slice(), part));
        rest = after;
    }
    let count = AtomicUsize::new(0);
    let cancelled = AtomicBool::new(false);
    let total = dims[2].max(1);
    let done = |n: usize| {
        let d = count.fetch_add(n, Ordering::Relaxed) + n;
        if !progress(d as f32 / total as f32) {
            cancelled.store(true, Ordering::Relaxed);
        }
    };
    let ctx = Planes { plane, cancelled: &cancelled, done: &done };
    let results: Vec<Result<()>> = parts
        .into_par_iter()
        .map(|(path, frames, out)| {
            if cancelled.load(Ordering::Relaxed) {
                bail!("cancelled");
            }
            if dicom { dicom_group(path, frames, out, &ctx, open) } else { image_group(path, frames, out, &ctx) }
        })
        .collect();
    if cancelled.load(Ordering::Relaxed) {
        bail!("cancelled");
    }
    // The first file's error, in slice order.
    results.into_iter().collect::<Result<()>>()?;
    Ok(values)
}

/// Decode `frames` (slice index, frame or page) of one file in parallel into their planes in
/// `out`; `decode` gives a frame's grey values and `tick` is called as each is done. It ends
/// as decoding the frames one after another and checking the sizes after did: at the first
/// frame that fails (with its error, or passing on its panic), else at the first plane of the
/// wrong size.
fn frames_into(frames: &[(usize, u32)], out: &mut [f32], ctx: &Planes, tick: &(dyn Fn() + Sync), decode: &(dyn Fn(u32) -> Result<Vec<f32>> + Sync)) -> Result<()> {
    frames_into_caught(frames, out, ctx, tick, decode).unwrap_or_else(|panic| std::panic::resume_unwind(panic))
}

/// [`frames_into`], giving the panic of the first frame that fails, when that frame panicked,
/// rather than passing it on. Frames after it are decoded too (in parallel), so their panics
/// are caught: they must not stand in for the error of an earlier frame.
fn frames_into_caught(frames: &[(usize, u32)], out: &mut [f32], ctx: &Planes, tick: &(dyn Fn() + Sync), decode: &(dyn Fn(u32) -> Result<Vec<f32>> + Sync)) -> std::thread::Result<Result<()>> {
    let plane = ctx.plane;
    let lens: Vec<std::thread::Result<Result<usize>>> = out
        .par_chunks_mut(plane)
        .zip(frames.par_iter())
        .map(|(dst, &(_, f))| {
            if ctx.cancelled.load(Ordering::Relaxed) {
                return Ok(Err(anyhow!("cancelled")));
            }
            let decoded = std::panic::catch_unwind(AssertUnwindSafe(|| decode(f)));
            if let Ok(Ok(p)) = &decoded {
                if p.len() == plane {
                    dst.copy_from_slice(p);
                }
                tick();
            }
            decoded.map(|r| r.map(|p| p.len()))
        })
        .collect();
    let mut wrong = None;
    for (len, &(k, _)) in lens.into_iter().zip(frames) {
        let len = match len? {
            Ok(len) => len,
            Err(e) => return Ok(Err(e)),
        };
        if len != plane && wrong.is_none() {
            wrong = Some((k, len));
        }
    }
    Ok(match wrong {
        Some((k, len)) => Err(anyhow!("slice {} has {len} pixels, expected {plane}", k + 1)),
        None => Ok(()),
    })
}

/// Colour samples to grey (ITU-R 601 luma); grey samples as they are.
fn grey(raw: Vec<f32>, n: usize) -> Vec<f32> {
    let samples = (raw.len() / n.max(1)).max(1);
    if samples >= 3 { raw.chunks(samples).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect() } else { raw }
}

/// Grey values (modality LUT applied) of frame `f` of a DICOM file, decoded on its own.
fn frame_alone(obj: &DefaultDicomObject, path: &Path, f: u32) -> Result<Vec<f32>> {
    let px = obj.decode_pixel_data_frame(f).map_err(|e| anyhow!("cannot decode {}: {e}", path.display()))?;
    let n = px.columns() as usize * px.rows() as usize;
    let raw: Vec<f32> = px.to_vec_frame(0).map_err(|e| anyhow!("cannot convert the pixels: {e}"))?;
    Ok(grey(raw, n))
}

/// How many frames of a multi-frame file to decode at a time, each on its own, when that gives
/// exactly the values of decoding the whole file (`None`: decode the whole file); `wanted`
/// frames are to be decoded from it. That needs encapsulated data with one fragment per frame,
/// and a rescale given for no frame or for every frame (given for some only, a whole decode
/// uses the first value for every frame but a frame decoded alone its own). RLE and JPEG 2000
/// frame decoders read only their fragment. The JPEG family's copies all the compressed data
/// for each frame, so it must be at most 128 MB, the copies for all the wanted frames at most
/// 16 GB (on the order of a second of CPU), and at most about 256 MB of them are made at once.
fn frames_at_once(obj: &DefaultDicomObject, wanted: usize) -> Option<usize> {
    use dicom_core::value::Value;
    use dicom_dictionary_std::tags;
    let Some(Value::PixelSequence(seq)) = obj.get(tags::PIXEL_DATA).map(|e| e.value()) else { return None };
    let frames = match obj.get(tags::NUMBER_OF_FRAMES).map(|e| e.to_int::<i32>()) {
        None => 1,
        Some(Ok(n)) if n > 0 => n as usize,
        Some(_) => return None,
    };
    if seq.fragments().len() != frames {
        return None;
    }
    let rescaled = obj.get(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE).and_then(|e| e.items()).map_or(0, |items| {
        items.iter().filter(|i| i.get(tags::PIXEL_VALUE_TRANSFORMATION_SEQUENCE).and_then(|s| s.items()).and_then(|s| s.first()).and_then(|s| s.get(tags::RESCALE_SLOPE)).is_some()).count()
    });
    if rescaled != 0 && rescaled != frames {
        return None;
    }
    match obj.meta().transfer_syntax().trim_end_matches(['\0', ' ']) {
        "1.2.840.10008.1.2.5" | "1.2.840.10008.1.2.4.90" | "1.2.840.10008.1.2.4.91" => Some(frames),
        "1.2.840.10008.1.2.4.50" | "1.2.840.10008.1.2.4.51" | "1.2.840.10008.1.2.4.57" | "1.2.840.10008.1.2.4.70" => {
            let size = seq.fragments().iter().map(Vec::len).sum::<usize>().max(1);
            let copies = (size as u64).saturating_mul(wanted as u64);
            (size <= 128 << 20 && copies <= 16 << 30).then(|| ((256 << 20) / size).max(1))
        }
        _ => None,
    }
}

/// Decode the frames `frames` of one DICOM file into `out` (modality LUT applied, colour
/// turned to grey). JPEG-LS frames and frames that decode alone (see [`frames_at_once`]) are
/// decoded in parallel; other multi-frame data are decoded whole once, then converted frame by
/// frame in parallel.
fn dicom_group(path: &Path, frames: &[(usize, u32)], out: &mut [f32], ctx: &Planes, open: Option<&OpenFiles>) -> Result<()> {
    // A file read again for a later volume stays open between the reads; `wanted` is how
    // many frames are decoded from it in all.
    let (kept, wanted) = match open.and_then(|o| Some((o, o.frames_read(path)?))) {
        Some((o, all)) => (Some(o.get(path)?), all),
        None => (None, frames.len()),
    };
    let opened;
    let file = match &kept {
        Some(f) => f.as_ref(),
        None => {
            opened = DicomFile::open(path)?;
            &opened
        }
    };
    let obj = &file.obj;
    let tick = || (ctx.done)(1);
    if crate::dicom::is_jpegls(obj.meta().transfer_syntax()) {
        let jls = crate::dicom::JlsFrames::new(obj)?;
        return frames_into(frames, out, ctx, &tick, &|f| Ok(jls.frame(f)?.2));
    }
    if frames.len() == 1 {
        return frames_into(frames, out, ctx, &tick, &|f| frame_alone(obj, path, f));
    }
    // Planes already counted by a frame-by-frame attempt that then failed.
    let mut counted = None;
    if let Some(at_once) = frames_at_once(obj, wanted) {
        let reported = AtomicUsize::new(0);
        let tick = || {
            reported.fetch_add(1, Ordering::Relaxed);
            (ctx.done)(1);
        };
        let alone = frames.chunks(at_once).zip(out.chunks_mut(at_once * ctx.plane)).try_for_each(|(some, part)| {
            // A frame whose decoder panicked fails the attempt too: the whole file is then
            // decoded, which ends as it did before (with that panic or an earlier error).
            frames_into_caught(some, part, ctx, &tick, &|f| frame_alone(obj, path, f)).unwrap_or_else(|_| Err(anyhow!("a frame could not be decoded on its own")))
        });
        match alone {
            Ok(()) => return Ok(()),
            // Cancelling is no reason to try the whole file.
            Err(e) if ctx.cancelled.load(Ordering::Relaxed) => return Err(e),
            // Otherwise decode the whole file, as before, for its values or its error.
            Err(_) => counted = Some(reported.load(Ordering::Relaxed)),
        }
    }
    let whole;
    let local;
    let px: &DecodedPixelData = match &kept {
        Some(f) => {
            whole = f.whole(path)?;
            &whole
        }
        None => {
            local = obj.decode_pixel_data().map_err(|e| anyhow!("cannot decode {}: {e}", path.display()))?;
            &local
        }
    };
    let n = px.columns() as usize * px.rows() as usize;
    let convert = |f: u32| {
        let raw: Vec<f32> = px.to_vec_frame(f).map_err(|e| anyhow!("cannot convert frame {f}: {e}"))?;
        Ok(grey(raw, n))
    };
    match counted {
        None => frames_into(frames, out, ctx, &tick, &convert),
        Some(c) => {
            frames_into(frames, out, ctx, &|| {}, &convert)?;
            (ctx.done)(frames.len() - c);
            Ok(())
        }
    }
}

/// Decode the pages `frames` of one image file into `out`. A run of pages of a multi-page
/// TIFF is shared among threads in stretches, each read with one decoder.
fn image_group(path: &Path, frames: &[(usize, u32)], out: &mut [f32], ctx: &Planes) -> Result<()> {
    let run = frames.windows(2).all(|w| w[1].1 == w[0].1 + 1);
    if frames.len() > 1 && run && crate::stack::is_tiff(path) {
        return tiff_pages_into(path, frames, out, ctx);
    }
    frames_into(frames, out, ctx, &|| (ctx.done)(1), &|page| Ok(crate::stack::read_plane(path, page)?.2))
}

/// A page of the wrong size, if any: its slice index and pixel count.
type WrongSize = Option<(usize, usize)>;

/// Consecutive pages of a multi-page TIFF into `out`, about two stretches per thread. Errors
/// are as [`frames_into`]'s: the first page that fails, else the first of the wrong size.
fn tiff_pages_into(path: &Path, frames: &[(usize, u32)], out: &mut [f32], ctx: &Planes) -> Result<()> {
    let plane = ctx.plane;
    let per = frames.len().div_ceil(2 * rayon::current_num_threads()).max(1);
    // Each stretch: its first page that fails, else its first page of the wrong size. A
    // stretch's panic is caught so that it passes on only when no earlier stretch failed.
    let stretches: Vec<std::thread::Result<Result<WrongSize>>> = out
        .par_chunks_mut(per * plane)
        .zip(frames.par_chunks(per))
        .map(|(dst, run)| {
            std::panic::catch_unwind(AssertUnwindSafe(|| {
                let mut wrong = None;
                crate::stack::read_pages(path, run[0].1, run.len(), &mut |i, values| {
                    if ctx.cancelled.load(Ordering::Relaxed) {
                        bail!("cancelled");
                    }
                    if values.len() == plane {
                        dst[i * plane..(i + 1) * plane].copy_from_slice(&values);
                    } else if wrong.is_none() {
                        wrong = Some((run[i].0, values.len()));
                    }
                    (ctx.done)(1);
                    Ok(())
                })?;
                Ok(wrong)
            }))
        })
        .collect();
    let mut wrong = None;
    for stretch in stretches {
        wrong = wrong.or(stretch.unwrap_or_else(|panic| std::panic::resume_unwind(panic))?);
    }
    match wrong {
        Some((k, len)) => bail!("slice {} has {len} pixels, expected {plane}", k + 1),
        None => Ok(()),
    }
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// A DICOM file opened for decoding.
struct DicomFile {
    obj: DefaultDicomObject,
    /// Its whole pixel data, decoded once while the file is kept open.
    whole: Mutex<Option<Arc<DecodedPixelData<'static>>>>,
}

impl DicomFile {
    fn open(path: &Path) -> Result<Self> {
        let obj = dicom_object::open_file(path).with_context(|| format!("cannot read {}", path.display()))?;
        Ok(Self { obj, whole: Mutex::new(None) })
    }

    /// The whole pixel data decoded (once).
    fn whole(&self, path: &Path) -> Result<Arc<DecodedPixelData<'static>>> {
        let mut whole = lock(&self.whole);
        if let Some(px) = &*whole {
            return Ok(px.clone());
        }
        let px = Arc::new(self.obj.decode_pixel_data().map_err(|e| anyhow!("cannot decode {}: {e}", path.display()))?.to_owned());
        *whole = Some(px.clone());
        Ok(px)
    }
}

/// A file of [`OpenFiles`], opened when first needed.
type OpenFile = Arc<Mutex<Option<Arc<DicomFile>>>>;

/// How [`OpenFiles`] reads a file it keeps open.
struct Reads {
    /// The last volume that reads it (its place in the list of volumes).
    last: usize,
    /// How many frames are read from it in all.
    frames: usize,
}

/// The DICOM files kept open while several volumes are loaded one after another: those read
/// more than once (by several of the volumes, or in several runs of slices of one), each from
/// its first read to the end of the last volume that reads it. A file read once is opened and
/// let go by that read alone, as by [`load_volume_at`].
struct OpenFiles {
    reads: HashMap<PathBuf, Reads>,
    files: Mutex<HashMap<PathBuf, OpenFile>>,
}

impl OpenFiles {
    /// For loading volumes `ts` of `v`, in that order.
    fn new(v: &Volume, ts: &[usize]) -> Self {
        // Each file's runs of slices, with its last volume and frame count.
        let mut runs: HashMap<&Path, (usize, Reads)> = HashMap::new();
        if let (Voxels::Dicom { slices }, Some(g)) = (&v.voxels, &v.geometry) {
            let z = g.dims[2];
            for (i, &t) in ts.iter().enumerate() {
                // The slices `load_at` reads for volume `t`, in the runs `load_planes` makes.
                let mut previous = None;
                for (p, _) in slices[t.saturating_mul(z).min(slices.len())..].iter().take(z) {
                    let (count, reads) = runs.entry(p).or_insert((0, Reads { last: i, frames: 0 }));
                    if previous != Some(p) {
                        *count += 1;
                    }
                    reads.last = i;
                    reads.frames += 1;
                    previous = Some(p);
                }
            }
        }
        let reads = runs.into_iter().filter(|(_, (count, _))| *count > 1).map(|(p, (_, reads))| (p.to_path_buf(), reads)).collect();
        Self { reads, files: Mutex::default() }
    }

    /// How many frames are read from `path` in all, if it is a file kept open.
    fn frames_read(&self, path: &Path) -> Option<usize> {
        self.reads.get(path).map(|r| r.frames)
    }

    /// Close the files that no volume after the `i`th reads.
    fn done_with(&self, i: usize) {
        lock(&self.files).retain(|p, _| self.reads.get(p).is_some_and(|r| r.last > i));
    }

    /// A file kept open, opened when first needed.
    fn get(&self, path: &Path) -> Result<Arc<DicomFile>> {
        let cell = lock(&self.files).entry(path.to_path_buf()).or_default().clone();
        let mut file = lock(&cell);
        if let Some(f) = &*file {
            return Ok(f.clone());
        }
        let f = Arc::new(DicomFile::open(path)?);
        *file = Some(f.clone());
        Ok(f)
    }
}

/// What a load in [`shared`] is of: the volume as described, which of its 3D volumes, and the
/// size, modification time and inode of every file it reads (a file changed on disk is read
/// again).
#[derive(Clone, PartialEq)]
struct Key {
    voxels: Voxels,
    geometry: Option<Geometry>,
    name: String,
    window: Option<(f64, f64)>,
    photometric: Option<String>,
    t: usize,
    stamps: Vec<(u64, i128, u64)>,
}

impl Key {
    fn new(v: &Volume, t: usize) -> Self {
        let mut paths: Vec<&Path> = match &v.voxels {
            Voxels::Raw { file, .. } => vec![file.as_path()],
            Voxels::Dicom { slices } | Voxels::Images { slices } => slices.iter().map(|(p, _)| p.as_path()).collect(),
            Voxels::Unsupported(_) => Vec::new(),
        };
        paths.dedup();
        let stamps = paths.iter().map(|p| std::fs::metadata(p).map(|m| stamp(&m)).unwrap_or((0, 0, 0))).collect();
        Self { voxels: v.voxels.clone(), geometry: v.geometry.clone(), name: v.name.clone(), window: v.window, photometric: v.photometric.clone(), t, stamps }
    }
}

fn stamp(m: &std::fs::Metadata) -> (u64, i128, u64) {
    let modified = m.modified().ok().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map_or(0, |d| d.as_nanos() as i128);
    #[cfg(unix)]
    let inode = std::os::unix::fs::MetadataExt::ino(m);
    #[cfg(not(unix))]
    let inode = 0;
    (m.len(), modified, inode)
}

/// How a load that others wait for ended.
#[derive(Clone)]
enum Outcome {
    Loaded(Arc<LoadedVolume>),
    Failed(String),
    Cancelled,
}

/// A load in progress: those waiting for it sleep on `ended` and show `fraction`.
struct Slot {
    outcome: Mutex<Option<Outcome>>,
    ended: Condvar,
    /// Progress so far (f32 bits).
    fraction: AtomicU32,
}

enum Entry {
    Ready(Weak<LoadedVolume>),
    Loading(Arc<Slot>),
}

/// Loads through [`shared`]. Volumes are held weakly: one is freed when its last user lets go.
static SHARED: Mutex<Vec<(Key, Entry)>> = Mutex::new(Vec::new());

/// [`load_volume_at`], shared between callers. While anyone still holds a volume, a call for
/// the same volume and time point, with its files unchanged, returns that volume again; a call
/// made while it is being loaded waits for that load instead of starting another (reporting
/// its progress, and still able to stop waiting through its own `progress`). When the caller
/// that started a load cancels it, one of those waiting starts it again. The checks of
/// [`load_volume_at`] run first, so their errors come at once.
///
/// It can block until another caller's load of the same volume ends, so call it from a job
/// thread (`Job::spawn`, `in_job_pool`), never from the UI thread. The volume stays shared only
/// while some caller holds the `Arc` it returns.
pub fn shared(v: &Volume, t: usize, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<Arc<LoadedVolume>> {
    checked(v, t, max_voxels)?;
    let key = Key::new(v, t);
    loop {
        let (slot, mine) = {
            let mut cache = lock(&SHARED);
            let mut waiting = None;
            if let Some(i) = cache.iter().position(|(k, _)| *k == key) {
                match &cache[i].1 {
                    Entry::Ready(held) => match held.upgrade() {
                        Some(vol) => return Ok(vol),
                        None => {
                            cache.swap_remove(i);
                        }
                    },
                    Entry::Loading(slot) => waiting = Some(slot.clone()),
                }
            }
            match waiting {
                Some(slot) => (slot, false),
                None => {
                    let slot = Arc::new(Slot { outcome: Mutex::new(None), ended: Condvar::new(), fraction: AtomicU32::new(0) });
                    cache.push((key.clone(), Entry::Loading(slot.clone())));
                    (slot, true)
                }
            }
        };
        if mine {
            return load_shared(v, t, max_voxels, progress, &slot);
        }
        if let Some(vol) = wait(&slot, progress)? {
            return Ok(vol);
        }
        // That load was cancelled: try again (perhaps loading it here).
    }
}

/// Wait for another call's load: its volume, its error, or `None` when it was cancelled.
fn wait(slot: &Slot, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<Option<Arc<LoadedVolume>>> {
    loop {
        {
            let mut outcome = lock(&slot.outcome);
            if outcome.is_none() {
                outcome = match slot.ended.wait_timeout(outcome, Duration::from_millis(50)) {
                    Ok((g, _)) => g,
                    Err(e) => e.into_inner().0,
                };
            }
            match &*outcome {
                Some(Outcome::Loaded(vol)) => return Ok(Some(vol.clone())),
                Some(Outcome::Failed(why)) => bail!("{why}"),
                Some(Outcome::Cancelled) => return Ok(None),
                None => {}
            }
        }
        // Still loading. This caller may stop waiting; the load goes on for the others.
        if !progress(f32::from_bits(slot.fraction.load(Ordering::Relaxed))) {
            bail!("cancelled");
        }
    }
}

/// Load for [`shared`], then tell those waiting how it ended (also when the load panics).
fn load_shared(v: &Volume, t: usize, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync), slot: &Arc<Slot>) -> Result<Arc<LoadedVolume>> {
    /// Ends the load as failed unless it ended normally, so a panic leaves no one waiting.
    struct Unwinding<'a>(&'a Arc<Slot>, bool);
    impl Drop for Unwinding<'_> {
        fn drop(&mut self) {
            if self.1 {
                finish(self.0, Outcome::Failed("the load stopped unexpectedly".into()));
            }
        }
    }
    let mut unwinding = Unwinding(slot, true);
    let cancelled = AtomicBool::new(false);
    let report = |f: f32| {
        slot.fraction.store(f.to_bits(), Ordering::Relaxed);
        let go = progress(f);
        if !go {
            cancelled.store(true, Ordering::Relaxed);
        }
        go
    };
    let loaded = load_volume_at(v, t, max_voxels, &report);
    unwinding.1 = false;
    match loaded {
        Ok(vol) => {
            let vol = Arc::new(vol);
            finish(slot, Outcome::Loaded(vol.clone()));
            Ok(vol)
        }
        Err(e) => {
            finish(slot, if cancelled.load(Ordering::Relaxed) { Outcome::Cancelled } else { Outcome::Failed(format!("{e:#}")) });
            Err(e)
        }
    }
}

/// Record how a load ended: in the cache first (so a caller trying again after a cancel starts
/// a new load), then for those waiting.
fn finish(slot: &Arc<Slot>, outcome: Outcome) {
    {
        let mut cache = lock(&SHARED);
        if let Some(i) = cache.iter().position(|(_, e)| matches!(e, Entry::Loading(s) if Arc::ptr_eq(s, slot))) {
            match &outcome {
                Outcome::Loaded(vol) => cache[i].1 = Entry::Ready(Arc::downgrade(vol)),
                _ => {
                    cache.swap_remove(i);
                }
            }
        }
        // Forget volumes no one holds any more (the new one is held, so it stays).
        cache.retain(|(_, e)| match e {
            Entry::Ready(held) => held.strong_count() > 0,
            Entry::Loading(_) => true,
        });
    }
    *lock(&slot.outcome) = Some(outcome);
    slot.ended.notify_all();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DataType, Format, Kind};
    use glam::DVec3;

    #[test]
    fn raw_and_stack_volumes_load_whole() {
        let dir = std::env::temp_dir().join(format!("mmv-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("v.raw");
        std::fs::write(&p, (0..24i16).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let g = Geometry::from_axes([4, 3, 2, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let mut v = Volume {
            format: Format::Nifti1,
            kind: Kind::Image,
            name: "v".into(),
            files: vec![p.clone()],
            bytes: 48,
            geometry: Some(g.clone()),
            data_type: Some(DataType::I16),
            fields: vec![],
            warnings: vec![],
            voxels: Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 },
            window: None,
            photometric: None,
            patient: None,
        };
        let l = load_volume(&v, 1000, &|_| true).unwrap();
        assert_eq!(l.get(3, 2, 1), 23.0);
        assert_eq!(l.range, (0.0, 23.0));
        assert!(load_volume(&v, 10, &|_| true).is_err());
        // Two PNG slices.
        let mut slices = Vec::new();
        for k in 0..2u8 {
            let f = dir.join(format!("s{k}.png"));
            image::GrayImage::from_fn(4, 3, |x, y| image::Luma([k * 12 + (y * 4 + x) as u8])).save(&f).unwrap();
            slices.push((f, 0));
        }
        v.voxels = Voxels::Images { slices };
        let l = load_volume(&v, 1000, &|_| true).unwrap();
        assert_eq!(l.values, (0..24).map(|x| x as f32).collect::<Vec<_>>());
        assert!(load_volume(&v, 1000, &|_| false).is_err());
        // A 4D file: two volumes one after the other, and two interleaved components.
        let p4 = dir.join("v4.raw");
        std::fs::write(&p4, (0..48i16).flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let g4 = Geometry::from_axes([4, 3, 2, 2], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        v.geometry = Some(g4);
        v.voxels = Voxels::Raw { file: p4, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 };
        assert_eq!(volume_count(&v), 2);
        assert_eq!(load_volume_at(&v, 1, 1000, &|_| true).unwrap().values[0], 24.0);
        assert!(load_volume_at(&v, 2, 1000, &|_| true).is_err());
        if let Voxels::Raw { interleave, .. } = &mut v.voxels {
            *interleave = 2;
        }
        let c1 = load_volume_at(&v, 1, 1000, &|_| true).unwrap();
        assert_eq!((c1.values[0], c1.values[23]), (1.0, 47.0));
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn temp_dir(what: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mmv-{what}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn volume(name: &str, dims: [usize; 4], voxels: Voxels) -> Volume {
        Volume {
            format: Format::Dicom,
            kind: Kind::Image,
            name: name.into(),
            files: vec![],
            bytes: 0,
            geometry: Some(Geometry::from_axes(dims, [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO)),
            data_type: Some(DataType::U16),
            fields: vec![],
            warnings: vec![],
            voxels,
            window: None,
            photometric: None,
            patient: None,
        }
    }

    /// A 12-bit 40 × 30 frame, different for each `k`.
    fn frame(k: usize) -> Vec<u16> {
        (0..40 * 30).map(|i| ((i * 37 + k * 101) % 4096) as u16).collect()
    }

    /// DICOM RLE of one 16-bit frame: two segments (high and low bytes), literal runs only.
    fn rle(px: &[u16]) -> Vec<u8> {
        let mut out = vec![0u8; 64];
        out[0..4].copy_from_slice(&2u32.to_le_bytes());
        for (s, shift) in [8u32, 0].into_iter().enumerate() {
            let at = out.len() as u32;
            out[4 + 4 * s..8 + 4 * s].copy_from_slice(&at.to_le_bytes());
            let bytes: Vec<u8> = px.iter().map(|v| (v >> shift) as u8).collect();
            for run in bytes.chunks(128) {
                out.push(run.len() as u8 - 1);
                out.extend_from_slice(run);
            }
        }
        if out.len() % 2 == 1 {
            out.push(0);
        }
        out
    }

    /// How a multi-frame test file carries rescale.
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Rescale {
        Absent,
        TopLevel,
        EveryFrame,
        SomeFrames,
    }

    /// An Enhanced multi-frame file (40 × 30, 12-bit) in transfer syntax `ts`: native frames
    /// when `fragments` is empty, else those fragments.
    fn write_multiframe(path: &Path, ts: &str, frames: usize, fragments: Vec<Vec<u8>>, rescale: Rescale) {
        use dicom_core::value::{DataSetSequence, PixelFragmentSequence, Value};
        use dicom_core::{DataElement, PrimitiveValue, VR};
        use dicom_dictionary_std::tags;
        use dicom_object::{FileMetaTableBuilder, InMemDicomObject};
        let class = "1.2.840.10008.5.1.4.1.1.4.1";
        let mut o = InMemDicomObject::new_empty();
        let s = |o: &mut InMemDicomObject, tag, vr, v: &str| o.put(DataElement::new(tag, vr, PrimitiveValue::from(v)));
        s(&mut o, tags::SOP_CLASS_UID, VR::UI, class);
        s(&mut o, tags::SOP_INSTANCE_UID, VR::UI, "1.2.3.77");
        s(&mut o, tags::PHOTOMETRIC_INTERPRETATION, VR::CS, "MONOCHROME2");
        s(&mut o, tags::NUMBER_OF_FRAMES, VR::IS, &frames.to_string());
        if rescale == Rescale::TopLevel {
            s(&mut o, tags::RESCALE_SLOPE, VR::DS, "2");
            s(&mut o, tags::RESCALE_INTERCEPT, VR::DS, "-7");
        }
        for (tag, v) in [(tags::ROWS, 30u16), (tags::COLUMNS, 40), (tags::BITS_ALLOCATED, 16), (tags::BITS_STORED, 12), (tags::HIGH_BIT, 11), (tags::PIXEL_REPRESENTATION, 0), (tags::SAMPLES_PER_PIXEL, 1)] {
            o.put(DataElement::new(tag, VR::US, PrimitiveValue::from(v)));
        }
        let per: Vec<InMemDicomObject> = (0..frames)
            .map(|i| {
                let mut f = InMemDicomObject::new_empty();
                let given = rescale == Rescale::EveryFrame || (rescale == Rescale::SomeFrames && i < frames / 2);
                if given {
                    let mut pv = InMemDicomObject::new_empty();
                    s(&mut pv, tags::RESCALE_SLOPE, VR::DS, &format!("{}", 1.0 + (i % 3) as f64 * 0.5));
                    s(&mut pv, tags::RESCALE_INTERCEPT, VR::DS, &format!("{}", -(i as f64) * 3.0));
                    f.put(DataElement::new(tags::PIXEL_VALUE_TRANSFORMATION_SEQUENCE, VR::SQ, DataSetSequence::from(vec![pv])));
                }
                f
            })
            .collect();
        o.put(DataElement::new(tags::PER_FRAME_FUNCTIONAL_GROUPS_SEQUENCE, VR::SQ, DataSetSequence::from(per)));
        if fragments.is_empty() {
            let all: Vec<u16> = (0..frames).flat_map(frame).collect();
            o.put(DataElement::new(tags::PIXEL_DATA, VR::OW, PrimitiveValue::U16(all.into())));
        } else {
            o.put(DataElement::new(tags::PIXEL_DATA, VR::OB, Value::PixelSequence(PixelFragmentSequence::new(vec![], fragments))));
        }
        let file = o.with_meta(FileMetaTableBuilder::new().transfer_syntax(ts).media_storage_sop_class_uid(class).media_storage_sop_instance_uid("1.2.3.77")).unwrap();
        file.write_to_file(path).unwrap();
    }

    /// Frame `f` as decoding the whole file gave it before.
    fn decoded_whole(path: &Path, f: u32) -> Vec<f32> {
        let obj = dicom_object::open_file(path).unwrap();
        if crate::dicom::is_jpegls(obj.meta().transfer_syntax()) {
            return crate::dicom::jpegls_frame(&obj, f).unwrap().2;
        }
        obj.decode_pixel_data().unwrap().to_vec_frame(f).unwrap()
    }

    #[test]
    fn multiframe_files_load_as_when_decoded_whole() {
        let dir = temp_dir("multiframe");
        let data = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
        let jpeg = std::fs::read(data.join("jpeg/grey12_lossless_sv1.jpg")).unwrap();
        let j2k = std::fs::read(data.join("j2k/grey12.j2k")).unwrap();
        let jls = std::fs::read(data.join("jpegls/grey12.jls")).unwrap();
        let even = |mut b: Vec<u8>| {
            if b.len() % 2 == 1 {
                b.push(0);
            }
            b
        };
        let frames = 8;
        type Fragment<'a> = &'a dyn Fn(usize) -> Vec<u8>;
        let syntaxes: [(&str, Fragment, bool); 5] = [
            ("1.2.840.10008.1.2.1", &|_| Vec::new(), false),
            ("1.2.840.10008.1.2.5", &|k| rle(&frame(k)), true),
            ("1.2.840.10008.1.2.4.70", &|_| even(jpeg.clone()), true),
            ("1.2.840.10008.1.2.4.90", &|_| even(j2k.clone()), true),
            ("1.2.840.10008.1.2.4.80", &|_| even(jls.clone()), false),
        ];
        for (ts, fragment, frame_by_frame) in syntaxes {
            for rescale in [Rescale::Absent, Rescale::TopLevel, Rescale::EveryFrame, Rescale::SomeFrames] {
                let p = dir.join(format!("{ts}-{rescale:?}.dcm"));
                let fragments: Vec<Vec<u8>> = if ts == "1.2.840.10008.1.2.1" { Vec::new() } else { (0..frames).map(fragment).collect() };
                write_multiframe(&p, ts, frames, fragments, rescale);
                let obj = dicom_object::open_file(&p).unwrap();
                assert_eq!(frames_at_once(&obj, frames).is_some(), frame_by_frame && rescale != Rescale::SomeFrames, "{ts} {rescale:?}");
                // So many frames wanted that the JPEG family's copies pass their budget.
                let jpeg = ts == "1.2.840.10008.1.2.4.70";
                assert_eq!(frames_at_once(&obj, 1 << 40).is_some(), frame_by_frame && rescale != Rescale::SomeFrames && !jpeg, "{ts} {rescale:?}");
                let expected: Vec<Vec<f32>> = (0..frames as u32).map(|f| decoded_whole(&p, f)).collect();
                // As one volume of 8 slices, and as 2 time points of 4.
                for (z, t) in [(8, 1), (4, 2)] {
                    let slices = (0..frames as u32).map(|f| (p.clone(), f)).collect();
                    let v = volume("mf", [40, 30, z, t], Voxels::Dicom { slices });
                    let mut all = Vec::new();
                    load_volumes_at(&v, &(0..t).collect::<Vec<_>>(), 1 << 20, &|_| true, &mut |i, vol| {
                        all.push((i, vol.values));
                        Ok(())
                    })
                    .unwrap();
                    for i in 0..t {
                        let got = load_volume_at(&v, i, 1 << 20, &|_| true).unwrap().values;
                        let want: Vec<f32> = expected[i * z..(i + 1) * z].concat();
                        let same = |a: &[f32]| a.len() == want.len() && a.iter().zip(&want).all(|(x, y)| x.to_bits() == y.to_bits());
                        assert!(same(&got), "{ts} {rescale:?} t {i}");
                        assert!(all[i].0 == i && same(&all[i].1), "{ts} {rescale:?} t {i}, several at once");
                    }
                }
                // Thumbnails from the file kept open are those from opening it each time.
                let slices = (0..frames as u32).map(|f| (p.clone(), f)).collect();
                let v = volume("mf", [40, 30, 8, 1], Voxels::Dicom { slices });
                for k in [0, 5] {
                    assert_eq!(crate::slice::dicom_slice_from(&obj, &v, k).unwrap(), crate::slice::dicom_slice(&v, k).unwrap());
                }
                // Planes of the wrong size give the error they gave before.
                let slices = (0..frames as u32).map(|f| (p.clone(), f)).collect();
                let wrong = volume("mf", [20, 30, 4, 2], Voxels::Dicom { slices });
                let e = load_volume_at(&wrong, 1, 1 << 20, &|_| true).unwrap_err();
                assert_eq!(e.to_string(), "slice 1 has 1200 pixels, expected 600", "{ts}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// Load volumes `ts` of `v` as [`load_volumes_at`] does, checking each against
    /// [`load_volume_at`]; gives the files kept open after each volume, and checks that every
    /// file kept open is freed by the end.
    fn open_after_each(v: &Volume, ts: &[usize]) -> Vec<Vec<PathBuf>> {
        let open = OpenFiles::new(v, ts);
        let mut after = Vec::new();
        let mut freed: Vec<Weak<DicomFile>> = Vec::new();
        let mut check = |t: usize, vol: LoadedVolume| {
            let want = load_volume_at(v, t, 1 << 20, &|_| true).unwrap().values;
            assert!(vol.values.len() == want.len() && vol.values.iter().zip(&want).all(|(a, b)| a.to_bits() == b.to_bits()), "t {t}");
            let files = lock(&open.files);
            for cell in files.values() {
                // Held by the open files alone.
                assert_eq!(Arc::strong_count(cell), 1);
                let file = lock(cell);
                let file = file.as_ref().expect("opened");
                assert_eq!(Arc::strong_count(file), 1);
                freed.push(Arc::downgrade(file));
            }
            let mut paths: Vec<PathBuf> = files.keys().cloned().collect();
            paths.sort();
            after.push(paths);
            Ok(())
        };
        load_volumes_with(v, ts, 1 << 20, &|_| true, &mut check, &open).unwrap();
        assert!(freed.iter().all(|f| f.strong_count() == 0));
        after
    }

    #[test]
    fn several_volumes_keep_open_only_the_files_read_again() {
        let dir = temp_dir("several");
        for ts in ["1.2.840.10008.1.2.1", "1.2.840.10008.1.2.5"] {
            let native = ts == "1.2.840.10008.1.2.1";
            let fragments = |first: usize, n: usize| if native { Vec::new() } else { (first..first + n).map(|k| rle(&frame(k))).collect() };
            // One file of 4 frames per time point, each different.
            let per_t: Vec<PathBuf> = (0..3)
                .map(|t| {
                    let p = dir.join(format!("{ts}-t{t}.dcm"));
                    write_multiframe(&p, ts, 4, fragments(t * 4, 4), [Rescale::Absent, Rescale::TopLevel, Rescale::EveryFrame][t]);
                    p
                })
                .collect();
            let slices = per_t.iter().flat_map(|p| (0..4).map(|f| (p.clone(), f))).collect();
            let v = volume("per-t", [40, 30, 4, 3], Voxels::Dicom { slices });
            assert!(OpenFiles::new(&v, &[0, 1, 2]).reads.is_empty(), "{ts}");
            assert_eq!(open_after_each(&v, &[0, 1, 2]), vec![Vec::<PathBuf>::new(); 3], "{ts}");
            // Asked for twice, the first file stays open until its second load.
            let none = Vec::new();
            assert_eq!(open_after_each(&v, &[0, 1, 0, 2]), vec![vec![per_t[0].clone()], vec![per_t[0].clone()], none.clone(), none.clone()], "{ts}");
            // All the time points in one file, 4 frames or 1 each: open until the last.
            let all = dir.join(format!("{ts}-all.dcm"));
            write_multiframe(&all, ts, 12, fragments(0, 12), Rescale::EveryFrame);
            for z in [4, 1] {
                let slices = (0..3 * z as u32).map(|f| (all.clone(), f)).collect();
                let v = volume("all", [40, 30, z, 3], Voxels::Dicom { slices });
                assert_eq!(open_after_each(&v, &[0, 1, 2]), vec![vec![all.clone()], vec![all.clone()], none.clone()], "{ts} z {z}");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn frames_decoded_in_parallel_end_as_when_decoded_in_turn() {
        let cancelled = AtomicBool::new(false);
        let ctx = Planes { plane: 2, cancelled: &cancelled, done: &|_| {} };
        let frames: Vec<(usize, u32)> = (0..64).map(|k| (k, k as u32)).collect();
        // Frames listed fail: 'e' with an error, 'p' with a panic, 'w' with a plane of the wrong size.
        let decode = |bad: &[(u32, char)], f: u32| -> Result<Vec<f32>> {
            match bad.iter().find(|b| b.0 == f).map(|b| b.1) {
                Some('e') => bail!("frame {f} is corrupt"),
                Some('p') => panic!("frame {f} panicked"),
                Some(_) => Ok(vec![0.0; 3]),
                None => Ok(vec![f as f32; 2]),
            }
        };
        let run = |bad: &[(u32, char)]| -> std::result::Result<String, String> {
            let mut out = vec![0.0; 128];
            let ended = std::panic::catch_unwind(AssertUnwindSafe(|| frames_into(&frames, &mut out, &ctx, &|| {}, &|f| decode(bad, f))));
            match ended {
                Ok(Ok(())) => {
                    assert_eq!(out, (0..128).map(|i| (i / 2) as f32).collect::<Vec<f32>>());
                    Ok("ok".into())
                }
                Ok(Err(e)) => Ok(e.to_string()),
                Err(panic) => Err(panic.downcast_ref::<String>().cloned().unwrap_or_default()),
            }
        };
        assert_eq!(run(&[]), Ok("ok".into()));
        assert_eq!(run(&[(10, 'e'), (40, 'p')]), Ok("frame 10 is corrupt".into()));
        assert_eq!(run(&[(10, 'p'), (40, 'e')]), Err("frame 10 panicked".into()));
        assert_eq!(run(&[(10, 'p'), (11, 'p'), (40, 'p')]), Err("frame 10 panicked".into()));
        assert_eq!(run(&[(5, 'w'), (50, 'e')]), Ok("frame 50 is corrupt".into()));
        assert_eq!(run(&[(5, 'w'), (50, 'w')]), Ok("slice 6 has 3 pixels, expected 2".into()));
        // The frame-by-frame attempt gets the panic back instead.
        let mut out = vec![0.0; 128];
        let caught = frames_into_caught(&frames, &mut out, &ctx, &|| {}, &|f| decode(&[(20, 'p'), (30, 'e')], f));
        assert_eq!(caught.unwrap_err().downcast_ref::<String>().map(String::as_str), Some("frame 20 panicked"));
    }

    #[test]
    fn multipage_tiffs_load_in_stretches() {
        let dir = temp_dir("tiffpages");
        // 45 pages of 16-bit grey and 23 of RGB.
        let grey = dir.join("grey.tif");
        {
            let mut e = tiff::encoder::TiffEncoder::new(std::fs::File::create(&grey).unwrap()).unwrap();
            for k in 0..45u16 {
                e.write_image::<tiff::encoder::colortype::Gray16>(8, 6, &(0..48).map(|i| k * 100 + i).collect::<Vec<u16>>()).unwrap();
            }
        }
        let rgb = dir.join("rgb.tif");
        {
            let mut e = tiff::encoder::TiffEncoder::new(std::fs::File::create(&rgb).unwrap()).unwrap();
            for k in 0..23u8 {
                e.write_image::<tiff::encoder::colortype::RGB8>(8, 6, &(0..144).map(|i| k.wrapping_mul(7).wrapping_add(i as u8)).collect::<Vec<u8>>()).unwrap();
            }
        }
        for (p, pages) in [(&grey, 45u32), (&rgb, 23)] {
            let slices = (0..pages).map(|k| (p.clone(), k)).collect();
            let v = volume("tiff", [8, 6, pages as usize, 1], Voxels::Images { slices });
            let got = load_volume(&v, 1 << 20, &|_| true).unwrap().values;
            let want: Vec<f32> = (0..pages).flat_map(|k| crate::stack::read_plane(p, k).unwrap().2).collect();
            assert_eq!(got, want);
            assert!(load_volume(&v, 1 << 20, &|f| f < 0.3).is_err(), "cancelled");
        }
        // A page of another size: the same error as when every page was read first.
        let odd = dir.join("odd.tif");
        {
            let mut e = tiff::encoder::TiffEncoder::new(std::fs::File::create(&odd).unwrap()).unwrap();
            for k in 0..20u16 {
                let w = if k == 13 { 9 } else { 8 };
                e.write_image::<tiff::encoder::colortype::Gray16>(w, 6, &vec![k; w as usize * 6]).unwrap();
            }
        }
        let slices = (0..20).map(|k| (odd.clone(), k)).collect();
        let v = volume("odd", [8, 6, 20, 1], Voxels::Images { slices });
        assert_eq!(load_volume(&v, 1 << 20, &|_| true).unwrap_err().to_string(), "slice 14 has 54 pixels, expected 48");
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A 64³ int16 raw volume in its own file.
    fn raw_volume(dir: &Path, name: &str, seed: i16) -> Volume {
        let p = dir.join(format!("{name}.raw"));
        std::fs::write(&p, (0..64 * 64 * 64).flat_map(|v: i32| (v as i16).wrapping_mul(seed).to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        let mut v = volume(name, [64, 64, 64, 1], Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 });
        v.format = Format::Nifti1;
        v
    }

    fn entries_for(name: &str) -> usize {
        lock(&SHARED).iter().filter(|(k, _)| k.name == name).count()
    }

    #[test]
    fn shared_loads_are_decoded_once() {
        let dir = temp_dir("shared-once");
        let v = raw_volume(&dir, "shared-once", 3);
        let direct = load_volume(&v, 1 << 20, &|_| true).unwrap();
        // Six callers at once, each a little slow to report progress: one load among them.
        let got: Vec<Arc<LoadedVolume>> = std::thread::scope(|s| {
            let handles: Vec<_> = (0..6)
                .map(|_| {
                    s.spawn(|| {
                        shared(&v, 0, 1 << 20, &|_| {
                            std::thread::sleep(Duration::from_millis(2));
                            true
                        })
                        .unwrap()
                    })
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        assert!(got.iter().all(|g| Arc::ptr_eq(g, &got[0])));
        assert_eq!(got[0].values, direct.values);
        assert_eq!(entries_for("shared-once"), 1);
        // Held, it is handed out again; let go, it is loaded afresh.
        assert!(Arc::ptr_eq(&shared(&v, 0, 1 << 20, &|_| true).unwrap(), &got[0]));
        drop(got);
        let again = shared(&v, 0, 1 << 20, &|_| true).unwrap();
        assert_eq!(again.values, direct.values);
        assert_eq!(entries_for("shared-once"), 1);
        // A changed file is read again; the checks come first.
        std::thread::sleep(Duration::from_millis(5));
        let changed = raw_volume(&dir, "shared-once", 5);
        let fresh = shared(&changed, 0, 1 << 20, &|_| true).unwrap();
        assert!(!Arc::ptr_eq(&fresh, &again) && fresh.values != again.values);
        assert_eq!(shared(&v, 0, 10, &|_| true).unwrap_err().to_string(), load_volume(&v, 10, &|_| true).unwrap_err().to_string());
        let _ = std::fs::remove_dir_all(&dir);
    }

    type Shared = Result<Arc<LoadedVolume>>;

    /// Start a load whose progress waits for a second caller to be waiting, then does `then`;
    /// returns the first and second callers' results.
    fn second_caller_after(name: &str, then: fn() -> bool) -> (std::thread::Result<Shared>, Shared, Volume) {
        let dir = temp_dir(name);
        let v = raw_volume(&dir, name, 7);
        let started = AtomicBool::new(false);
        let waiting = AtomicBool::new(false);
        std::thread::scope(|s| {
            let first = s.spawn(|| {
                shared(&v, 0, 1 << 20, &|_| {
                    started.store(true, Ordering::SeqCst);
                    let t = std::time::Instant::now();
                    while !waiting.load(Ordering::SeqCst) && t.elapsed() < Duration::from_secs(10) {
                        std::thread::sleep(Duration::from_millis(1));
                    }
                    then()
                })
            });
            while !started.load(Ordering::SeqCst) {
                std::thread::sleep(Duration::from_millis(1));
            }
            let second = shared(&v, 0, 1 << 20, &|_| {
                waiting.store(true, Ordering::SeqCst);
                true
            });
            (first.join(), second, v.clone())
        })
    }

    #[test]
    fn a_cancelled_shared_load_is_started_again_by_a_waiting_caller() {
        let (first, second, v) = second_caller_after("shared-cancel", || false);
        assert_eq!(first.unwrap().unwrap_err().to_string(), "cancelled");
        assert_eq!(second.unwrap().values, load_volume(&v, 1 << 20, &|_| true).unwrap().values);
    }

    #[test]
    fn a_panicking_shared_load_fails_its_waiting_callers() {
        let (first, second, v) = second_caller_after("shared-panic", || panic!("test panic in a progress callback"));
        assert!(first.is_err(), "the first caller panicked");
        assert_eq!(second.unwrap_err().to_string(), "the load stopped unexpectedly");
        // Nothing is left half-loaded: the next call loads.
        assert_eq!(shared(&v, 0, 1 << 20, &|_| true).unwrap().values, load_volume(&v, 1 << 20, &|_| true).unwrap().values);
    }
}
