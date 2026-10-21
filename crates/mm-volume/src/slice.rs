//! Slices of a volume for display: raw formats (NIfTI, Analyze, NRRD) are loaded whole once
//! (first 3D volume, capped in size), DICOM slices are decoded file by file (or frame by
//! frame). Every slice is turned to the radiological convention: anterior (axial) or
//! superior (coronal, sagittal) up, the patient's right on the left of the screen, and
//! anterior on the left of sagittal slices.

use crate::geometry::Geometry;
use crate::model::{DataType, Volume, Voxels};
use anyhow::{bail, Context, Result};
use glam::DVec3;
use std::io::Read;

/// A slice ready to show: grey values, row by row from the top left.
#[derive(Clone, Debug, PartialEq)]
pub struct SliceImage {
    pub width: usize,
    pub height: usize,
    pub values: Vec<f32>,
    /// Millimetres per pixel across and down.
    pub mm_per_pixel: [f64; 2],
    /// Show inverted (DICOM MONOCHROME1).
    pub invert: bool,
    /// Window (centre, width) from the 1st to the 99th percentile of the values.
    pub auto_window: (f64, f64),
}

/// The first 3D volume of a raw-format file, in memory.
#[derive(Clone, Debug, PartialEq)]
pub struct VolumeData {
    pub dims: [usize; 3],
    pub values: Vec<f32>,
}

/// How many slices a volume has (along its third axis).
pub fn slice_count(v: &Volume) -> usize {
    match &v.voxels {
        Voxels::Dicom { slices } | Voxels::Images { slices } => v.geometry.as_ref().map(|g| g.dims[2]).unwrap_or(slices.len()).min(slices.len()),
        _ => v.geometry.as_ref().map(|g| g.dims[2]).unwrap_or(0),
    }
}

/// Load the first 3D volume of a raw-format volume (at most `max_voxels` voxels).
pub fn load_raw(v: &Volume, max_voxels: usize) -> Result<VolumeData> {
    load_raw_at(v, 0, max_voxels)
}

/// Load 3D volume `t` of a raw-format volume: the `t`-th component when components are
/// interleaved per voxel, else the `t`-th block of a 4D file.
pub fn load_raw_at(v: &Volume, t: usize, max_voxels: usize) -> Result<VolumeData> {
    load_raw_into(v, t, max_voxels, &|_| true)
}

/// About how many bytes [`load_raw_into`] reads and decodes at a time.
const RAW_STEP: usize = 4 << 20;

/// [`load_raw_at`], reading and decoding about 4 MB at a time (so only the values are held
/// whole, not the file's bytes as well). `progress` gets 0.1 to 1 as the data come in;
/// returning `false` cancels.
pub(crate) fn load_raw_into(v: &Volume, t: usize, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync)) -> Result<VolumeData> {
    load_raw_stepped(v, t, max_voxels, progress, RAW_STEP)
}

/// [`load_raw_into`] with about `step` bytes per step.
fn load_raw_stepped(v: &Volume, t: usize, max_voxels: usize, progress: &(dyn Fn(f32) -> bool + Sync), step: usize) -> Result<VolumeData> {
    let Voxels::Raw { file, file_offset, gzip, skip, interleave, dtype, big_endian, slope, intercept } = &v.voxels else { bail!("not a raw-format volume") };
    let g = v.geometry.as_ref().context("no geometry")?;
    let dims = [g.dims[0], g.dims[1], g.dims[2]];
    let n = dims.iter().product::<usize>();
    if n > max_voxels {
        bail!("too large to preview ({} million voxels)", n / 1_000_000);
    }
    let stride = (*interleave).max(1);
    let size = dtype.size();
    let volumes = g.dims[3].max(1);
    if t >= volumes {
        bail!("there is no volume {} (of {volumes})", t + 1);
    }
    // Bytes before volume t's first value.
    let before = if stride > 1 { t.min(stride - 1) * size } else { t * n * size };
    let mut f = std::fs::File::open(file).with_context(|| format!("cannot open {}", file.display()))?;
    use std::io::Seek;
    f.seek(std::io::SeekFrom::Start(*file_offset))?;
    let mut buffered = std::io::BufReader::new(f);
    let mut reader: Box<dyn Read> = if !*gzip {
        Box::new(buffered)
    } else if std::io::BufRead::fill_buf(&mut buffered)?.starts_with(&[0x1f, 0x8b]) {
        Box::new(flate2::read::MultiGzDecoder::new(buffered))
    } else {
        Box::new(flate2::read::ZlibDecoder::new(buffered))
    };
    let to_skip = *skip + before as u64;
    if to_skip > 0 {
        std::io::copy(&mut reader.by_ref().take(to_skip), &mut std::io::sink())?;
    }
    let rescale = |values: &mut [f32]| {
        if *slope != 1.0 || *intercept != 0.0 {
            for x in values {
                *x = (*x as f64 * slope + intercept) as f32;
            }
        }
    };
    let short = "the file ends before its data does";
    let mut values: Vec<f32>;
    if *dtype == DataType::F32 && !*big_endian && stride == 1 && cfg!(target_endian = "little") {
        // Little-endian floats on a little-endian machine: the file's bytes are the values.
        values = vec![0.0f32; n];
        let mut done = 0;
        while done < n {
            let m = (step / 4).max(1).min(n - done);
            let part = &mut values[done..done + m];
            // SAFETY: the bytes of `m` initialised f32s, every bit pattern of which is a valid f32.
            let bytes = unsafe { std::slice::from_raw_parts_mut(part.as_mut_ptr().cast::<u8>(), m * 4) };
            reader.read_exact(bytes).context(short)?;
            rescale(part);
            done += m;
            if !progress(0.1 + 0.9 * done as f32 / n as f32) {
                bail!("cancelled");
            }
        }
    } else {
        // Whole records of `stride` values, so a step never splits a voxel. Interleaved data are
        // read from component t on, so the last record stops `before` bytes short (left zero).
        let record = stride * size;
        let want = if stride > 1 { n * record - before } else { n * size };
        let per_step = (step / record).max(1);
        let mut buf = vec![0u8; per_step.min(n) * record];
        values = Vec::with_capacity(n);
        let mut done = 0;
        while done < n {
            let m = per_step.min(n - done);
            let chunk = &mut buf[..m * record];
            let read = (want - done * record).min(m * record);
            reader.read_exact(&mut chunk[..read]).context(short)?;
            chunk[read..].fill(0);
            let from = values.len();
            decode_into(chunk, *dtype, *big_endian, stride, &mut values);
            rescale(&mut values[from..]);
            done += m;
            if !progress(0.1 + 0.9 * done as f32 / n as f32) {
                bail!("cancelled");
            }
        }
    }
    Ok(VolumeData { dims, values })
}

/// Append the values of component 0 of every `stride`-th voxel in `bytes` to `out`.
fn decode_into(bytes: &[u8], t: DataType, big: bool, stride: usize, out: &mut Vec<f32>) {
    let size = t.size();
    let records = bytes.chunks_exact(size * stride);
    // Values one after the other (no interleave) are read in fixed-size pieces, which the
    // compiler turns into much faster code.
    macro_rules! num {
        ($ty:ty) => {{
            const S: usize = std::mem::size_of::<$ty>();
            let value = |b: [u8; S]| (if big { <$ty>::from_be_bytes(b) } else { <$ty>::from_le_bytes(b) }) as f32;
            if stride == 1 {
                out.extend(bytes.chunks_exact(S).map(|r| value(r.try_into().unwrap())))
            } else {
                out.extend(records.map(|r| value(r[..S].try_into().unwrap())))
            }
        }};
    }
    match t {
        DataType::U8 => out.extend(records.map(|r| r[0] as f32)),
        DataType::I8 => out.extend(records.map(|r| r[0] as i8 as f32)),
        DataType::U16 => num!(u16),
        DataType::I16 => num!(i16),
        DataType::U32 => num!(u32),
        DataType::I32 => num!(i32),
        DataType::U64 => num!(u64),
        DataType::I64 => num!(i64),
        DataType::F32 => num!(f32),
        DataType::F64 => num!(f64),
        DataType::Rgb24 | DataType::Rgba32 => out.extend(records.map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32)),
        DataType::Complex64 => out.extend(records.map(|p| {
            let f = |o: usize| { let b: [u8; 4] = p[o..o + 4].try_into().unwrap(); if big { f32::from_be_bytes(b) } else { f32::from_le_bytes(b) } };
            f(0).hypot(f(4))
        })),
        DataType::Complex128 => out.extend(records.map(|p| {
            let f = |o: usize| { let b: [u8; 8] = p[o..o + 8].try_into().unwrap(); if big { f64::from_be_bytes(b) } else { f64::from_le_bytes(b) } };
            f(0).hypot(f(8)) as f32
        })),
    }
}

/// Which voxel axis (0 = i, 1 = j) runs across the screen and which down, and whether each
/// is reversed, for the radiological convention.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DisplayAxes {
    pub across: usize,
    pub flip_across: bool,
    pub flip_down: bool,
}

pub fn display_axes(g: Option<&Geometry>) -> DisplayAxes {
    let Some(g) = g else { return DisplayAxes { across: 0, flip_across: false, flip_down: false } };
    let [di, dj, dk] = g.directions();
    let n = dk.abs();
    // Screen right and screen up in patient (RAS) terms, by the plane of the slice.
    let (right, up) = if n.z >= n.x && n.z >= n.y {
        (DVec3::new(-1.0, 0.0, 0.0), DVec3::Y) // axial: patient left on the right, anterior up
    } else if n.y >= n.x {
        (DVec3::new(-1.0, 0.0, 0.0), DVec3::Z) // coronal
    } else {
        (DVec3::new(0.0, -1.0, 0.0), DVec3::Z) // sagittal: anterior on the left
    };
    let across = if di.dot(right).abs() >= dj.dot(right).abs() { 0 } else { 1 };
    let (a, d) = if across == 0 { (di, dj) } else { (dj, di) };
    DisplayAxes { across, flip_across: a.dot(right) < 0.0, flip_down: d.dot(up) > 0.0 }
}

/// The three standard views of a volume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Plane {
    Axial,
    Coronal,
    Sagittal,
}

impl Plane {
    pub const ALL: [Plane; 3] = [Plane::Axial, Plane::Coronal, Plane::Sagittal];

    pub fn label(self) -> &'static str {
        match self {
            Plane::Axial => "Axial",
            Plane::Coronal => "Coronal",
            Plane::Sagittal => "Sagittal",
        }
    }
}

/// How one plane of a volume is shown: the voxel axis it slices, the voxel axes running
/// across and down the screen, and whether each runs backwards (radiological convention, as
/// for thumbnails).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneAxes {
    pub slice: usize,
    pub across: usize,
    pub down: usize,
    pub flip_across: bool,
    pub flip_down: bool,
}

/// For each world axis (R, A, S), the voxel axis closest to it (each voxel axis used once).
pub fn voxel_axis_of_world(g: &Geometry) -> [usize; 3] {
    let dirs = g.directions();
    let mut used = [false; 3];
    let mut out = [0usize; 3];
    let mut order: Vec<usize> = (0..3).collect();
    order.sort_by(|&a, &b| dirs[b].abs().max_element().total_cmp(&dirs[a].abs().max_element()));
    for j in order {
        let d = dirs[j];
        let Some(w) = (0..3).filter(|&w| !used[w]).max_by(|&x, &y| d[x].abs().total_cmp(&d[y].abs())) else { continue };
        used[w] = true;
        out[w] = j;
    }
    out
}

pub fn plane_axes(g: &Geometry, plane: Plane) -> PlaneAxes {
    let v = voxel_axis_of_world(g);
    let dirs = g.directions();
    // (slice, across with the world axis screen-right is the negative of, down with the
    // world axis that points up).
    let (slice, across, across_world, down, up_world) = match plane {
        Plane::Axial => (v[2], v[0], 0, v[1], 1),
        Plane::Coronal => (v[1], v[0], 0, v[2], 2),
        Plane::Sagittal => (v[0], v[1], 1, v[2], 2),
    };
    PlaneAxes { slice, across, down, flip_across: dirs[across][across_world] > 0.0, flip_down: dirs[down][up_world] > 0.0 }
}

/// Lay out slice values (`nx` × `ny`, x fastest) for the screen.
fn orient(values: &[f32], nx: usize, ny: usize, axes: DisplayAxes, spacing: [f64; 2]) -> (usize, usize, Vec<f32>, [f64; 2]) {
    let (w, h) = if axes.across == 0 { (nx, ny) } else { (ny, nx) };
    let mut out = vec![0.0f32; w * h];
    for y in 0..h {
        let v = if axes.flip_down { h - 1 - y } else { y };
        for x in 0..w {
            let u = if axes.flip_across { w - 1 - x } else { x };
            let (i, j) = if axes.across == 0 { (u, v) } else { (v, u) };
            out[y * w + x] = values[j * nx + i];
        }
    }
    let mm = if axes.across == 0 { spacing } else { [spacing[1], spacing[0]] };
    (w, h, out, mm)
}

/// Window from the 1st to the 99th percentile (of the non-zero values when most are zero,
/// as in a skull-stripped brain).
pub fn auto_window(values: &[f32]) -> (f64, f64) {
    let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
    if v.is_empty() {
        return (0.5, 1.0);
    }
    let zeros = v.iter().filter(|&&x| x == 0.0).count();
    if zeros * 2 > v.len() && zeros < v.len() {
        v.retain(|&x| x != 0.0);
    }
    // The two percentiles by selection rather than a full sort. total_cmp orders every value,
    // so the value selected at a rank is the one sorting would put there.
    let rank = |p: f64| ((v.len() - 1) as f64 * p).round() as usize;
    let (r_lo, r_hi) = (rank(0.01), rank(0.99));
    let hi = *v.select_nth_unstable_by(r_hi, f32::total_cmp).1;
    // Everything before rank r_hi is now at most `hi`, so rank r_lo is found among those.
    let lo = if r_lo < r_hi { *v[..r_hi].select_nth_unstable_by(r_lo, f32::total_cmp).1 } else { hi };
    let (lo, hi) = (lo as f64, hi as f64);
    let (lo, hi) = if hi > lo {
        (lo, hi)
    } else {
        let min = v.iter().copied().min_by(f32::total_cmp).unwrap();
        let max = v.iter().copied().max_by(f32::total_cmp).unwrap();
        (min as f64, max as f64 + 1.0)
    };
    ((lo + hi) * 0.5, hi - lo)
}

/// Slice `k` of a raw-format volume already in memory.
pub fn raw_slice(v: &Volume, data: &VolumeData, k: usize) -> Result<SliceImage> {
    let [nx, ny, nz] = data.dims;
    if k >= nz {
        bail!("slice {k} out of range");
    }
    let plane = &data.values[k * nx * ny..(k + 1) * nx * ny];
    let g = v.geometry.as_ref();
    let spacing = g.map(|g| [g.spacing[0], g.spacing[1]]).unwrap_or([1.0, 1.0]);
    let (width, height, values, mm_per_pixel) = orient(plane, nx, ny, display_axes(g), spacing);
    let auto_window = auto_window(&values);
    Ok(SliceImage { width, height, values, mm_per_pixel, invert: false, auto_window })
}

/// Decode slice `k` of a DICOM volume.
pub fn dicom_slice(v: &Volume, k: usize) -> Result<SliceImage> {
    let Voxels::Dicom { slices } = &v.voxels else { bail!("not a DICOM volume") };
    let (path, _) = slices.get(k).context("slice out of range")?;
    let obj = dicom_object::open_file(path).with_context(|| format!("cannot read {}", path.display()))?;
    dicom_slice_from(&obj, v, k)
}

/// Decode slice `k` of a DICOM volume from its file, already open (so a multi-frame file can
/// stay open while its slices are shown one after another).
pub fn dicom_slice_from(obj: &dicom_object::DefaultDicomObject, v: &Volume, k: usize) -> Result<SliceImage> {
    use dicom_pixeldata::PixelDecoder;
    let Voxels::Dicom { slices } = &v.voxels else { bail!("not a DICOM volume") };
    let (_, frame) = slices.get(k).context("slice out of range")?;
    let (nx, ny, grey) = if crate::dicom::is_jpegls(obj.meta().transfer_syntax()) {
        crate::dicom::jpegls_frame(obj, *frame)?
    } else {
        let px = obj.decode_pixel_data_frame(*frame).map_err(|e| anyhow::anyhow!("cannot decode the pixels: {e}"))?;
        let (nx, ny) = (px.columns() as usize, px.rows() as usize);
        let raw: Vec<f32> = px.to_vec_frame(0).map_err(|e| anyhow::anyhow!("cannot convert the pixels: {e}"))?;
        let samples = (raw.len() / (nx * ny).max(1)).max(1);
        (nx, ny, if samples >= 3 { raw.chunks(samples).map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect() } else { raw })
    };
    let g = v.geometry.as_ref();
    let spacing = g.map(|g| [g.spacing[0], g.spacing[1]]).unwrap_or([1.0, 1.0]);
    let (width, height, values, mm_per_pixel) = orient(&grey, nx, ny, display_axes(g), spacing);
    let auto_window = auto_window(&values);
    Ok(SliceImage { width, height, values, mm_per_pixel, invert: v.photometric.as_deref() == Some("MONOCHROME1"), auto_window })
}

/// Decode slice `k` of a volume made of 2D images.
pub fn image_slice(v: &Volume, k: usize) -> Result<SliceImage> {
    let Voxels::Images { slices } = &v.voxels else { bail!("not an image stack") };
    let (path, page) = slices.get(k).context("slice out of range")?;
    let (nx, ny, grey) = crate::stack::read_plane(path, *page)?;
    let g = v.geometry.as_ref();
    let spacing = g.map(|g| [g.spacing[0], g.spacing[1]]).unwrap_or([1.0, 1.0]);
    let (width, height, values, mm_per_pixel) = orient(&grey, nx, ny, display_axes(g), spacing);
    let auto_window = auto_window(&values);
    Ok(SliceImage { width, height, values, mm_per_pixel, invert: false, auto_window })
}

/// Decode slice `k` of a volume stored slice by slice (DICOM files or 2D images).
pub fn file_slice(v: &Volume, k: usize) -> Result<SliceImage> {
    match &v.voxels {
        Voxels::Images { .. } => image_slice(v, k),
        _ => dicom_slice(v, k),
    }
}

/// Map a slice to 8-bit grey with a window (centre, width).
pub fn to_grey8(s: &SliceImage, window: (f64, f64)) -> Vec<u8> {
    let (c, w) = window;
    let lo = c - w * 0.5;
    let scale = 255.0 / w.max(f64::MIN_POSITIVE);
    s.values
        .iter()
        .map(|&x| {
            let g = ((x as f64 - lo) * scale).round().clamp(0.0, 255.0) as u8;
            if s.invert { 255 - g } else { g }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Format, Kind};

    fn volume_with(axes: [DVec3; 3], voxels: Voxels) -> Volume {
        Volume {
            format: Format::Nifti1,
            kind: Kind::Image,
            name: "t".into(),
            files: vec![],
            bytes: 0,
            geometry: Some(Geometry::from_axes([3, 2, 2, 1], [1.0, 2.0, 3.0], axes, DVec3::ZERO)),
            data_type: Some(DataType::I16),
            fields: vec![],
            warnings: vec![],
            voxels,
            window: None,
            photometric: None,
            patient: None,
        }
    }

    #[test]
    fn plane_axes_match_the_thumbnails() {
        // DICOM axial (LPS): the axial plane slices k with no flips, like the thumbnail.
        let lps = Geometry::from_axes([3, 2, 4, 1], [1.0; 3], [-DVec3::X, -DVec3::Y, DVec3::Z], DVec3::ZERO);
        assert_eq!(plane_axes(&lps, Plane::Axial), PlaneAxes { slice: 2, across: 0, down: 1, flip_across: false, flip_down: false });
        // Coronal of the same: across i, down k flipped (superior up).
        assert_eq!(plane_axes(&lps, Plane::Coronal), PlaneAxes { slice: 1, across: 0, down: 2, flip_across: false, flip_down: true });
        // Sagittal PIR (i → P, j → I, k → R): slices k, anterior on the left, superior up.
        let pir = Geometry::from_axes([3, 2, 4, 1], [1.0; 3], [-DVec3::Y, -DVec3::Z, DVec3::X], DVec3::ZERO);
        assert_eq!(plane_axes(&pir, Plane::Sagittal), PlaneAxes { slice: 2, across: 0, down: 1, flip_across: false, flip_down: false });
        let a = plane_axes(&pir, Plane::Axial);
        assert_eq!((a.slice, a.across, a.down, a.flip_across, a.flip_down), (1, 2, 0, true, false));
        for g in [&lps, &pir] {
            let d = display_axes(Some(g));
            let p = plane_axes(g, if g == &lps { Plane::Axial } else { Plane::Sagittal });
            assert_eq!((d.across, d.flip_across, d.flip_down), (p.across, p.flip_across, p.flip_down));
        }
    }

    #[test]
    fn radiological_display() {
        // A RAS array (i → R, j → A): flip across (patient right on the left) and down
        // (anterior up).
        let g = Geometry::from_axes([3, 2, 1, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let a = display_axes(Some(&g));
        assert_eq!(a, DisplayAxes { across: 0, flip_across: true, flip_down: true });
        // DICOM axial (LPS in RAS terms): already as displayed.
        let g = Geometry::from_axes([3, 2, 1, 1], [1.0; 3], [-DVec3::X, -DVec3::Y, DVec3::Z], DVec3::ZERO);
        assert_eq!(display_axes(Some(&g)), DisplayAxes { across: 0, flip_across: false, flip_down: false });
        // Sagittal PIR (i → P, j → I): anterior on the left, superior up, no flips.
        let g = Geometry::from_axes([3, 2, 1, 1], [1.0; 3], [-DVec3::Y, -DVec3::Z, DVec3::X], DVec3::ZERO);
        assert_eq!(display_axes(Some(&g)), DisplayAxes { across: 0, flip_across: false, flip_down: false });
        let (w, h, out, _) = orient(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, DisplayAxes { across: 0, flip_across: true, flip_down: true }, [1.0, 1.0]);
        assert_eq!((w, h), (3, 2));
        assert_eq!(out, vec![6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        let (w, h, out, mm) = orient(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 3, 2, DisplayAxes { across: 1, flip_across: false, flip_down: false }, [1.0, 2.0]);
        assert_eq!((w, h, mm), (2, 3, [2.0, 1.0]));
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn raw_volumes_plain_gzip_and_interleaved() {
        let dir = std::env::temp_dir().join(format!("mmv-slice-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let values: Vec<i16> = (0..12).collect();
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        let lps = [-DVec3::X, -DVec3::Y, DVec3::Z];
        // Plain, with a 10-byte header before the data.
        let p = dir.join("plain.raw");
        std::fs::write(&p, [vec![9u8; 10], bytes.clone()].concat()).unwrap();
        let v = volume_with(lps, Voxels::Raw { file: p, file_offset: 10, gzip: false, skip: 0, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 2.0, intercept: 1.0 });
        let d = load_raw(&v, 1000).unwrap();
        assert_eq!(d.values, values.iter().map(|&x| 2.0 * x as f32 + 1.0).collect::<Vec<_>>());
        let s = raw_slice(&v, &d, 1).unwrap();
        assert_eq!((s.width, s.height), (3, 2));
        assert_eq!(s.values, vec![13.0, 15.0, 17.0, 19.0, 21.0, 23.0]);
        // gzip, with 4 bytes to skip inside the compressed stream (as in .nii.gz).
        use std::io::Write;
        let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
        e.write_all(&[[0u8; 4].to_vec(), bytes.clone()].concat()).unwrap();
        let p = dir.join("data.gz");
        std::fs::write(&p, e.finish().unwrap()).unwrap();
        let v = volume_with(lps, Voxels::Raw { file: p, file_offset: 0, gzip: true, skip: 4, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 });
        assert_eq!(load_raw(&v, 1000).unwrap().values[11], 11.0);
        // Two interleaved layers: component 0 only.
        let inter: Vec<u8> = values.iter().flat_map(|v| [v.to_le_bytes(), (-1i16).to_le_bytes()]).flatten().collect();
        let p = dir.join("inter.raw");
        std::fs::write(&p, inter).unwrap();
        let v = volume_with(lps, Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 2, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 });
        assert_eq!(load_raw(&v, 1000).unwrap().values, values.iter().map(|&x| x as f32).collect::<Vec<_>>());
        // Too big, or cut short.
        assert!(load_raw(&v, 5).is_err());
        let p = dir.join("short.raw");
        std::fs::write(&p, &bytes[..10]).unwrap();
        let v = volume_with(lps, Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::I16, big_endian: false, slope: 1.0, intercept: 0.0 });
        assert!(load_raw(&v, 1000).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn windows() {
        let mut v: Vec<f32> = (0..1000).map(|x| x as f32).collect();
        let (c, w) = auto_window(&v);
        assert!((c - 499.5).abs() < 2.0 && (w - 980.0).abs() < 3.0, "{c} {w}");
        // Mostly background: the window follows the tissue.
        v.extend(std::iter::repeat_n(0.0, 5000));
        let (c2, _) = auto_window(&v);
        assert!((c2 - c).abs() < 5.0);
        let s = SliceImage { width: 3, height: 1, values: vec![0.0, 50.0, 100.0], mm_per_pixel: [1.0; 2], invert: true, auto_window: (50.0, 100.0) };
        let g = to_grey8(&s, (50.0, 100.0));
        assert_eq!((g[0], g[2]), (255, 0));
        assert!((127..=128).contains(&g[1]));
    }

    /// The window as it was computed before, by sorting.
    fn sorted_window(values: &[f32]) -> (f64, f64) {
        let mut v: Vec<f32> = values.iter().copied().filter(|x| x.is_finite()).collect();
        if v.is_empty() {
            return (0.5, 1.0);
        }
        let zeros = v.iter().filter(|&&x| x == 0.0).count();
        if zeros * 2 > v.len() && zeros < v.len() {
            v.retain(|&x| x != 0.0);
        }
        v.sort_by(f32::total_cmp);
        let q = |p: f64| v[((v.len() - 1) as f64 * p).round() as usize] as f64;
        let (lo, hi) = (q(0.01), q(0.99));
        let (lo, hi) = if hi > lo { (lo, hi) } else { (v[0] as f64, *v.last().unwrap() as f64 + 1.0) };
        ((lo + hi) * 0.5, hi - lo)
    }

    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut s = seed | 1;
        move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        }
    }

    #[test]
    fn windows_by_selection_match_sorting() {
        let mut r = rng(17);
        let mut cases: Vec<Vec<f32>> = vec![vec![], vec![3.0], vec![2.0, 1.0], vec![0.0, -0.0, 0.0], vec![-0.0, 0.0, 5.0], vec![f32::NAN, 3.0, f32::INFINITY], vec![7.0; 100], vec![-0.0; 9]];
        for n in [2usize, 3, 7, 50, 99, 100, 101, 1000, 4321] {
            for kind in 0..6 {
                cases.push((0..n).map(|k| match kind {
                    0 => (r() % 4096) as f32,
                    1 => if k % 3 == 0 { 0.0 } else { (r() % 900) as f32 + 300.0 },
                    2 => match r() % 5 { 0 => 0.0, 1 => -0.0, 2 => (r() % 3) as f32, _ => (r() >> 40) as f32 * 1e-3 - 5.0 },
                    3 => f32::from_bits(r() as u32),
                    4 => ((r() % 7) as f32 - 3.0) * 0.5,
                    _ => if r().is_multiple_of(50) { (r() % 100) as f32 } else { 0.0 },
                }).collect());
            }
        }
        for v in &cases {
            let (a, b) = (sorted_window(v), auto_window(v));
            assert!(a.0.to_bits() == b.0.to_bits() && a.1.to_bits() == b.1.to_bits(), "{} values: {a:?} / {b:?}", v.len());
        }
    }

    /// How values were decoded before.
    fn old_decode(bytes: &[u8], t: DataType, big: bool, stride: usize) -> Vec<f32> {
        let size = t.size();
        let voxel = |k: usize| &bytes[k * stride * size..k * stride * size + size];
        let n = bytes.len() / (size * stride);
        macro_rules! num {
            ($ty:ty) => {
                (0..n).map(|k| {
                    let b: [u8; std::mem::size_of::<$ty>()] = voxel(k).try_into().unwrap();
                    (if big { <$ty>::from_be_bytes(b) } else { <$ty>::from_le_bytes(b) }) as f32
                }).collect()
            };
        }
        match t {
            DataType::U8 => (0..n).map(|k| voxel(k)[0] as f32).collect(),
            DataType::I8 => (0..n).map(|k| voxel(k)[0] as i8 as f32).collect(),
            DataType::U16 => num!(u16),
            DataType::I16 => num!(i16),
            DataType::U32 => num!(u32),
            DataType::I32 => num!(i32),
            DataType::U64 => num!(u64),
            DataType::I64 => num!(i64),
            DataType::F32 => num!(f32),
            DataType::F64 => num!(f64),
            DataType::Rgb24 | DataType::Rgba32 => (0..n).map(|k| {
                let p = voxel(k);
                0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32
            }).collect(),
            DataType::Complex64 => (0..n).map(|k| {
                let p = voxel(k);
                let f = |o: usize| { let b: [u8; 4] = p[o..o + 4].try_into().unwrap(); if big { f32::from_be_bytes(b) } else { f32::from_le_bytes(b) } };
                f(0).hypot(f(4))
            }).collect(),
            DataType::Complex128 => (0..n).map(|k| {
                let p = voxel(k);
                let f = |o: usize| { let b: [u8; 8] = p[o..o + 8].try_into().unwrap(); if big { f64::from_be_bytes(b) } else { f64::from_le_bytes(b) } };
                f(0).hypot(f(8)) as f32
            }).collect(),
        }
    }

    /// Volume `t` read as before: the whole data into one buffer, then decoded.
    fn whole_read(v: &Volume, t: usize) -> Result<VolumeData> {
        let Voxels::Raw { file, file_offset, gzip, skip, interleave, dtype, big_endian, slope, intercept } = &v.voxels else { bail!("not raw") };
        let g = v.geometry.as_ref().unwrap();
        let dims = [g.dims[0], g.dims[1], g.dims[2]];
        let n = dims.iter().product::<usize>();
        let (stride, size) = ((*interleave).max(1), dtype.size());
        let before = if stride > 1 { t.min(stride - 1) * size } else { t * n * size };
        let mut f = std::fs::File::open(file)?;
        use std::io::Seek;
        f.seek(std::io::SeekFrom::Start(*file_offset))?;
        let mut buffered = std::io::BufReader::new(f);
        let mut reader: Box<dyn Read> = if !*gzip {
            Box::new(buffered)
        } else if std::io::BufRead::fill_buf(&mut buffered)?.starts_with(&[0x1f, 0x8b]) {
            Box::new(flate2::read::MultiGzDecoder::new(buffered))
        } else {
            Box::new(flate2::read::ZlibDecoder::new(buffered))
        };
        std::io::copy(&mut reader.by_ref().take(*skip + before as u64), &mut std::io::sink())?;
        let mut bytes = vec![0u8; n * stride * size];
        let want = if stride > 1 { n * stride * size - before } else { n * size };
        reader.read_exact(&mut bytes[..want]).context("the file ends before its data does")?;
        let mut values = old_decode(&bytes, *dtype, *big_endian, stride);
        if *slope != 1.0 || *intercept != 0.0 {
            for x in &mut values {
                *x = (*x as f64 * slope + intercept) as f32;
            }
        }
        Ok(VolumeData { dims, values })
    }

    #[test]
    fn raw_volumes_read_in_steps_match_one_whole_read() {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("mmv-steps-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let mut r = rng(5);
        let dims = [37usize, 23, 5];
        let n = dims.iter().product::<usize>();
        let types = [DataType::U8, DataType::I8, DataType::U16, DataType::I16, DataType::U32, DataType::I32, DataType::U64, DataType::I64, DataType::F32, DataType::F64, DataType::Rgb24, DataType::Rgba32, DataType::Complex64, DataType::Complex128];
        let mut checked = 0;
        for (case, dtype) in types.iter().copied().cycle().take(14 * 6).enumerate() {
            let big = case % 2 == 1;
            let (stride, volumes) = if case % 3 == 0 { (3, 3) } else { (1, 1 + case % 3) };
            let (skip, offset) = ((case % 4) as u64 * 7, (case % 5) as u64 * 13);
            let (slope, intercept) = if case % 4 == 1 { (0.37, -12.5) } else { (1.0, 0.0) };
            // Random bytes (large integers, NaNs and all), cut short now and then.
            let len = skip as usize + n * dtype.size() * if stride > 1 { stride } else { volumes };
            let mut data: Vec<u8> = (0..len).map(|_| r() as u8).collect();
            if case % 7 == 6 {
                data.truncate(data.len() - 1 - (r() % 50) as usize);
            }
            let compress = case % 3;
            let body = match compress {
                0 => data,
                1 => {
                    let mut e = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::fast());
                    e.write_all(&data).unwrap();
                    e.finish().unwrap()
                }
                _ => {
                    let mut e = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::fast());
                    e.write_all(&data).unwrap();
                    e.finish().unwrap()
                }
            };
            let p = dir.join(format!("v{case}.raw"));
            std::fs::write(&p, [vec![0xa5u8; offset as usize], body].concat()).unwrap();
            let mut v = volume_with([DVec3::X, DVec3::Y, DVec3::Z], Voxels::Raw { file: p, file_offset: offset, gzip: compress > 0, skip, interleave: stride, dtype, big_endian: big, slope, intercept });
            v.geometry = Some(Geometry::from_axes([dims[0], dims[1], dims[2], volumes], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
            for t in 0..volumes {
                let old = whole_read(&v, t);
                for step in [RAW_STEP, 1000, 77, 1] {
                    let new = load_raw_stepped(&v, t, usize::MAX, &|_| true, step);
                    match (&old, &new) {
                        (Ok(a), Ok(b)) => {
                            assert_eq!(a.dims, b.dims);
                            assert!(a.values.iter().zip(&b.values).all(|(x, y)| x.to_bits() == y.to_bits()) && a.values.len() == b.values.len(), "case {case} {dtype:?} t {t} step {step}");
                        }
                        (Err(a), Err(b)) => assert_eq!(format!("{a:#}"), format!("{b:#}"), "case {case}"),
                        _ => panic!("case {case} {dtype:?} t {t} step {step}: {:?} / {:?}", old.as_ref().err(), new.as_ref().err()),
                    }
                    checked += 1;
                }
            }
        }
        assert!(checked > 300);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn raw_loads_report_progress_and_cancel() {
        let dir = std::env::temp_dir().join(format!("mmv-cancel-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("v.raw");
        std::fs::write(&p, (0..12i32).flat_map(|v| (v * 100_000).to_le_bytes()).collect::<Vec<u8>>()).unwrap();
        for dtype in [DataType::I32, DataType::F32] {
            let v = volume_with([DVec3::X, DVec3::Y, DVec3::Z], Voxels::Raw { file: p.clone(), file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype, big_endian: false, slope: 1.0, intercept: 0.0 });
            let seen = std::sync::Mutex::new(Vec::new());
            load_raw_stepped(&v, 0, 100, &|f| {
                seen.lock().unwrap().push(f);
                true
            }, 8).unwrap();
            let seen = seen.into_inner().unwrap();
            assert_eq!(seen.len(), 6, "{dtype:?}");
            assert!(seen.windows(2).all(|w| w[0] < w[1]) && (seen.last().unwrap() - 1.0).abs() < 1e-6);
            let e = load_raw_stepped(&v, 0, 100, &|f| f < 0.5, 8).unwrap_err();
            assert_eq!(e.to_string(), "cancelled");
        }
        assert_eq!(load_raw(&volume_with([DVec3::X, DVec3::Y, DVec3::Z], Voxels::Raw { file: p, file_offset: 0, gzip: false, skip: 0, interleave: 1, dtype: DataType::I32, big_endian: false, slope: 1.0, intercept: 0.0 }), 100).unwrap().values[11], 1_100_000.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
