//! Writing volumes as NIfTI-1 (`.nii`, or gzip-compressed `.nii.gz`): one 3D grid of uint8,
//! int16 or float32 values with its placement (sform and qform both set from the grid), so
//! derived images and label maps open in Polymix, 3D Slicer, ITK-SNAP, FSL and nibabel.
//! `.nii.gz` files are compressed in parallel but are still one ordinary gzip stream.

use crate::geometry::Geometry;
use anyhow::{bail, Context, Result};
use flate2::{Compression, FlushCompress, Status};
use glam::{DMat3, DVec3};
use rayon::prelude::*;
use std::borrow::Cow;
use std::io::Write;
use std::path::Path;

/// Voxel values to write, in voxel order (i fastest).
#[derive(Clone, Copy, Debug)]
pub enum Data<'a> {
    U8(&'a [u8]),
    I16(&'a [i16]),
    F32(&'a [f32]),
}

impl Data<'_> {
    fn len(&self) -> usize {
        match self {
            Data::U8(v) => v.len(),
            Data::I16(v) => v.len(),
            Data::F32(v) => v.len(),
        }
    }

    /// NIfTI datatype code and bits per voxel.
    fn code(&self) -> (i16, i16) {
        match self {
            Data::U8(_) => (2, 8),
            Data::I16(_) => (4, 16),
            Data::F32(_) => (16, 32),
        }
    }

    /// Bytes of the values, little endian.
    fn byte_len(&self) -> usize {
        match self {
            Data::U8(v) => v.len(),
            Data::I16(v) => v.len() * 2,
            Data::F32(v) => v.len() * 4,
        }
    }

    /// Bytes `from..from + out.len()` of the little-endian values into `out` (both ends on
    /// whole values).
    fn bytes_into(&self, from: usize, out: &mut [u8]) {
        match self {
            Data::U8(v) => out.copy_from_slice(&v[from..from + out.len()]),
            Data::I16(v) => out.as_chunks_mut::<2>().0.iter_mut().zip(&v[from / 2..]).for_each(|(o, x)| *o = x.to_le_bytes()),
            Data::F32(v) => out.as_chunks_mut::<4>().0.iter_mut().zip(&v[from / 4..]).for_each(|(o, x)| *o = x.to_le_bytes()),
        }
    }
}

/// Bytes of the header-then-values stream compressed at a time by each thread for `.nii.gz`.
/// A multiple of 4, as is the 352-byte header, so pieces start and end on whole values.
const GZ_PIECE: usize = 8 << 20;

/// Bytes `at..at + out.len()` of the header followed by the values.
fn stream_into(header: &[u8], data: &Data, at: usize, out: &mut [u8]) {
    let head = header.len().saturating_sub(at).min(out.len());
    if head > 0 {
        out[..head].copy_from_slice(&header[at..at + head]);
    }
    if head < out.len() {
        data.bytes_into(at + head - header.len(), &mut out[head..]);
    }
}

/// Write `header` then `data` as a single gzip member (as `GzEncoder` with
/// `Compression::fast()` would, so every gzip reader takes it). Pieces of `piece` bytes are
/// compressed in parallel as raw deflate, each ended on a byte boundary (a sync flush) and the
/// last one finished, which together make one deflate stream; their CRC-32s are combined in
/// order. A piece that barely compresses is stored instead, which is far quicker to write and
/// to read back. Pieces are done a pool's worth at a time to bound memory.
fn write_gzip(w: &mut dyn Write, header: &[u8], data: &Data, piece: usize) -> Result<()> {
    let total = header.len() + data.byte_len();
    let pieces = total.div_ceil(piece);
    // No name or time, XFL 4 (fastest compression), OS unknown: flate2's header for fast().
    w.write_all(&[0x1f, 0x8b, 8, 0, 0, 0, 0, 0, 4, 255])?;
    let mut crc = flate2::Crc::new();
    let wave = rayon::current_num_threads().max(1);
    for first in (0..pieces).step_by(wave) {
        let done: Vec<Result<(Vec<u8>, flate2::Crc)>> = (first..(first + wave).min(pieces))
            .into_par_iter()
            .map(|i| {
                let at = i * piece;
                let len = piece.min(total - at);
                // Bytes past the header are compressed where they are; other values are
                // turned into little-endian bytes first.
                let bytes: Cow<[u8]> = match data {
                    Data::U8(v) if at >= header.len() => Cow::Borrowed(&v[at - header.len()..at - header.len() + len]),
                    _ => {
                        let mut b = vec![0u8; len];
                        stream_into(header, data, at, &mut b);
                        Cow::Owned(b)
                    }
                };
                let mut c = flate2::Crc::new();
                c.update(&bytes);
                let level = if barely_compresses(&bytes)? { Compression::none() } else { Compression::fast() };
                Ok((deflate(&bytes, level, i + 1 == pieces)?, c))
            })
            .collect();
        for d in done {
            let (z, c) = d?;
            w.write_all(&z)?;
            crc.combine(&c);
        }
    }
    w.write_all(&crc.sum().to_le_bytes())?;
    // ISIZE: the length modulo 2^32.
    w.write_all(&(total as u32).to_le_bytes())?;
    Ok(())
}

/// Raw deflate of `input`, ended on a byte boundary so the next piece's stream can follow, or
/// finished when it is the `last` piece.
fn deflate(input: &[u8], level: Compression, last: bool) -> Result<Vec<u8>> {
    let mut c = flate2::Compress::new(level, false);
    let flush = if last { FlushCompress::Finish } else { FlushCompress::Sync };
    let mut out = Vec::with_capacity(input.len() + input.len() / 64 + 1024);
    loop {
        let before = (c.total_in(), c.total_out());
        let status = c.compress_vec(&input[c.total_in() as usize..], &mut out, flush)?;
        let room = out.len() < out.capacity();
        let finished = if last { status == Status::StreamEnd } else { c.total_in() as usize == input.len() && room };
        if finished {
            return Ok(out);
        }
        if !room {
            out.reserve(out.capacity() / 2 + 1024);
        } else if (c.total_in(), c.total_out()) == before {
            bail!("the compressor stopped before the end of the data");
        }
    }
}

/// Whether a piece barely compresses (more than 95% of its size left), judged on four samples
/// spread over it.
fn barely_compresses(bytes: &[u8]) -> Result<bool> {
    let sample = (bytes.len() / 8).min(64 << 10);
    if sample < 1024 {
        return Ok(false);
    }
    let step = bytes.len() / 4;
    let mut packed = 0;
    for k in 0..4 {
        packed += deflate(&bytes[k * step..k * step + sample], Compression::fast(), true)?.len();
    }
    Ok(packed as f64 > 0.95 * (4 * sample) as f64)
}

/// What the values mean (NIfTI intent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Intent {
    None,
    /// Integer labels (segments).
    Label,
}

/// Quaternion (b, c, d) and qfac of a grid's rotation, as NIfTI stores it (a left-handed grid
/// flips its third axis and sets qfac = -1).
fn quaternion(dirs: [DVec3; 3]) -> ([f64; 3], f64) {
    let mut r = DMat3::from_cols(dirs[0], dirs[1], dirs[2]);
    let qfac = if r.determinant() < 0.0 { -1.0 } else { 1.0 };
    if qfac < 0.0 {
        r.z_axis = -r.z_axis;
    }
    let m = |i: usize, j: usize| r.col(j)[i];
    let trace = m(0, 0) + m(1, 1) + m(2, 2);
    let (a, b, c, d) = if trace > 0.0 {
        let a = 0.5 * (1.0 + trace).sqrt();
        (a, 0.25 * (m(2, 1) - m(1, 2)) / a, 0.25 * (m(0, 2) - m(2, 0)) / a, 0.25 * (m(1, 0) - m(0, 1)) / a)
    } else if m(0, 0) > m(1, 1) && m(0, 0) > m(2, 2) {
        let b = 0.5 * (1.0 + m(0, 0) - m(1, 1) - m(2, 2)).sqrt();
        (0.25 * (m(2, 1) - m(1, 2)) / b, b, 0.25 * (m(0, 1) + m(1, 0)) / b, 0.25 * (m(0, 2) + m(2, 0)) / b)
    } else if m(1, 1) > m(2, 2) {
        let c = 0.5 * (1.0 + m(1, 1) - m(0, 0) - m(2, 2)).sqrt();
        (0.25 * (m(0, 2) - m(2, 0)) / c, 0.25 * (m(0, 1) + m(1, 0)) / c, c, 0.25 * (m(1, 2) + m(2, 1)) / c)
    } else {
        let d = 0.5 * (1.0 + m(2, 2) - m(0, 0) - m(1, 1)).sqrt();
        (0.25 * (m(1, 0) - m(0, 1)) / d, 0.25 * (m(0, 2) + m(2, 0)) / d, 0.25 * (m(1, 2) + m(2, 1)) / d, d)
    };
    // NIfTI keeps a ≥ 0.
    let s = if a < 0.0 { -1.0 } else { 1.0 };
    ([b * s, c * s, d * s], qfac)
}

/// Write `data` (on the grid `geometry`, `dims` voxels) as NIfTI-1, gzip-compressed when the
/// name ends in `.gz`. The file appears complete or not at all.
pub fn nifti(path: &Path, data: Data, dims: [usize; 3], geometry: &Geometry, description: &str, intent: Intent) -> Result<()> {
    if data.len() != dims.iter().product::<usize>() {
        bail!("{} values for a {}×{}×{} grid", data.len(), dims[0], dims[1], dims[2]);
    }
    if dims.iter().any(|&d| d > i16::MAX as usize) {
        bail!("too many voxels along an axis for NIfTI-1");
    }
    let mut h = vec![0u8; 352];
    let put_i16 = |h: &mut [u8], at: usize, v: i16| h[at..at + 2].copy_from_slice(&v.to_le_bytes());
    let put_f32 = |h: &mut [u8], at: usize, v: f64| h[at..at + 4].copy_from_slice(&(v as f32).to_le_bytes());
    h[0..4].copy_from_slice(&348i32.to_le_bytes());
    h[38] = b'r';
    put_i16(&mut h, 40, 3);
    for (k, &d) in dims.iter().enumerate() {
        put_i16(&mut h, 42 + 2 * k, d as i16);
    }
    for k in 3..7 {
        put_i16(&mut h, 42 + 2 * k, 1);
    }
    if intent == Intent::Label {
        put_i16(&mut h, 68, 1002);
    }
    let (code, bits) = data.code();
    put_i16(&mut h, 70, code);
    put_i16(&mut h, 72, bits);
    let ([b, c, d], qfac) = quaternion(geometry.directions());
    put_f32(&mut h, 76, qfac);
    for k in 0..3 {
        put_f32(&mut h, 80 + 4 * k, geometry.spacing[k]);
    }
    put_f32(&mut h, 108, 352.0);
    put_f32(&mut h, 112, 1.0);
    h[123] = 2; // millimetres
    let desc = description.as_bytes();
    h[148..148 + desc.len().min(79)].copy_from_slice(&desc[..desc.len().min(79)]);
    put_i16(&mut h, 252, 1); // qform: scanner
    put_i16(&mut h, 254, 1); // sform: scanner
    put_f32(&mut h, 256, b);
    put_f32(&mut h, 260, c);
    put_f32(&mut h, 264, d);
    let a = &geometry.affine;
    for (k, row) in a.iter().take(3).enumerate() {
        put_f32(&mut h, 268 + 4 * k, row[3]);
    }
    for (row, at) in a.iter().zip([280usize, 296, 312]) {
        for (k, &x) in row.iter().enumerate() {
            put_f32(&mut h, at + 4 * k, x);
        }
    }
    h[344..348].copy_from_slice(b"n+1\0");

    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let tmp = path.with_extension("partial");
    {
        let mut f = std::io::BufWriter::new(std::fs::File::create(&tmp).with_context(|| format!("cannot write {}", path.display()))?);
        if name.ends_with(".gz") {
            write_gzip(&mut f, &h, &data, GZ_PIECE)?;
        } else {
            // The values go out as they are, or a megabyte at a time, without a copy of them all.
            f.write_all(&h)?;
            if let Data::U8(v) = data {
                f.write_all(v)?;
            } else {
                let len = data.byte_len();
                let mut buf = vec![0u8; len.min(1 << 20)];
                let mut at = 0;
                while at < len {
                    let n = buf.len().min(len - at);
                    data.bytes_into(at, &mut buf[..n]);
                    f.write_all(&buf[..n])?;
                    at += n;
                }
            }
        }
        f.flush()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{DataType, Kind};
    use crate::slice::load_raw;

    #[test]
    fn float_round_trip() {
        let dir = std::env::temp_dir().join(format!("mmv-write-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let axes = [DVec3::new(0.8, 0.6, 0.0), DVec3::new(-0.6, 0.8, 0.0), DVec3::new(0.0, 0.0, -1.0)];
        let g = Geometry::from_axes([4, 3, 2, 1], [0.5, 0.75, 2.0], axes, DVec3::new(10.0, -20.0, 30.0));
        let values: Vec<f32> = (0..24).map(|v| v as f32 * 0.25 - 1.0).collect();
        let p = dir.join("n4.nii.gz");
        nifti(&p, Data::F32(&values), [4, 3, 2], &g, "test", Intent::None).unwrap();
        let head = crate::nifti::read_head(&p, true, 352).unwrap();
        let mut h = crate::nifti::parse(&head).unwrap();
        let v = crate::nifti::volume(&p, true, &h).unwrap();
        assert_eq!(v.kind, Kind::Image);
        assert_eq!(v.data_type, Some(DataType::F32));
        assert_eq!(load_raw(&v, 100).unwrap().values, values);
        for (r, e) in v.geometry.as_ref().unwrap().affine.iter().flatten().zip(g.affine.iter().flatten()) {
            assert!((r - e).abs() < 1e-4);
        }
        // The qform alone gives the same grid.
        h.sform_code = 0;
        let q = Geometry::from_affine([4, 3, 2, 1], h.affine().0);
        for (r, e) in q.affine.iter().flatten().zip(g.affine.iter().flatten()) {
            assert!((r - e).abs() < 1e-4, "qform {:?} vs {:?}", q.affine, g.affine);
        }
        assert!(nifti(&p, Data::U8(&[1, 2]), [4, 3, 2], &g, "", Intent::Label).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    fn noise(n: usize) -> impl Iterator<Item = u32> {
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        (0..n).map(move |_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s as u32
        })
    }

    /// Everything a gzip stream inflates to, checking that it is a single member with nothing
    /// after it (the single-member decoder also checks the CRC and the length).
    fn inflate_one_member(gz: &[u8]) -> Vec<u8> {
        use std::io::Read;
        let mut d = flate2::bufread::GzDecoder::new(gz);
        let mut out = Vec::new();
        d.read_to_end(&mut out).unwrap();
        assert!(d.into_inner().is_empty(), "one member, nothing after it");
        let mut again = Vec::new();
        flate2::read::MultiGzDecoder::new(gz).read_to_end(&mut again).unwrap();
        assert_eq!(again, out);
        out
    }

    #[test]
    fn gzip_written_in_pieces_is_one_member_of_the_same_bytes() {
        let header: Vec<u8> = (0..352).map(|i| (i * 7) as u8).collect();
        // Stretches of zeros (compressed) and of noise (stored).
        let mut bits = noise(400_000);
        let values: Vec<f32> = (0..400_000).map(|k| if (k / 30_000) % 2 == 0 { 0.0 } else { f32::from_bits(bits.next().unwrap()) }).collect();
        let labels: Vec<u8> = (0..70_001).map(|k| (k / 1000 % 3) as u8).collect();
        let small: Vec<i16> = (0..3001).map(|k| k as i16 * 11 - 900).collect();
        for (data, pieces) in [(Data::F32(&values), vec![GZ_PIECE, 256 << 10, 64 << 10, 4096]), (Data::U8(&labels), vec![GZ_PIECE, 4096, 100]), (Data::I16(&small), vec![GZ_PIECE, 100, 8])] {
            let mut expected = header.clone();
            let mut bytes = vec![0u8; data.byte_len()];
            data.bytes_into(0, &mut bytes);
            expected.extend(bytes);
            for piece in pieces {
                let mut gz = Vec::new();
                write_gzip(&mut gz, &header, &data, piece).unwrap();
                assert_eq!(inflate_one_member(&gz), expected, "pieces of {piece}");
            }
        }
        // Noise is stored, zeros and labels compressed.
        let noisy: Vec<u8> = noise(1 << 18).flat_map(u32::to_le_bytes).collect();
        assert!(barely_compresses(&noisy).unwrap());
        assert!(!barely_compresses(&vec![0u8; 1 << 20]).unwrap());
        assert!(!barely_compresses(&labels).unwrap());
    }

    #[test]
    fn plain_and_compressed_files_hold_the_same_bytes() {
        let dir = std::env::temp_dir().join(format!("mmv-write-same-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let g = Geometry::from_axes([100, 70, 100, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let values: Vec<i16> = (0..700_000).map(|k| (k % 977) as i16 - 300).collect();
        let (plain, packed) = (dir.join("v.nii"), dir.join("v.nii.gz"));
        nifti(&plain, Data::I16(&values), [100, 70, 100], &g, "same", Intent::None).unwrap();
        nifti(&packed, Data::I16(&values), [100, 70, 100], &g, "same", Intent::None).unwrap();
        let file = std::fs::read(&plain).unwrap();
        assert_eq!(&file[352..], values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>().as_slice());
        assert_eq!(inflate_one_member(&std::fs::read(&packed).unwrap()), file);
        assert!(!dir.join("v.partial").exists());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
