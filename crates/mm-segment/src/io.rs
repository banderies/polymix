//! Saving label maps: NIfTI-1 (`.nii` / `.nii.gz`, uint8, label intent, sform and qform set
//! from the grid) and 3D Slicer segmentations (`.seg.nrrd`, with segment names and colours),
//! both readable by the Volume workspace and by Slicer, ITK-SNAP, FSL and nibabel.

use anyhow::{bail, Context, Result};
use glam::{DMat3, DVec3};
use mm_volume::geometry::Geometry;
use std::io::Write;
use std::path::Path;

/// One segment of a label map.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentInfo {
    pub name: String,
    pub colour: [u8; 3],
    /// Its value in the label map (1–255).
    pub label: u8,
}

fn write_maybe_gzip(path: &Path, bytes: &[u8]) -> Result<()> {
    let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
    let tmp = path.with_extension("partial");
    {
        let f = std::io::BufWriter::new(std::fs::File::create(&tmp).with_context(|| format!("cannot write {}", path.display()))?);
        if name.ends_with(".gz") {
            let mut e = flate2::write::GzEncoder::new(f, flate2::Compression::default());
            e.write_all(bytes)?;
            e.finish()?.flush()?;
        } else {
            let mut f = f;
            f.write_all(bytes)?;
            f.flush()?;
        }
    }
    std::fs::rename(&tmp, path)?;
    Ok(())
}

/// Quaternion (b, c, d) and qfac of the rotation part of a voxel → RAS matrix, as NIfTI
/// stores it (a left-handed grid flips its third axis and sets qfac = -1).
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

/// Save a uint8 label map as NIfTI-1 (gzip-compressed when the name ends in `.gz`).
pub fn write_nifti(path: &Path, labels: &[u8], dims: [usize; 3], geometry: &Geometry, description: &str) -> Result<()> {
    if labels.len() != dims.iter().product::<usize>() {
        bail!("label map size does not match the grid");
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
    put_i16(&mut h, 68, 1002); // intent: label
    put_i16(&mut h, 70, 2); // uint8
    put_i16(&mut h, 72, 8);
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
    h.extend_from_slice(labels);
    write_maybe_gzip(path, &h)
}

/// Save a label map as a 3D Slicer segmentation (`.seg.nrrd`, one layer, gzip).
pub fn write_seg_nrrd(path: &Path, labels: &[u8], dims: [usize; 3], geometry: &Geometry, segments: &[SegmentInfo]) -> Result<()> {
    if labels.len() != dims.iter().product::<usize>() {
        bail!("label map size does not match the grid");
    }
    let m = geometry.matrix();
    let col = |v: DVec3| format!("({},{},{})", v.x, v.y, v.z);
    let mut head = String::new();
    head += "NRRD0004\n# Written by Polymix\ntype: unsigned char\ndimension: 3\nspace: right-anterior-superior\n";
    head += &format!("sizes: {} {} {}\n", dims[0], dims[1], dims[2]);
    head += &format!("space directions: {} {} {}\n", col(m.x_axis.truncate()), col(m.y_axis.truncate()), col(m.z_axis.truncate()));
    head += "kinds: domain domain domain\nencoding: gzip\n";
    head += &format!("space origin: {}\n", col(m.w_axis.truncate()));
    head += "Segmentation_MasterRepresentation:=Binary labelmap\nSegmentation_ContainedRepresentationNames:=Binary labelmap|\n";
    head += "Segmentation_ReferenceImageExtentOffset:=0 0 0\n";
    let extent = format!("0 {} 0 {} 0 {}", dims[0] - 1, dims[1] - 1, dims[2] - 1);
    for (n, s) in segments.iter().enumerate() {
        let [r, g, b] = s.colour.map(|c| c as f64 / 255.0);
        head += &format!("Segment{n}_ID:=Segment_{}\n", s.label);
        head += &format!("Segment{n}_Name:={}\n", s.name.replace(['\n', '\r'], " "));
        head += &format!("Segment{n}_NameAutoGenerated:=0\n");
        head += &format!("Segment{n}_Color:={r:.4} {g:.4} {b:.4}\n");
        head += &format!("Segment{n}_ColorAutoGenerated:=0\n");
        head += &format!("Segment{n}_LabelValue:={}\n", s.label);
        head += &format!("Segment{n}_Layer:=0\n");
        head += &format!("Segment{n}_Extent:={extent}\n");
        head += &format!("Segment{n}_Tags:=|\n");
    }
    head += "\n";
    let mut data = Vec::new();
    {
        let mut e = flate2::write::GzEncoder::new(&mut data, flate2::Compression::default());
        e.write_all(labels)?;
        e.finish()?;
    }
    let tmp = path.with_extension("partial");
    std::fs::write(&tmp, [head.as_bytes(), &data].concat()).with_context(|| format!("cannot write {}", path.display()))?;
    std::fs::rename(&tmp, path)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::model::{Kind, Voxels};
    use mm_volume::slice::load_raw;

    fn oblique() -> Geometry {
        let axes = [DVec3::new(0.8, 0.6, 0.0), DVec3::new(-0.6, 0.8, 0.0), DVec3::new(0.0, 0.0, -1.0)];
        Geometry::from_axes([4, 3, 2, 1], [0.5, 0.75, 2.0], axes, DVec3::new(10.0, -20.0, 30.0))
    }

    #[test]
    fn nifti_round_trip_through_the_reader() {
        let dir = std::env::temp_dir().join(format!("mmseg-io-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let labels: Vec<u8> = (0..24).map(|v| (v % 3) as u8).collect();
        let g = oblique();
        for name in ["seg.nii", "seg.nii.gz"] {
            let p = dir.join(name);
            write_nifti(&p, &labels, [4, 3, 2], &g, "Polymix test").unwrap();
            let gz = name.ends_with(".gz");
            let head = mm_volume::nifti::read_head(&p, gz, 352).unwrap();
            let mut h = mm_volume::nifti::parse(&head).unwrap();
            let v = mm_volume::nifti::volume(&p, gz, &h).unwrap();
            assert_eq!(v.kind, Kind::LabelMap);
            let read = v.geometry.as_ref().unwrap();
            for (r, e) in read.affine.iter().flatten().zip(g.affine.iter().flatten()) {
                assert!((r - e).abs() < 1e-4, "{:?} vs {:?}", read.affine, g.affine);
            }
            assert!(matches!(v.voxels, Voxels::Raw { .. }));
            assert_eq!(load_raw(&v, 100).unwrap().values, labels.iter().map(|&l| l as f32).collect::<Vec<_>>());
            // The qform alone gives the same grid.
            h.sform_code = 0;
            let (q, from) = h.affine();
            assert_eq!(from, "qform");
            let qm = Geometry::from_affine([4, 3, 2, 1], q);
            for (r, e) in qm.affine.iter().flatten().zip(g.affine.iter().flatten()) {
                assert!((r - e).abs() < 1e-4, "qform {:?} vs {:?}", qm.affine, g.affine);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn seg_nrrd_reads_back_as_a_segmentation() {
        let dir = std::env::temp_dir().join(format!("mmseg-nrrd-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let labels: Vec<u8> = (0..24).map(|v| (v % 3) as u8).collect();
        let p = dir.join("vessels.seg.nrrd");
        let segs = [SegmentInfo { name: "Aorta".into(), colour: [220, 60, 50], label: 1 }, SegmentInfo { name: "Branch".into(), colour: [80, 160, 230], label: 2 }];
        write_seg_nrrd(&p, &labels, [4, 3, 2], &oblique(), &segs).unwrap();
        let h = mm_volume::nrrd::read(&p, 1 << 20).unwrap();
        assert_eq!(h.segment_names(), vec!["Aorta".to_string(), "Branch".to_string()]);
        let v = mm_volume::nrrd::volume(&p, &h).unwrap();
        assert_eq!(v.kind, Kind::Segmentation);
        let read = v.geometry.as_ref().unwrap();
        for (r, e) in read.affine.iter().flatten().zip(oblique().affine.iter().flatten()) {
            assert!((r - e).abs() < 1e-9);
        }
        assert_eq!(load_raw(&v, 100).unwrap().values[5], 2.0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
