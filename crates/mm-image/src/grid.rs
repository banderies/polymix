//! Changing an image's grid without resampling: crop to a box, pad, reorient to standard axes
//! (the voxels are reordered; the anatomy stays where it is), and fixes for a wrong header
//! (mirror the image across a patient axis, or swap two axes of the header).

use anyhow::{bail, Result};
use glam::{DMat4, DVec3, DVec4};
use mm_volume::geometry::{axcodes, Geometry};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;

fn world_of(g: &Geometry, ijk: [f64; 3]) -> DVec3 {
    g.world(DVec3::new(ijk[0], ijk[1], ijk[2]))
}

/// The voxels `lo..=hi` (per axis) as a new image, placed where they were.
pub fn crop(vol: &LoadedVolume, lo: [usize; 3], hi: [usize; 3]) -> Result<LoadedVolume> {
    for a in 0..3 {
        if lo[a] > hi[a] || hi[a] >= vol.dims[a] {
            bail!("the box is outside the image");
        }
    }
    let dims = [0, 1, 2].map(|a| hi[a] - lo[a] + 1);
    let mut values = vec![0.0f32; dims[0] * dims[1] * dims[2]];
    values.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..dims[1] {
            let src = vol.index(lo[0], lo[1] + y, lo[2] + z);
            plane[y * dims[0]..(y + 1) * dims[0]].copy_from_slice(&vol.values[src..src + dims[0]]);
        }
    });
    let g = shifted(&vol.geometry, dims, [lo[0] as f64, lo[1] as f64, lo[2] as f64]);
    let mut out = crate::image::on_grid(&vol.name, g, values);
    out.invert = vol.invert;
    Ok(out)
}

/// The grid `g` resized to `dims`, starting at its voxel `start` (may be negative).
fn shifted(g: &Geometry, dims: [usize; 3], start: [f64; 3]) -> Geometry {
    let m = g.matrix();
    let origin = world_of(g, start);
    let m2 = DMat4::from_cols(m.x_axis, m.y_axis, m.z_axis, origin.extend(1.0));
    Geometry::from_affine([dims[0], dims[1], dims[2], 1], m2)
}

/// Add `before` / `after` voxels on each axis, filled with `value`.
pub fn pad(vol: &LoadedVolume, before: [usize; 3], after: [usize; 3], value: f32) -> Result<LoadedVolume> {
    let dims = [0, 1, 2].map(|a| vol.dims[a] + before[a] + after[a]);
    if dims.iter().product::<usize>() > 1_000_000_000 {
        bail!("the padded image would be too large");
    }
    let mut values = vec![value; dims[0] * dims[1] * dims[2]];
    values.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(z, plane)| {
        if z < before[2] || z >= before[2] + vol.dims[2] {
            return;
        }
        for y in before[1]..before[1] + vol.dims[1] {
            let src = vol.index(0, y - before[1], z - before[2]);
            plane[y * dims[0] + before[0]..y * dims[0] + before[0] + vol.dims[0]].copy_from_slice(&vol.values[src..src + vol.dims[0]]);
        }
    });
    let g = shifted(&vol.geometry, dims, [-(before[0] as f64), -(before[1] as f64), -(before[2] as f64)]);
    let mut out = crate::image::on_grid(&vol.name, g, values);
    out.invert = vol.invert;
    Ok(out)
}

/// The box (inclusive voxel ranges) around the object found by [`crate::auto_mask`], grown by
/// `margin_mm` and kept inside the image.
pub fn auto_box(vol: &LoadedVolume, margin_mm: f64) -> Option<([usize; 3], [usize; 3])> {
    let m = mm_segment::Mask { dims: vol.dims, data: crate::auto_mask(vol) };
    let (lo, hi) = m.bounds()?;
    let pad = vol.geometry.spacing.map(|s| (margin_mm / s.max(1e-9)).ceil() as usize);
    Some(([0, 1, 2].map(|a| lo[a].saturating_sub(pad[a])), [0, 1, 2].map(|a| (hi[a] + pad[a]).min(vol.dims[a] - 1))))
}

/// Reorder the voxels so the axes point as `target` says (three letters, one of R/L, A/P,
/// S/I each, e.g. "RAS" or "LPS"); every voxel keeps its place in the patient.
pub fn reorient(vol: &LoadedVolume, target: &str) -> Result<LoadedVolume> {
    let t: Vec<char> = target.to_uppercase().chars().collect();
    let world = |c: char| -> Option<(usize, f64)> {
        Some(match c {
            'R' => (0, 1.0),
            'L' => (0, -1.0),
            'A' => (1, 1.0),
            'P' => (1, -1.0),
            'S' => (2, 1.0),
            'I' => (2, -1.0),
            _ => return None,
        })
    };
    let Some(want) = (t.len() == 3).then(|| t.iter().map(|&c| world(c)).collect::<Option<Vec<_>>>()).flatten() else {
        bail!("orientation must be three letters like RAS or LPS");
    };
    if want.iter().map(|w| w.0).collect::<std::collections::BTreeSet<_>>().len() != 3 {
        bail!("orientation {target} names an axis twice");
    }
    let of_world = mm_volume::slice::voxel_axis_of_world(&vol.geometry);
    let dirs = vol.geometry.directions();
    // New axis a takes old axis src[a], reversed when it points the other way.
    let src: [usize; 3] = [0, 1, 2].map(|a| of_world[want[a].0]);
    let flip: [bool; 3] = [0, 1, 2].map(|a| dirs[src[a]][want[a].0] * want[a].1 < 0.0);
    let dims = [0, 1, 2].map(|a| vol.dims[src[a]]);
    let mut values = vec![0.0f32; vol.values.len()];
    values.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(z, plane)| {
        let mut old = [0usize; 3];
        for y in 0..dims[1] {
            for x in 0..dims[0] {
                for (a, &n) in [x, y, z].iter().enumerate() {
                    old[src[a]] = if flip[a] { dims[a] - 1 - n } else { n };
                }
                plane[y * dims[0] + x] = vol.get(old[0], old[1], old[2]);
            }
        }
    });
    let m = vol.geometry.matrix();
    let cols = [m.x_axis, m.y_axis, m.z_axis];
    let col = |a: usize| if flip[a] { -cols[src[a]] } else { cols[src[a]] };
    let mut first = [0.0f64; 3];
    for a in 0..3 {
        first[src[a]] = if flip[a] { (dims[a] - 1) as f64 } else { 0.0 };
    }
    let origin = world_of(&vol.geometry, first);
    let g = Geometry::from_affine([dims[0], dims[1], dims[2], 1], DMat4::from_cols(col(0), col(1), col(2), origin.extend(1.0)));
    let mut out = crate::image::on_grid(&vol.name, g, values);
    out.invert = vol.invert;
    Ok(out)
}

/// The image mirrored across the patient axis `world` (0 = left–right, 1 = front–back,
/// 2 = up–down) about its centre: for a scan whose header has a side the wrong way round.
pub fn mirror(vol: &LoadedVolume, world: usize) -> Result<LoadedVolume> {
    if world > 2 {
        bail!("no such axis");
    }
    let a = mm_volume::slice::voxel_axis_of_world(&vol.geometry)[world];
    let d = vol.dims;
    let mut values = vec![0.0f32; vol.values.len()];
    values.par_chunks_mut(d[0] * d[1]).enumerate().for_each(|(z, plane)| {
        for y in 0..d[1] {
            for x in 0..d[0] {
                let mut p = [x, y, z];
                p[a] = d[a] - 1 - p[a];
                plane[y * d[0] + x] = vol.get(p[0], p[1], p[2]);
            }
        }
    });
    Ok(crate::image::with_values(vol, &vol.name, values))
}

/// Swap two voxel axes in the header (the data stay; what they are labelled changes): for a
/// scan whose header calls, say, coronal slices axial.
pub fn swap_header_axes(vol: &LoadedVolume, a: usize, b: usize) -> Result<LoadedVolume> {
    if a > 2 || b > 2 || a == b {
        bail!("pick two different axes");
    }
    let g = &vol.geometry;
    let dirs = g.directions();
    let mut new_dirs = dirs;
    new_dirs.swap(a, b);
    let mut spacing = g.spacing;
    spacing.swap(a, b);
    // Keep the grid's centre where it was.
    let centre_ijk = vol.dims.map(|n| (n as f64 - 1.0) * 0.5);
    let centre = world_of(g, centre_ijk);
    let m = DMat4::from_cols(
        (new_dirs[0] * spacing[0]).extend(0.0),
        (new_dirs[1] * spacing[1]).extend(0.0),
        (new_dirs[2] * spacing[2]).extend(0.0),
        DVec4::new(0.0, 0.0, 0.0, 1.0),
    );
    let origin = centre - m.transform_point3(DVec3::new(centre_ijk[0], centre_ijk[1], centre_ijk[2]));
    let m = DMat4::from_cols(m.x_axis, m.y_axis, m.z_axis, origin.extend(1.0));
    let mut out = crate::image::on_grid(&vol.name, Geometry::from_affine([vol.dims[0], vol.dims[1], vol.dims[2], 1], m), vol.values.clone());
    out.invert = vol.invert;
    Ok(out)
}

/// The orientation code of an image's axes ("LPS", "RAS", …).
pub fn orientation(vol: &LoadedVolume) -> String {
    axcodes(vol.geometry.directions())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(dims: [usize; 3], axes: [DVec3; 3]) -> LoadedVolume {
        let values = (0..dims.iter().product::<usize>()).map(|v| v as f32).collect();
        LoadedVolume::from_values("r", dims, values, Geometry::from_axes([dims[0], dims[1], dims[2], 1], [0.5, 1.0, 2.0], axes, DVec3::new(10.0, 20.0, 30.0)))
    }

    #[test]
    fn crop_and_pad_keep_positions() {
        let v = ramp([6, 5, 4], [DVec3::X, DVec3::Y, DVec3::Z]);
        let c = crop(&v, [1, 2, 1], [4, 3, 2]).unwrap();
        assert_eq!(c.dims, [4, 2, 2]);
        assert_eq!(c.get(0, 0, 0), v.get(1, 2, 1));
        assert!((c.geometry.world(DVec3::ZERO) - v.geometry.world(DVec3::new(1.0, 2.0, 1.0))).length() < 1e-9);
        let p = pad(&c, [1, 0, 2], [0, 3, 1], -1.0).unwrap();
        assert_eq!(p.dims, [5, 5, 5]);
        assert_eq!(p.get(1, 0, 2), c.get(0, 0, 0));
        assert_eq!(p.get(0, 0, 0), -1.0);
        assert!((p.geometry.world(DVec3::new(1.0, 0.0, 2.0)) - c.geometry.world(DVec3::ZERO)).length() < 1e-9);
        assert!(crop(&v, [0, 0, 0], [6, 0, 0]).is_err());
    }

    #[test]
    fn reorient_keeps_every_voxel_in_place() {
        // A sagittal PIR grid → RAS and LPS.
        let v = ramp([4, 3, 5], [-DVec3::Y, -DVec3::Z, DVec3::X]);
        for target in ["RAS", "LPS", "SAR"] {
            let r = reorient(&v, target).unwrap();
            assert_eq!(orientation(&r), target);
            assert_eq!(r.voxel_count(), v.voxel_count());
            // Same value at the same place.
            for (i, j, k) in [(0, 0, 0), (3, 2, 4), (1, 2, 3)] {
                let p = v.geometry.world(DVec3::new(i as f64, j as f64, k as f64));
                let q = r.geometry.matrix().inverse().transform_point3(p).round();
                assert_eq!(r.get(q.x as usize, q.y as usize, q.z as usize), v.get(i, j, k), "{target}");
            }
        }
        assert!(reorient(&v, "RRS").is_err());
    }

    #[test]
    fn mirror_and_swap() {
        let v = ramp([4, 3, 2], [DVec3::X, DVec3::Y, DVec3::Z]);
        let m = mirror(&v, 0).unwrap();
        assert_eq!(m.get(0, 1, 1), v.get(3, 1, 1));
        let s = swap_header_axes(&v, 0, 1).unwrap();
        assert_eq!(orientation(&s), "ARS");
        assert_eq!(s.values, v.values);
        let centre = |g: &Geometry| g.world(DVec3::new(1.5, 1.0, 0.5));
        assert!((centre(&s.geometry) - centre(&v.geometry)).length() < 1e-9);
    }

    #[test]
    fn auto_box_finds_the_object() {
        let dims = [20, 20, 20];
        let values = (0..8000).map(|n| {
            let (i, j, k) = (n % 20, (n / 20) % 20, n / 400);
            if (5..12).contains(&i) && (6..15).contains(&j) && (3..9).contains(&k) { 100.0 } else { 1.0 }
        }).collect();
        let v = LoadedVolume::from_values("b", dims, values, Geometry::from_axes([20, 20, 20, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO));
        assert_eq!(auto_box(&v, 2.0), Some(([3, 4, 1], [13, 16, 10])));
    }
}
