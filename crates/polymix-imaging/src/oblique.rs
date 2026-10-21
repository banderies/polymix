//! Oblique planes: a slice view's plane turned away from the voxel grid, resampled from the
//! volume. Each plane is a frame in RAS millimetres — right and down on the screen, and its
//! normal (the next slice) — through the cursor; the image covers the volume's bounding sphere
//! and stays put on the screen while the plane moves along its normal.

use crate::layers::SlabMode;
use crate::PLANES;
use glam::{DMat4, DVec3};
use mm_volume::load::LoadedVolume;
use mm_volume::slice::Plane;
use rayon::prelude::*;
use std::ops::Range;

/// A plane's frame in RAS: right and down on the screen, and its normal (unit vectors).
pub type Frame = [DVec3; 3];

/// The radiological frames of the three planes (as the views along the voxel grid show
/// them): axial seen from the feet, sagittal from the patient's right... with the patient's
/// right on the screen's left.
pub fn standard_frame(p: Plane) -> Frame {
    match p {
        Plane::Axial => [DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z],
        Plane::Sagittal => [DVec3::NEG_Y, DVec3::NEG_Z, DVec3::X],
        Plane::Coronal => [DVec3::NEG_X, DVec3::NEG_Z, DVec3::Y],
    }
}

/// The three views' frames turned with a vessel running along `tangent`: the view whose
/// standard plane is most nearly across the vessel shows its cross-section, the other two
/// contain it. Each keeps its standard view's sides (no mirror images), as near its standard
/// orientation as the vessel allows. Also returns which view is across.
pub fn along_vessel(tangent: DVec3) -> ([Frame; 3], usize) {
    let t = tangent.normalize_or(DVec3::Z);
    let std = PLANES.map(standard_frame);
    let k = (0..3).max_by(|&a, &b| std[a][2].dot(t).abs().total_cmp(&std[b][2].dot(t).abs())).unwrap_or(0);
    // `want` without its part along `n` (another direction if nothing is left).
    let flat = |want: DVec3, n: DVec3| {
        let d = want - n * want.dot(n);
        if d.length() > 1e-6 { d.normalize() } else { n.any_orthonormal_vector() }
    };
    // A frame of normal `n`, down near the standard view `j`'s, its sides as that view's.
    let frame = |j: usize, n: DVec3| {
        let down = flat(std[j][1], n);
        let hand = std[j][0].dot(std[j][1].cross(std[j][2])).signum();
        [down.cross(n) * hand, down, n]
    };
    let mut out = std;
    let n_across = if std[k][2].dot(t) < 0.0 { -t } else { t };
    out[k] = frame(k, n_across);
    let others: Vec<usize> = (0..3).filter(|&j| j != k).collect();
    let n1 = flat(std[others[0]][2], t);
    let n1 = if n1.dot(std[others[0]][2]) < 0.0 { -n1 } else { n1 };
    let n2 = t.cross(n1).normalize();
    let n2 = if n2.dot(std[others[1]][2]) < 0.0 { -n2 } else { n2 };
    out[others[0]] = frame(others[0], n1);
    out[others[1]] = frame(others[1], n2);
    (out, k)
}

/// Rotate `f` by `angle` (radians) about the axis `axis` (unit).
pub fn rotated(f: Frame, axis: DVec3, angle: f64) -> Frame {
    let q = glam::DQuat::from_axis_angle(axis, angle);
    f.map(|v| (q * v).normalize())
}

/// Where an oblique plane's image lies: pixel (u, v)'s centre is at
/// `origin + (u + 0.5) · right · px + (v + 0.5) · down · px`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ObliqueGrid {
    pub frame: Frame,
    pub origin: DVec3,
    /// Millimetres per pixel.
    pub px: f64,
    pub size: [usize; 2],
    /// Voxel coordinates → RAS, and back.
    pub to_world: DMat4,
    pub to_voxel: DMat4,
}

/// Largest image side (pixels).
const MAX_SIDE: usize = 1024;

impl ObliqueGrid {
    /// The plane of `frame` through `cursor` (voxel coordinates), its image covering the
    /// volume's bounding sphere.
    pub fn new(vol: &LoadedVolume, frame: Frame, cursor: DVec3) -> Self {
        let to_world = vol.geometry.matrix();
        let to_voxel = to_world.inverse();
        let d = vol.dims.map(|n| n as f64 - 1.0);
        let corners: Vec<DVec3> = (0..8).map(|c| to_world.transform_point3(DVec3::new(if c & 1 > 0 { d[0] } else { 0.0 }, if c & 2 > 0 { d[1] } else { 0.0 }, if c & 4 > 0 { d[2] } else { 0.0 }))).collect();
        let centre = corners.iter().copied().sum::<DVec3>() / 8.0;
        let radius = corners.iter().map(|c| c.distance(centre)).fold(0.0, f64::max) + vol.geometry.spacing.iter().cloned().fold(0.0, f64::max);
        let mut px = vol.geometry.spacing.iter().cloned().fold(f64::INFINITY, f64::min);
        if 2.0 * radius / px > MAX_SIDE as f64 {
            px = 2.0 * radius / MAX_SIDE as f64;
        }
        let n = (2.0 * radius / px).ceil() as usize;
        // The image centred on the volume's centre as seen in the plane through the cursor.
        let c = to_world.transform_point3(cursor);
        let on_plane = centre - frame[2] * (centre - c).dot(frame[2]);
        let origin = on_plane - (frame[0] + frame[1]) * (n as f64 * 0.5 * px);
        Self { frame, origin, px, size: [n, n], to_world, to_voxel }
    }

    /// RAS position of pixel coordinates (continuous: the pixel's centre is at + 0.5).
    pub fn world(&self, u: f64, v: f64) -> DVec3 {
        self.origin + self.frame[0] * (u * self.px) + self.frame[1] * (v * self.px)
    }

    /// Pixel coordinates of a RAS position, and its distance from the plane in pixels.
    pub fn pixel(&self, w: DVec3) -> (f64, f64, f64) {
        let d = w - self.origin;
        (d.dot(self.frame[0]) / self.px, d.dot(self.frame[1]) / self.px, d.dot(self.frame[2]) / self.px)
    }

    /// The image's values, row by row: trilinear (`smooth`) or nearest, over a slab of
    /// `half_mm` on each side combined as `mode`; 0 outside the volume.
    pub fn sample(&self, vol: &LoadedVolume, values: &[f32], smooth: bool, half_mm: f64, mode: SlabMode) -> Vec<f32> {
        let [w, h] = self.size;
        let steps = self.slab_steps(half_mm, mode);
        (0..w * h)
            .into_par_iter()
            .map(|p| {
                let base = self.world((p % w) as f64 + 0.5, (p / w) as f64 + 0.5);
                let at = |s: i64| {
                    let v = self.to_voxel.transform_point3(base + self.frame[2] * (s as f64 * self.px));
                    if smooth { trilinear(vol.dims, values, v) } else { nearest(vol.dims, values, v) }
                };
                let mut acc = at(0);
                for s in (-steps..=steps).filter(|&s| s != 0) {
                    let x = at(s);
                    acc = match mode {
                        SlabMode::Max => acc.max(x),
                        SlabMode::Min => acc.min(x),
                        _ => acc + x,
                    };
                }
                if mode == SlabMode::Mean && steps > 0 {
                    acc /= (2 * steps + 1) as f32;
                }
                acc
            })
            .collect()
    }

    /// Samples along the normal on each side of the plane that [`Self::sample`] combines for a
    /// slab of `half_mm` on each side (0 or less: the plane alone). The values depend on the
    /// slab through this alone.
    pub(crate) fn slab_steps(&self, half_mm: f64, mode: SlabMode) -> i64 {
        if mode == SlabMode::Off {
            0
        } else {
            (half_mm / self.px).round() as i64
        }
    }

    /// The labels under each pixel (nearest voxel).
    pub fn sample_labels(&self, vol: &LoadedVolume, labels: &[u8]) -> Vec<u8> {
        let [w, h] = self.size;
        (0..w * h).into_par_iter().map(|p| self.label_at(vol.dims, labels, p % w, p / w)).collect()
    }

    /// Re-sample the labels of pixels `x` × `y` into `out` (the whole image's labels, row by
    /// row, as `sample_labels` makes them), leaving the other pixels as they are.
    pub(crate) fn sample_labels_in(&self, vol: &LoadedVolume, labels: &[u8], out: &mut [u8], x: Range<usize>, y: Range<usize>) {
        // On this thread: even a whole image of the largest size is a few milliseconds, which
        // tasks on a busy machine would take longer to finish (see slice_view::PARALLEL_WORK).
        let w = self.size[0];
        for (r, row) in out[y.start * w..y.end * w].chunks_mut(w).enumerate() {
            for (i, l) in row[x.clone()].iter_mut().enumerate() {
                *l = self.label_at(vol.dims, labels, x.start + i, y.start + r);
            }
        }
    }

    /// The label under pixel (`x`, `y`): its centre's nearest voxel's, 0 outside the volume.
    /// The whole image and its parts are sampled by this alone, so they agree.
    fn label_at(&self, dims: [usize; 3], labels: &[u8], x: usize, y: usize) -> u8 {
        let v = self.to_voxel.transform_point3(self.world(x as f64 + 0.5, y as f64 + 0.5));
        let i = [0, 1, 2].map(|a| v[a].round());
        if (0..3).any(|a| i[a] < 0.0 || i[a] >= dims[a] as f64) {
            return 0;
        }
        labels[(i[2] as usize * dims[1] + i[1] as usize) * dims[0] + i[0] as usize]
    }
}

fn nearest(dims: [usize; 3], values: &[f32], v: DVec3) -> f32 {
    let i = [0, 1, 2].map(|a| v[a].round());
    if (0..3).any(|a| i[a] < 0.0 || i[a] >= dims[a] as f64) {
        return 0.0;
    }
    values[(i[2] as usize * dims[1] + i[1] as usize) * dims[0] + i[0] as usize]
}

fn trilinear(dims: [usize; 3], values: &[f32], v: DVec3) -> f32 {
    if (0..3).any(|a| v[a] < -0.5 || v[a] > dims[a] as f64 - 0.5) {
        return 0.0;
    }
    let c = [0, 1, 2].map(|a| v[a].clamp(0.0, (dims[a] - 1) as f64));
    let i0 = c.map(|x| x.floor() as usize);
    let i1 = [0, 1, 2].map(|a| (i0[a] + 1).min(dims[a] - 1));
    let f = [0, 1, 2].map(|a| (c[a] - i0[a] as f64) as f32);
    let get = |i: usize, j: usize, k: usize| values[(k * dims[1] + j) * dims[0] + i];
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    let x00 = lerp(get(i0[0], i0[1], i0[2]), get(i1[0], i0[1], i0[2]), f[0]);
    let x10 = lerp(get(i0[0], i1[1], i0[2]), get(i1[0], i1[1], i0[2]), f[0]);
    let x01 = lerp(get(i0[0], i0[1], i1[2]), get(i1[0], i0[1], i1[2]), f[0]);
    let x11 = lerp(get(i0[0], i1[1], i1[2]), get(i1[0], i1[1], i1[2]), f[0]);
    lerp(lerp(x00, x10, f[1]), lerp(x01, x11, f[1]), f[2])
}

/// The patient direction a RAS direction points to most: R, L, A, P, S or I.
pub fn direction_letter(d: DVec3) -> &'static str {
    let a = d.abs();
    if a.x >= a.y && a.x >= a.z {
        if d.x >= 0.0 { "R" } else { "L" }
    } else if a.y >= a.z {
        if d.y >= 0.0 { "A" } else { "P" }
    } else if d.z >= 0.0 {
        "S"
    } else {
        "I"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::geometry::Geometry;

    fn vol() -> LoadedVolume {
        // Values are i (the x voxel index); 1 mm voxels, origin at 0.
        let g = Geometry::from_axes([10, 10, 10, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        LoadedVolume::from_values("t", [10, 10, 10], (0..1000).map(|n| (n % 10) as f32).collect(), g)
    }

    #[test]
    fn frames_are_orthonormal() {
        let check = |f: Frame| {
            for v in f {
                assert!((v.length() - 1.0).abs() < 1e-9);
            }
            assert!(f[0].dot(f[1]).abs() < 1e-9 && f[0].dot(f[2]).abs() < 1e-9 && f[1].dot(f[2]).abs() < 1e-9);
            // The normal is the screen's (towards or away from the viewer: it is the direction
            // of the next slice, as in the views along the grid).
            assert!((f[0].cross(f[1]).dot(f[2]).abs() - 1.0).abs() < 1e-9);
        };
        for p in Plane::ALL {
            check(standard_frame(p));
        }
        // A vessel running up (the axial view across it), and one running left to right (the
        // sagittal view across it), either way along.
        for (t, across) in [(DVec3::new(0.3, 0.2, 1.0), 0), (DVec3::new(-0.3, 0.2, -1.0), 0), (DVec3::new(1.0, 0.1, 0.2), 1), (DVec3::new(-1.0, 0.1, 0.2), 1)] {
            let (frames, k) = along_vessel(t);
            assert_eq!(k, across);
            frames.iter().for_each(|f| check(*f));
            assert!(frames[k][2].cross(t).length() < 1e-9, "across the vessel");
            for (j, f) in frames.iter().enumerate().filter(|(j, _)| *j != k) {
                assert!(f[2].dot(t).abs() < 1e-9, "the others contain it");
                let s = standard_frame(PLANES[j]);
                assert!(f[2].dot(s[2]) > 0.0 && f[1].dot(s[1]) > 0.0, "near the standard view");
            }
            // No mirror images: each view's sides as its standard view's.
            for (j, f) in frames.iter().enumerate() {
                let s = standard_frame(PLANES[j]);
                assert_eq!(f[0].dot(f[1].cross(f[2])).signum(), s[0].dot(s[1].cross(s[2])).signum());
            }
            assert!(frames[0][2].dot(frames[1][2]).abs() < 1e-9 && frames[1][2].dot(frames[2][2]).abs() < 1e-9);
        }
        check(rotated(standard_frame(Plane::Coronal), DVec3::Z, 0.7));
    }

    #[test]
    fn an_axial_plane_samples_the_grid() {
        let v = vol();
        let g = ObliqueGrid::new(&v, standard_frame(Plane::Axial), DVec3::new(4.0, 4.0, 3.0));
        // The cursor's voxel lies on the plane, and its value is found there.
        let (u, w, d) = g.pixel(v.geometry.world(DVec3::new(4.0, 4.0, 3.0)));
        assert!(d.abs() < 1e-9);
        let img = g.sample(&v, &v.values, true, 0.0, SlabMode::Off);
        let at = img[(w.floor() as usize) * g.size[0] + u.floor() as usize];
        assert!((at - 4.0).abs() < 0.51, "{at}");
        // Right on the screen is the patient's left (−x): values fall to the right.
        let row = w.floor() as usize * g.size[0];
        assert!(img[row + u.floor() as usize + 2] < img[row + u.floor() as usize]);
        // Outside the volume: 0.
        assert_eq!(img[0], 0.0);
    }

    #[test]
    fn labels_sampled_in_parts_match_the_whole() {
        // An anisotropic, flipped grid and a turned plane, so many pixels sample near voxel
        // boundaries; labels are a hash of the voxel.
        let g = Geometry::from_axes([23, 17, 13, 1], [0.7, 0.9, 1.3], [DVec3::NEG_X, DVec3::NEG_Y, DVec3::Z], DVec3::new(5.0, 3.0, -4.0));
        let v = LoadedVolume::from_values("t", [23, 17, 13], vec![0.0; 23 * 17 * 13], g);
        let mut labels: Vec<u8> = (0..v.values.len()).map(|n| ((n as u32).wrapping_mul(2654435761) >> 27) as u8).collect();
        let frame = rotated(standard_frame(Plane::Coronal), DVec3::new(0.4, 1.0, 0.3).normalize(), 0.6);
        let ob = ObliqueGrid::new(&v, frame, DVec3::new(11.0, 8.0, 6.0));
        let [w, h] = ob.size;
        // The whole image as it was sampled before (verbatim).
        let before: Vec<u8> = (0..w * h)
            .map(|p| {
                let x = ob.to_voxel.transform_point3(ob.world((p % w) as f64 + 0.5, (p / w) as f64 + 0.5));
                let i = [0, 1, 2].map(|a| x[a].round());
                if (0..3).any(|a| i[a] < 0.0 || i[a] >= v.dims[a] as f64) {
                    return 0;
                }
                labels[(i[2] as usize * v.dims[1] + i[1] as usize) * v.dims[0] + i[0] as usize]
            })
            .collect();
        let mut img = ob.sample_labels(&v, &labels);
        assert_eq!(img, before);
        // Change the labels, then re-sample parts: the image is whole sampling's wherever
        // the parts reach, and as it was elsewhere.
        labels.iter_mut().for_each(|l| *l = l.wrapping_add(1));
        let whole = ob.sample_labels(&v, &labels);
        for (x, y) in [(3..9, 0..1), (w - 1..w, 5..h), (4..w - 2, 7..19), (0..w, 0..h)] {
            let old = img.clone();
            ob.sample_labels_in(&v, &labels, &mut img, x.clone(), y.clone());
            for p in 0..w * h {
                let inside = x.contains(&(p % w)) && y.contains(&(p / w));
                assert_eq!(img[p], if inside { whole[p] } else { old[p] });
            }
        }
    }
}
