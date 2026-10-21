//! A mask to a closed surface mesh. The mask becomes a signed distance field (exact
//! Euclidean distances in millimetres, zero half-way between inside and outside voxels),
//! smoothed with a Gaussian, and its zero level is meshed by `mm_solid::mesh_implicit`:
//! the result is watertight, 2-manifold and outward oriented. The field lives on the
//! mask's bounding box only, so small structures in big volumes stay cheap.

use crate::mask::Mask;
use crate::ops::distance_sq;
use crate::smooth::convolve;
use anyhow::{bail, Context, Result};
use glam::DVec3;
use mm_core::{Aabb, Mesh};
use mm_volume::geometry::Geometry;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MeshParams {
    /// Gaussian smoothing of the surface (standard deviation, mm); 0 keeps the voxel steps.
    pub smoothing_mm: f64,
    /// Edge length of the meshing grid (mm); `None` uses the smallest voxel size.
    pub cell_mm: Option<f64>,
    /// Write coordinates in DICOM / LPS patient space (as 3D Slicer exports models) instead
    /// of RAS.
    pub lps: bool,
}

impl Default for MeshParams {
    fn default() -> Self {
        Self { smoothing_mm: 0.5, cell_mm: None, lps: true }
    }
}

/// Values on a box of voxels that may reach past the volume's edges (outside = not set).
struct Crop {
    /// Voxel index of the first sample (may be negative).
    origin: [i64; 3],
    dims: [usize; 3],
    values: Vec<f32>,
}

impl Crop {
    fn at(&self, i: usize, j: usize, k: usize) -> f32 {
        self.values[(k * self.dims[1] + j) * self.dims[0] + i]
    }

    /// Trilinear value at continuous voxel coordinates of the full grid; `far` outside.
    fn sample(&self, p: DVec3, far: f32) -> f32 {
        let q = p - DVec3::new(self.origin[0] as f64, self.origin[1] as f64, self.origin[2] as f64);
        let max = |a: usize| self.dims[a] as f64 - 1.0;
        if q.x < 0.0 || q.y < 0.0 || q.z < 0.0 || q.x > max(0) || q.y > max(1) || q.z > max(2) {
            return far;
        }
        let (i, j, k) = ((q.x.floor() as usize).min(self.dims[0].saturating_sub(2)), (q.y.floor() as usize).min(self.dims[1].saturating_sub(2)), (q.z.floor() as usize).min(self.dims[2].saturating_sub(2)));
        let (fx, fy, fz) = ((q.x - i as f64) as f32, (q.y - j as f64) as f32, (q.z - k as f64) as f32);
        let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
        let i1 = (i + 1).min(self.dims[0] - 1);
        let j1 = (j + 1).min(self.dims[1] - 1);
        let k1 = (k + 1).min(self.dims[2] - 1);
        let c00 = lerp(self.at(i, j, k), self.at(i1, j, k), fx);
        let c10 = lerp(self.at(i, j1, k), self.at(i1, j1, k), fx);
        let c01 = lerp(self.at(i, j, k1), self.at(i1, j, k1), fx);
        let c11 = lerp(self.at(i, j1, k1), self.at(i1, j1, k1), fx);
        lerp(lerp(c00, c10, fy), lerp(c01, c11, fy), fz)
    }
}

/// Separable Gaussian blur with per-axis standard deviation in voxels.
fn gaussian(values: &mut [f32], dims: [usize; 3], sigma: [f64; 3]) {
    let mut src = vec![0f32; values.len()];
    for axis in 0..3 {
        let s = sigma[axis];
        if s < 0.2 || dims[axis] < 2 {
            continue;
        }
        let r = (3.0 * s).ceil() as i64;
        let w: Vec<f32> = (-r..=r).map(|x| (-(x * x) as f64 / (2.0 * s * s)).exp() as f32).collect();
        let total: f32 = w.iter().sum();
        // Each sum from +0, then over the weights' total; the ends of lines repeat (the crop has
        // a margin, so they are far).
        convolve(values, &mut src, dims, axis, &w, 0.0, |a| a / total);
    }
}

/// The signed distance (mm, negative inside) of `mask` on its bounding box plus `pad`
/// voxels, smoothed.
fn smoothed_distance(mask: &Mask, spacing: [f64; 3], smoothing_mm: f64) -> Result<Crop> {
    let (lo, hi) = mask.bounds().context("the segment is empty")?;
    let sigma = spacing.map(|s| smoothing_mm / s);
    let pad = sigma.map(|s| (3.0 * s).ceil() as i64 + 3);
    let origin = [0, 1, 2].map(|a| lo[a] as i64 - pad[a]);
    let dims = [0, 1, 2].map(|a| (hi[a] - lo[a]) + 1 + 2 * pad[a] as usize);
    let n = dims[0] * dims[1] * dims[2];
    if n > 400_000_000 {
        bail!("the segment spans too many voxels to mesh ({} million)", n / 1_000_000);
    }
    let mut inside = vec![0u8; n];
    inside.par_chunks_mut(dims[0] * dims[1]).enumerate().for_each(|(z, p)| {
        let k = origin[2] + z as i64;
        if k < 0 || k >= mask.dims[2] as i64 {
            return;
        }
        for y in 0..dims[1] {
            let j = origin[1] + y as i64;
            if j < 0 || j >= mask.dims[1] as i64 {
                continue;
            }
            for x in 0..dims[0] {
                let i = origin[0] + x as i64;
                if i >= 0 && i < mask.dims[0] as i64 && mask.get(i as usize, j as usize, k as usize) {
                    p[y * dims[0] + x] = 1;
                }
            }
        }
    });
    let outside: Vec<u8> = inside.par_iter().map(|&v| 1 - v).collect();
    let to_inside = distance_sq(dims, &inside, spacing);
    let to_outside = distance_sq(dims, &outside, spacing);
    // Zero half-way between neighbouring inside and outside voxels.
    let half = 0.5 * spacing.iter().copied().fold(f64::INFINITY, f64::min);
    let mut values: Vec<f32> = (0..n).into_par_iter().map(|q| if inside[q] != 0 { -(to_outside[q].sqrt() - half) as f32 } else { (to_inside[q].sqrt() - half) as f32 }).collect();
    gaussian(&mut values, dims, sigma);
    Ok(Crop { origin, dims, values })
}

/// Mesh `mask` (on the grid `geometry`) as a closed surface. `progress` gets the fraction
/// done; returning `false` cancels.
pub fn mask_to_mesh(mask: &Mask, geometry: &Geometry, params: &MeshParams, progress: Option<&dyn Fn(f32) -> bool>) -> Result<Mesh> {
    let spacing = geometry.spacing;
    let crop = smoothed_distance(mask, spacing, params.smoothing_mm.max(0.0))?;
    let to_voxel = geometry.matrix().inverse();
    let h = params.cell_mm.unwrap_or_else(|| spacing.iter().copied().fold(f64::INFINITY, f64::min)).max(1e-3);
    // World bounds of the cropped box.
    let mut bounds = Aabb { min: DVec3::splat(f64::INFINITY), max: DVec3::splat(f64::NEG_INFINITY) };
    for c in 0..8 {
        let v = DVec3::new(
            (crop.origin[0] + if c & 1 > 0 { crop.dims[0] as i64 - 1 } else { 0 }) as f64,
            (crop.origin[1] + if c & 2 > 0 { crop.dims[1] as i64 - 1 } else { 0 }) as f64,
            (crop.origin[2] + if c & 4 > 0 { crop.dims[2] as i64 - 1 } else { 0 }) as f64,
        );
        let p = geometry.world(v);
        bounds.min = bounds.min.min(p);
        bounds.max = bounds.max.max(p);
    }
    let far = 1e3f32;
    let f = |p: DVec3| crop.sample(to_voxel.transform_point3(p), far) as f64;
    // A Gaussian-smoothed distance field changes by at most 1 mm per mm; trilinear
    // interpolation across anisotropic voxels can exceed that slightly.
    let mut mesh = mm_solid::mesh_implicit(f, 1.5, &bounds, h, progress).map_err(|e| anyhow::anyhow!("meshing failed: {e}"))?;
    if params.lps {
        // RAS → LPS is a half turn about the z axis: orientation is kept.
        for p in mesh.positions_mut() {
            *p = DVec3::new(-p.x, -p.y, p.z);
        }
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_is_the_old_one_to_the_bit() {
        use crate::reference::{mesh_gaussian, unit, DIMS};
        for &dims in &DIMS {
            let n = dims.iter().product::<usize>();
            let v: Vec<f32> = (0..n).map(|q| (unit(q as u64 + 77) * 20.0 - 10.0) as f32).collect();
            for sigma in [[0.0, 0.0, 0.0], [0.2, 0.5, 1.3], [2.0, 0.19, 0.7], [f64::NAN, 1.0, 9.0], [0.6, 0.6, 0.6]] {
                let (mut a, mut b) = (v.clone(), v.clone());
                mesh_gaussian(&mut a, dims, sigma);
                gaussian(&mut b, dims, sigma);
                assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()), "{dims:?} {sigma:?}");
            }
        }
    }

    #[test]
    fn a_ball_meshes_closed_with_the_right_size() {
        let n = 24;
        let mut m = Mask::new([n, n, 12]);
        // A ball of radius 8 mm on 1 × 1 × 2 mm voxels, placed obliquely.
        let c = DVec3::new(11.5, 11.5, 11.0);
        for k in 0..12 {
            for j in 0..n {
                for i in 0..n {
                    if (DVec3::new(i as f64, j as f64, k as f64 * 2.0) - c).length() <= 8.0 {
                        m.set(i, j, k, true);
                    }
                }
            }
        }
        let axes = [DVec3::new(1.0, 1.0, 0.0).normalize(), DVec3::new(-1.0, 1.0, 0.0).normalize(), DVec3::Z];
        let g = Geometry::from_axes([n, n, 12, 1], [1.0, 1.0, 2.0], axes, DVec3::new(5.0, -3.0, 10.0));
        let mesh = mask_to_mesh(&m, &g, &MeshParams { smoothing_mm: 0.8, cell_mm: Some(0.75), lps: false }, None).unwrap();
        assert!(mesh.triangle_count() > 500);
        assert_eq!(mm_solid::cleanup::bowtie_count(&mesh), 0);
        // Closed: every edge has two triangles.
        assert!(mesh.is_closed(), "open edges: {}", mesh.boundary_edge_ids().count());
        // Centred where the ball is, with about its radius.
        let centre_world = g.world(DVec3::new(11.5, 11.5, 5.5));
        let pts = mesh.positions();
        let mean = pts.iter().fold(DVec3::ZERO, |a, p| a + *p) / pts.len() as f64;
        assert!((mean - centre_world).length() < 0.5, "{mean} vs {centre_world}");
        let r = pts.iter().map(|p| (*p - centre_world).length()).sum::<f64>() / pts.len() as f64;
        assert!((r - 8.0).abs() < 0.8, "radius {r}");
        // LPS flips x and y.
        let lps = mask_to_mesh(&m, &g, &MeshParams { smoothing_mm: 0.8, cell_mm: Some(0.75), lps: true }, None).unwrap();
        let mean2 = lps.positions().iter().fold(DVec3::ZERO, |a, p| a + *p) / lps.vertex_count() as f64;
        assert!((mean2 - DVec3::new(-mean.x, -mean.y, mean.z)).length() < 0.5);
        assert!(mask_to_mesh(&Mask::new([4, 4, 4]), &g, &MeshParams::default(), None).is_err());
    }
}
