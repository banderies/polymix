//! Segment statistics, as 3D Slicer's Segment Statistics module reports them: size, the
//! intensities of the volume under the segment, where it is, and its shape. The shape
//! measures follow the definitions of ITK's label shape statistics, which Slicer's labelmap
//! statistics use (Lehmann 2007, "Label object representation and manipulation with ITK",
//! Insight Journal), measured as follows.
//!
//! - Surface area: the voxels' exposed faces, each weighted by the cosine of the angle between
//!   it and the surface's normal there (a normal-based voxel face estimator, among those
//!   reviewed by Windreich, Kiryati & Lohmann 2003, "Voxel-based surface area estimation: from
//!   theory to practice", Pattern Recognit. 36(11):2531–2541). A surface of normal n shows
//!   faces across each voxel axis a over |n_a| of its area, so the faces alone overestimate by
//!   |n_x| + |n_y| + |n_z| (a ball's area by half), and weighted by |n_a| they add up to the
//!   area. The normal at a face is the gradient, at the face's centre, of the segment blurred
//!   by a Gaussian of 1.5 voxels, summed over the voxels within 4.5 voxels of it (no field
//!   over the whole box is needed). Balls of radius 3 to 100 voxels come out within 1%; flat
//!   faces exactly, but the normal turns at a box's edges and corners, which rounds them (a
//!   voxel box reads 5–7% low, 13% for one 3 voxels thin; an oblique rod of radius 1.5–4
//!   voxels with flat ends 6–1% low). A smaller Gaussian biases the normals toward the face's
//!   own axis (balls 3% high); marching tetrahedra on the voxels' signed distance, unsmoothed,
//!   keep the voxel terraces (balls 13% high).
//! - Feret diameter: the largest distance between two voxel centres, which lies between two
//!   vertices of their convex hull. The candidates are the voxels at an end of their row, of
//!   their column and of their pillar (every hull vertex is), compared by pairs when there are
//!   up to a few thousand; past that, the extreme candidates along 1024 directions are compared
//!   by pairs and the best pair is improved by farthest-point steps until it stops growing (an
//!   error of at most D (1 − cos θ) for the directions' largest gap θ, a fraction of a voxel,
//!   and in practice none).
//! - Principal axes: from the second moments of the segment as a solid (the voxel centres'
//!   covariance plus each voxel's own, a twelfth of its edge vectors' outer products), in
//!   millimetres through the volume's axes; each length is the diameter of the solid ellipsoid
//!   with the same moments, 2 √(5 λ) for an eigenvalue λ (exactly the diameter of a ball).
//! - Elongation: ITK's and Slicer's, √(λ_max / λ_mid): the ratio of the two longest principal
//!   axes, at least 1 (4 for a 40 × 10 × 10 box).
//! - Roundness: ITK's and Slicer's, the area of a sphere of the segment's volume over the
//!   surface area (1 for a ball, less for other shapes).

use crate::mask::Mask;
use glam::{DMat3, DVec3};
use mm_volume::load::LoadedVolume;
use rayon::prelude::*;
use std::f64::consts::PI;

/// Statistics of a segment over a volume.
#[derive(Clone, Debug, PartialEq)]
pub struct SegmentStats {
    pub voxels: usize,
    pub volume_mm3: f64,
    /// Area (mm²) of the smooth surface through the voxels' boundary: their exposed faces
    /// weighted by the cosine to the local surface normal (see the module notes); within 1% of
    /// a ball's area.
    pub surface_mm2: f64,
    /// Mean of the (finite) volume values under the segment; NaN when there are none.
    pub mean: f64,
    /// Population standard deviation of the values (divided by their count).
    pub sd: f64,
    pub min: f32,
    pub max: f32,
    /// The middle value (the mean of the two middle ones for an even count).
    pub median: f32,
    /// Centre of mass (unweighted voxel centres) in RAS mm.
    pub centroid_ras: [f64; 3],
    /// Inclusive voxel bounds.
    pub bounds: ([usize; 3], [usize; 3]),
    /// Largest distance between two of its (surface) voxel centres, mm.
    pub feret_mm: f64,
    /// Roundness as in Slicer: the area of a sphere of the same volume over the surface area
    /// (1 for a sphere).
    pub roundness: f64,
    /// Elongation as in Slicer (ITK): the ratio of the two longest principal axes,
    /// √(λ_max / λ_mid) ≥ 1.
    pub elongation: f64,
    /// Principal axis lengths (mm), from the second moments of the voxels as a solid (sorted,
    /// largest first): the diameters 2 √(5 λ) of the ellipsoid with the same moments.
    pub principal_mm: [f64; 3],
}

/// Sums over the segment's voxels: count, first and second moments of the voxel positions
/// (about a fixed point), and the finite volume values.
#[derive(Default)]
struct Sums {
    n: usize,
    s: [f64; 3],
    /// xx, yy, zz, xy, xz, yz.
    ss: [f64; 6],
    values: Vec<f32>,
}

impl Sums {
    fn add(mut self, o: Sums) -> Sums {
        self.n += o.n;
        (0..3).for_each(|a| self.s[a] += o.s[a]);
        (0..6).for_each(|a| self.ss[a] += o.ss[a]);
        self.values.extend(o.values);
        self
    }
}

/// Statistics of the voxels of `mask` over `vol` (`None` when the mask is empty).
pub fn segment_stats(mask: &Mask, vol: &LoadedVolume) -> Option<SegmentStats> {
    assert_eq!(mask.dims, vol.dims, "mask and volume of different sizes");
    let (lo, hi) = mask.bounds()?;
    let [nx, ny, _] = mask.dims;
    // Positions about the box's centre keep the second moments precise.
    let c = DVec3::new((lo[0] + hi[0]) as f64, (lo[1] + hi[1]) as f64, (lo[2] + hi[2]) as f64) * 0.5;
    let sums = (lo[2]..=hi[2])
        .into_par_iter()
        .map(|k| {
            let mut s = Sums::default();
            for j in lo[1]..=hi[1] {
                let row = (k * ny + j) * nx;
                for i in lo[0]..=hi[0] {
                    if mask.data[row + i] == 0 {
                        continue;
                    }
                    let p = [i as f64 - c.x, j as f64 - c.y, k as f64 - c.z];
                    s.n += 1;
                    (0..3).for_each(|a| s.s[a] += p[a]);
                    let ss = [p[0] * p[0], p[1] * p[1], p[2] * p[2], p[0] * p[1], p[0] * p[2], p[1] * p[2]];
                    (0..6).for_each(|a| s.ss[a] += ss[a]);
                    let v = vol.values[row + i];
                    if v.is_finite() {
                        s.values.push(v);
                    }
                }
            }
            s
        })
        .reduce(Sums::default, Sums::add);
    let n = sums.n as f64;
    let a = DMat3::from_mat4(vol.geometry.matrix());
    let volume_mm3 = n * a.determinant().abs();

    // Intensities.
    let mut values = sums.values;
    let (mean, sd, min, max, median) = if values.is_empty() {
        (f64::NAN, f64::NAN, f32::NAN, f32::NAN, f32::NAN)
    } else {
        let m = values.len() as f64;
        let mean = values.par_iter().map(|&v| v as f64).sum::<f64>() / m;
        let var = values.par_iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / m;
        let (min, max) = values.par_iter().fold(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), &v| (a.min(v), b.max(v))).reduce(|| (f32::INFINITY, f32::NEG_INFINITY), |(a, b), (c, d)| (a.min(c), b.max(d)));
        let (mid, even) = (values.len() / 2, values.len() % 2 == 0);
        let (below, &mut upper, _) = values.select_nth_unstable_by(mid, f32::total_cmp);
        let median = if even { 0.5 * (upper + below.iter().copied().fold(f32::NEG_INFINITY, f32::max)) } else { upper };
        (mean, var.sqrt(), min, max, median)
    };

    // Position and second moments (voxel units, then millimetres through the volume's axes).
    let m = DVec3::from(sums.s) / n;
    let centroid_ras = vol.geometry.world(c + m).to_array();
    let ss = sums.ss.map(|v| v / n);
    let cov = DMat3::from_cols(
        DVec3::new(ss[0] - m.x * m.x + 1.0 / 12.0, ss[3] - m.x * m.y, ss[4] - m.x * m.z),
        DVec3::new(ss[3] - m.x * m.y, ss[1] - m.y * m.y + 1.0 / 12.0, ss[5] - m.y * m.z),
        DVec3::new(ss[4] - m.x * m.z, ss[5] - m.y * m.z, ss[2] - m.z * m.z + 1.0 / 12.0),
    );
    let lambda = sym_eigenvalues(a * cov * a.transpose()).map(|l| l.max(0.0));
    let principal_mm = lambda.map(|l| 2.0 * (5.0 * l).sqrt());
    let elongation = if lambda[1] > 0.0 { (lambda[0] / lambda[1]).sqrt() } else { 1.0 };

    let surface_mm2 = surface_area(mask, lo, hi, a);
    let r_eq = (3.0 * volume_mm3 / (4.0 * PI)).cbrt();
    let roundness = if surface_mm2 > 0.0 { 4.0 * PI * r_eq * r_eq / surface_mm2 } else { 0.0 };
    Some(SegmentStats {
        voxels: sums.n,
        volume_mm3,
        surface_mm2,
        mean,
        sd,
        min,
        max,
        median,
        centroid_ras,
        bounds: (lo, hi),
        feret_mm: feret(mask, lo, hi, a),
        roundness,
        elongation,
        principal_mm,
    })
}

/// Eigenvalues of a symmetric 3 × 3 matrix, largest first: the trigonometric solution of its
/// characteristic cubic (O. K. Smith 1961, "Eigenvalues of a symmetric 3 × 3 matrix", Commun.
/// ACM 4(4):168).
fn sym_eigenvalues(m: DMat3) -> [f64; 3] {
    let (a, b, c) = (m.x_axis.x, m.y_axis.y, m.z_axis.z);
    let off = m.y_axis.x.powi(2) + m.z_axis.x.powi(2) + m.z_axis.y.powi(2);
    let q = (a + b + c) / 3.0;
    let p2 = (a - q).powi(2) + (b - q).powi(2) + (c - q).powi(2) + 2.0 * off;
    if p2 <= 1e-24 * q * q {
        return [q; 3];
    }
    let p = (p2 / 6.0).sqrt();
    let r = ((m - DMat3::from_diagonal(DVec3::splat(q))) * (1.0 / p)).determinant() * 0.5;
    let phi = r.clamp(-1.0, 1.0).acos() / 3.0;
    let l1 = q + 2.0 * p * phi.cos();
    let l3 = q + 2.0 * p * (phi + 2.0 * PI / 3.0).cos();
    [l1, 3.0 * q - l1 - l3, l3]
}

/// Voxel offsets around a face, with their weighted direction to it.
type Window = Vec<([i64; 3], DVec3)>;

/// Surface area (mm²) of the voxels from `lo` to `hi` of `mask` (see the module notes); `a`
/// maps voxel steps to millimetres.
fn surface_area(mask: &Mask, lo: [usize; 3], hi: [usize; 3], a: DMat3) -> f64 {
    const SIGMA: f64 = 1.5;
    let rw = (3.0 * SIGMA).ceil() as i64;
    let dims = mask.dims.map(|d| d as i64);
    let inside = |p: [i64; 3]| p[0] >= 0 && p[1] >= 0 && p[2] >= 0 && p[0] < dims[0] && p[1] < dims[1] && p[2] < dims[2] && mask.data[((p[2] * dims[1] + p[1]) * dims[0] + p[0]) as usize] != 0;
    // Area and unit normal (mm) of the faces across each voxel axis.
    let cols = [a.x_axis, a.y_axis, a.z_axis];
    let faces: [(f64, DVec3); 3] = std::array::from_fn(|x| {
        let n = cols[(x + 1) % 3].cross(cols[(x + 2) % 3]);
        (n.length(), n.normalize_or_zero())
    });
    // Gradients in voxel steps become normals in millimetres through the inverse transpose.
    let to_mm = a.inverse().transpose();
    // For the face on side `sign` of a voxel across `axis`: the voxels around the face's
    // centre c (offsets from the voxel) and the Gaussian-weighted c − v pointing out of them.
    let window = |axis: usize, sign: i64| -> Window {
        let mut w = Vec::new();
        for u in -rw..=rw + 1 {
            for v in -rw..=rw {
                for t in -rw..=rw {
                    let mut o = [0i64; 3];
                    o[axis] = u * sign;
                    (o[(axis + 1) % 3], o[(axis + 2) % 3]) = (v, t);
                    let mut c = DVec3::ZERO;
                    c[axis] = 0.5 * sign as f64;
                    let d = c - DVec3::new(o[0] as f64, o[1] as f64, o[2] as f64);
                    if d.length() <= 3.0 * SIGMA {
                        w.push((o, d * (-0.5 * d.length_squared() / (SIGMA * SIGMA)).exp()));
                    }
                }
            }
        }
        w
    };
    let windows: [[Window; 2]; 3] = std::array::from_fn(|x| [window(x, 1), window(x, -1)]);
    (lo[2]..=hi[2])
        .into_par_iter()
        .map(|k| {
            let mut area = 0.0;
            for j in lo[1]..=hi[1] {
                for i in lo[0]..=hi[0] {
                    let p = [i as i64, j as i64, k as i64];
                    if !inside(p) {
                        continue;
                    }
                    for (axis, &(face, normal)) in faces.iter().enumerate() {
                        for (s, sign) in [1i64, -1].into_iter().enumerate() {
                            let mut q = p;
                            q[axis] += sign;
                            if inside(q) {
                                continue;
                            }
                            let g: DVec3 = windows[axis][s].iter().filter(|(o, _)| inside([p[0] + o[0], p[1] + o[1], p[2] + o[2]])).map(|w| w.1).sum();
                            let n = (to_mm * g).normalize_or_zero();
                            // No normal (a checkerboard, say): the face counts whole.
                            area += face * if n == DVec3::ZERO { 1.0 } else { n.dot(normal).abs() };
                        }
                    }
                }
            }
            area
        })
        .sum()
}

/// Largest distance (mm) between two voxel centres of `mask` (within `lo`..=`hi`); `a` maps
/// voxel steps to millimetres.
fn feret(mask: &Mask, lo: [usize; 3], hi: [usize; 3], a: DMat3) -> f64 {
    let [nx, ny, _] = mask.dims;
    let d = [0, 1, 2].map(|x| hi[x] - lo[x] + 1);
    let set = |i: usize, j: usize, k: usize| mask.data[(k * ny + j) * nx + i] != 0;
    const NONE: (usize, usize) = (usize::MAX, 0);
    let ext = |e: &mut (usize, usize), v: usize| *e = (e.0.min(v), e.1.max(v));
    let is_end = |e: (usize, usize), v: usize| v == e.0 || v == e.1;
    // Ends of each row (along i) and column (along j), slice by slice; of each pillar (along
    // k), row of slices by row of slices.
    type Ends = Vec<(usize, usize)>;
    let slices: Vec<(Ends, Ends)> = (lo[2]..=hi[2])
        .into_par_iter()
        .map(|k| {
            let (mut rows, mut cols) = (vec![NONE; d[1]], vec![NONE; d[0]]);
            for j in lo[1]..=hi[1] {
                for i in (lo[0]..=hi[0]).filter(|&i| set(i, j, k)) {
                    ext(&mut rows[j - lo[1]], i);
                    ext(&mut cols[i - lo[0]], j);
                }
            }
            (rows, cols)
        })
        .collect();
    let pillars: Vec<Ends> = (lo[1]..=hi[1])
        .into_par_iter()
        .map(|j| {
            let mut p = vec![NONE; d[0]];
            for k in lo[2]..=hi[2] {
                for i in (lo[0]..=hi[0]).filter(|&i| set(i, j, k)) {
                    ext(&mut p[i - lo[0]], k);
                }
            }
            p
        })
        .collect();
    // Hull vertices are ends of every line of voxels through them.
    let pts: Vec<DVec3> = slices
        .par_iter()
        .enumerate()
        .flat_map_iter(|(z, (rows, cols))| {
            let pillars = &pillars;
            rows.iter().enumerate().filter(|(_, e)| e.0 != usize::MAX).flat_map(move |(y, &(i0, i1))| {
                let ends = if i0 == i1 { vec![i0] } else { vec![i0, i1] };
                ends.into_iter().filter(move |&i| is_end(cols[i - lo[0]], lo[1] + y) && is_end(pillars[y][i - lo[0]], lo[2] + z)).map(move |i| a * DVec3::new((i - lo[0]) as f64, y as f64, z as f64))
            })
        })
        .collect();
    diameter(&pts)
}

/// Largest distance between two of `p`: by pairs for up to a few thousand points; otherwise
/// the points extreme along 1024 directions (a Fibonacci lattice on the half sphere) by pairs,
/// then farthest-point steps from the best pair's ends until it stops growing.
fn diameter(p: &[DVec3]) -> f64 {
    // The farthest pair among `idx`, as (squared distance, first, second).
    let pairs = |idx: &[usize]| {
        idx.par_iter()
            .enumerate()
            .map(|(n, &u)| idx[n + 1..].iter().map(|&v| ((p[u] - p[v]).length_squared(), u, v)).fold((0.0, u, u), |b, x| if x.0 > b.0 { x } else { b }))
            .reduce(|| (0.0, 0, 0), |b, x| if x.0 > b.0 { x } else { b })
    };
    if p.len() <= 3000 {
        return pairs(&(0..p.len()).collect::<Vec<_>>()).0.sqrt();
    }
    const DIRS: usize = 1024;
    let golden = PI * (3.0 - 5f64.sqrt());
    let mut ext: Vec<usize> = (0..DIRS)
        .into_par_iter()
        .flat_map_iter(|n| {
            let z = 1.0 - (n as f64 + 0.5) / DIRS as f64;
            let (s, c) = (golden * n as f64).sin_cos();
            let d = DVec3::new(c, s, 0.0) * (1.0 - z * z).sqrt() + DVec3::Z * z;
            let ((mut lo, mut tlo), (mut hi, mut thi)) = ((0, f64::INFINITY), (0, f64::NEG_INFINITY));
            for (m, q) in p.iter().enumerate() {
                let t = q.dot(d);
                if t < tlo {
                    (lo, tlo) = (m, t);
                }
                if t > thi {
                    (hi, thi) = (m, t);
                }
            }
            [lo, hi]
        })
        .collect();
    ext.sort_unstable();
    ext.dedup();
    let (mut best, mut u, mut v) = pairs(&ext);
    let farthest = |from: DVec3| p.par_iter().enumerate().map(|(m, q)| ((*q - from).length_squared(), m)).reduce(|| (0.0, 0), |b, x| if x.0 > b.0 { x } else { b });
    loop {
        let ((du, fu), (dv, fv)) = (farthest(p[u]), farthest(p[v]));
        if du.max(dv) <= best {
            break;
        }
        if du >= dv {
            (best, v) = (du, fu);
        } else {
            (best, u) = (dv, fv);
        }
    }
    best.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::geometry::Geometry;
    use std::time::Instant;

    fn volume(dims: [usize; 3], spacing: [f64; 3], values: Vec<f32>) -> LoadedVolume {
        LoadedVolume::from_values("v", dims, values, Geometry::from_axes([dims[0], dims[1], dims[2], 1], spacing, [DVec3::X, DVec3::Y, DVec3::Z], DVec3::new(-20.0, 5.0, 100.0)))
    }

    /// Voxels within `r` mm of `c` (mm from voxel 0) on a grid of `spacing`.
    fn ball(dims: [usize; 3], spacing: [f64; 3], c: DVec3, r: f64) -> Mask {
        let mut m = Mask::new(dims);
        for k in 0..dims[2] {
            for j in 0..dims[1] {
                for i in 0..dims[0] {
                    let p = DVec3::new(i as f64 * spacing[0], j as f64 * spacing[1], k as f64 * spacing[2]);
                    m.set(i, j, k, (p - c).length() <= r);
                }
            }
        }
        m
    }

    fn close(a: f64, b: f64, rel: f64) -> bool {
        (a - b).abs() <= rel * b.abs()
    }

    #[test]
    fn stats_of_a_ball() {
        let (n, r) = (27, 10.0);
        let m = ball([n; 3], [1.0; 3], DVec3::splat(13.0), r);
        let values: Vec<f32> = m.data.iter().map(|&v| if v != 0 { 2.0 } else { 7.0 }).collect();
        let vol = volume([n; 3], [1.0; 3], values);
        let s = segment_stats(&m, &vol).unwrap();
        assert_eq!(s.voxels, m.count());
        assert!(close(s.volume_mm3, 4.0 / 3.0 * PI * r.powi(3), 0.03), "volume {}", s.volume_mm3);
        assert!(close(s.surface_mm2, 4.0 * PI * r * r, 0.05), "surface {}", s.surface_mm2);
        assert!((0.95..=1.05).contains(&s.roundness), "roundness {}", s.roundness);
        assert!((s.elongation - 1.0).abs() < 0.05, "elongation {}", s.elongation);
        assert!((s.feret_mm - 2.0 * r).abs() <= 1.0, "feret {}", s.feret_mm);
        for l in s.principal_mm {
            assert!((l - 2.0 * r).abs() < 0.5, "{:?}", s.principal_mm);
        }
        assert_eq!((s.mean, s.sd, s.min, s.max, s.median), (2.0, 0.0, 2.0, 2.0, 2.0));
        let want = vol.geometry.world(DVec3::splat(13.0));
        assert!((DVec3::from(s.centroid_ras) - want).length() < 1e-9, "{:?}", s.centroid_ras);
        assert_eq!(s.bounds, ([3; 3], [23; 3]));
    }

    #[test]
    fn stats_of_a_box_and_of_values() {
        let dims = [50, 20, 20];
        let mut m = Mask::new(dims);
        let mut values = vec![0f32; 50 * 20 * 20];
        let mut k_values = Vec::new();
        for k in 5..15 {
            for j in 5..15 {
                for i in 5..45 {
                    m.set(i, j, k, true);
                    let v = (i + 3 * j + 7 * k) as f32;
                    values[m.index(i, j, k)] = v;
                    k_values.push(v as f64);
                }
            }
        }
        let vol = volume(dims, [1.0; 3], values);
        let s = segment_stats(&m, &vol).unwrap();
        assert!((s.elongation - 4.0).abs() < 1e-9, "elongation {}", s.elongation);
        assert!((s.principal_mm[0] / s.principal_mm[1] - 4.0).abs() < 1e-9);
        assert!((s.principal_mm[1] - s.principal_mm[2]).abs() < 1e-9);
        assert!((s.feret_mm - (39f64 * 39.0 + 81.0 + 81.0).sqrt()).abs() < 1e-9, "feret {}", s.feret_mm);
        // The faces are exact; the edges and corners are rounded a little.
        assert!(s.surface_mm2 < 1800.0 && s.surface_mm2 > 0.9 * 1800.0, "surface {}", s.surface_mm2);
        assert!(s.roundness < 0.9);
        let mean = k_values.iter().sum::<f64>() / k_values.len() as f64;
        let sd = (k_values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / k_values.len() as f64).sqrt();
        assert!((s.mean - mean).abs() < 1e-9 && (s.sd - sd).abs() < 1e-9, "{} {} vs {mean} {sd}", s.mean, s.sd);
        k_values.sort_by(f64::total_cmp);
        let median = 0.5 * (k_values[1999] + k_values[2000]);
        assert_eq!(s.median as f64, median);
        assert_eq!((s.min, s.max), (5.0 + 15.0 + 35.0, 44.0 + 42.0 + 98.0));
    }

    #[test]
    fn stats_with_anisotropic_spacing() {
        let (dims, spacing) = ([40, 40, 16], [0.6, 0.6, 2.0]);
        let c = DVec3::new(12.0, 11.7, 15.0);
        let m = ball(dims, spacing, c, 9.0);
        let vol = volume(dims, spacing, vec![1.0; 40 * 40 * 16]);
        let s = segment_stats(&m, &vol).unwrap();
        assert!((s.volume_mm3 - s.voxels as f64 * 0.72).abs() < 1e-9);
        assert!(close(s.volume_mm3, 4.0 / 3.0 * PI * 729.0, 0.03), "volume {}", s.volume_mm3);
        assert!(close(s.surface_mm2, 4.0 * PI * 81.0, 0.06), "surface {}", s.surface_mm2);
        let want = vol.geometry.world(c / DVec3::from(spacing));
        assert!((DVec3::from(s.centroid_ras) - want).length() < 0.1, "{:?} vs {want}", s.centroid_ras);
        assert!((s.feret_mm - 18.0).abs() <= 1.0, "feret {}", s.feret_mm);
        assert!(s.elongation < 1.06, "elongation {}", s.elongation);
    }

    #[test]
    fn stats_of_empty_and_tiny_masks() {
        let vol = volume([4, 4, 4], [1.0; 3], vec![3.0; 64]);
        assert_eq!(segment_stats(&Mask::new([4, 4, 4]), &vol), None);
        let mut m = Mask::new([4, 4, 4]);
        m.set(0, 0, 0, true);
        let s = segment_stats(&m, &vol).unwrap();
        assert_eq!((s.voxels, s.mean, s.sd, s.median, s.feret_mm), (1, 3.0, 0.0, 3.0, 0.0));
        assert!(s.surface_mm2 > 0.0 && s.roundness.is_finite() && s.elongation == 1.0);
        m.set(3, 3, 3, true);
        let s = segment_stats(&m, &vol).unwrap();
        assert!((s.feret_mm - 27f64.sqrt()).abs() < 1e-12);
        assert_eq!(s.median, 3.0);
        // NaN values are left out.
        let mut nan = vol.clone();
        nan.values[0] = f32::NAN;
        let s = segment_stats(&m, &nan).unwrap();
        assert_eq!((s.mean, s.min, s.max), (3.0, 3.0, 3.0));
    }

    #[test]
    fn stats_feret_on_many_candidates_and_timing() {
        // A ball of about a million voxels; its surface has well over 3000 hull candidates.
        let n = 128;
        let r = 62.0;
        let m = ball([n; 3], [1.0; 3], DVec3::splat(63.5), r);
        let vol = volume([n; 3], [1.0; 3], vec![1.0; n * n * n]);
        let t = Instant::now();
        let s = segment_stats(&m, &vol).unwrap();
        let ms = t.elapsed().as_secs_f64() * 1e3;
        eprintln!("segment_stats of {} voxels: {ms:.0} ms", s.voxels);
        assert!(s.voxels > 900_000);
        assert!((s.feret_mm - 2.0 * r).abs() <= 1.0, "feret {}", s.feret_mm);
        assert!(close(s.surface_mm2, 4.0 * PI * r * r, 0.03), "surface {}", s.surface_mm2);
        // An oblique rod: the Feret diameter by directions matches the pairs.
        let mut rod = Mask::new([n; 3]);
        for t in 0..1200 {
            let p = DVec3::new(10.0, 20.0, 15.0) + DVec3::new(0.7, 0.5, 0.45) * t as f64 * 0.1;
            for o in 0..27 {
                let q = p + DVec3::new((o % 3) as f64 - 1.0, ((o / 3) % 3) as f64 - 1.0, (o / 9) as f64 - 1.0) * 2.0;
                rod.set(q.x as usize, q.y as usize, q.z as usize, true);
            }
        }
        let pts: Vec<DVec3> = (0..rod.len()).filter(|&q| rod.data[q] != 0).map(|q| { let [i, j, k] = rod.ijk(q); DVec3::new(i as f64, j as f64, k as f64) }).collect();
        assert!(pts.len() > 3000);
        let exact = pts.iter().flat_map(|a| pts.iter().map(move |b| (*a - *b).length_squared())).fold(0.0, f64::max).sqrt();
        assert!((diameter(&pts) - exact).abs() < 1e-9);
        let s = segment_stats(&rod, &vol).unwrap();
        assert!((s.feret_mm - exact).abs() < 1e-9, "{} vs {exact}", s.feret_mm);
    }
}
