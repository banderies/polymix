//! Property tests: surface extraction must always be closed, edge-manifold, bowtie-free
//! and consistently oriented, for arbitrary (including adversarial) fields.

mod common;

use common::*;
use glam::DVec3;
use mm_core::{measure, Aabb, Mesh};
use mm_solid::{blocky, build_field, cleanup, extract, mesh_implicit, FieldSource, GridSpec};

/// Values on a dense lattice (nearest-sample lookup): the most adversarial input since
/// neighbouring samples are uncorrelated.
struct DenseField {
    grid: GridSpec,
    n: [usize; 3],
    values: Vec<f64>,
}

impl DenseField {
    fn new(bricks: [u32; 3], mut gen: impl FnMut([usize; 3]) -> f64) -> Self {
        let grid = GridSpec::new(DVec3::new(-1.0, 2.0, 0.5), 0.37, bricks);
        let s = grid.samples();
        let n = [s[0] as usize, s[1] as usize, s[2] as usize];
        let mut values = Vec::with_capacity(n[0] * n[1] * n[2]);
        for z in 0..n[2] {
            for y in 0..n[1] {
                for x in 0..n[0] {
                    values.push(gen([x, y, z]));
                }
            }
        }
        Self { grid, n, values }
    }
}

impl FieldSource for DenseField {
    fn eval(&self, p: DVec3) -> f64 {
        let q = ((p - self.grid.origin) / self.grid.cell_size).round();
        let i = [q.x as usize, q.y as usize, q.z as usize];
        self.values[i[0] + self.n[0] * (i[1] + self.n[1] * i[2])].clamp(-1.0, 1.0)
    }
    fn far(&self) -> f64 {
        1.0
    }
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        let b = grid.bricks;
        (0..b[2]).flat_map(|z| (0..b[1]).flat_map(move |y| (0..b[0]).map(move |x| [x, y, z]))).collect()
    }
}

fn mesh_of(src: &dyn FieldSource, grid: GridSpec) -> (Mesh, Mesh) {
    let field = build_field(src, grid, None).unwrap();
    let (p, t) = extract::marching_tetrahedra(&field);
    let (m, rep) = Mesh::from_triangles(&p, &t, None);
    assert_eq!(rep.nonmanifold_split, 0, "non-manifold edges: {rep:?}");
    assert_eq!(rep.duplicate + rep.degenerate, 0, "{rep:?}");
    let (bp, bt) = blocky::voxel_faces(&field);
    let (b, rep) = Mesh::from_triangles(&bp, &bt, None);
    assert_eq!(rep.nonmanifold_split, 0, "blocky non-manifold edges: {rep:?}");
    (m, b)
}

#[test]
fn random_noise_fields_are_manifold() {
    let mut rng = Rng::new(1);
    let mut nonempty = 0;
    for trial in 0..64 {
        let kind = trial % 4;
        let density = rng.range(0.2, 0.8);
        let mut r2 = Rng::new(1000 + trial as u64);
        let src = DenseField::new([2, 2, 3], |_| match kind {
            // Binary noise: maximally ambiguous cube configurations.
            0 => {
                if r2.f() < density {
                    -1.0
                } else {
                    1.0
                }
            }
            // Continuous noise.
            1 => r2.range(-1.0, 1.0),
            // Values with many exact zeros.
            2 => [-1.0, 0.0, 1.0][r2.below(3)],
            // Tiny magnitudes near the quantization step.
            _ => r2.range(-1.0, 1.0) * 1e-4,
        });
        let (m, b) = mesh_of(&src, src.grid);
        if m.triangle_count() > 0 {
            nonempty += 1;
            assert_closed_manifold(&m, &format!("noise trial {trial} (kind {kind})"));
            assert!(measure::volume(&m) > 0.0);
            let mut c = m.clone();
            cleanup::collapse_short_edges(&mut c, 0.3 * src.grid.cell_size);
            cleanup::flip_to_improve(&mut c, 2);
            assert_closed_manifold(&c, &format!("noise trial {trial} after cleanup"));
        }
        if b.triangle_count() > 0 {
            assert_closed_manifold(&b, &format!("blocky noise trial {trial}"));
            assert!(measure::volume(&b) > 0.0);
        }
    }
    assert!(nonempty >= 60, "only {nonempty} non-trivial trials");
}

/// Noise confined to a spherical shell between a uniform inside core and a uniform
/// outside, with no seed bricks at all: classification + propagation must find every
/// sign change, and inactive bricks must border the noise consistently.
struct ShellNoise {
    r0: f64,
    r1: f64,
    seed: u64,
    h: f64,
}
impl FieldSource for ShellNoise {
    fn eval(&self, p: DVec3) -> f64 {
        let r = p.length();
        if r < self.r0 {
            return -1.0;
        }
        if r > self.r1 {
            return 1.0;
        }
        // Deterministic hash noise per lattice sample.
        let q = (p / self.h).round();
        let mut x = (q.x as i64 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
            ^ (q.y as i64 as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
            ^ (q.z as i64 as u64).wrapping_mul(0x1656_67B1_9E37_79F9)
            ^ self.seed;
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^= x >> 29;
        if x & 1 == 0 {
            -0.5
        } else {
            0.5
        }
    }
    fn far(&self) -> f64 {
        1.0
    }
    fn seed_bricks(&self, _: &GridSpec) -> Vec<[u32; 3]> {
        Vec::new()
    }
}

#[test]
fn unseeded_noise_shell_is_manifold() {
    for seed in 0..6u64 {
        let h = 0.05;
        let src = ShellNoise { r0: 0.5 + 0.03 * seed as f64, r1: 0.62 + 0.04 * seed as f64, seed, h };
        // Lattice aligned with the noise hash (origin on a multiple of h).
        let grid = GridSpec::new(DVec3::splat(-1.2), h, [6, 6, 6]);
        let (m, b) = mesh_of(&src, grid);
        assert!(m.triangle_count() > 1000);
        assert_closed_manifold(&m, &format!("noise shell {seed}"));
        assert_closed_manifold(&b, &format!("blocky noise shell {seed}"));
        // The core is enclosed: total volume at least the core's.
        let core = 4.0 / 3.0 * std::f64::consts::PI * src.r0.powi(3);
        assert!(measure::volume(&m) > 0.9 * core);
    }
}

#[test]
fn random_sphere_unions_are_manifold() {
    for seed in 0..40u64 {
        let mut rng = Rng::new(seed + 7);
        let n = 1 + rng.below(6);
        let spheres: Vec<(DVec3, f64)> = (0..n).map(|_| (rng.vec(1.0), rng.range(0.15, 0.7))).collect();
        let f = |p: DVec3| spheres.iter().map(|&(c, r)| p.distance(c) - r).fold(f64::INFINITY, f64::min);
        let h = rng.range(0.04, 0.12);
        let bounds = Aabb { min: DVec3::splat(-1.8), max: DVec3::splat(1.8) };
        let m = mesh_implicit(f, 1.0, &bounds, h, None).unwrap();
        assert_solid(&m, &format!("sphere union seed {seed}"));
        // Volume sanity: between the largest single sphere and the sum of all.
        let vmax: f64 = spheres.iter().map(|s| 4.0 / 3.0 * std::f64::consts::PI * s.1.powi(3)).fold(0.0, f64::max);
        let vsum: f64 = spheres.iter().map(|s| 4.0 / 3.0 * std::f64::consts::PI * s.1.powi(3)).sum();
        let v = measure::volume(&m);
        assert!(v > 0.85 * vmax && v < 1.1 * vsum + 0.01, "seed {seed}: volume {v} not in [{vmax}, {vsum}]");
    }
}

#[test]
fn random_metaballs_are_manifold() {
    // Smooth sums of Gaussian blobs produce near-tangent necks and thin bridges.
    for seed in 0..20u64 {
        let mut rng = Rng::new(seed + 100);
        let n = 2 + rng.below(6);
        let s = rng.range(0.25, 0.5);
        let blobs: Vec<DVec3> = (0..n).map(|_| rng.vec(0.8)).collect();
        let thresh = rng.range(0.3, 0.9);
        let f = |p: DVec3| thresh - blobs.iter().map(|&c| (-(p.distance_squared(c)) / (s * s)).exp()).sum::<f64>();
        let lip = n as f64 * 0.86 / s;
        let bounds = Aabb { min: DVec3::splat(-1.8), max: DVec3::splat(1.8) };
        let m = mesh_implicit(f, lip, &bounds, rng.range(0.05, 0.1), None).unwrap();
        if m.triangle_count() > 0 {
            assert_solid(&m, &format!("metaballs seed {seed}"));
        }
    }
}

/// A sphere field that only reveals one seed brick: the builder must find the rest of the
/// surface through propagation alone.
struct HiddenSphere {
    r: f64,
}
impl FieldSource for HiddenSphere {
    fn eval(&self, p: DVec3) -> f64 {
        (p.length() - self.r).clamp(-0.3, 0.3)
    }
    fn far(&self) -> f64 {
        0.3
    }
    fn seed_bricks(&self, grid: &GridSpec) -> Vec<[u32; 3]> {
        let b = grid.brick_of(DVec3::new(self.r, 0.0, 0.0));
        vec![[b[0] as u32, b[1] as u32, b[2] as u32]]
    }
}

#[test]
fn propagation_recovers_unseeded_surface() {
    let src = HiddenSphere { r: 1.0 };
    let grid = GridSpec::covering(&Aabb { min: DVec3::splat(-1.0), max: DVec3::splat(1.0) }, 0.05, 0.3).unwrap();
    let (m, b) = mesh_of(&src, grid);
    assert_solid(&m, "hidden sphere");
    assert_solid(&b, "hidden sphere (blocky)");
    let v = measure::volume(&m);
    let exact = 4.0 / 3.0 * std::f64::consts::PI;
    assert!((v - exact).abs() / exact < 0.01, "volume {v}");
    assert_eq!(component_count(&m), 1);
}

/// Inside region touching the grid boundary: the forced-outside boundary layer must close it.
#[test]
fn field_clipped_by_domain_is_closed() {
    let f = |p: DVec3| p.z - 0.3; // half-space below z = 0.3
    let bounds = Aabb { min: DVec3::splat(-1.0), max: DVec3::splat(1.0) };
    let m = mesh_implicit(f, 1.0, &bounds, 0.1, None).unwrap();
    assert_solid(&m, "clipped half-space");
}

/// A field whose `uniform_sign` lies (claims certainty for blocks that straddle the zero
/// set) and that provides no seeds: the builder must still produce a consistent field.
struct LyingPlane {
    c: f64,
}
impl FieldSource for LyingPlane {
    fn eval(&self, p: DVec3) -> f64 {
        (p.x - self.c + 0.3 * (p.y * 3.0).sin()).clamp(-0.2, 0.2)
    }
    fn far(&self) -> f64 {
        0.2
    }
    fn seed_bricks(&self, _: &GridSpec) -> Vec<[u32; 3]> {
        Vec::new()
    }
    fn uniform_sign(&self, c: DVec3, _r: f64) -> Option<bool> {
        Some(self.eval(c) < 0.0)
    }
}

#[test]
fn wrong_uniform_heuristic_is_repaired() {
    for (i, c) in [0.123, -0.31, 0.77].into_iter().enumerate() {
        let src = LyingPlane { c };
        // 40³ bricks so that 16³-brick uniform blocks exist on both sides of the surface.
        let grid = GridSpec::new(DVec3::splat(-2.0), 4.0 / 319.0, [40, 40, 40]);
        let (m, _) = mesh_of(&src, grid);
        assert_solid(&m, &format!("lying plane {i}"));
        assert_eq!(component_count(&m), 1);
    }
}

/// Closed surface away from the domain boundary, no seeds, lying `uniform_sign`: only the
/// uniform-block conflict refinement can discover it.
struct LyingSphere;
impl FieldSource for LyingSphere {
    fn eval(&self, p: DVec3) -> f64 {
        (p.length() - 1.0).clamp(-0.2, 0.2)
    }
    fn far(&self) -> f64 {
        0.2
    }
    fn seed_bricks(&self, _: &GridSpec) -> Vec<[u32; 3]> {
        Vec::new()
    }
    fn uniform_sign(&self, c: DVec3, _r: f64) -> Option<bool> {
        Some(self.eval(c) < 0.0)
    }
}

#[test]
fn wrong_uniform_heuristic_closed_surface() {
    let grid = GridSpec::new(DVec3::splat(-2.0), 4.0 / 319.0, [40, 40, 40]);
    let (m, b) = mesh_of(&LyingSphere, grid);
    assert_solid(&m, "lying sphere");
    assert_solid(&b, "lying sphere (blocky)");
    let v = measure::volume(&m);
    assert!((v - 4.0 / 3.0 * std::f64::consts::PI).abs() < 0.02, "volume {v}");
}
