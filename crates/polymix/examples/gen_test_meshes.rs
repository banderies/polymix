//! Generate synthetic test meshes that mimic raw vessel segmentations:
//! `cargo run -p polymix --example gen_test_meshes --release -- <out_dir>`

use glam::DVec3;
use mm_core::{primitives, Mesh};
use std::path::PathBuf;

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64) / ((1u64 << 53) as f64)
    }
}

fn bezier(a: DVec3, b: DVec3, c: DVec3, n: usize) -> Vec<DVec3> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            a * (1.0 - t) * (1.0 - t) + b * 2.0 * t * (1.0 - t) + c * t * t
        })
        .collect()
}

fn tube(path: Vec<DVec3>, r0: f64, r1: f64) -> Mesh {
    let n = path.len();
    let radii: Vec<f64> = (0..n).map(|i| r0 + (r1 - r0) * i as f64 / (n - 1) as f64).collect();
    primitives::tube_along(&path, &radii, 40)
}

fn add_noise(m: &mut Mesh, amp: f64, rng: &mut Lcg) {
    let normals = m.compute_vertex_normals();
    let ids: Vec<u32> = m.vertex_ids().collect();
    for v in ids {
        let p = m.vertex(v) + normals[v as usize] * (rng.f() - 0.5) * 2.0 * amp;
        m.set_vertex(v, p);
    }
}

fn vessel() -> Mesh {
    let mut rng = Lcg(42);
    let mut m = Mesh::new();
    let trunk = tube(bezier(DVec3::new(0.0, 0.0, 0.0), DVec3::new(1.0, 0.0, 20.0), DVec3::new(0.0, 0.0, 40.0), 60), 3.0, 2.8);
    let left = tube(bezier(DVec3::new(0.0, 0.0, 39.0), DVec3::new(-6.0, 0.0, 52.0), DVec3::new(-16.0, 1.0, 70.0), 60), 2.3, 1.9);
    let right = tube(bezier(DVec3::new(0.0, 0.0, 39.0), DVec3::new(7.0, 2.0, 50.0), DVec3::new(15.0, 5.0, 68.0), 60), 2.1, 1.7);
    let side = tube(bezier(DVec3::new(0.0, 0.0, 20.0), DVec3::new(5.0, 0.0, 22.0), DVec3::new(10.0, -2.0, 26.0), 30), 1.0, 0.8);
    for part in [trunk, left, right, side] {
        m.append_mesh(&part, 0);
    }
    // Saccular aneurysm at the bifurcation apex.
    let mut sac = primitives::icosphere(3.6, 4);
    sac.transform(|p| p + DVec3::new(0.5, 0.5, 44.5));
    m.append_mesh(&sac, 0);
    add_noise(&mut m, 0.07, &mut rng);
    // Floating segmentation fragments.
    for k in 0..6 {
        let mut f = primitives::icosphere(0.4 + 0.2 * rng.f(), 1);
        let c = DVec3::new(rng.f() * 30.0 - 15.0, rng.f() * 10.0 - 5.0, rng.f() * 70.0);
        f.transform(|p| p + c);
        m.append_mesh(&f, 0);
        let _ = k;
    }
    // A couple of holes in the trunk wall.
    let holes: Vec<u32> = m
        .triangle_ids()
        .filter(|&t| {
            let c = m.tri_centroid(t);
            (c - DVec3::new(3.0, 0.0, 12.0)).length() < 1.2 || (c - DVec3::new(-2.5, 1.5, 30.0)).length() < 0.8
        })
        .collect();
    m.remove_triangles(holes);
    m
}

fn main() -> anyhow::Result<()> {
    let out: PathBuf = std::env::args().nth(1).map(Into::into).unwrap_or_else(|| "test_meshes".into());
    std::fs::create_dir_all(&out)?;
    let opts = mm_io::WriteOptions::default();
    let v = vessel();
    println!("vessel: {} tris", v.triangle_count());
    mm_io::write_mesh(&out.join("synthetic_vessel.stl"), &v, &opts)?;
    let mut big = primitives::icosphere(50.0, 8);
    add_noise(&mut big, 0.05, &mut Lcg(7));
    println!("big sphere: {} tris", big.triangle_count());
    mm_io::write_mesh(&out.join("big_sphere_1M.stl"), &big, &opts)?;
    let bunny = primitives::cube(20.0, 8);
    mm_io::write_mesh(&out.join("cube.stl"), &bunny, &opts)?;
    Ok(())
}
