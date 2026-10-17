//! Render the app icon (a shaded low-poly sphere on a rounded dark tile) to PNG:
//! `cargo run -p polymix --example gen_icon -- icon_1024.png`

use glam::{DVec3, Vec3};

fn main() -> anyhow::Result<()> {
    let out = std::env::args().nth(1).unwrap_or_else(|| "icon_1024.png".into());
    let n = 1024usize;
    let mut img = vec![[0u8; 4]; n * n];
    let mesh = mm_core::primitives::icosphere(1.0, 1);
    // Rounded-square background with a vertical gradient.
    let r = 0.18 * n as f32;
    let (lo, hi) = (n as f32 * 0.06, n as f32 * 0.94);
    for y in 0..n {
        for x in 0..n {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let dx = (lo + r - fx).max(fx - (hi - r)).max(0.0);
            let dy = (lo + r - fy).max(fy - (hi - r)).max(0.0);
            let d = (dx * dx + dy * dy).sqrt() - r;
            let inside = fx >= lo && fx <= hi && fy >= lo && fy <= hi;
            let a = if inside { (0.5 - d).clamp(0.0, 1.0) } else { 0.0 };
            let t = fy / n as f32;
            let c = Vec3::new(0.21, 0.22, 0.26).lerp(Vec3::new(0.07, 0.07, 0.09), t);
            img[y * n + x] = [(c.x * 255.0) as u8, (c.y * 255.0) as u8, (c.z * 255.0) as u8, (a * 255.0) as u8];
        }
    }
    // Rotate the sphere a little and rasterize front-facing triangles with flat shading.
    let rot = glam::DQuat::from_rotation_y(0.5) * glam::DQuat::from_rotation_x(-0.35);
    let light = DVec3::new(-0.5, 0.7, 0.6).normalize();
    let scale = n as f64 * 0.33;
    let center = n as f64 * 0.5;
    let mut tris: Vec<([DVec3; 3], DVec3)> = mesh
        .triangle_ids()
        .map(|t| {
            let p = mesh.tri_positions(t).map(|q| rot * q);
            let nrm = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
            (p, nrm)
        })
        .filter(|(_, nrm)| nrm.z > 0.0)
        .collect();
    tris.sort_by(|a, b| (a.0[0].z + a.0[1].z + a.0[2].z).partial_cmp(&(b.0[0].z + b.0[1].z + b.0[2].z)).unwrap());
    let orange = DVec3::new(1.0, 0.55, 0.16);
    for (p, nrm) in tris {
        let s: Vec<(f64, f64)> = p.iter().map(|q| (center + q.x * scale, center - q.y * scale)).collect();
        let shade = 0.35 + 0.65 * nrm.dot(light).max(0.0);
        let col = orange * shade + DVec3::splat(0.25 * nrm.dot(light).max(0.0).powi(8));
        let (minx, maxx) = (s.iter().map(|v| v.0).fold(f64::MAX, f64::min) as usize, s.iter().map(|v| v.0).fold(0.0, f64::max) as usize + 1);
        let (miny, maxy) = (s.iter().map(|v| v.1).fold(f64::MAX, f64::min) as usize, s.iter().map(|v| v.1).fold(0.0, f64::max) as usize + 1);
        let edge = |a: (f64, f64), b: (f64, f64), p: (f64, f64)| (b.0 - a.0) * (p.1 - a.1) - (b.1 - a.1) * (p.0 - a.0);
        for y in miny..maxy.min(n) {
            for x in minx..maxx.min(n) {
                let q = (x as f64 + 0.5, y as f64 + 0.5);
                let (e0, e1, e2) = (edge(s[0], s[1], q), edge(s[1], s[2], q), edge(s[2], s[0], q));
                if (e0 <= 0.0 && e1 <= 0.0 && e2 <= 0.0) || (e0 >= 0.0 && e1 >= 0.0 && e2 >= 0.0) {
                    // Thin dark edges between facets.
                    let len = |a: (f64, f64), b: (f64, f64)| ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
                    let dmin = (e0.abs() / len(s[0], s[1])).min(e1.abs() / len(s[1], s[2])).min(e2.abs() / len(s[2], s[0]));
                    let k = if dmin < 3.0 { 0.55 } else { 1.0 };
                    let c = (col * k).clamp(DVec3::ZERO, DVec3::ONE);
                    img[y * n + x] = [(c.x * 255.0) as u8, (c.y * 255.0) as u8, (c.z * 255.0) as u8, 255];
                }
            }
        }
    }
    let f = std::io::BufWriter::new(std::fs::File::create(&out)?);
    let mut enc = png::Encoder::new(f, n as u32, n as u32);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&img.concat())?;
    println!("wrote {out}");
    Ok(())
}
