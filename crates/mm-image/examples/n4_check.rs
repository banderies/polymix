//! Run N4 on an image and print timings and field statistics; with a second argument, also
//! save the middle axial slice before and after as PNGs there (local checks only):
//! `cargo run -p mm-image --release --example n4_check -- image.nii.gz [out_dir]`.

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("an image file"));
    let out = std::env::args().nth(2).map(std::path::PathBuf::from);
    let found = mm_volume::scan::classify(&path, std::fs::metadata(&path)?.len());
    let mm_volume::scan::Found::Volume(v) = found else { anyhow::bail!("not a volume") };
    let t = Instant::now();
    let vol = mm_volume::load::load_volume(&v, 600_000_000, &|_| true)?;
    println!("loaded {:?} spacing {:?} in {:.2?}", vol.dims, vol.geometry.spacing, t.elapsed());
    let t = Instant::now();
    let r = mm_image::n4(&vol, None, &mm_image::N4Params::default(), &|_| true)?;
    let masked: Vec<f32> = r.bias.iter().zip(&r.mask).filter(|(_, &m)| m > 0).map(|(&b, _)| b).collect();
    let (lo, hi) = masked.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), &x| (a.min(x), b.max(x)));
    println!("N4 in {:.2?}: iterations {:?}, mask {:.0}% of the grid, field {:.2}–{:.2} inside the mask", t.elapsed(), r.iterations, 100.0 * masked.len() as f64 / r.mask.len() as f64, lo, hi);
    if let Some(dir) = out {
        let axes = mm_volume::slice::plane_axes(&vol.geometry, mm_volume::slice::Plane::Axial);
        let k = vol.dims[axes.slice] / 2;
        for (name, values, invert) in [("before", &vol.values, false), ("after", &r.corrected, false), ("bias", &r.bias, false)] {
            let (w, h) = (vol.dims[axes.across], vol.dims[axes.down]);
            let mut img = image::GrayImage::new(w as u32, h as u32);
            let pick = |x: usize, y: usize| {
                let mut ijk = [0usize; 3];
                ijk[axes.slice] = k;
                ijk[axes.across] = if axes.flip_across { w - 1 - x } else { x };
                ijk[axes.down] = if axes.flip_down { h - 1 - y } else { y };
                values[vol.index(ijk[0], ijk[1], ijk[2])]
            };
            let sample: Vec<f32> = (0..h).flat_map(|y| (0..w).map(move |x| (x, y))).map(|(x, y)| pick(x, y)).collect();
            let (c, ww) = mm_volume::slice::auto_window(&sample);
            for y in 0..h {
                for x in 0..w {
                    let g = (((pick(x, y) as f64 - (c - ww / 2.0)) / ww).clamp(0.0, 1.0) * 255.0) as u8;
                    img.put_pixel(x as u32, y as u32, image::Luma([if invert { 255 - g } else { g }]));
                }
            }
            img.save(dir.join(format!("n4_{name}.png")))?;
        }
    }
    Ok(())
}
