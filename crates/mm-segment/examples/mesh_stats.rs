//! Mesh the non-zero voxels of a label map and print timings and mesh statistics (no
//! files written): `cargo run -p mm-segment --release --example mesh_stats -- labels.nii.gz`.

use mm_segment::{mask_to_mesh, Mask, MeshParams};
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let path = std::path::PathBuf::from(std::env::args().nth(1).expect("a NIfTI or NRRD label map"));
    let t = Instant::now();
    let found = mm_volume::scan::classify(&path, std::fs::metadata(&path)?.len());
    let mm_volume::scan::Found::Volume(v) = found else { anyhow::bail!("not a volume") };
    let vol = mm_volume::load::load_volume(&v, 600_000_000, &|_| true)?;
    println!("loaded {:?} in {:.2?}", vol.dims, t.elapsed());
    let mut mask = Mask::new(vol.dims);
    mask.data.iter_mut().zip(&vol.values).for_each(|(m, &x)| *m = (x > 0.5) as u8);
    println!("{} voxels set", mask.count());
    for smoothing in [0.0, 0.5, 1.0] {
        let t = Instant::now();
        let mesh = mask_to_mesh(&mask, &vol.geometry, &MeshParams { smoothing_mm: smoothing, ..Default::default() }, None)?;
        println!("smoothing {smoothing} mm: {} triangles, closed {}, in {:.2?}", mesh.triangle_count(), mesh.is_closed(), t.elapsed());
    }
    Ok(())
}
