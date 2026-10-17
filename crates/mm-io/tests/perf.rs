//! Timing of large-file IO. Run with:
//! `cargo test -p mm-io --release --test perf -- --ignored --nocapture`

use mm_core::{primitives, Mesh};
use mm_io::*;
use std::path::PathBuf;
use std::time::Instant;

fn time<T>(label: &str, f: impl FnOnce() -> T) -> T {
    let t = Instant::now();
    let r = f();
    println!("{label:<34} {:>8.1} ms", t.elapsed().as_secs_f64() * 1e3);
    r
}

#[test]
#[ignore]
fn perf_1m_triangles() {
    let dir = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("mm-io-perf");
    std::fs::create_dir_all(&dir).unwrap();
    // 12 * 289^2 = 1,002,252 triangles, closed, valence ~6.
    let m = time("generate cube(289)", || primitives::cube(100.0, 289));
    println!("mesh: {} vertices, {} triangles", m.vertex_count(), m.triangle_count());

    let (pos, tris, _) = m.to_arrays();
    time("Mesh::from_triangles (reference)", || Mesh::from_triangles(&pos, &tris, None));

    let stl = dir.join("big.stl");
    time("write binary STL", || write_mesh(&stl, &m, &WriteOptions::default()).unwrap());
    println!("  size {:.1} MB", std::fs::metadata(&stl).unwrap().len() as f64 / 1e6);
    let bytes = std::fs::read(&stl).unwrap();
    let got = time("read binary STL (from memory)", || read_mesh_from_bytes(&bytes, Format::Stl).unwrap());
    let got2 = time("read binary STL (from file)", || read_mesh(&stl).unwrap());
    assert!(got.mesh.is_closed());
    assert_eq!(got.mesh.vertex_count(), m.vertex_count());
    assert_eq!(got2.mesh.triangle_count(), m.triangle_count());

    for (name, opts) in [
        ("big_ascii.stl", WriteOptions { stl_ascii: true, ..Default::default() }),
        ("big.obj", WriteOptions::default()),
        ("big.ply", WriteOptions::default()),
        ("big.off", WriteOptions::default()),
        ("big.3mf", WriteOptions::default()),
    ] {
        let p = dir.join(name);
        time(&format!("write {name}"), || write_mesh(&p, &m, &opts).unwrap());
        let mb = std::fs::metadata(&p).unwrap().len() as f64 / 1e6;
        let got = time(&format!("read {name} ({mb:.0} MB)"), || read_mesh(&p).unwrap());
        assert_eq!(got.mesh.triangle_count(), m.triangle_count());
        assert!(got.mesh.is_closed());
    }
}
