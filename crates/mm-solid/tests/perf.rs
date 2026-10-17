//! Performance tests (run with `--release -- --ignored --nocapture`).

mod common;

use common::*;
use glam::DVec3;
use mm_core::{measure, primitives, Mesh};
use mm_solid::{build_field, cell_size_for_resolution, cleanup, extract, make_solid, GridSpec, MakeSolidParams, MeshSdf};
use std::time::Instant;

/// Synthetic vessel tree: a binary tree of curved open tubes whose ends overlap at the
/// junctions (like separately segmented vessel pieces). Roughly `2^depth · rings · slices · 2`
/// triangles.
fn vessel_tree(depth: usize, rings: usize, slices: usize) -> Mesh {
    let mut out = Mesh::new();
    let mut stack = vec![(DVec3::ZERO, DVec3::Z, 2.0, 30.0, 0usize, 0u64)];
    while let Some((start, dir, radius, length, level, seed)) = stack.pop() {
        let mut rng = Rng::new(seed * 7919 + level as u64 + 1);
        let (u, _) = mm_core::geom::orthonormal_basis(dir);
        let bend = u * rng.range(-0.2, 0.2);
        let path: Vec<DVec3> = (0..rings)
            .map(|i| {
                let t = i as f64 / (rings - 1) as f64;
                start + (dir * t + bend * (t * (1.0 - t) * 4.0)) * length
            })
            .collect();
        let radii: Vec<f64> = (0..rings).map(|i| radius * (1.0 - 0.15 * i as f64 / rings as f64)).collect();
        let tube = primitives::tube_along(&path, &radii, slices);
        out.append_mesh(&tube, level as i32);
        if level + 1 < depth {
            let end = *path.last().unwrap();
            let tangent = (path[rings - 1] - path[rings - 2]).normalize();
            let (a, b) = mm_core::geom::orthonormal_basis(tangent);
            let phi = rng.range(0.0, std::f64::consts::TAU);
            let side = a * phi.cos() + b * phi.sin();
            for s in [-1.0, 1.0] {
                let ang: f64 = rng.range(0.35, 0.6);
                let d = (tangent * ang.cos() + side * (s * ang.sin())).normalize();
                // Start slightly inside the parent so the pieces overlap.
                let st = end - tangent * radius * 0.5;
                stack.push((st, d, radius * 0.72, length * 0.8, level + 1, seed * 2 + (s > 0.0) as u64));
            }
        }
    }
    out
}

fn run_breakdown(name: &str, mesh: &Mesh, cells: u32) {
    let h = cell_size_for_resolution(&mesh.bounds(), cells);
    let t0 = Instant::now();
    let sdf = MeshSdf::new(mesh, h, 0.0, 0.5);
    let t_build = t0.elapsed();
    let grid = GridSpec::covering(&mesh.bounds(), h, sdf.band() + 2.0 * h).unwrap();
    let t1 = Instant::now();
    let field = build_field(&sdf, grid, None).unwrap();
    let t_field = t1.elapsed();
    let t2 = Instant::now();
    let (p, t) = extract::marching_tetrahedra(&field);
    let t_mt = t2.elapsed();
    let t3 = Instant::now();
    let (mut m, _) = Mesh::from_triangles(&p, &t, None);
    let t_mesh = t3.elapsed();
    let raw = m.triangle_count();
    let t4 = Instant::now();
    cleanup::collapse_short_edges(&mut m, 0.3 * h);
    let t_collapse = t4.elapsed();
    let t5 = Instant::now();
    cleanup::flip_to_improve(&mut m, 3);
    let t_flip = t5.elapsed();
    let s = grid.samples();
    println!(
        "{name} @ {cells} cells (grid {}x{}x{}, h={h:.4}): input {} tris\n  trees {:?} | field {:?} ({} active bricks, {:.1} MB, {:.1} M samples) | MT {:?} ({raw} tris) | Mesh build {:?} | collapse {:?} -> {} tris | flips {:?}\n  total {:?}",
        s[0],
        s[1],
        s[2],
        mesh.triangle_count(),
        t_build,
        t_field,
        field.active_bricks(),
        field.memory_bytes() as f64 / 1e6,
        field.active_bricks() as f64 * 512.0 / 1e6,
        t_mt,
        t_mesh,
        t_collapse,
        m.triangle_count(),
        t_flip,
        t0.elapsed()
    );
    assert_solid(&m, name);
}

fn run_make_solid(name: &str, mesh: &Mesh, cells: u32) -> std::time::Duration {
    let p = MakeSolidParams::with_resolution(mesh, cells);
    let t = Instant::now();
    let out = make_solid(mesh, &p, None).unwrap();
    let el = t.elapsed();
    println!(
        "make_solid {name} ({} tris) @ {cells} cells: {:?} -> {} tris, {} shell(s), volume {:.1}",
        mesh.triangle_count(),
        el,
        out.triangle_count(),
        mm_core::topology::connected_components(&out).len(),
        measure::volume(&out)
    );
    assert_solid(&out, name);
    el
}

// make_solid only (for clean timing / peak-memory numbers under `/usr/bin/time -l`).

#[test]
#[ignore]
fn perf_sphere_1m() {
    let mesh = primitives::icosphere(50.0, 8); // 1.31M triangles
    let el = run_make_solid("icosphere", &mesh, 256);
    assert!(el.as_secs_f64() < 10.0, "too slow: {el:?}");
}

#[test]
#[ignore]
fn perf_vessels_1m() {
    let mesh = vessel_tree(6, 130, 64); // 1.04M triangles
    let el = run_make_solid("vessel tree", &mesh, 256);
    assert!(el.as_secs_f64() < 10.0, "too slow: {el:?}");
}

#[test]
#[ignore]
fn perf_sphere_1m_512() {
    run_make_solid("icosphere", &primitives::icosphere(50.0, 8), 512);
}

#[test]
#[ignore]
fn perf_vessels_1m_512() {
    run_make_solid("vessel tree", &vessel_tree(6, 130, 64), 512);
}

#[test]
#[ignore]
fn perf_vessels_1m_1024() {
    run_make_solid("vessel tree", &vessel_tree(6, 130, 64), 1024);
}

// Per-phase breakdowns.

#[test]
#[ignore]
fn breakdown_sphere() {
    let mesh = primitives::icosphere(50.0, 8);
    run_breakdown("icosphere", &mesh, 256);
    run_breakdown("icosphere", &mesh, 512);
}

#[test]
#[ignore]
fn breakdown_vessels() {
    let mesh = vessel_tree(6, 130, 64);
    run_breakdown("vessel tree", &mesh, 256);
    run_breakdown("vessel tree", &mesh, 512);
}
