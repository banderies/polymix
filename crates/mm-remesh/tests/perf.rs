//! Performance checks; run with
//! `CARGO_TARGET_DIR=target/agent-remesh cargo test -p mm-remesh --release --test perf -- --ignored --nocapture --test-threads=1`

mod common;

use common::*;
use mm_core::{primitives, DVec3, Mesh};
use mm_remesh::*;
use std::time::Instant;

/// Curved, tapering, slightly noisy tube (vessel-like), open at both ends.
fn vessel(slices: usize, rings: usize, seed: u64) -> Mesh {
    let path: Vec<DVec3> = (0..rings)
        .map(|i| {
            let s = i as f64 / (rings - 1) as f64;
            DVec3::new(8.0 * (2.0 * s).sin(), 6.0 * s * s, 40.0 * s)
        })
        .collect();
    let radii: Vec<f64> = (0..rings).map(|i| 2.5 - 1.5 * i as f64 / (rings - 1) as f64).collect();
    let mut m = primitives::tube_along(&path, &radii, slices);
    let mut rng = Lcg(seed);
    for v in m.vertex_ids().collect::<Vec<_>>() {
        let p = m.vertex(v);
        let d = DVec3::new(rng.f() - 0.5, rng.f() - 0.5, rng.f() - 0.5) * 0.004;
        m.set_vertex(v, p + d);
    }
    m
}

fn report(label: &str, m: &Mesh, secs: f64) {
    let st = edge_length_stats(m);
    println!(
        "{label}: {:.3}s  tris {}  edge mean {:.4} sd {:.4}  q min {:.3} mean {:.3}",
        secs,
        m.triangle_count(),
        st.mean,
        st.std_dev,
        min_quality(m),
        mean_quality(m)
    );
}

#[test]
#[ignore]
fn perf_remesh_500k() {
    let base = vessel(280, 1150, 1);
    println!("input: {} tris, mean edge {:.4}", base.triangle_count(), edge_length_stats(&base).mean);
    let mean = edge_length_stats(&base).mean;
    for (label, factor, iters) in [("same density, 1 it", 1.0, 1), ("same density, 5 it", 1.0, 5), ("coarsen 2x, 5 it", 2.0, 5), ("refine 0.7x, 3 it", 0.7, 3)] {
        let mut m = base.clone();
        let t = Instant::now();
        let rep = remesh(
            &mut m,
            &RemeshParams { iterations: iters, boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(mean * factor) },
            None,
        )
        .unwrap();
        let secs = t.elapsed().as_secs_f64();
        report(label, &m, secs);
        println!("    splits {} collapses {} flips {}  {:?}", rep.splits, rep.collapses, rep.flips, rep.timings);
        m.check_validity().unwrap();
    }
}

#[test]
#[ignore]
fn perf_remesh_icosphere_327k() {
    let base = primitives::icosphere(10.0, 7);
    let mean = edge_length_stats(&base).mean;
    let mut m = base.clone();
    let t = Instant::now();
    let rep = remesh(&mut m, &RemeshParams { iterations: 1, ..RemeshParams::with_target(mean) }, None).unwrap();
    report("icosphere 327k, 1 it", &m, t.elapsed().as_secs_f64());
    println!("    {:?}", rep.timings);
    let mut m = base.clone();
    let t = Instant::now();
    let rep = remesh(&mut m, &RemeshParams { iterations: 3, ..RemeshParams::with_target(mean * 0.8) }, None).unwrap();
    report("icosphere 327k -> 0.8x, 3 it", &m, t.elapsed().as_secs_f64());
    println!("    {:?}", rep.timings);
    let mut m = base.clone();
    let t = Instant::now();
    remesh(&mut m, &RemeshParams { iterations: 3, adaptive: true, ..RemeshParams::with_target(mean) }, None).unwrap();
    report("icosphere 327k, adaptive 3 it", &m, t.elapsed().as_secs_f64());
}

#[test]
#[ignore]
fn perf_reduce_1m_to_100k() {
    let base = primitives::icosphere(10.0, 8);
    println!("input: {} tris", base.triangle_count());
    let mut m = base.clone();
    let t = Instant::now();
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(100_000)), None).unwrap();
    report("reduce icosphere 1.31M -> 100k", &m, t.elapsed().as_secs_f64());
    println!("    {rep:?}");
    m.check_validity().unwrap();

    let base = vessel(350, 1430, 2);
    println!("input: {} tris", base.triangle_count());
    let mut m = base.clone();
    let t = Instant::now();
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(100_000)), None).unwrap();
    report("reduce vessel 1M -> 100k", &m, t.elapsed().as_secs_f64());
    m.check_validity().unwrap();

    let mut m = base.clone();
    let t = Instant::now();
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::MaxDeviation(0.02)), None).unwrap();
    report("reduce vessel 1M, max deviation 0.02", &m, t.elapsed().as_secs_f64());
    println!("    {:?}", rep.max_deviation);
    m.check_validity().unwrap();
}

#[test]
#[ignore]
fn perf_reduce_maxdev_only() {
    let base = vessel(350, 1430, 2);
    let mut m = base.clone();
    let t = Instant::now();
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::MaxDeviation(0.02)), None).unwrap();
    report("reduce vessel 1M, max deviation 0.02", &m, t.elapsed().as_secs_f64());
    println!("    {:?}", rep.max_deviation);
}
