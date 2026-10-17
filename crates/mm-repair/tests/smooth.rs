mod common;

use common::*;
use mm_core::{measure, primitives, DVec3, Mesh};
use mm_repair::{
    fair_region, fair_region_with, smooth_vertices, smooth_vertices_with_progress, FairParams, LaplacianWeights,
    RepairError, SmoothMethod, SmoothParams,
};

fn noisy_sphere(sub: usize, amp: f64, seed: u64) -> Mesh {
    let mut m = primitives::icosphere(1.0, sub);
    let mut rng = Lcg(seed);
    let ids: Vec<u32> = m.vertex_ids().collect();
    for v in ids {
        let p = m.vertex(v);
        m.set_vertex(v, p * (1.0 + amp * rng.sym()));
    }
    m
}

fn radial_stats(m: &Mesh) -> (f64, f64) {
    let n = m.vertex_count() as f64;
    let mean = m.vertex_ids().map(|v| m.vertex(v).length()).sum::<f64>() / n;
    let rms = (m.vertex_ids().map(|v| (m.vertex(v).length() - mean).powi(2)).sum::<f64>() / n).sqrt();
    (mean, rms)
}

#[test]
fn taubin_denoises_without_shrinking() {
    let base = noisy_sphere(4, 0.02, 3);
    let (_, rms0) = radial_stats(&base);
    let mut taubin = base.clone();
    let p = SmoothParams { iterations: 20, strength: 0.5, method: SmoothMethod::Taubin { mu: -0.53 }, ..Default::default() };
    let moved = smooth_vertices(&mut taubin, None, &p);
    assert_eq!(moved, base.vertex_count());
    assert_valid(&taubin);
    let (mean_t, rms_t) = radial_stats(&taubin);

    let mut lap = base.clone();
    let p = SmoothParams { iterations: 20, strength: 0.5, method: SmoothMethod::Laplacian, ..Default::default() };
    smooth_vertices(&mut lap, None, &p);
    let (mean_l, rms_l) = radial_stats(&lap);
    println!("noise rms {rms0:.5}; taubin mean r {mean_t:.5} rms {rms_t:.5}; laplacian mean r {mean_l:.5} rms {rms_l:.5}");
    assert!(rms_t < rms0 / 3.0, "Taubin should remove most noise");
    assert!(mean_t > 0.99, "Taubin shrank the sphere to {mean_t}");
    assert!(mean_l < mean_t, "plain Laplacian shrinks more than Taubin");
    assert!(rms_l < rms0 / 3.0);
}

#[test]
fn all_weight_schemes_smooth() {
    let base = noisy_sphere(3, 0.02, 11);
    let (_, rms0) = radial_stats(&base);
    for w in [LaplacianWeights::Uniform, LaplacianWeights::Cotangent, LaplacianWeights::MeanValue] {
        let mut m = base.clone();
        let p = SmoothParams { iterations: 10, strength: 0.5, weights: w, ..Default::default() };
        smooth_vertices(&mut m, None, &p);
        assert_valid(&m);
        let (mean, rms) = radial_stats(&m);
        println!("{w:?}: mean {mean:.5} rms {rms:.5}");
        assert!(rms < rms0 / 2.0, "{w:?} did not smooth: {rms} vs {rms0}");
        assert!(mean > 0.98);
    }
}

#[test]
fn volume_preservation() {
    let base = noisy_sphere(3, 0.03, 5);
    let v0 = measure::volume(&base);
    let mut m = base.clone();
    let p = SmoothParams { iterations: 30, strength: 0.6, method: SmoothMethod::Laplacian, preserve_volume: true, ..Default::default() };
    smooth_vertices(&mut m, None, &p);
    let v1 = measure::volume(&m);
    println!("volume {v0:.5} -> {v1:.5}");
    assert!((v1 - v0).abs() < 1e-3 * v0);
    let mut shr = base.clone();
    smooth_vertices(&mut shr, None, &SmoothParams { preserve_volume: false, ..p });
    assert!(measure::volume(&shr) < 0.97 * v0, "without preservation Laplacian smoothing shrinks");
}

#[test]
fn selection_only_moves_selected_and_boundary_is_kept() {
    // Noisy open grid; smooth a central selection, then everything with boundary preserved.
    let mut m = primitives::plane_grid(2.0, 2.0, 20, 20);
    let mut rng = Lcg(9);
    let ids: Vec<u32> = m.vertex_ids().collect();
    for &v in &ids {
        let p = m.vertex(v);
        m.set_vertex(v, p + DVec3::Z * 0.05 * rng.sym());
    }
    let before = m.clone();
    let sel: Vec<u32> = ids.iter().copied().filter(|&v| m.vertex(v).truncate().length() < 0.5).collect();
    let moved = smooth_vertices(&mut m, Some(&sel), &SmoothParams::default());
    assert_eq!(moved, sel.len());
    for &v in &ids {
        if !sel.contains(&v) {
            assert_eq!(m.vertex(v), before.vertex(v));
        }
    }
    let rough = |m: &Mesh, vs: &[u32]| vs.iter().map(|&v| m.vertex(v).z.abs()).sum::<f64>() / vs.len() as f64;
    assert!(rough(&m, &sel) < 0.5 * rough(&before, &sel));

    let mut all = before.clone();
    smooth_vertices(&mut all, None, &SmoothParams { preserve_boundary: true, ..Default::default() });
    for &v in &ids {
        if before.is_boundary_vertex(v) {
            assert_eq!(all.vertex(v), before.vertex(v));
        }
    }
    let mut free = before.clone();
    smooth_vertices(&mut free, None, &SmoothParams { preserve_boundary: false, ..Default::default() });
    assert_valid(&free);
    let corner = ids.iter().copied().find(|&v| before.is_boundary_vertex(v) && before.vertex(v).x > 0.99 && before.vertex(v).y.abs() < 0.5).unwrap();
    assert_ne!(free.vertex(corner), before.vertex(corner));
    // Boundary vertices are smoothed along the boundary curve (only the grid corners round
    // off, which propagates a little along the straight edge).
    assert!((free.vertex(corner).x - 1.0).abs() < 1e-2);
}

#[test]
fn smoothing_cancel() {
    let mut m = noisy_sphere(2, 0.02, 1);
    let r = smooth_vertices_with_progress(&mut m, None, &SmoothParams { iterations: 50, ..Default::default() }, Some(&|f| f < 0.2));
    assert_eq!(r, Err(RepairError::Cancelled));
    assert_valid(&m);
}

/// Unit icosphere with a smooth dent of depth `depth` in the cap around +Z.
fn dented_sphere(sub: usize, cap_angle: f64, depth: f64) -> (Mesh, Vec<u32>) {
    let mut m = primitives::icosphere(1.0, sub);
    let cos_cap = cap_angle.cos();
    let region: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).normalize().z > cos_cap).collect();
    for &v in &region {
        let p = m.vertex(v);
        let a = p.normalize().z.clamp(-1.0, 1.0).acos() / cap_angle; // 0 at pole, 1 at rim
        let bump = depth * (1.0 - a * a).powi(2);
        m.set_vertex(v, p * (1.0 - bump));
    }
    (m, region)
}

#[test]
fn fair_region_restores_dented_sphere() {
    let cap = 25f64.to_radians();
    let (dented, region) = dented_sphere(4, cap, 0.15);
    let (d0, _) = max_rms(&dented, &region, |p| p.length() - 1.0);
    let mut results = Vec::new();
    for (order, w) in [(2, LaplacianWeights::Cotangent), (2, LaplacianWeights::Uniform), (1, LaplacianWeights::Cotangent)] {
        let mut m = dented.clone();
        let st = fair_region(&mut m, &region, order, w).unwrap();
        assert!(st.converged, "{st:?}");
        assert_valid(&m);
        let (mx, rms) = max_rms(&m, &region, |p| p.length() - 1.0);
        println!("order {order} {w:?}: dent {d0:.4} -> max {mx:.5} rms {rms:.5} ({} its)", st.iterations);
        results.push(mx);
        // Fixed vertices did not move.
        for v in m.vertex_ids() {
            if !region.contains(&v) {
                assert_eq!(m.vertex(v), dented.vertex(v));
            }
        }
    }
    assert!(results[0] < 0.01, "bilaplacian fairing should restore the sphere: {}", results[0]);
    assert!(results[1] < 0.015);
    assert!(results[2] > results[0], "membrane flattens more than thin plate");
}

#[test]
fn fair_region_options_and_errors() {
    let cap = 20f64.to_radians();
    let (dented, region) = dented_sphere(3, cap, 0.1);
    let mut m = dented.clone();
    assert!(matches!(fair_region(&mut m, &region, 3, LaplacianWeights::Uniform), Err(RepairError::InvalidParameter(_))));
    let p = FairParams { reweight_passes: 2, weights: LaplacianWeights::MeanValue, ..Default::default() };
    fair_region_with(&mut m, &region, &p, None).unwrap();
    let (mx, _) = max_rms(&m, &region, |p| p.length() - 1.0);
    assert!(mx < 0.02, "{mx}");
    // Cancelling leaves the mesh untouched.
    let mut c = dented.clone();
    let p = FairParams { tolerance: 1e-14, max_iterations: 100_000, ..Default::default() };
    assert_eq!(fair_region_with(&mut c, &region, &p, Some(&|_| false)), Err(RepairError::Cancelled));
    for v in c.vertex_ids() {
        assert_eq!(c.vertex(v), dented.vertex(v));
    }
    // Selecting a whole closed component: nothing is fixed, so nothing moves.
    let mut s = primitives::icosphere(1.0, 2);
    let all: Vec<u32> = s.vertex_ids().collect();
    let before = s.clone();
    fair_region(&mut s, &all, 2, LaplacianWeights::Cotangent).unwrap();
    for v in s.vertex_ids() {
        assert_eq!(s.vertex(v), before.vertex(v));
    }
}

#[test]
fn bumpy_tube_patch_is_flattened_into_the_vessel() {
    // A vessel-like tube with a blob on its side; fairing the blob restores the cylinder.
    let mut m = primitives::cylinder(1.0, 6.0, 48, 48, false);
    let c = DVec3::new(1.0, 0.0, 3.0);
    let region: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).distance(c) < 0.8).collect();
    for &v in &region {
        let p = m.vertex(v);
        let t = 1.0 - p.distance(c) / 0.8;
        let radial = DVec3::new(p.x, p.y, 0.0).normalize();
        m.set_vertex(v, p + radial * 0.3 * t * t);
    }
    fair_region(&mut m, &region, 2, LaplacianWeights::Cotangent).unwrap();
    assert_valid(&m);
    let (mx, rms) = max_rms(&m, &region, |p| p.truncate().length() - 1.0);
    println!("tube blob: |rho-1| max {mx:.5} rms {rms:.5}");
    // A thin plate cannot reproduce a cylinder exactly; for a ±47° patch the expected
    // deviation is ~0.8⁴/24 ≈ 0.017 (vs. the 0.3 blob).
    assert!(mx < 0.025);
}

/// Timing on a ~1.3M triangle mesh (`cargo test --release -p mm-repair -- --ignored --nocapture`).
#[test]
#[ignore]
fn perf_smoothing_and_fairing_1m() {
    let m = noisy_sphere(8, 0.002, 77);
    println!("{} triangles", m.triangle_count());
    for (name, w) in [("uniform", LaplacianWeights::Uniform), ("cotangent", LaplacianWeights::Cotangent)] {
        let mut c = m.clone();
        let t = std::time::Instant::now();
        smooth_vertices(&mut c, None, &SmoothParams { iterations: 10, weights: w, ..Default::default() });
        println!("taubin x10 ({name}) on all {} vertices: {:.3}s", c.vertex_count(), t.elapsed().as_secs_f64());
    }
    for cap_deg in [10.0f64, 25.0] {
        let cos = cap_deg.to_radians().cos();
        let region: Vec<u32> = m.vertex_ids().filter(|&v| m.vertex(v).normalize().z > cos).collect();
        let mut c = m.clone();
        let t = std::time::Instant::now();
        let st = fair_region(&mut c, &region, 2, LaplacianWeights::Cotangent).unwrap();
        let (mx, _) = max_rms(&c, &region, |p| p.length() - 1.0);
        println!(
            "fair_region order 2 on {} vertices: {:.3}s ({} its, converged {}), |r-1| max {mx:.5}",
            region.len(),
            t.elapsed().as_secs_f64(),
            st.iterations,
            st.converged
        );
    }
    m.check_validity().unwrap();
}
