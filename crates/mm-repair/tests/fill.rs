mod common;

use common::*;
use mm_core::{primitives, topology, DVec3, Mesh};
use mm_repair::{fill_hole, FillMode, FillParams, RepairError};
use std::time::Instant;

const MODES: [FillMode; 3] = [FillMode::Minimal, FillMode::Flat, FillMode::Smooth];

#[test]
fn sphere_hole_all_modes() {
    let mut base = primitives::icosphere(1.0, 3);
    let removed = punch_hole(&mut base, DVec3::new(0.0, 0.3, 1.0).normalize(), 0.45);
    assert!(removed > 20);
    let lps = loops(&base);
    assert_eq!(lps.len(), 1);
    let lp = &lps[0];
    let (c, n, loop_dev) = loop_plane(&base, lp);
    for mode in MODES {
        let mut m = base.clone();
        let r = fill_hole(&mut m, lp, &FillParams::new(mode)).unwrap();
        assert_watertight(&m);
        assert!(!r.new_triangles.is_empty());
        assert!(r.new_triangles.iter().all(|&t| m.is_triangle(t)));
        match mode {
            FillMode::Minimal => {
                assert!(r.new_vertices.is_empty());
                assert_eq!(r.new_triangles.len(), lp.len() - 2);
            }
            FillMode::Flat => {
                assert!(r.new_vertices.len() > 5);
                let (mx, _) = max_rms(&m, &r.new_vertices, |p| (p - c).dot(n));
                println!("flat: max plane distance {mx:.5} (loop deviation {loop_dev:.5})");
                assert!(mx <= loop_dev + 1e-9, "flat fill leaves the loop's plane: {mx} > {loop_dev}");
            }
            FillMode::Smooth => {
                assert!(r.new_vertices.len() > 5);
                let (mx, rms) = max_rms(&m, &r.new_vertices, |p| p.length() - 1.0);
                println!("smooth: |r-1| max {mx:.5} rms {rms:.5} over {} vertices", r.new_vertices.len());
                assert!(mx < 0.01, "smooth fill deviates from the sphere by {mx}");
            }
        }
        // Group of new triangles is new.
        let g = m.tri_group(r.new_triangles[0]);
        assert!(r.new_triangles.iter().all(|&t| m.tri_group(t) == g));
        assert!(g > base.max_group_id());
    }
}

#[test]
fn several_sphere_holes_smooth_fill_close_to_sphere() {
    let mut m = primitives::icosphere(2.0, 4);
    let centers = [DVec3::X, -DVec3::X, DVec3::Y, DVec3::new(0.3, -0.6, 0.74).normalize()];
    for c in centers {
        punch_hole(&mut m, c * 2.0, 0.5);
    }
    let lps = loops(&m);
    assert_eq!(lps.len(), 4);
    let mut new_verts = Vec::new();
    for lp in &lps {
        let r = fill_hole(&mut m, lp, &FillParams::new(FillMode::Smooth)).unwrap();
        new_verts.extend(r.new_vertices);
    }
    assert_watertight(&m);
    let (mx, rms) = max_rms(&m, &new_verts, |p| p.length() - 2.0);
    println!("4 holes radius-2 sphere: |r-2| max {mx:.5} rms {rms:.5}");
    assert!(mx < 0.01);
}

#[test]
fn cylinder_both_ends() {
    let base = primitives::cylinder(1.0, 2.0, 24, 6, false);
    for mode in MODES {
        let mut m = base.clone();
        let lps = loops(&m);
        assert_eq!(lps.len(), 2);
        let mut new_verts = Vec::new();
        for lp in &lps {
            let r = fill_hole(&mut m, lp, &FillParams::new(mode)).unwrap();
            new_verts.extend(r.new_vertices);
        }
        assert_watertight(&m);
        if mode == FillMode::Flat {
            assert!(!new_verts.is_empty());
            for &v in &new_verts {
                let p = m.vertex(v);
                assert!(p.z.abs() < 1e-9 || (p.z - 2.0).abs() < 1e-9, "flat cap vertex off the end plane: {p}");
                assert!(p.truncate().length() < 1.0);
            }
            // Volume matches the prism.
            let vol = mm_core::measure::volume(&m);
            let prism = 0.5 * 24.0 * (2.0 * std::f64::consts::PI / 24.0).sin() * 2.0;
            assert!((vol - prism).abs() < 1e-9, "{vol} vs {prism}");
        }
        if mode == FillMode::Smooth {
            // Caps bulge outwards (continuing the straight walls gives a rounded cap).
            assert!(new_verts.iter().any(|&v| m.vertex(v).z > 2.0 + 1e-3));
            assert!(new_verts.iter().any(|&v| m.vertex(v).z < -1e-3));
        }
    }
}

#[test]
fn tube_side_hole_smooth_is_cylindrical() {
    let mut m = primitives::cylinder(1.0, 4.0, 48, 32, false);
    punch_hole(&mut m, DVec3::new(1.0, 0.0, 2.0), 0.6);
    let lps: Vec<_> = loops(&m).into_iter().filter(|l| l.centroid(&m).z > 0.5 && l.centroid(&m).z < 3.5).collect();
    assert_eq!(lps.len(), 1);
    let r = fill_hole(&mut m, &lps[0], &FillParams::new(FillMode::Smooth)).unwrap();
    assert_valid(&m);
    assert_eq!(loops(&m).len(), 2);
    let (mx, rms) = max_rms(&m, &r.new_vertices, |p| p.truncate().length() - 1.0);
    println!("tube side hole: |rho-1| max {mx:.5} rms {rms:.5} over {} vertices", r.new_vertices.len());
    assert!(mx < 0.02);
}

#[test]
fn large_hole_over_500_edges() {
    let mut base = primitives::icosphere(1.0, 7);
    punch_hole(&mut base, DVec3::Z, 0.7);
    let lps = loops(&base);
    assert_eq!(lps.len(), 1);
    let n = lps[0].len();
    println!("large loop: {n} edges");
    assert!(n >= 500, "loop has only {n} edges");
    for mode in MODES {
        let mut m = base.clone();
        let t0 = Instant::now();
        let r = fill_hole(&mut m, &lps[0], &FillParams::new(mode)).unwrap();
        let dt = t0.elapsed().as_secs_f64();
        assert_watertight(&m);
        let (mx, rms) = max_rms(&m, &r.new_vertices, |p| p.length() - 1.0);
        println!(
            "{mode:?}: {} tris, {} verts in {dt:.3}s; |r-1| max {mx:.4} rms {rms:.4}",
            r.new_triangles.len(),
            r.new_vertices.len()
        );
        if mode == FillMode::Smooth {
            assert!(mx < 0.05, "smooth large fill deviates {mx}");
        }
    }
}

#[test]
fn saddle_loop() {
    let f = |x: f64, y: f64| 0.3 * (x * x - y * y);
    let mut base = primitives::plane_grid(2.0, 2.0, 40, 40);
    base.transform(|p| DVec3::new(p.x, p.y, f(p.x, p.y)));
    punch_hole(&mut base, DVec3::ZERO, 0.5);
    let mut lps = loops(&base);
    assert_eq!(lps.len(), 2);
    lps.sort_by_key(|l| l.len());
    lps.truncate(1);
    let (_, _, dev) = loop_plane(&base, &lps[0]);
    assert!(dev > 0.05, "loop should be non-planar ({dev})");
    for mode in MODES {
        let mut m = base.clone();
        let r = fill_hole(&mut m, &lps[0], &FillParams::new(mode)).unwrap();
        assert_valid(&m);
        assert!(m.is_consistently_oriented());
        assert_eq!(loops(&m).len(), 1, "only the outer grid border should remain");
        let (mx, rms) = max_rms(&m, &r.new_vertices, |p| p.z - f(p.x, p.y));
        println!("saddle {mode:?}: |z - f| max {mx:.5} rms {rms:.5}");
        match mode {
            FillMode::Smooth => assert!(mx < 0.01),
            FillMode::Flat => assert!(mx < 0.002),
            FillMode::Minimal => {}
        }
    }
}

#[test]
fn wrapped_band_hole_on_capped_tube() {
    // A hole wrapping 270° around a tube: its projection onto any plane overlaps itself.
    let mut base = primitives::cylinder(1.0, 4.0, 32, 16, true);
    let tris: Vec<u32> = base
        .triangle_ids()
        .filter(|&t| {
            let c = base.tri_centroid(t);
            let ang = c.y.atan2(c.x);
            (c.z - 2.0).abs() < 0.5 && ang.abs() < 0.75 * std::f64::consts::PI && c.truncate().length() > 0.5
        })
        .collect();
    base.remove_triangles(tris);
    let lps = loops(&base);
    assert_eq!(lps.len(), 1);
    for mode in MODES {
        let mut m = base.clone();
        fill_hole(&mut m, &lps[0], &FillParams::new(mode)).unwrap();
        assert_watertight(&m);
    }
}

#[test]
fn existing_chord_is_never_used() {
    // Square pyramid without its base (hole A-B-C-D), plus a fin triangle (A, C, F) that
    // makes A–C an existing edge outside the hole.
    let pos = vec![
        DVec3::new(-1.0, -1.0, 0.0), // A 0
        DVec3::new(1.0, -1.0, 0.0),  // B 1
        DVec3::new(1.0, 1.0, 0.0),   // C 2
        DVec3::new(-1.0, 1.0, 0.0),  // D 3
        DVec3::new(0.0, 0.0, 1.0),   // E 4
        DVec3::new(0.0, 0.0, 2.0),   // F 5
    ];
    let tris = [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4], [0, 2, 5]];
    let (base, _) = Mesh::from_triangles(&pos, &tris, None);
    let lp = loops(&base).into_iter().find(|l| l.len() == 4).expect("square loop");
    for mode in MODES {
        let mut m = base.clone();
        let r = fill_hole(&mut m, &lp, &FillParams::new(mode)).unwrap();
        assert_valid(&m);
        let ac = m.find_edge(0, 2).unwrap();
        assert!(m.is_boundary_edge(ac), "{mode:?}: fill used the existing A-C edge");
        for &t in &r.new_triangles {
            let tv = m.triangle(t);
            assert!(!(tv.contains(&0) && tv.contains(&2)));
        }
    }
}

#[test]
fn loop_through_same_vertex_twice() {
    // Remove two triangles of one vertex's fan that are not adjacent: the hole touches v twice.
    let mut base = primitives::icosphere(1.0, 2);
    let v = base.vertex_ids().find(|&v| base.vertex_valence(v) == 6).unwrap();
    let fan = mm_repair::ops::vertex_fans(&base, v);
    assert_eq!(fan.len(), 1);
    // Order the fan by walking: pick a triangle and the one two steps away (non-adjacent).
    let tris = &fan[0];
    let t0 = tris[0];
    let far = *tris
        .iter()
        .find(|&&t| t != t0 && !base.tri_neighbors(t0).contains(&t) && {
            let s: Vec<u32> = base.triangle(t).to_vec();
            s.contains(&v)
        })
        .unwrap();
    base.remove_triangles([t0, far]);
    assert!(base.is_bowtie_vertex(v));
    let lps = loops(&base);
    for mode in MODES {
        let mut m = base.clone();
        for lp in &lps {
            fill_hole(&mut m, lp, &FillParams::new(mode)).unwrap();
        }
        assert_valid(&m);
        assert!(m.is_closed());
        assert!(m.is_consistently_oriented());
    }
}

#[test]
fn stale_or_bad_loops_are_rejected_without_changes() {
    let mut m = primitives::icosphere(1.0, 2);
    punch_hole(&mut m, DVec3::Z, 0.4);
    let lp = loops(&m).remove(0);
    fill_hole(&mut m, &lp, &FillParams::default()).unwrap();
    let snapshot = (m.vertex_count(), m.triangle_count(), m.edge_count());
    let err = fill_hole(&mut m, &lp, &FillParams::default()).unwrap_err();
    assert!(matches!(err, RepairError::InvalidLoop(_)), "{err}");
    assert_eq!(snapshot, (m.vertex_count(), m.triangle_count(), m.edge_count()));

    // A lone triangle's boundary cannot be filled (it would duplicate the triangle).
    let (mut t, _) = Mesh::from_triangles(&[DVec3::ZERO, DVec3::X, DVec3::Y], &[[0, 1, 2]], None);
    let lp = loops(&t).remove(0);
    assert_eq!(fill_hole(&mut t, &lp, &FillParams::default()).unwrap_err(), RepairError::IsolatedTriangle);
    assert_eq!(t.triangle_count(), 1);
    assert_valid(&t);

    let p = FillParams { target_edge_length: Some(-1.0), ..Default::default() };
    let mut m = primitives::cylinder(1.0, 1.0, 8, 1, false);
    let lp = loops(&m).remove(0);
    assert!(matches!(fill_hole(&mut m, &lp, &p), Err(RepairError::InvalidParameter(_))));
}

#[test]
fn reversed_loop_is_accepted_and_target_length_respected() {
    let mut m = primitives::cylinder(1.0, 2.0, 32, 4, false);
    let mut lp = loops(&m).remove(0);
    lp.vertices.reverse();
    lp.edges.reverse();
    let params = FillParams { mode: FillMode::Flat, target_edge_length: Some(0.1), group: Some(7), ..Default::default() };
    let r = fill_hole(&mut m, &lp, &params).unwrap();
    assert_valid(&m);
    assert!(m.is_consistently_oriented());
    assert!(r.new_triangles.iter().all(|&t| m.tri_group(t) == 7));
    let mut lens = Vec::new();
    let mut seen = std::collections::HashSet::new();
    for &t in &r.new_triangles {
        for e in m.tri_edges(t) {
            if seen.insert(e) && !m.is_boundary_edge(e) {
                lens.push(m.edge_length(e));
            }
        }
    }
    let mean = lens.iter().sum::<f64>() / lens.len() as f64;
    println!("flat fill with target 0.1: {} tris, mean edge {mean:.4}", r.new_triangles.len());
    assert!(mean > 0.06 && mean < 0.14, "mean edge {mean}");
    assert!(topology::boundary_loops(&m).len() == 1);
}

#[test]
fn large_wrapped_hole_uses_fan_fallback() {
    // > DP_MAX_LOOP edges and a self-overlapping projection: neither the dynamic program
    // nor planar ear clipping applies, so the centroid fan + refinement path runs.
    let mut base = primitives::cylinder(1.0, 4.0, 240, 40, true);
    let tris: Vec<u32> = base
        .triangle_ids()
        .filter(|&t| {
            let c = base.tri_centroid(t);
            (c.z - 2.0).abs() < 0.5 && c.y.atan2(c.x).abs() < 0.75 * std::f64::consts::PI && c.truncate().length() > 0.5
        })
        .collect();
    base.remove_triangles(tris);
    let lps = loops(&base);
    assert_eq!(lps.len(), 1);
    assert!(lps[0].len() > mm_repair::fill::DP_MAX_LOOP, "{}", lps[0].len());
    for mode in MODES {
        let mut m = base.clone();
        let t0 = Instant::now();
        let r = fill_hole(&mut m, &lps[0], &FillParams::new(mode)).unwrap();
        assert_watertight(&m);
        let (mx, rms) = max_rms(&m, &r.new_vertices, |p| p.truncate().length() - 1.0);
        println!(
            "wrapped {mode:?}: {} edges, {} new verts in {:.3}s, |rho-1| max {mx:.3} rms {rms:.3}",
            lps[0].len(),
            r.new_vertices.len(),
            t0.elapsed().as_secs_f64()
        );
    }
}

#[test]
fn dynamic_program_at_its_size_limit() {
    let n = mm_repair::fill::DP_MAX_LOOP;
    let base = primitives::cylinder(1.0, 1.0, n, 2, false);
    let lp = loops(&base).remove(0);
    assert_eq!(lp.len(), n);
    let mut m = base.clone();
    let t0 = Instant::now();
    let r = fill_hole(&mut m, &lp, &FillParams::new(FillMode::Minimal)).unwrap();
    println!("minimal DP fill of a {n}-edge loop: {:.3}s", t0.elapsed().as_secs_f64());
    assert_eq!(r.new_triangles.len(), n - 2);
    assert!(r.new_vertices.is_empty());
    assert_valid(&m);
    assert_eq!(loops(&m).len(), 1);
    // Planar loop: the minimal fill is flat.
    for &t in &r.new_triangles {
        for p in m.tri_positions(t) {
            assert!(p.z.abs() < 1e-12 || (p.z - 1.0).abs() < 1e-12);
        }
    }
}
