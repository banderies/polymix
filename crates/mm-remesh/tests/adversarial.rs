//! Adversarial inputs: slivers, needles, caps, zero-length edges, tiny components,
//! high-valence vertices, non-manifold builder splits, noise, random regions.

mod common;

use common::*;
use mm_core::{measure, primitives, Bvh, DVec3, Mesh};
use mm_remesh::*;

fn random_edge(m: &Mesh, rng: &mut Lcg) -> u32 {
    loop {
        let e = rng.below(m.max_eid()) as u32;
        if m.is_edge(e) {
            return e;
        }
    }
}

/// Riddle a mesh with needles (near-endpoint splits), caps (pokes next to an edge) and
/// zero-length edges, keeping the surface geometry unchanged.
fn add_slivers(m: &mut Mesh, n: usize, seed: u64) {
    let mut rng = Lcg(seed);
    for _ in 0..n {
        match rng.below(4) {
            0 => {
                let e = random_edge(m, &mut rng);
                m.split_edge_at(e, 1e-7).unwrap();
            }
            1 => {
                let e = random_edge(m, &mut rng);
                m.split_edge_at(e, 0.0).unwrap(); // exact duplicate position
            }
            2 => {
                let e = random_edge(m, &mut rng);
                let t = m.edge_triangles(e)[0];
                m.poke_triangle(t, DVec3::new(0.5, 0.5 - 1e-9, 1e-9)).unwrap();
            }
            _ => {
                let e = random_edge(m, &mut rng);
                m.split_edge_at(e, 0.5).unwrap();
            }
        }
    }
}

#[test]
fn sliver_plane_remesh_and_reduce() {
    let mut base = primitives::plane_grid(1.0, 1.0, 16, 16);
    add_slivers(&mut base, 400, 11);
    assert_valid(&base);
    println!("input min quality {:.2e}", min_quality(&base));
    let segs = boundary_segments(&base);

    let mut m = base.clone();
    let rep = remesh(&mut m, &RemeshParams { boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(0.06) }, None)
        .unwrap();
    assert_valid(&m);
    println!("{rep:?} min q {:.3} mean q {:.3}", min_quality(&m), mean_quality(&m));
    for v in m.vertex_ids() {
        assert!(m.vertex(v).z.abs() < 1e-12);
    }
    for v in boundary_vertices(&m) {
        assert!(dist_to_segments(m.vertex(v), &segs) < 1e-9);
    }
    assert!(min_quality(&m) > 0.3, "slivers survived: {}", min_quality(&m));
    let area: f64 = m.triangle_ids().map(|t| m.tri_area(t)).sum();
    assert!((area - 1.0).abs() < 1e-3, "area {area}");

    let mut m = base.clone();
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(20.0)), None).unwrap();
    assert_valid(&m);
    for v in m.vertex_ids() {
        assert!(m.vertex(v).z.abs() < 1e-9);
    }

    let mut m = base.clone();
    collapse_short_edges(&mut m, 1e-3, None, None).unwrap();
    assert_valid(&m);
}

#[test]
fn sliver_sphere_stays_closed() {
    let mut base = primitives::icosphere(1.0, 3);
    add_slivers(&mut base, 600, 3);
    let v0 = measure::volume(&base);
    for (i, adaptive) in [false, true].into_iter().enumerate() {
        let mut m = base.clone();
        remesh(&mut m, &RemeshParams { adaptive, ..RemeshParams::with_target(0.08) }, None).unwrap();
        assert_valid(&m);
        assert!(m.is_closed());
        assert!(((measure::volume(&m) - v0) / v0).abs() < 0.01, "case {i}");
        assert!(min_quality(&m) > 0.2, "case {i}: min q {}", min_quality(&m));
    }
    let mut m = base.clone();
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(300)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(m.triangle_count() <= 302);
    let mut m = base.clone();
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::MaxDeviation(0.01)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
}

#[test]
fn high_valence_caps_are_fixed() {
    // Capped cylinder: each cap centre has valence 96.
    let mut m = primitives::cylinder(1.0, 2.0, 96, 4, true);
    let maxv0 = m.vertex_ids().map(|v| m.vertex_valence(v)).max().unwrap();
    assert!(maxv0 >= 96);
    let rep = remesh(&mut m, &RemeshParams { preserve_sharp_edges: Some(45.0), ..RemeshParams::with_target(0.12) }, None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    let maxv = m.vertex_ids().map(|v| m.vertex_valence(v)).max().unwrap();
    println!("{rep:?} max valence {maxv} min q {:.3}", min_quality(&m));
    assert!(maxv <= 10, "max valence {maxv}");
    // Rim creases stay sharp: every triangle lies on one cap or on the side wall.
    let rmin = (std::f64::consts::PI / 96.0).cos() - 1e-9;
    for t in m.triangle_ids() {
        let p = m.tri_positions(t);
        let on_cap = |z: f64| p.iter().all(|q| (q.z - z).abs() < 1e-9);
        let on_side = p.iter().all(|q| (q.x * q.x + q.y * q.y).sqrt() > rmin);
        assert!(on_cap(0.0) || on_cap(2.0) || on_side, "triangle straddles the rim: {p:?}");
    }
    let mut m = primitives::cylinder(1.0, 2.0, 96, 4, true);
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(30.0)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
}

#[test]
fn tiny_components_and_isolated_triangles() {
    let mut m = primitives::icosphere(1.0, 3);
    for k in 0..5 {
        let mut t = primitives::icosphere(1e-4, 0);
        t.transform(|p| p + DVec3::new(3.0 + k as f64, 0.0, 0.0));
        m.append_mesh(&t, 10 + k);
        let tet_pos = [DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z].map(|p| p * 1e-5 + DVec3::new(0.0, 3.0 + k as f64, 0.0));
        let (tet, _) = Mesh::from_triangles(&tet_pos, &[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]], None);
        m.append_mesh(&tet, 20 + k);
    }
    let a = m.append_vertex(DVec3::new(0.0, 0.0, 5.0));
    let b = m.append_vertex(DVec3::new(1e-6, 0.0, 5.0));
    let c = m.append_vertex(DVec3::new(0.0, 1e-6, 5.0));
    m.append_triangle([a, b, c], 99).unwrap();
    let mut quad = primitives::plane_grid(1e-5, 1e-5, 1, 1);
    quad.transform(|p| p + DVec3::new(0.0, 0.0, -5.0));
    m.append_mesh(&quad, 98);
    assert_valid(&m);
    let base = m.clone();
    let ncomp0 = mm_core::topology::connected_components(&base).len();

    for mode in [BoundaryMode::Fixed, BoundaryMode::ConstrainToBoundary, BoundaryMode::Free] {
        let mut m = base.clone();
        remesh(&mut m, &RemeshParams { boundary: mode, ..RemeshParams::with_target(0.1) }, None).unwrap();
        assert_valid(&m);
        assert_eq!(mm_core::topology::connected_components(&m).len(), ncomp0, "{mode:?}");
    }
    let mut m = base.clone();
    reduce(&mut m, &ReduceParams { preserve_boundary: false, ..ReduceParams::with_target(ReduceTarget::Percentage(10.0)) }, None).unwrap();
    assert_valid(&m);
    let mut m = base.clone();
    collapse_short_edges(&mut m, 0.01, None, None).unwrap();
    assert_valid(&m);
}

#[test]
fn builder_split_nonmanifold_and_bowtie_input() {
    // Fan of three triangles on one edge (builder duplicates one) plus a bowtie vertex.
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.5, 1.0, 0.0),
        DVec3::new(0.5, -1.0, 0.0),
        DVec3::new(0.5, 0.0, 1.0),
        DVec3::new(2.0, 1.0, 0.0),
        DVec3::new(2.0, -1.0, 0.0),
    ];
    let tris = [[0, 1, 2], [1, 0, 3], [0, 1, 4], [1, 5, 6]];
    let (base, rep) = Mesh::from_triangles(&pos, &tris, None);
    assert_eq!(rep.nonmanifold_split, 1);
    for mode in [BoundaryMode::Fixed, BoundaryMode::ConstrainToBoundary, BoundaryMode::Free] {
        let mut m = base.clone();
        remesh(&mut m, &RemeshParams { boundary: mode, ..RemeshParams::with_target(0.1) }, None).unwrap();
        assert_valid(&m);
        // The bowtie vertex 1 is never moved or removed.
        assert_eq!(m.vertex(1), pos[1]);
    }
    let mut m = base.clone();
    reduce(&mut m, &ReduceParams { preserve_boundary: false, ..ReduceParams::with_target(ReduceTarget::TriangleCount(1)) }, None).unwrap();
    assert_valid(&m);
}

#[test]
fn noisy_sphere() {
    let mut m = primitives::icosphere(1.0, 4);
    let mut rng = Lcg(77);
    for v in m.vertex_ids().collect::<Vec<_>>() {
        let p = m.vertex(v) * (1.0 + 0.02 * (rng.f() - 0.5));
        m.set_vertex(v, p);
    }
    let orig = m.clone();
    remesh(&mut m, &RemeshParams::with_target(0.05), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    let bvh = Bvh::build(&orig);
    let max_d = m.vertex_ids().map(|v| dist_to(&orig, &bvh, m.vertex(v))).fold(0.0, f64::max);
    assert!(max_d < 1e-9, "{max_d}");
    let mut m = orig.clone();
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(10.0)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
}

#[test]
fn random_regions_and_parameters_fuzz() {
    let mut rng = Lcg(2024);
    for case in 0..24 {
        let mut m = match case % 4 {
            0 => primitives::icosphere(1.0, 3),
            1 => primitives::cylinder(1.0, 2.0, 20, 6, case % 8 == 1),
            2 => primitives::cube(2.0, 4),
            _ => {
                let mut g = primitives::plane_grid(2.0, 2.0, 14, 14);
                g.transform(|p| DVec3::new(p.x, p.y, 0.2 * (p.x * 2.0).sin()));
                g
            }
        };
        if case % 3 == 0 {
            add_slivers(&mut m, 60, case as u64);
        }
        for t in m.triangle_ids().collect::<Vec<_>>() {
            if m.tri_centroid(t).y > 0.4 {
                m.set_tri_group(t, 1);
            }
        }
        // Random blob region (possibly several pieces).
        let c = DVec3::new(rng.f() - 0.5, rng.f() - 0.5, rng.f() - 0.5) * 2.0;
        let r = 0.4 + rng.f();
        let region: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).distance(c) < r).collect();
        let use_region = case % 2 == 0;
        let before = snapshot_outside(&m, if use_region { &region } else { &[] });
        let mean = edge_length_stats(&m).mean;
        let mut work = m.clone();
        let new_region = if case % 5 == 4 {
            let rep = reduce(
                &mut work,
                &ReduceParams {
                    region: use_region.then(|| region.clone()),
                    preserve_boundary: rng.below(2) == 0,
                    preserve_group_boundaries: rng.below(2) == 0,
                    ..ReduceParams::with_target(ReduceTarget::Percentage(20.0 + 50.0 * rng.f()))
                },
                None,
            )
            .unwrap();
            rep.region_triangles
        } else {
            let rep = remesh(
                &mut work,
                &RemeshParams {
                    region: use_region.then(|| region.clone()),
                    boundary: [BoundaryMode::Fixed, BoundaryMode::ConstrainToBoundary, BoundaryMode::Free][rng.below(3)],
                    preserve_sharp_edges: if rng.below(2) == 0 { Some(40.0) } else { None },
                    preserve_group_boundaries: rng.below(2) == 0,
                    adaptive: rng.below(3) == 0,
                    iterations: 3 + rng.below(6),
                    ..RemeshParams::with_target(mean * (0.4 + 1.6 * rng.f()))
                },
                None,
            )
            .unwrap();
            rep.region_triangles
        };
        assert_valid(&work);
        assert_eq!(work.is_closed(), m.is_closed(), "case {case}");
        if use_region {
            let nr = new_region.unwrap();
            tri_set(&work, &nr);
            assert_eq!(snapshot_outside(&work, &nr), before, "case {case}");
        }
    }
}

#[test]
fn extreme_parameters() {
    // Target far larger than the object: collapses as far as topology allows.
    let mut m = primitives::icosphere(1.0, 3);
    remesh(&mut m, &RemeshParams::with_target(5.0), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(m.triangle_count() < 200, "{}", m.triangle_count());
    let mut m = primitives::plane_grid(1.0, 1.0, 10, 10);
    remesh(&mut m, &RemeshParams { boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(10.0) }, None).unwrap();
    assert_valid(&m);
    // Much finer target: one pass refines all the way down.
    let mut m = primitives::icosphere(1.0, 2);
    let mean = edge_length_stats(&m).mean;
    remesh(&mut m, &RemeshParams { iterations: 2, ..RemeshParams::with_target(mean / 8.0) }, None).unwrap();
    assert_valid(&m);
    assert!(m.triangle_count() > 64 * 320 / 2);
    // Zero iterations is a no-op.
    let mut m = primitives::icosphere(1.0, 2);
    let before = m.to_arrays();
    remesh(&mut m, &RemeshParams { iterations: 0, ..RemeshParams::with_target(0.1) }, None).unwrap();
    assert_eq!(before, m.to_arrays());
    // Reduce to nothing: stops at the smallest valid closed mesh.
    let mut m = primitives::icosphere(1.0, 3);
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::TriangleCount(0)), None).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert!(m.triangle_count() >= 4 && m.triangle_count() <= 20, "{}", m.triangle_count());
    // Regions with duplicate and dead ids; a region covering everything.
    let mut m = primitives::icosphere(1.0, 3);
    let mut region: Vec<u32> = m.triangle_ids().take(100).collect();
    region.extend_from_slice(&[0, 1, 2, 999_999, u32::MAX]);
    let rep = remesh(&mut m, &RemeshParams { region: Some(region), ..RemeshParams::with_target(0.05) }, None).unwrap();
    assert_valid(&m);
    tri_set(&m, rep.region_triangles.as_ref().unwrap());
    let mut m = primitives::icosphere(1.0, 3);
    let all: Vec<u32> = m.triangle_ids().collect();
    let rep = remesh(&mut m, &RemeshParams { region: Some(all), ..RemeshParams::with_target(0.1) }, None).unwrap();
    assert_valid(&m);
    assert_eq!(rep.region_triangles.unwrap().len(), m.triangle_count());
}

/// Boundary faces of the voxels inside a sphere, welded on the integer grid: the classic
/// staircase surface of a thresholded segmentation, including non-manifold edges and
/// bowtie vertices where voxels touch only diagonally.
fn voxel_blob(r: i32) -> Mesh {
    use std::collections::HashMap;
    let inside = |x: i32, y: i32, z: i32| {
        let (fx, fy, fz) = (x as f64 + 0.5, y as f64 + 0.5, z as f64 + 0.5);
        fx * fx + fy * fy * 1.3 + fz * fz * 0.8 + 3.0 * (fx * 0.7).sin() < (r * r) as f64
    };
    let mut ids: HashMap<(i32, i32, i32), u32> = HashMap::new();
    let mut pos = Vec::new();
    let mut tris = Vec::new();
    let mut vid = |p: (i32, i32, i32), pos: &mut Vec<DVec3>| {
        *ids.entry(p).or_insert_with(|| {
            pos.push(DVec3::new(p.0 as f64, p.1 as f64, p.2 as f64));
            (pos.len() - 1) as u32
        })
    };
    for x in -r - 2..=r + 2 {
        for y in -r - 2..=r + 2 {
            for z in -r - 2..=r + 2 {
                if !inside(x, y, z) {
                    continue;
                }
                for axis in 0..3 {
                    for s in [0, 1] {
                        let mut n = [x, y, z];
                        n[axis] += if s == 1 { 1 } else { -1 };
                        if inside(n[0], n[1], n[2]) {
                            continue;
                        }
                        let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
                        let corner = |du: i32, dv: i32| {
                            let mut c = [x, y, z];
                            c[axis] += s;
                            c[u] += du;
                            c[v] += dv;
                            (c[0], c[1], c[2])
                        };
                        let q = [corner(0, 0), corner(1, 0), corner(1, 1), corner(0, 1)].map(|c| vid(c, &mut pos));
                        if s == 1 {
                            tris.push([q[0], q[1], q[2]]);
                            tris.push([q[0], q[2], q[3]]);
                        } else {
                            tris.push([q[0], q[2], q[1]]);
                            tris.push([q[0], q[3], q[2]]);
                        }
                    }
                }
            }
        }
    }
    Mesh::from_triangles(&pos, &tris, None).0
}

#[test]
fn voxel_staircase_surface() {
    let base = voxel_blob(7);
    assert_valid(&base);
    let closed = base.is_closed();
    let ncomp = mm_core::topology::connected_components(&base).len();
    println!("voxel blob: {} tris, closed {closed}, {ncomp} components", base.triangle_count());

    for (sharp, adaptive) in [(None, false), (Some(60.0), false), (None, true)] {
        let mut m = base.clone();
        let rep = remesh(
            &mut m,
            &RemeshParams { preserve_sharp_edges: sharp, adaptive, boundary: BoundaryMode::ConstrainToBoundary, ..RemeshParams::with_target(0.7) },
            None,
        )
        .unwrap();
        assert_valid(&m);
        assert_eq!(m.is_closed(), closed);
        println!("  remesh sharp={sharp:?} adaptive={adaptive}: {} tris, min q {:.3}, mean q {:.3}", rep.final_triangles, min_quality(&m), mean_quality(&m));
        if sharp.is_some() {
            // Staircase steps are kept: every triangle stays axis-aligned.
            for t in m.triangle_ids() {
                assert!(m.tri_normal(t).abs().max_element() > 1.0 - 1e-6);
            }
        }
    }
    let mut m = base.clone();
    reduce(&mut m, &ReduceParams::with_target(ReduceTarget::Percentage(20.0)), None).unwrap();
    assert_valid(&m);
    assert_eq!(m.is_closed(), closed);
    let mut m = base.clone();
    let rep = reduce(&mut m, &ReduceParams::with_target(ReduceTarget::MaxDeviation(0.2)), None).unwrap();
    assert_valid(&m);
    println!("  reduce max-dev 0.2: {} tris", rep.final_triangles);
    let h = hausdorff_one_sided(&base, &m, 2, 9);
    println!("  hausdorff {h:.4}, tracked max {:?}", rep.max_deviation);
    // Tracked samples obey the bound; between samples, 90-degree concave steps finer
    // than the tolerance can exceed it a little (documented limitation).
    assert!(rep.max_deviation.unwrap() <= 0.2 * 0.95 + 1e-12);
    assert!(h <= 0.2 * 1.25, "{h}");
}
