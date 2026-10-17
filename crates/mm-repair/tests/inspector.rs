mod common;

use common::*;
use mm_core::{measure, primitives, topology, DVec3, Mesh};
use mm_repair::ops;
use mm_repair::{
    auto_repair, inspect, repair_single, Defect, FillMode, FillParams, InspectParams, RepairError, RepairParams,
};
use std::time::Instant;

/// Order the triangles around vertex `v` of a closed manifold region by walking the fan.
fn ordered_fan(pos: &[DVec3], tris: &[[u32; 3]], v: u32) -> Vec<usize> {
    let fan: Vec<usize> = (0..tris.len()).filter(|&i| tris[i].contains(&v)).collect();
    let n = pos[v as usize].normalize();
    let (u, w) = mm_core::geom::orthonormal_basis(n);
    let mut keyed: Vec<(f64, usize)> = fan
        .iter()
        .map(|&i| {
            let c = (pos[tris[i][0] as usize] + pos[tris[i][1] as usize] + pos[tris[i][2] as usize]) / 3.0 - pos[v as usize];
            (c.dot(w).atan2(c.dot(u)), i)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.total_cmp(&b.0));
    keyed.into_iter().map(|k| k.1).collect()
}

/// An icosphere with every defect the Inspector handles:
/// two holes, a bowtie vertex, a cracked (duplicated-vertex) seam, flipped triangles and
/// three floating fragments.
fn broken_sphere() -> Mesh {
    let base = primitives::icosphere(1.0, 3);
    let (mut pos, tris, _) = base.to_arrays();
    let centroid = |t: &[u32; 3], pos: &[DVec3]| (pos[t[0] as usize] + pos[t[1] as usize] + pos[t[2] as usize]) / 3.0;
    // Holes.
    let holes = [DVec3::new(0.0, 0.0, 1.0), DVec3::new(0.0, 0.8, -0.6)];
    let mut keep: Vec<[u32; 3]> =
        tris.iter().copied().filter(|t| holes.iter().all(|h| centroid(t, &pos).distance(*h) > 0.3)).collect();
    // Bowtie: drop two opposite triangles of a valence-6 fan away from the holes and seam.
    let target = DVec3::new(-0.3, -0.85, 0.2).normalize();
    let v = (0..pos.len() as u32)
        .filter(|&v| keep.iter().filter(|t| t.contains(&v)).count() == 6)
        .min_by(|&a, &b| pos[a as usize].distance(target).total_cmp(&pos[b as usize].distance(target)))
        .unwrap();
    let fan = ordered_fan(&pos, &keep, v);
    let (d1, d2) = (fan[1], fan[4]);
    keep = keep.into_iter().enumerate().filter(|(i, _)| *i != d1 && *i != d2).map(|(_, t)| t).collect();
    // Cracked seam: triangles with centroid x > 0 get private copies of shared vertices.
    let mut side = vec![[false; 2]; pos.len()];
    for t in &keep {
        let s = (centroid(t, &pos).x > 0.0) as usize;
        for &q in t {
            side[q as usize][s] = true;
        }
    }
    let mut dup = vec![u32::MAX; pos.len()];
    for q in 0..pos.len() {
        if side[q][0] && side[q][1] && q as u32 != v {
            dup[q] = pos.len() as u32;
            let p = pos[q];
            pos.push(p);
        }
    }
    for t in keep.iter_mut() {
        if centroid(t, &pos).x > 0.0 {
            for q in t.iter_mut() {
                if dup[*q as usize] != u32::MAX {
                    *q = dup[*q as usize];
                }
            }
        }
    }
    // Flipped triangles.
    for (i, t) in keep.iter_mut().enumerate() {
        if i % 9 == 4 {
            t.swap(1, 2);
        }
    }
    // Floating fragments: a tiny tetrahedron, a lone triangle, a small open patch.
    let add = |pts: &[DVec3], ts: &[[u32; 3]], pos: &mut Vec<DVec3>, keep: &mut Vec<[u32; 3]>| {
        let o = pos.len() as u32;
        pos.extend_from_slice(pts);
        keep.extend(ts.iter().map(|t| [t[0] + o, t[1] + o, t[2] + o]));
    };
    let s = 0.05;
    let c = DVec3::new(2.0, 0.0, 0.0);
    add(
        &[c, c + DVec3::X * s, c + DVec3::Y * s, c + DVec3::Z * s],
        &[[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]],
        &mut pos,
        &mut keep,
    );
    let c = DVec3::new(0.0, 2.0, 0.0);
    add(&[c, c + DVec3::X * s, c + DVec3::Z * s], &[[0, 1, 2]], &mut pos, &mut keep);
    let g = primitives::plane_grid(0.1, 0.1, 3, 3);
    let (gp, gt, _) = g.to_arrays();
    let gp: Vec<DVec3> = gp.iter().map(|p| *p + DVec3::new(0.0, 0.0, 2.0)).collect();
    add(&gp, &gt, &mut pos, &mut keep);
    let (m, rep) = Mesh::from_triangles(&pos, &keep, None);
    assert_eq!(rep.nonmanifold_split, 0);
    m
}

#[test]
fn inspector_finds_all_defects() {
    let m = broken_sphere();
    assert_valid(&m);
    let r = inspect(&m, &InspectParams::default());
    println!(
        "holes {} (edges {:?}), bowties {}, small comps {}, degenerate {}, inconsistent {}, cracks {}, comps {}",
        r.holes.len(),
        r.holes.iter().map(|h| h.num_edges).collect::<Vec<_>>(),
        r.bowtie_vertices.len(),
        r.small_components.len(),
        r.degenerate_triangles.len(),
        r.inconsistent_orientation_edges,
        r.crack_vertices,
        r.num_components
    );
    assert!(r.holes.len() >= 4, "two holes, the bowtie hole(s) and the seam");
    assert_eq!(r.bowtie_vertices.len(), 1);
    assert!(r.holes.iter().any(|h| h.touches_bowtie));
    assert_eq!(r.small_components.len(), 3);
    assert!(r.small_components.iter().all(|c| c.area <= 0.0101 && c.centroid.length() > 1.5));
    assert!(r.inconsistent_orientation_edges > 50);
    assert!(r.crack_vertices > 20);
    assert!(r.degenerate_triangles.is_empty());
    assert_eq!(r.num_components, 5, "two sphere halves + three fragments");
    assert!(!r.is_clean() && !r.is_closed());
    assert_eq!(r.defects().len(), r.holes.len() + 1 + 3);
    for h in &r.holes {
        assert!(h.perimeter > 0.0 && h.num_edges == h.edge_loop.len());
    }
}

#[test]
fn auto_repair_produces_one_closed_oriented_component() {
    for mode in [FillMode::Minimal, FillMode::Flat, FillMode::Smooth] {
        let mut m = broken_sphere();
        let calls = std::cell::Cell::new(0usize);
        let prog = |f: f32| {
            assert!((0.0..=1.0).contains(&f));
            calls.set(calls.get() + 1);
            true
        };
        let rep = auto_repair(&mut m, &RepairParams { fill_mode: mode, ..Default::default() }, Some(&prog));
        println!("{mode:?}: {rep:#?}");
        assert!(calls.get() > 5);
        assert_watertight(&m);
        assert!(rep.is_watertight());
        assert!(rep.welded_vertices > 20);
        // The bowtie sits on a hole boundary: filling its loop as two sub-loops resolves it.
        assert_eq!(rep.bowties_split, 0);
        assert_eq!(rep.components_removed, 3);
        assert!(rep.holes_filled >= 3);
        assert_eq!(rep.holes_failed, 0);
        assert!(rep.triangles_flipped > 50);
        let after = inspect(&m, &InspectParams::default());
        assert!(after.is_clean(), "{after:?}");
        let vol = measure::volume(&m);
        let sphere = measure::volume(&primitives::icosphere(1.0, 3));
        println!("{mode:?}: volume {vol:.4} (intact icosphere {sphere:.4})");
        assert!((vol - sphere).abs() < 0.06 * sphere);
        if mode == FillMode::Smooth {
            assert!((vol - sphere).abs() < 0.01 * sphere);
        }
    }
}

#[test]
fn auto_repair_can_be_cancelled_safely() {
    for stop_at in [0.0f32, 0.12, 0.3, 0.5] {
        let mut m = broken_sphere();
        let rep = auto_repair(&mut m, &RepairParams::default(), Some(&|f| f <= stop_at));
        assert!(rep.cancelled);
        assert_valid(&m);
    }
}

#[test]
fn repair_single_defects() {
    let mut m = broken_sphere();
    let fill = FillParams::new(FillMode::Smooth);
    // Remove one fragment.
    let r = inspect(&m, &InspectParams::default());
    let comp = r.small_components[0].triangles.clone();
    let out = repair_single(&mut m, &Defect::SmallComponent(comp.clone()), &fill).unwrap();
    assert_eq!(out.removed_triangles, comp.len());
    assert!(matches!(repair_single(&mut m, &Defect::SmallComponent(comp), &fill), Err(RepairError::StaleDefect(_))));
    // Split the bowtie.
    let bow = r.bowtie_vertices[0];
    let out = repair_single(&mut m, &Defect::BowtieVertex(bow), &fill).unwrap();
    assert_eq!(out.new_vertices.len(), 1);
    assert!(!m.is_bowtie_vertex(bow));
    assert!(repair_single(&mut m, &Defect::BowtieVertex(bow), &fill).is_err());
    // Fill the hole around +Z (clicked by the user).
    let r = inspect(&m, &InspectParams::default());
    let hole = r.holes.iter().min_by(|a, b| a.centroid.distance(DVec3::Z).total_cmp(&b.centroid.distance(DVec3::Z))).unwrap();
    let n_holes = r.holes.len();
    let out = repair_single(&mut m, &Defect::Hole(hole.edge_loop.clone()), &fill).unwrap();
    assert!(!out.fill.new_triangles.is_empty());
    assert_valid(&m);
    assert_eq!(ops::boundary_loops(&m).len(), n_holes - 1);
    // Stale hole.
    assert!(repair_single(&mut m, &Defect::Hole(hole.edge_loop.clone()), &fill).is_err());
    assert_valid(&m);
}

#[test]
fn degenerate_triangles_are_removed() {
    // Grid with a needle (two vertices made coincident) and a cap (a vertex moved onto an edge).
    let mut m = primitives::plane_grid(1.0, 1.0, 6, 6);
    let find = |m: &Mesh, x: f64, y: f64| m.vertex_ids().find(|&v| m.vertex(v).distance(DVec3::new(x, y, 0.0)) < 1e-9).unwrap();
    let h = 1.0 / 6.0;
    // Needle: pull a vertex almost onto its neighbour.
    let a = find(&m, -0.5 + 2.0 * h, -0.5 + 2.0 * h);
    m.set_vertex(a, DVec3::new(-0.5 + 3.0 * h - 1e-9, -0.5 + 2.0 * h, 0.0));
    // Cap: move a vertex onto the midpoint of the diagonal of a neighbouring quad.
    let b = find(&m, -0.5 + 4.0 * h, -0.5 + 4.0 * h);
    m.set_vertex(b, DVec3::new(-0.5 + 4.5 * h, -0.5 + 4.5 * h, 0.0));
    let degen = ops::find_degenerate_triangles(&m, 1e-6);
    println!("degenerate before: {}", degen.len());
    assert!(degen.len() >= 2);
    let rep = ops::remove_degenerate_triangles(&mut m, 1e-6);
    println!("{rep:?}");
    assert_valid(&m);
    assert_eq!(rep.remaining, 0);
    assert!(ops::find_degenerate_triangles(&m, 1e-6).is_empty());
    assert_eq!(topology::boundary_loops(&m).len(), 1);
}

#[test]
fn weld_rejects_nonmanifold_merges() {
    // Three triangles sharing an edge: from_triangles splits the third off with coincident
    // vertices; welding them back would create a non-manifold edge and must be refused.
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.5, 1.0, 0.0),
        DVec3::new(0.5, -1.0, 0.0),
        DVec3::new(0.5, 0.0, 1.0),
    ];
    let (mut m, rep) = Mesh::from_triangles(&pos, &[[0, 1, 2], [1, 0, 3], [0, 1, 4]], None);
    assert_eq!(rep.nonmanifold_split, 1);
    let w = ops::weld_boundary_vertices(&mut m, 1e-9);
    assert_valid(&m);
    assert!(w.rejected >= 1);
    assert_eq!(m.triangle_count(), 3);
}

/// Performance check on a ~1.3M triangle mesh with dozens of holes (run with
/// `cargo test --release -p mm-repair -- --ignored --nocapture`).
#[test]
#[ignore]
fn perf_auto_repair_1m_triangles() {
    let t0 = Instant::now();
    let mut m = primitives::icosphere(50.0, 8);
    let n0 = m.triangle_count();
    let mut rng = Lcg(42);
    let mut n_holes = 0;
    for _ in 0..40 {
        let d = DVec3::new(rng.sym(), rng.sym(), rng.sym()).normalize_or(DVec3::Z);
        let r = 0.6 + 2.4 * (rng.sym() * 0.5 + 0.5);
        if punch_hole(&mut m, d * 50.0, r) > 0 {
            n_holes += 1;
        }
    }
    // Some noise-like floating debris.
    for k in 0..10 {
        let c = DVec3::new(80.0 + k as f64, 0.0, 0.0);
        let s = 0.2;
        let o = m.append_vertex(c);
        let a = m.append_vertex(c + DVec3::X * s);
        let b = m.append_vertex(c + DVec3::Y * s);
        m.append_triangle([o, a, b], 0).unwrap();
    }
    let loops = topology::boundary_loops(&m).len();
    println!(
        "setup {:.2}s: {} triangles ({} originally), {} holes punched, {} boundary loops",
        t0.elapsed().as_secs_f64(),
        m.triangle_count(),
        n0,
        n_holes,
        loops
    );
    let t = Instant::now();
    let r = inspect(&m, &InspectParams::default());
    println!("inspect: {:.3}s ({} holes, {} small comps)", t.elapsed().as_secs_f64(), r.holes.len(), r.small_components.len());
    for mode in [FillMode::Smooth, FillMode::Flat, FillMode::Minimal] {
        let mut mm = m.clone();
        let t = Instant::now();
        let rep = auto_repair(&mut mm, &RepairParams { fill_mode: mode, ..Default::default() }, None);
        let dt = t.elapsed().as_secs_f64();
        println!(
            "auto_repair {mode:?}: {dt:.3}s; filled {} holes (+{} tris), removed {} comps, watertight {}",
            rep.holes_filled,
            rep.fill_triangles_added,
            rep.components_removed,
            rep.is_watertight()
        );
        assert!(rep.is_watertight());
    }
    m.check_validity().unwrap();
}

/// Worst case for the crack welder: an unwelded triangle soup (every corner duplicated),
/// with some triangles flipped.
#[test]
#[ignore]
fn perf_auto_repair_triangle_soup() {
    for sub in [6usize, 7] {
        let base = primitives::icosphere(20.0, sub);
        let (pos, tris, _) = base.to_arrays();
        let mut sp = Vec::with_capacity(tris.len() * 3);
        let mut st = Vec::with_capacity(tris.len());
        for (i, t) in tris.iter().enumerate() {
            let o = sp.len() as u32;
            for &q in t {
                sp.push(pos[q as usize]);
            }
            st.push(if i % 7 == 0 { [o, o + 2, o + 1] } else { [o, o + 1, o + 2] });
        }
        let (mut m, _) = Mesh::from_triangles(&sp, &st, None);
        let t = Instant::now();
        let rep = auto_repair(&mut m, &RepairParams::default(), None);
        println!(
            "soup of {} triangles: auto_repair {:.3}s, welded {} vertices, watertight {}",
            tris.len(),
            t.elapsed().as_secs_f64(),
            rep.welded_vertices,
            rep.is_watertight()
        );
        assert!(rep.is_watertight());
        assert_eq!(m.vertex_count(), base.vertex_count());
    }
}

#[test]
fn auto_repair_on_clean_and_empty_meshes() {
    let mut m = primitives::icosphere(1.0, 3);
    let before = m.clone();
    let rep = auto_repair(&mut m, &RepairParams::default(), None);
    assert!(rep.is_watertight());
    assert_eq!((rep.welded_vertices, rep.holes_filled, rep.triangles_flipped, rep.degenerate_fixed), (0, 0, 0, 0));
    assert_eq!(m.triangle_count(), before.triangle_count());
    for v in m.vertex_ids() {
        assert_eq!(m.vertex(v), before.vertex(v));
    }
    assert!(inspect(&m, &InspectParams::default()).is_clean());

    let mut e = Mesh::new();
    let rep = auto_repair(&mut e, &RepairParams::default(), None);
    assert_eq!(rep.num_components, 0);
    assert!(inspect(&e, &InspectParams::default()).is_clean());

    // An inward-oriented closed mesh is flipped outward.
    let mut inv = primitives::icosphere(1.0, 2);
    inv.reverse_orientation();
    let rep = auto_repair(&mut inv, &RepairParams::default(), None);
    assert_eq!(rep.triangles_flipped, inv.triangle_count());
    assert!(measure::volume(&inv) > 0.0);
}

/// A ~3M triangle vessel-like tube (curved centreline, varying radius, open ends) with
/// dozens of punched holes and a cracked seam.
#[test]
#[ignore]
fn perf_auto_repair_3m_vessel() {
    let t0 = Instant::now();
    let n_rings = 7500;
    let path: Vec<DVec3> = (0..n_rings)
        .map(|i| {
            let s = i as f64 / (n_rings - 1) as f64;
            DVec3::new(20.0 * (3.0 * s).sin(), 15.0 * (2.0 * s).cos(), 150.0 * s)
        })
        .collect();
    let radii: Vec<f64> = (0..n_rings).map(|i| 2.0 + 0.5 * (i as f64 * 0.01).sin()).collect();
    let tube = primitives::tube_along(&path, &radii, 200);
    // Crack: duplicate the vertices of one ring (an STL-style seam).
    let (pos, mut tris, _) = tube.to_arrays();
    let mut pos = pos;
    let seam = 3000u32;
    let base = pos.len() as u32;
    for j in 0..200 {
        let p = pos[(seam * 200 + j) as usize];
        pos.push(p);
    }
    for t in tris.iter_mut() {
        if t.iter().all(|&q| q / 200 >= seam) && t.iter().any(|&q| q / 200 == seam) {
            for q in t.iter_mut() {
                if *q / 200 == seam {
                    *q = base + *q % 200;
                }
            }
        }
    }
    let (mut m, _) = Mesh::from_triangles(&pos, &tris, None);
    let mut rng = Lcg(5);
    let mut punched = 0;
    for _ in 0..40 {
        let i = (rng.next_u32() as usize) % (n_rings - 200) + 100;
        let v = (i * 200 + (rng.next_u32() as usize % 200)) as u32;
        if !m.is_vertex(v) {
            continue;
        }
        let c = m.vertex(v);
        if punch_hole(&mut m, c, 0.8) > 0 {
            punched += 1;
        }
    }
    println!(
        "vessel setup {:.2}s: {} triangles, {} holes punched, {} loops",
        t0.elapsed().as_secs_f64(),
        m.triangle_count(),
        punched,
        ops::boundary_loops(&m).len()
    );
    let t = Instant::now();
    let r = inspect(&m, &InspectParams::default());
    println!("inspect: {:.3}s ({} holes, {} crack vertices)", t.elapsed().as_secs_f64(), r.holes.len(), r.crack_vertices);
    let t = Instant::now();
    let rep = auto_repair(&mut m, &RepairParams::default(), None);
    println!(
        "auto_repair Smooth (3M vessel): {:.3}s; welded {}, filled {} holes (+{} tris), watertight {}\n  phases: {:?}",
        t.elapsed().as_secs_f64(),
        rep.welded_vertices,
        rep.holes_filled,
        rep.fill_triangles_added,
        rep.is_watertight(),
        rep.timings.iter().map(|(n, s)| format!("{n} {s:.3}s")).collect::<Vec<_>>()
    );
    assert!(rep.is_watertight());
    m.check_validity().unwrap();
}

#[test]
fn close_cracks_zips_t_junctions() {
    use mm_core::{primitives, DVec3};
    // Left: 4 rows along the seam; right: 8 rows; they touch at x = 0 without sharing vertices.
    let mut left = primitives::plane_grid(2.0, 4.0, 2, 4);
    left.transform(|p| p - DVec3::new(1.0, 0.0, 0.0));
    let mut right = primitives::plane_grid(2.0, 4.0, 2, 8);
    right.transform(|p| p + DVec3::new(1.0 + 1e-7, 0.0, 0.0));
    let mut m = left.clone();
    m.append_mesh(&right, 0);
    let seam_open = |m: &mm_core::Mesh| m.boundary_edge_ids().filter(|&e| m.edge_midpoint(e).x.abs() < 1e-3).count();
    assert!(seam_open(&m) > 0);
    let r = mm_repair::close_cracks(&mut m, 1e-5);
    m.check_validity().unwrap();
    assert!(r.splits >= 4, "the coarse side's seam edges get split ({r:?})");
    assert_eq!(seam_open(&m), 0, "the seam is closed ({r:?})");
    assert_eq!(mm_core::topology::connected_components(&m).len(), 1);
}
