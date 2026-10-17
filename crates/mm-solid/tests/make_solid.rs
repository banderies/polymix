//! End-to-end tests of make_solid and the SDF operations on messy inputs.

mod common;

use common::*;
use glam::DVec3;
use mm_core::{measure, primitives, topology, Mesh};
use mm_solid::{hollow, make_solid, offset_solid, sdf_boolean, BoolOp, MakeSolidParams, SolidError, SolidMode};
use std::f64::consts::PI;

fn sphere_vol(r: f64) -> f64 {
    4.0 / 3.0 * PI * r * r * r
}

fn params(h: f64) -> MakeSolidParams {
    MakeSolidParams { cell_size: h, ..Default::default() }
}

#[test]
fn icosphere_becomes_sphere() {
    let input = primitives::icosphere(1.0, 4);
    let h = 0.05;
    let out = make_solid(&input, &params(h), None).unwrap();
    assert_solid(&out, "icosphere");
    assert_eq!(component_count(&out), 1);
    let (max_dev, mean_dev) = distance_to(&out, &input);
    assert!(max_dev <= 0.5 * h, "max deviation {max_dev} > half a cell");
    assert!(mean_dev <= 0.1 * h, "mean deviation {mean_dev}");
    for v in out.vertex_ids() {
        let r = out.vertex(v).length();
        assert!((r - 1.0).abs() <= 0.5 * h, "radius {r}");
    }
    let (vi, vo) = (measure::volume(&input), measure::volume(&out));
    assert!((vo - vi).abs() / vi < 0.02, "volume {vo} vs {vi}");
    let ma = min_angle_deg(&out);
    assert!(ma > 1.0, "min angle {ma}");
}

#[test]
fn accurate_mode_hugs_the_input() {
    let input = primitives::icosphere(1.0, 3);
    let h = 0.06;
    let fast = make_solid(&input, &params(h), None).unwrap();
    let acc = make_solid(&input, &MakeSolidParams { mode: SolidMode::Accurate, ..params(h) }, None).unwrap();
    assert_solid(&acc, "accurate");
    let (_, mean_fast) = distance_to(&fast, &input);
    let (max_acc, mean_acc) = distance_to(&acc, &input);
    assert!(mean_acc < 0.5 * mean_fast, "accurate {mean_acc} vs fast {mean_fast}");
    assert!(max_acc <= 0.5 * h);
}

#[test]
fn blocky_mode_is_manifold() {
    let input = primitives::icosphere(1.0, 3);
    let h = 0.1;
    let out = make_solid(&input, &MakeSolidParams { mode: SolidMode::Blocky, ..params(h) }, None).unwrap();
    assert_solid(&out, "blocky");
    // Every vertex sits on the voxel-corner lattice; every face is axis aligned.
    for t in out.triangle_ids() {
        let n = out.tri_normal(t);
        assert!(n.abs().max_element() > 0.999, "face normal {n}");
    }
    let v = measure::volume(&out);
    assert!((v - sphere_vol(1.0)).abs() / sphere_vol(1.0) < 0.08, "blocky volume {v}");
}

fn sphere_with_hole(cut_z: f64) -> Mesh {
    let s = primitives::icosphere(1.0, 4);
    let keep: Vec<u32> = s.triangle_ids().filter(|&t| s.tri_centroid(t).z < cut_z).collect();
    s.submesh(&keep).0
}

#[test]
fn sphere_with_big_hole_is_closed() {
    let input = sphere_with_hole(0.5);
    assert!(!input.is_closed());
    let out = make_solid(&input, &params(0.05), None).unwrap();
    assert_solid(&out, "holed sphere");
    assert_eq!(component_count(&out), 1);
    // Roughly the sphere minus the cap above z≈0.5 (cap volume πh²(3r−h)/3 with h = 0.5).
    let cap = PI * 0.25 * (3.0 - 0.5) / 3.0;
    let expect = sphere_vol(1.0) - cap;
    let v = measure::volume(&out);
    assert!((v - expect).abs() / expect < 0.05, "volume {v} vs ~{expect}");
}

#[test]
fn preserve_open_boundaries() {
    let input = sphere_with_hole(0.5);
    let p = MakeSolidParams { close_holes: false, ..params(0.05) };
    let out = make_solid(&input, &p, None).unwrap();
    out.check_validity().unwrap();
    assert!(!out.is_closed(), "hole should stay open");
    assert!(out.is_consistently_oriented());
    assert_eq!(cleanup_bowties(&out), 0);
    let loops = topology::boundary_loops(&out);
    assert_eq!(loops.len(), 1, "expected one open boundary");
    // The opening sits near the input rim (z ≈ 0.5, radius ≈ 0.87).
    let c = loops[0].centroid(&out);
    assert!((c.z - 0.5).abs() < 0.15, "loop centroid {c}");
}

fn cleanup_bowties(m: &Mesh) -> usize {
    m.vertex_ids().filter(|&v| m.is_bowtie_vertex(v)).count()
}

#[test]
fn overlapping_spheres_merge() {
    let mut a = primitives::icosphere(1.0, 3);
    let mut b = primitives::icosphere(1.0, 3);
    b.transform(|p| p + DVec3::X);
    a.append_mesh(&b, 1);
    let out = make_solid(&a, &params(0.05), None).unwrap();
    assert_solid(&out, "two spheres");
    assert_eq!(component_count(&out), 1);
    let lens = PI * (4.0 + 1.0) * 1.0 / 12.0;
    let expect = 2.0 * sphere_vol(1.0) - lens;
    let v = measure::volume(&out);
    assert!((v - expect).abs() / expect < 0.03, "volume {v} vs {expect}");
}

#[test]
fn doubled_jittered_mesh_is_cleaned() {
    let s = primitives::icosphere(1.0, 3);
    let mut rng = Rng::new(3);
    let mut tris = triangles_of(&s);
    let mut jittered = s.clone();
    for v in jittered.vertex_ids().collect::<Vec<_>>() {
        let p = jittered.vertex(v) + rng.vec(0.004);
        jittered.set_vertex(v, p);
    }
    tris.extend(triangles_of(&jittered));
    // Also flip a third of all triangles to simulate inconsistent orientation.
    for (i, t) in tris.iter_mut().enumerate() {
        if i % 3 == 0 {
            t.swap(1, 2);
        }
    }
    let soup = weld_soup(&tris);
    let out = make_solid(&soup, &params(0.05), None).unwrap();
    assert_solid(&out, "doubled");
    assert_eq!(component_count(&out), 1);
    let v = measure::volume(&out);
    assert!((v - measure::volume(&s)).abs() / measure::volume(&s) < 0.05, "volume {v}");
}

#[test]
fn self_intersecting_tube_and_inverted_shell() {
    // A tube along a figure-eight crosses itself.
    let path: Vec<DVec3> = (0..=120)
        .map(|i| {
            let t = i as f64 / 120.0 * 2.0 * PI;
            DVec3::new(t.sin(), (2.0 * t).sin() * 0.5, 0.0)
        })
        .collect();
    let radii = vec![0.12; path.len()];
    let tube = primitives::tube_along(&path, &radii, 16);
    let out = make_solid(&tube, &params(0.03), None).unwrap();
    assert_solid(&out, "figure-eight tube");
    assert_eq!(component_count(&out), 1);

    let mut inv = primitives::icosphere(1.0, 3);
    inv.reverse_orientation();
    let out = make_solid(&inv, &params(0.08), None).unwrap();
    assert_solid(&out, "inverted sphere");
    assert!((measure::volume(&out) - sphere_vol(1.0)).abs() / sphere_vol(1.0) < 0.05);
}

#[test]
fn thin_open_tube_closes() {
    let r = 0.2;
    let path: Vec<DVec3> = (0..60).map(|i| DVec3::new((i as f64 * 0.1).sin() * 0.8, i as f64 * 0.05, 0.0)).collect();
    let len: f64 = path.windows(2).map(|w| w[0].distance(w[1])).sum();
    let tube = primitives::tube_along(&path, &vec![r; path.len()], 24);
    let h = r / 4.0;
    let out = make_solid(&tube, &params(h), None).unwrap();
    assert_solid(&out, "thin tube");
    assert_eq!(component_count(&out), 1);
    let expect = PI * r * r * len;
    let v = measure::volume(&out);
    assert!((v - expect).abs() / expect < 0.1, "tube volume {v} vs {expect}");
}

#[test]
fn offsets_scale_volume() {
    let s = primitives::icosphere(1.0, 4);
    let h = 0.05;
    let grown = offset_solid(&s, 0.2, h, None).unwrap();
    assert_solid(&grown, "grown");
    let shrunk = offset_solid(&s, -0.2, h, None).unwrap();
    assert_solid(&shrunk, "shrunk");
    let vs = measure::volume(&s);
    let rs = (vs / sphere_vol(1.0)).cbrt(); // effective radius of the polyhedron
    let vg = measure::volume(&grown);
    let vsh = measure::volume(&shrunk);
    assert!((vg - sphere_vol(rs + 0.2)).abs() / sphere_vol(rs + 0.2) < 0.03, "grown {vg}");
    assert!((vsh - sphere_vol(rs - 0.2)).abs() / sphere_vol(rs - 0.2) < 0.03, "shrunk {vsh}");
    // Large grow (many cells) still works and stays a single closed shell.
    let big = offset_solid(&s, 0.6, 0.1, None).unwrap();
    assert_solid(&big, "big offset");
    assert!((measure::volume(&big) - sphere_vol(rs + 0.6)).abs() / sphere_vol(rs + 0.6) < 0.03);
}

#[test]
fn booleans_have_sensible_volumes() {
    let a = primitives::icosphere(1.0, 3);
    let mut b = primitives::icosphere(1.0, 3);
    b.transform(|p| p + DVec3::X);
    let h = 0.05;
    let lens = PI * 5.0 / 12.0;
    let va = measure::volume(&a);
    let scale = va / sphere_vol(1.0);
    let u = sdf_boolean(&a, &b, BoolOp::Union, h, None).unwrap();
    assert_solid(&u, "union");
    assert_eq!(component_count(&u), 1);
    let d = sdf_boolean(&a, &b, BoolOp::Difference, h, None).unwrap();
    assert_solid(&d, "difference");
    let i = sdf_boolean(&a, &b, BoolOp::Intersection, h, None).unwrap();
    assert_solid(&i, "intersection");
    let (vu, vd, vi) = (measure::volume(&u), measure::volume(&d), measure::volume(&i));
    let close = |x: f64, y: f64, tol: f64| (x - y).abs() / y < tol;
    assert!(close(vu, (2.0 * sphere_vol(1.0) - lens) * scale, 0.03), "union {vu}");
    assert!(close(vd, (sphere_vol(1.0) - lens) * scale, 0.05), "difference {vd}");
    assert!(close(vi, lens * scale, 0.06), "intersection {vi}");
    assert!(close(vu, vd + va, 0.03), "union = (A−B) + B: {vu} vs {}", vd + va);
}

#[test]
fn hollow_sphere() {
    let s = primitives::icosphere(1.0, 4);
    let out = hollow(&s, 0.2, 0.04, None).unwrap();
    assert_closed_manifold(&out, "hollow");
    let comps = topology::connected_components(&out);
    assert_eq!(comps.len(), 2, "outer and inner shells");
    let vols: Vec<f64> = comps.iter().map(|c| mm_solid::cleanup::shell_volume(&out, c)).collect();
    assert!(vols.iter().any(|&v| v > 0.0) && vols.iter().any(|&v| v < 0.0), "inner shell faces inward: {vols:?}");
    let rs = (measure::volume(&s) / sphere_vol(1.0)).cbrt();
    let expect = sphere_vol(rs) - sphere_vol(rs - 0.2);
    let v = measure::volume(&out);
    assert!((v - expect).abs() / expect < 0.05, "shell volume {v} vs {expect}");
}

#[test]
fn min_thickness_turns_sheet_into_slab() {
    let sheet = primitives::plane_grid(1.0, 1.0, 20, 20);
    let t = 0.1;
    let p = MakeSolidParams { min_thickness: Some(t), ..params(0.02) };
    let out = make_solid(&sheet, &p, None).unwrap();
    assert_solid(&out, "slab");
    let v = measure::volume(&out);
    // Slab plus rounded rims: between t·A and (t + rim) · A.
    assert!(v > 0.9 * t && v < 1.3 * t, "slab volume {v}");
    let b = out.bounds();
    assert!((b.extents().z - t).abs() < 0.03, "thickness {}", b.extents().z);
    // Accurate mode must not project the slab back onto the sheet.
    let acc = make_solid(&sheet, &MakeSolidParams { mode: SolidMode::Accurate, ..p.clone() }, None).unwrap();
    assert_solid(&acc, "accurate slab");
    assert!((acc.bounds().extents().z - t).abs() < 0.03, "accurate slab thickness {}", acc.bounds().extents().z);

    // A thin closed box (wall thickness 0.04) is thickened to ~min thickness,
    // while a thick one is left alone.
    let mut thin = primitives::cube(1.0, 4);
    thin.transform(|p| DVec3::new(p.x, p.y, p.z * 0.04));
    let out = make_solid(&thin, &MakeSolidParams { min_thickness: Some(0.12), ..params(0.02) }, None).unwrap();
    assert_solid(&out, "thickened box");
    let ez = out.bounds().extents().z;
    assert!(ez > 0.1 && ez < 0.16, "thickened box height {ez}");
    let thick = primitives::cube(1.0, 4);
    let out = make_solid(&thick, &MakeSolidParams { min_thickness: Some(0.12), ..params(0.04) }, None).unwrap();
    assert!((measure::volume(&out) - 1.0).abs() < 0.03, "thick box unchanged");
}

#[test]
fn small_shells_are_removed() {
    let mut m = primitives::icosphere(1.0, 3);
    let mut speck = primitives::icosphere(0.04, 1);
    speck.transform(|p| p + DVec3::new(3.0, 0.0, 0.0));
    m.append_mesh(&speck, 1);
    let out = make_solid(&m, &params(0.05), None).unwrap();
    assert_solid(&out, "with speck");
    assert_eq!(component_count(&out), 1);
    let keep = make_solid(&m, &MakeSolidParams { min_shell_volume: Some(0.0), ..params(0.02) }, None).unwrap();
    assert_eq!(component_count(&keep), 2);
}

#[test]
fn cancellation_and_errors() {
    let s = primitives::icosphere(1.0, 2);
    let calls = std::cell::Cell::new(0);
    let cancel = |_: f32| {
        calls.set(calls.get() + 1);
        calls.get() < 3
    };
    let r = make_solid(&s, &params(0.05), Some(&cancel));
    assert_eq!(r.unwrap_err(), SolidError::Cancelled);
    let last = std::cell::Cell::new(-1.0f32);
    let mono = |f: f32| {
        assert!(f >= last.get() - 1e-6, "progress went backwards {f} < {}", last.get());
        last.set(f);
        true
    };
    make_solid(&s, &params(0.05), Some(&mono)).unwrap();
    assert!((last.get() - 1.0).abs() < 1e-6);
    assert_eq!(make_solid(&Mesh::new(), &params(0.1), None).unwrap_err(), SolidError::EmptyInput);
    assert!(matches!(make_solid(&s, &params(1e-6), None), Err(SolidError::GridTooLarge(_))));
}

#[test]
fn huge_sparse_grid_stays_small() {
    // Two tiny spheres 400 units apart at cell 0.1: a ~4000³-sample grid in total, but
    // only a few hundred bricks are active.
    let mut a = primitives::icosphere(1.0, 3);
    let mut b = primitives::icosphere(1.0, 3);
    b.transform(|p| p + DVec3::splat(400.0 / 3f64.sqrt()));
    a.append_mesh(&b, 1);
    let sdf = mm_solid::MeshSdf::new(&a, 0.1, 0.0, 0.5);
    let grid = mm_solid::GridSpec::covering(&a.bounds(), 0.1, sdf.band() + 0.2).unwrap();
    assert!(grid.samples().iter().all(|&n| n > 2000));
    let field = mm_solid::build_field(&sdf, grid, None).unwrap();
    assert!(field.active_bricks() < 5000, "{} active bricks", field.active_bricks());
    assert!(field.memory_bytes() < 64 << 20, "{} bytes", field.memory_bytes());
    let (p, t) = mm_solid::extract::marching_tetrahedra(&field);
    let (m, _) = Mesh::from_triangles(&p, &t, None);
    assert_closed_manifold(&m, "two far spheres");
    assert_eq!(component_count(&m), 2);
}

#[test]
fn slider_helpers() {
    let s = primitives::icosphere(2.0, 2);
    let p = MakeSolidParams::with_resolution(&s, 64);
    assert!((p.cell_size - 4.0 / 64.0).abs() < 1e-3);
    let lo = MakeSolidParams::from_sliders(&s, 0.0, 0.0);
    let hi = MakeSolidParams::from_sliders(&s, 1.0, 1.0);
    assert!(lo.cell_size > hi.cell_size * 15.0);
    let coarse = make_solid(&s, &MakeSolidParams::from_sliders(&s, 0.3, 0.0), None).unwrap();
    let fine = make_solid(&s, &MakeSolidParams::from_sliders(&s, 0.3, 1.0), None).unwrap();
    assert_solid(&coarse, "coarse density");
    assert_solid(&fine, "fine density");
    assert!(coarse.triangle_count() < fine.triangle_count());
}

#[test]
fn far_from_origin_and_nan_vertices() {
    // Coordinates typical of scanner space (large offsets, small features).
    let mut s = primitives::icosphere(1.0, 3);
    let off = DVec3::new(1.0e4, -2.5e4, 3.0e4);
    s.transform(|p| p + off);
    let out = make_solid(&s, &params(0.05), None).unwrap();
    assert_solid(&out, "far from origin");
    let (max_dev, _) = distance_to(&out, &s);
    assert!(max_dev < 0.5 * 0.05, "deviation {max_dev}");
    // A NaN vertex only drops its triangles; the hole is capped.
    let mut bad = primitives::icosphere(1.0, 3);
    let v = bad.vertex_ids().next().unwrap();
    bad.set_vertex(v, DVec3::new(f64::NAN, 0.0, 0.0));
    let out = make_solid(&bad, &params(0.05), None).unwrap();
    assert_solid(&out, "nan vertex");
    assert!((measure::volume(&out) - measure::volume(&primitives::icosphere(1.0, 3))).abs() < 0.05);
}

#[test]
fn random_triangle_soup() {
    // Pure garbage: random overlapping triangles. The result must still be a valid solid
    // (or empty).
    for seed in 0..6u64 {
        let mut rng = Rng::new(seed + 50);
        let tris: Vec<[DVec3; 3]> = (0..300)
            .map(|_| {
                let c = rng.vec(1.0);
                [c + rng.vec(0.3), c + rng.vec(0.3), c + rng.vec(0.3)]
            })
            .collect();
        let soup = weld_soup(&tris);
        let out = make_solid(&soup, &MakeSolidParams { winding_threshold: 0.2, ..params(0.04) }, None).unwrap();
        if out.triangle_count() > 0 {
            assert_solid(&out, &format!("soup {seed}"));
        }
        let thick = make_solid(&soup, &MakeSolidParams { min_thickness: Some(0.05), ..params(0.04) }, None).unwrap();
        assert_solid(&thick, &format!("thickened soup {seed}"));
    }
}

#[test]
fn open_vessel_keeps_inlet_and_outlet() {
    let path: Vec<DVec3> = (0..40).map(|i| DVec3::new((i as f64 * 0.15).sin() * 0.3, 0.0, i as f64 * 0.05)).collect();
    let tube = primitives::tube_along(&path, &vec![0.25; path.len()], 32);
    let closed = make_solid(&tube, &params(0.04), None).unwrap();
    assert_solid(&closed, "capped vessel");
    let open = make_solid(&tube, &MakeSolidParams { close_holes: false, ..params(0.04) }, None).unwrap();
    open.check_validity().unwrap();
    assert!(open.is_consistently_oriented());
    assert_eq!(open.vertex_ids().filter(|&v| open.is_bowtie_vertex(v)).count(), 0);
    let loops = topology::boundary_loops(&open);
    assert_eq!(loops.len(), 2, "inlet and outlet");
    for l in &loops {
        let c = l.centroid(&open);
        let near_end = c.distance(path[0]).min(c.distance(*path.last().unwrap()));
        assert!(near_end < 0.1, "loop centroid {c} not at a tube end");
    }
    assert_eq!(topology::connected_components(&open).len(), 1);
}

#[test]
fn min_thickness_inflates_thin_vessels() {
    let path: Vec<DVec3> = (0..30).map(|i| DVec3::new(0.0, 0.0, i as f64 * 0.05)).collect();
    let r = 0.02;
    let tube = primitives::tube_along(&path, &vec![r; path.len()], 12);
    let t = 0.1;
    let out = make_solid(&tube, &MakeSolidParams { min_thickness: Some(t), ..params(0.01) }, None).unwrap();
    assert_solid(&out, "inflated tube");
    let e = out.bounds().extents();
    assert!(e.x > 0.9 * t && e.x < 1.3 * t, "diameter {}", e.x);
    assert_eq!(component_count(&out), 1);
}
