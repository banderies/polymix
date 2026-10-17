//! Performance sanity checks on a ~1.3M-triangle mesh. Run with
//! `cargo test -p mm-edit --release -- --ignored --nocapture perf`.

use super::*;
use crate::*;
use mm_core::{Bvh, DVec3, Plane};
use std::time::Instant;

#[test]
#[ignore]
fn perf_plane_cut_1m() {
    let t = Instant::now();
    let base = primitives::icosphere(1.0, 8);
    println!("icosphere(8): {} triangles, built in {:?}", base.triangle_count(), t.elapsed());
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 0.1), DVec3::new(0.1, 0.2, 1.0));
    for (name, keep, fill) in [
        ("above/none", KeepSide::Above, CutFill::None),
        ("above/flat", KeepSide::Above, CutFill::Flat),
        ("both/flat", KeepSide::Both, CutFill::Flat),
    ] {
        let mut m = base.clone();
        let t = Instant::now();
        let r = plane_cut(&mut m, plane, &PlaneCutParams { keep, fill, ..Default::default() }).unwrap();
        let dt = t.elapsed();
        println!(
            "plane_cut {name}: {:?}  (split {}, removed {}, loop {} verts, {} cap tris)",
            dt,
            r.split_vertices.len(),
            r.removed_triangles,
            r.cut_loops.first().map(|l| l.len()).unwrap_or(0),
            r.cap_triangles.len()
        );
        assert!(dt.as_secs_f64() < 2.0);
        m.check_validity().unwrap();
    }
    let mut m = base.clone();
    let t = Instant::now();
    let r = local_plane_cut(&mut m, plane, DVec3::new(1.0, 0.0, 0.0), KeepSide::Below, CutFill::Flat).unwrap();
    println!("local_plane_cut: {:?} ({} removed)", t.elapsed(), r.cut.removed_triangles);
    let t = Instant::now();
    let bvh = Bvh::build(&base);
    println!("bvh build: {:?}", t.elapsed());
    let t = Instant::now();
    let x = estimate_vessel_cross_section(&base, Some(&bvh), DVec3::new(0.0, 1.0, 0.0), Some(0.2)).unwrap();
    println!("suggest_cut_plane (bvh): {:?} axis {:?}", t.elapsed(), x.axis);
}

#[test]
#[ignore]
fn perf_vessel_tube() {
    // A long, finely tessellated tube (~1M triangles), as from a vessel segmentation.
    let path: Vec<DVec3> = (0..=2500)
        .map(|i| {
            let s = i as f64 / 2500.0;
            DVec3::new(20.0 * s, 3.0 * (6.0 * s).sin(), 2.0 * (4.0 * s).cos())
        })
        .collect();
    let m0 = primitives::tube_along(&path, &vec![1.5; path.len()], 200);
    println!("tube: {} triangles", m0.triangle_count());
    let click = m0.vertex(m0.vertex_ids().nth(200 * 1200).unwrap());
    let bvh = Bvh::build(&m0);
    let t = Instant::now();
    let plane = suggest_cut_plane(&m0, Some(&bvh), click, None).unwrap();
    println!("suggest_cut_plane: {:?}", t.elapsed());
    let mut m = m0.clone();
    let t = Instant::now();
    let r = local_plane_cut(&mut m, plane, click, KeepSide::Below, CutFill::FlatRemeshed { target_edge_length: 0.05 })
        .unwrap();
    println!("local_plane_cut + remeshed cap: {:?} ({} cap tris)", t.elapsed(), r.cut.cap_triangles.len());
    m.check_validity().unwrap();
    assert!(m.is_consistently_oriented());
    let lp = r.cut.cut_loops[0].clone();
    let mut e = m0.clone();
    let rr = local_plane_cut(&mut e, plane, click, KeepSide::Below, CutFill::None).unwrap();
    let t = Instant::now();
    let fe = extrude_boundary_loop(
        &mut e,
        &rr.cut.cut_loops[0],
        &FlowExtensionParams { length: ExtensionLength::Diameters(5.0), make_circular: true, cap: true, ..Default::default() },
    )
    .unwrap();
    println!("flow extension (5 D, circular, remeshed cap): {:?} ({} wall tris)", t.elapsed(), fe.wall_triangles.len());
    e.check_validity().unwrap();
    assert!(e.is_consistently_oriented() && !lp.is_empty());
}

fn min_angle_deg(m: &Mesh, t: u32) -> f64 {
    let p = m.tri_positions(t);
    (0..3)
        .map(|i| {
            let (a, b, c) = (p[i], p[(i + 1) % 3], p[(i + 2) % 3]);
            (b - a).angle_between(c - a).to_degrees()
        })
        .fold(180.0, f64::min)
}

#[test]
#[ignore]
fn perf_remeshed_cap_quality() {
    let base = primitives::icosphere(1.0, 8);
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 0.1), DVec3::new(0.1, 0.2, 1.0));
    for (name, fill) in [
        ("flat", CutFill::Flat),
        ("remeshed h=0.02", CutFill::FlatRemeshed { target_edge_length: 0.02 }),
        ("remeshed h=avg edge", CutFill::FlatRemeshed { target_edge_length: base.average_edge_length() }),
    ] {
        let mut m = base.clone();
        let t = Instant::now();
        let r = plane_cut(&mut m, plane, &PlaneCutParams { keep: KeepSide::Below, fill, ..Default::default() }).unwrap();
        let dt = t.elapsed();
        let angles: Vec<f64> = r.cap_triangles.iter().map(|&t| min_angle_deg(&m, t)).collect();
        let worst = angles.iter().cloned().fold(180.0, f64::min);
        let below20 = angles.iter().filter(|&&a| a < 20.0).count();
        println!(
            "cap {name}: {:?}, {} tris, worst min-angle {:.2} deg, {:.2}% below 20 deg",
            dt,
            r.cap_triangles.len(),
            worst,
            100.0 * below20 as f64 / angles.len() as f64
        );
        m.check_validity().unwrap();
        assert!(m.is_consistently_oriented() && m.is_closed());
    }
}
