//! End-to-end test of the vessel → CFD workflow on a synthetic, deliberately messy vessel
//! tree (overlapping tubes, aneurysm, holes, floating fragments).

use crate::scene::SceneObject;
use crate::tools::vessel::{find_vessel_ends, EndKind};
use glam::DVec3;
use mm_core::{primitives, topology, Bvh, Mesh};

fn bezier(a: DVec3, b: DVec3, c: DVec3, n: usize) -> Vec<DVec3> {
    (0..=n)
        .map(|i| {
            let t = i as f64 / n as f64;
            a * (1.0 - t) * (1.0 - t) + b * 2.0 * t * (1.0 - t) + c * t * t
        })
        .collect()
}

fn messy_vessel() -> Mesh {
    let tube = |p: Vec<DVec3>, r0: f64, r1: f64| {
        let n = p.len();
        let radii: Vec<f64> = (0..n).map(|i| r0 + (r1 - r0) * i as f64 / (n - 1) as f64).collect();
        primitives::tube_along(&p, &radii, 32)
    };
    let mut m = Mesh::new();
    m.append_mesh(&tube(bezier(DVec3::ZERO, DVec3::new(1.0, 0.0, 20.0), DVec3::new(0.0, 0.0, 40.0), 40), 3.0, 2.8), 0);
    m.append_mesh(&tube(bezier(DVec3::new(0.0, 0.0, 39.0), DVec3::new(-6.0, 0.0, 52.0), DVec3::new(-16.0, 1.0, 70.0), 40), 2.3, 1.9), 0);
    m.append_mesh(&tube(bezier(DVec3::new(0.0, 0.0, 39.0), DVec3::new(7.0, 2.0, 50.0), DVec3::new(15.0, 5.0, 68.0), 40), 2.1, 1.7), 0);
    let mut sac = primitives::icosphere(3.6, 3);
    sac.transform(|p| p + DVec3::new(0.5, 0.5, 44.5));
    m.append_mesh(&sac, 0);
    let mut frag = primitives::icosphere(0.3, 1);
    frag.transform(|p| p + DVec3::new(10.0, 0.0, 10.0));
    m.append_mesh(&frag, 0);
    let holes: Vec<u32> = m.triangle_ids().filter(|&t| (m.tri_centroid(t) - DVec3::new(3.0, 0.0, 12.0)).length() < 1.0).collect();
    m.remove_triangles(holes);
    m
}

#[test]
fn vessel_to_cfd_pipeline() {
    let mut raw = messy_vessel();
    assert!(!raw.is_closed());

    // 0. Inspector auto-repair: holes filled, fragment removed, every shell closed.
    let rep = mm_repair::auto_repair(&mut raw, &mm_repair::RepairParams::default(), None);
    raw.check_validity().unwrap();
    assert!(rep.holes_filled >= 1 && rep.components_removed >= 1, "{rep:?}");
    assert!(raw.is_closed());

    // 1. Make Solid: one closed manifold shell (fragments removed by volume threshold).
    let params = mm_solid::MakeSolidParams::from_sliders(&raw, 0.5, 0.5);
    let mut m = mm_solid::make_solid(&raw, &params, None).expect("make solid");
    m.check_validity().unwrap();
    assert!(m.is_closed());
    let comps = topology::connected_components(&m);
    let largest = comps.iter().map(|c| c.len()).max().unwrap();
    let small: Vec<u32> = comps.iter().filter(|c| c.len() < largest).flatten().copied().collect();
    m.remove_triangles(small);
    assert_eq!(topology::connected_components(&m).len(), 1);

    // 2. Trim the three branch ends square, like clicking near each tip.
    let tips = [DVec3::new(0.0, 0.0, 3.0), DVec3::new(-14.0, 1.0, 66.0), DVec3::new(13.5, 4.5, 64.5)];
    for tip in tips {
        let bvh = Bvh::build(&m);
        let on_surface = bvh.nearest(&m, tip, f64::INFINITY).unwrap().point;
        let x = mm_edit::estimate_vessel_cross_section(&m, Some(&bvh), on_surface, None).expect("cross-section");
        let target = m.average_edge_length();
        mm_edit::local_plane_cut(&mut m, x.plane, on_surface, mm_edit::KeepSide::Below, mm_edit::CutFill::FlatRemeshed { target_edge_length: target })
            .expect("local cut");
        m.check_validity().unwrap();
        assert!(m.is_closed(), "trim keeps the mesh closed");
    }
    let ends = find_vessel_ends(&m);
    let caps: Vec<i32> = ends.iter().filter_map(|e| if let EndKind::Cap(g) = e.kind { Some(g) } else { None }).collect();
    assert_eq!(caps.len(), 3, "three flat caps after trimming: {ends:?}");

    // 3. Flow extension on each cap (remove cap, extrude 3 diameters, re-cap with same group).
    for e in ends.iter().filter(|e| matches!(e.kind, EndKind::Cap(_))) {
        let EndKind::Cap(g) = e.kind else { unreachable!() };
        let tris: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).collect();
        m.remove_triangles(tris);
        let lp = topology::boundary_loops(&m)
            .into_iter()
            .min_by(|a, b| a.centroid(&m).distance(e.center).partial_cmp(&b.centroid(&m).distance(e.center)).unwrap())
            .unwrap();
        let p = mm_edit::FlowExtensionParams {
            length: mm_edit::ExtensionLength::Diameters(3.0),
            segments: 12,
            make_circular: true,
            cap: true,
            ..Default::default()
        };
        let r = mm_edit::extrude_boundary_loop(&mut m, &lp.vertices, &p).expect("flow extension");
        for &t in &r.cap_triangles {
            m.set_tri_group(t, g);
        }
        for &t in &r.wall_triangles {
            m.set_tri_group(t, 0);
        }
        assert!(r.direction.dot(e.normal) > 0.9, "extension points outward");
        m.check_validity().unwrap();
        assert!(m.is_closed());
    }

    // 4. Remesh for CFD, keeping patch borders.
    let rp = mm_remesh::RemeshParams { target_edge_length: m.average_edge_length(), iterations: 5, preserve_group_boundaries: true, ..Default::default() };
    mm_remesh::remesh(&mut m, &rp, None).expect("remesh");
    m.check_validity().unwrap();
    assert!(m.is_closed());
    assert!(m.is_consistently_oriented());
    assert!(mm_core::measure::volume(&m) > 0.0);

    // 5. Patches: wall + 3 caps, exported as a multi-solid STL.
    let mut obj = SceneObject::new(1, "vessel", m);
    let stats = crate::tools::patches::group_stats(&obj.mesh);
    assert_eq!(stats.len(), 4, "wall + 3 caps: {stats:?}");
    obj.group_names.insert(0, "wall".into());
    let dir = std::env::temp_dir().join(format!("polymix_pipeline_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vessel_patches.stl");
    let n = crate::tools::patches::write_multisolid_stl(&path, &obj).unwrap();
    assert_eq!(n, 4);
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(text.matches("endsolid").count(), 4);
    assert!(text.contains("solid wall"));
    // Plain STL round trip stays closed.
    let stl = dir.join("vessel.stl");
    mm_io::write_mesh(&stl, &obj.mesh, &mm_io::WriteOptions::default()).unwrap();
    let back = mm_io::read_mesh(&stl).unwrap();
    assert!(back.mesh.is_closed());
    std::fs::remove_dir_all(&dir).ok();
}
