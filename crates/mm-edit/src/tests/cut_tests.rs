use super::*;
use crate::*;
use mm_core::{DVec3, Plane};

fn cut(m: &Mesh, plane: Plane, keep: KeepSide, fill: CutFill) -> (Mesh, PlaneCutResult) {
    let mut c = m.clone();
    let r = plane_cut(&mut c, plane, &PlaneCutParams { keep, fill, ..Default::default() }).unwrap();
    assert_valid(&c);
    (c, r)
}

#[test]
fn sphere_cut_above_below_both() {
    let base = primitives::icosphere(1.0, 3);
    let v0 = volume(&base);
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 0.3), DVec3::new(0.2, 0.1, 1.0));
    let (a, ra) = cut(&base, plane, KeepSide::Above, CutFill::Flat);
    let (b, rb) = cut(&base, plane, KeepSide::Below, CutFill::Flat);
    for (m, r) in [(&a, &ra), (&b, &rb)] {
        assert!(m.is_closed());
        assert_eq!(r.cut_loops.len(), 1);
        assert!(r.loop_capped[0]);
        assert!(!r.cap_triangles.is_empty());
        assert_eq!(n_components(m), 1);
    }
    for v in a.vertex_ids() {
        assert!(plane.signed_distance(a.vertex(v)) > -1e-9);
    }
    for v in b.vertex_ids() {
        assert!(plane.signed_distance(b.vertex(v)) < 1e-9);
    }
    // Caps face the discarded side.
    for &t in &ra.cap_triangles {
        assert!(a.tri_normal(t).dot(plane.normal) < -0.999);
    }
    for &t in &rb.cap_triangles {
        assert!(b.tri_normal(t).dot(plane.normal) > 0.999);
    }
    let (va, vb) = (volume(&a), volume(&b));
    assert!(va > 0.0 && vb > 0.0);
    // Sliver snapping moves a few vertices by < 1% of an edge, so the sum is close...
    assert!((va + vb - v0).abs() < 1e-5 * v0, "{va} + {vb} != {v0}");
    // ...and exact without snapping.
    let exact = |keep| {
        let mut m = base.clone();
        plane_cut(&mut m, plane, &PlaneCutParams { keep, fill: CutFill::Flat, snap_tolerance: 0.0 }).unwrap();
        assert_valid(&m);
        assert!(m.is_closed());
        volume(&m)
    };
    let (ea, eb) = (exact(KeepSide::Above), exact(KeepSide::Below));
    assert!((ea + eb - v0).abs() < 1e-12 * v0, "{ea} + {eb} != {v0}");

    let (c, rc) = cut(&base, plane, KeepSide::Both, CutFill::Flat);
    assert!(c.is_closed());
    assert_eq!(n_components(&c), 2);
    assert_eq!(rc.cut_loops.len(), 2);
    assert!(rc.loop_capped.iter().all(|&x| x));
    assert_eq!(rc.removed_triangles, 0);
    assert!((volume(&c) - (va + vb)).abs() < 1e-9 * v0);
    let mut vols: Vec<f64> = connected_components(&c)
        .iter()
        .map(|comp| {
            comp.iter()
                .map(|&t| {
                    let [p, q, r] = c.tri_positions(t);
                    p.dot(q.cross(r)) / 6.0
                })
                .sum::<f64>()
        })
        .collect();
    vols.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let mut want = [va, vb];
    want.sort_by(|x, y| x.partial_cmp(y).unwrap());
    assert!((vols[0] - want[0]).abs() < 1e-9 && (vols[1] - want[1]).abs() < 1e-9);

    // Slice without caps: two open pieces.
    let (d, rd) = cut(&base, plane, KeepSide::Both, CutFill::None);
    assert_eq!(n_components(&d), 2);
    assert_eq!(n_loops(&d), 2);
    assert!(rd.cap_triangles.is_empty());
}

#[test]
fn remeshed_cap_has_target_edge_length() {
    let base = primitives::icosphere(1.0, 3);
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 0.1), DVec3::Z);
    let target = 0.08;
    let (flat, _) = cut(&base, plane, KeepSide::Below, CutFill::Flat);
    let (m, r) = cut(&base, plane, KeepSide::Below, CutFill::FlatRemeshed { target_edge_length: target });
    assert!(m.is_closed());
    assert!((volume(&m) - volume(&flat)).abs() < 1e-9);
    let caps: std::collections::HashSet<u32> = r.cap_triangles.iter().copied().collect();
    let mut lens = Vec::new();
    for &t in &r.cap_triangles {
        for e in m.tri_edges(t) {
            let [t0, t1] = m.edge_triangles(e);
            if caps.contains(&t0) && caps.contains(&t1) {
                lens.push(m.edge_length(e));
            }
        }
        assert!(m.tri_normal(t).dot(DVec3::Z) > 0.999);
    }
    let mean = lens.iter().sum::<f64>() / lens.len() as f64;
    assert!(mean > 0.6 * target && mean < 1.3 * target, "mean cap edge {mean}");
    let max = lens.iter().cloned().fold(0.0, f64::max);
    assert!(max < 1.5 * target, "max cap edge {max}");
    let rad: f64 = (1.0f64 - 0.01).sqrt();
    let expect = PI * rad * rad / (3f64.sqrt() / 4.0 * target * target);
    let n = r.cap_triangles.len() as f64;
    assert!(n > 0.5 * expect && n < 2.0 * expect, "{n} cap triangles, expected ~{expect}");
}

#[test]
fn torus_cut_gives_annular_cap() {
    let (big, small) = (2.0, 0.6);
    let base = torus(big, small, 64, 32);
    assert_valid(&base);
    let v0 = volume(&base);
    let z = 0.05;
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, z), DVec3::Z);
    let (a, ra) = cut(&base, plane, KeepSide::Above, CutFill::Flat);
    let (b, _) = cut(&base, plane, KeepSide::Below, CutFill::Flat);
    assert!(a.is_closed() && b.is_closed());
    assert_eq!(ra.cut_loops.len(), 2);
    assert_eq!(ra.cap_groups.len(), 1, "outer + hole form a single cap region");
    assert!((volume(&a) + volume(&b) - v0).abs() < 1e-9 * v0);
    let cap_area: f64 = ra.cap_triangles.iter().map(|&t| a.tri_area(t)).sum();
    let rr = (small * small - z * z).sqrt();
    let want = PI * ((big + rr).powi(2) - (big - rr).powi(2));
    assert!((cap_area - want).abs() < 0.03 * want, "cap area {cap_area} vs {want}");
    // Cap must not cover the hole: no cap triangle centroid within the inner radius.
    for &t in &ra.cap_triangles {
        let c = a.tri_centroid(t);
        let rxy = (c.x * c.x + c.y * c.y).sqrt();
        assert!(rxy > big - rr - 1e-6);
    }
    // Exactly through the vertex rings at z = 0 (snapping) and remeshed.
    let plane0 = Plane::from_point_normal(DVec3::ZERO, DVec3::Z);
    let (c, rc) = cut(&base, plane0, KeepSide::Below, CutFill::FlatRemeshed { target_edge_length: 0.1 });
    assert!(c.is_closed());
    assert!(rc.split_vertices.is_empty());
    assert!((volume(&c) - 0.5 * v0).abs() < 1e-9 * v0);
    // Vertical plane through the axis: two separate disks.
    let (d, rd) = cut(&base, Plane::from_point_normal(DVec3::ZERO, DVec3::X), KeepSide::Above, CutFill::Flat);
    assert!(d.is_closed());
    assert_eq!(rd.cap_groups.len(), 2);
}

#[test]
fn island_inside_hole() {
    // A sphere floating in the hole of a torus: loops = torus outer (outer), torus inner
    // (hole), sphere section (outer inside the hole).
    let mut m = torus(2.0, 0.6, 64, 32);
    let s = primitives::icosphere(0.5, 3);
    m.append_mesh(&s, 1);
    let v0 = volume(&m);
    let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 0.02), DVec3::Z);
    let (a, ra) = cut(&m, plane, KeepSide::Above, CutFill::Flat);
    let (b, _) = cut(&m, plane, KeepSide::Below, CutFill::Flat);
    assert!(a.is_closed() && b.is_closed());
    assert_eq!(ra.cut_loops.len(), 3);
    assert_eq!(ra.cap_groups.len(), 2);
    assert!((volume(&a) + volume(&b) - v0).abs() < 1e-9 * v0);
}

#[test]
fn cut_through_vertices_snaps() {
    // uv sphere has a vertex ring exactly on z = 0.
    let base = primitives::uv_sphere(1.0, 16, 8);
    let (a, ra) = cut(&base, Plane::from_point_normal(DVec3::ZERO, DVec3::Z), KeepSide::Above, CutFill::Flat);
    assert!(a.is_closed());
    assert!(ra.split_vertices.is_empty());
    assert_eq!(ra.cut_loops[0].len(), 16);
    // Slightly off the ring: vertices are snapped instead of making slivers.
    let (b, rb) =
        cut(&base, Plane::from_point_normal(DVec3::new(0.0, 0.0, 1e-4), DVec3::Z), KeepSide::Above, CutFill::Flat);
    assert!(b.is_closed());
    assert!(rb.split_vertices.is_empty());
    assert_eq!(rb.snapped_vertices.len(), 16);
    let min_area = b.triangle_ids().map(|t| b.tri_area(t)).fold(f64::INFINITY, f64::min);
    assert!(min_area > 1e-4);
    // Cube cut along a grid line.
    let cube = primitives::cube(2.0, 2);
    let (c, rc) = cut(&cube, Plane::from_point_normal(DVec3::ZERO, DVec3::Z), KeepSide::Below, CutFill::Flat);
    assert!(c.is_closed());
    assert!(rc.split_vertices.is_empty());
    assert!((volume(&c) - 4.0).abs() < 1e-12);
    // Cube cut exactly at a face: nothing to remove or cap.
    let (d, rd) = cut(&cube, Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.0), DVec3::Z), KeepSide::Below, CutFill::Flat);
    assert!(d.is_closed());
    assert_eq!(rd.removed_triangles, 0);
    assert!((volume(&d) - 8.0).abs() < 1e-12);
    let (e, _) = cut(&cube, Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.0), DVec3::Z), KeepSide::Above, CutFill::Flat);
    assert_eq!(e.triangle_count(), 0);
}

#[test]
fn cut_open_mesh_and_miss() {
    let grid = primitives::plane_grid(2.0, 2.0, 10, 10);
    let (g, r) = cut(&grid, Plane::from_point_normal(DVec3::new(0.13, 0.0, 0.0), DVec3::X), KeepSide::Above, CutFill::Flat);
    assert_eq!(r.cut_loops.len(), 1);
    assert!(!r.loop_capped[0], "open curve on an open mesh must not be capped");
    assert!(g.vertex_ids().all(|v| g.vertex(v).x > 0.13 - 1e-9));
    let sphere = primitives::icosphere(1.0, 2);
    let (s, rs) = cut(&sphere, Plane::from_point_normal(DVec3::new(0.0, 0.0, 5.0), DVec3::Z), KeepSide::Below, CutFill::Flat);
    assert_eq!(s.triangle_count(), sphere.triangle_count());
    assert!(rs.cut_loops.is_empty());
}

#[test]
fn cap_planar_loops_closes_tube() {
    let mut m = primitives::cylinder(0.5, 2.0, 20, 4, false);
    let loops: Vec<Vec<u32>> = boundary_loops(&m).into_iter().map(|l| l.vertices).collect();
    let r = cap_planar_loops(&mut m, &loops, None, CutFill::Flat).unwrap();
    assert_valid(&m);
    assert!(m.is_closed());
    assert_eq!(r.groups.len(), 2);
    assert!(volume(&m) > 0.0);
    let want = 0.5 * 20.0 * 0.25 * (TAU / 20.0).sin() * 2.0;
    assert!((volume(&m) - want).abs() < 1e-9);
}

#[test]
fn intersection_curves_preview() {
    let m = u_tube(3.0, 3.0, 0.4, 24);
    let curves = plane_intersection_curves(&m, Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.0), DVec3::Z));
    assert_eq!(curves.len(), 2);
    for c in &curves {
        assert!(c.closed);
        assert!((c.equivalent_diameter() - 0.8).abs() < 0.02);
        assert!((c.centroid.z - 1.0).abs() < 1e-9);
    }
    let near = nearest_intersection_curve(&m, Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.0), DVec3::Z), DVec3::new(3.2, 0.0, 1.0))
        .unwrap();
    assert!((near.centroid.x - 3.0).abs() < 1e-3);
}

struct Lcg(u64);
impl Lcg {
    fn f(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    fn unit(&mut self) -> DVec3 {
        loop {
            let v = DVec3::new(self.f() * 2.0 - 1.0, self.f() * 2.0 - 1.0, self.f() * 2.0 - 1.0);
            if v.length() > 0.1 && v.length() < 1.0 {
                return v.normalize();
            }
        }
    }
}

#[test]
fn random_planes_keep_meshes_valid_and_closed() {
    let mut rng = Lcg(12345);
    let mut shapes = vec![
        primitives::icosphere(1.0, 3),
        torus(1.5, 0.5, 40, 20),
        primitives::cube(1.6, 4),
        primitives::uv_sphere(1.0, 24, 12),
    ];
    // Two interlocked tori: cuts produce several nested and separate loops.
    let mut t2 = torus(1.5, 0.5, 40, 20);
    t2.transform(|p| DVec3::new(p.x + 1.5, p.z, -p.y));
    let mut tt = torus(1.5, 0.5, 40, 20);
    tt.append_mesh(&t2, 1);
    shapes.push(tt);
    let fills = [CutFill::Flat, CutFill::FlatRemeshed { target_edge_length: 0.15 }];
    let mut cuts = 0;
    let trials: usize = std::env::var("MM_EDIT_FUZZ").ok().and_then(|s| s.parse().ok()).unwrap_or(12);
    for shape in &shapes {
        let v0 = volume(shape);
        for _ in 0..trials {
            let n = rng.unit();
            let d = (rng.f() * 2.0 - 1.0) * 0.8;
            let plane = Plane { normal: n, d };
            for fill in fills {
                let mut a = shape.clone();
                let ra = plane_cut(&mut a, plane, &PlaneCutParams { keep: KeepSide::Above, fill, snap_tolerance: 0.0 })
                    .unwrap();
                let mut b = shape.clone();
                plane_cut(&mut b, plane, &PlaneCutParams { keep: KeepSide::Below, fill, snap_tolerance: 0.0 }).unwrap();
                let mut c = shape.clone();
                let rc = plane_cut(&mut c, plane, &PlaneCutParams { keep: KeepSide::Both, fill, ..Default::default() })
                    .unwrap();
                for (m, name) in [(&a, "above"), (&b, "below"), (&c, "both")] {
                    m.check_validity().unwrap();
                    assert!(m.is_consistently_oriented(), "{name} not oriented");
                    assert!(m.is_closed(), "{name} not closed (plane {plane:?})");
                }
                assert!(ra.loop_capped.iter().all(|&x| x));
                assert!(rc.loop_capped.iter().all(|&x| x));
                let (va, vb) = (volume(&a), volume(&b));
                assert!((va + vb - v0).abs() < 1e-9 * v0.abs().max(1.0), "{va} + {vb} vs {v0}");
                assert!(va >= -1e-12 && vb >= -1e-12, "negative volume {va} {vb} shape v0={v0} plane {plane:?} fill {fill:?} loops {}", ra.cut_loops.len());
                cuts += 1;
            }
        }
    }
    assert!(cuts >= 100);
}

#[test]
fn random_local_cuts_stay_valid() {
    let mut rng = Lcg(777);
    let base = u_tube(3.0, 3.0, 0.4, 20);
    let trials: usize = std::env::var("MM_EDIT_FUZZ").ok().and_then(|s| s.parse().ok()).unwrap_or(25);
    for _ in 0..trials {
        // Random point on the surface, random tilt.
        // Seeds off the existing open ends: a plane through a boundary vertex makes the cut
        // touch the old boundary there, merging both loops (valid, but not cappable).
        let v = (rng.f() * base.vertex_count() as f64) as usize % base.vertex_count();
        let vid = base.vertex_ids().nth(v).unwrap();
        if base.is_boundary_vertex(vid) {
            continue;
        }
        let p = base.vertex(vid);
        let n = (DVec3::Z + rng.unit() * 0.4).normalize();
        let plane = Plane::from_point_normal(p, n);
        for keep in [KeepSide::Above, KeepSide::Below, KeepSide::Both] {
            let mut m = base.clone();
            match local_plane_cut(&mut m, plane, p, keep, CutFill::Flat) {
                Ok(r) => {
                    m.check_validity().unwrap();
                    assert!(m.is_consistently_oriented());
                    if r.curve_closed {
                        assert!(
                            r.cut.loop_capped.iter().any(|&c| c),
                            "keep {keep:?} plane {plane:?} p {p:?} loops {:?} capped {:?} split {} snapped {} removed {}",
                            r.cut.cut_loops.iter().map(|l| l.len()).collect::<Vec<_>>(),
                            r.cut.loop_capped,
                            r.cut.split_vertices.len(),
                            r.cut.snapped_vertices.len(),
                            r.cut.removed_triangles
                        );
                    }
                }
                Err(EditError::CurveDoesNotSeparate) | Err(EditError::NoIntersection) => {
                    m.check_validity().unwrap();
                }
                Err(e) => panic!("{e}"),
            }
        }
    }
}
