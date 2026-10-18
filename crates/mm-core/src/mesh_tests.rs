//! Stress tests for the local topology edits in `mesh.rs`.

use crate::mesh::*;
use crate::primitives;
use glam::DVec3;

/// Tiny deterministic PRNG so tests are reproducible without extra dependencies.
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
}

fn random_live_edge(m: &Mesh, rng: &mut Lcg) -> u32 {
    loop {
        let e = rng.below(m.max_eid()) as u32;
        if m.is_edge(e) {
            return e;
        }
    }
}

#[test]
fn split_interior_and_boundary() {
    let mut m = primitives::plane_grid(1.0, 1.0, 2, 2);
    let nt = m.triangle_count();
    let interior = m.edge_ids().find(|&e| !m.is_boundary_edge(e)).unwrap();
    let info = m.split_edge(interior).unwrap();
    m.check_validity().unwrap();
    assert_eq!(m.triangle_count(), nt + 2);
    assert!(m.is_vertex(info.v_new));
    let boundary = m.edge_ids().find(|&e| m.is_boundary_edge(e)).unwrap();
    m.split_edge(boundary).unwrap();
    m.check_validity().unwrap();
    assert_eq!(m.triangle_count(), nt + 3);
    assert!(m.is_consistently_oriented());
}

#[test]
fn flip_preserves_orientation() {
    let mut m = primitives::icosphere(1.0, 1);
    let mut rng = Lcg(7);
    for _ in 0..200 {
        let e = random_live_edge(&m, &mut rng);
        let _ = m.flip_edge(e);
        m.check_validity().unwrap();
    }
    assert!(m.is_consistently_oriented());
    assert!(m.is_closed());
}

#[test]
fn collapse_reduces_counts() {
    let mut m = primitives::icosphere(1.0, 2);
    let (nv, nt) = (m.vertex_count(), m.triangle_count());
    let e = m.edge_ids().next().unwrap();
    let [a, b] = m.edge_vertices(e);
    let info = m.collapse_edge(a, b).unwrap();
    m.check_validity().unwrap();
    assert_eq!(m.vertex_count(), nv - 1);
    assert_eq!(m.triangle_count(), nt - 2);
    assert!(!m.is_vertex(info.v_removed));
    assert!(m.is_consistently_oriented());
}

#[test]
fn collapse_boundary_edge() {
    let mut m = primitives::plane_grid(1.0, 1.0, 3, 3);
    let e = m.edge_ids().find(|&e| m.is_boundary_edge(e)).unwrap();
    let [a, b] = m.edge_vertices(e);
    let nt = m.triangle_count();
    m.collapse_edge(a, b).unwrap();
    m.check_validity().unwrap();
    assert_eq!(m.triangle_count(), nt - 1);
}

#[test]
fn tetrahedron_cannot_collapse() {
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
    ];
    let tris = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]];
    let (mut m, rep) = Mesh::from_triangles(&pos, &tris, None);
    assert_eq!(rep, BuildReport::default());
    assert!(m.is_closed());
    for e in m.edge_ids().collect::<Vec<_>>() {
        let [a, b] = m.edge_vertices(e);
        assert!(m.collapse_edge(a, b).is_err());
    }
    m.check_validity().unwrap();
}

#[test]
fn random_edit_sequence_stays_valid() {
    for seed in 0..6u64 {
        let mut m = if seed % 2 == 0 { primitives::icosphere(1.0, 2) } else { primitives::plane_grid(2.0, 2.0, 6, 6) };
        let closed = m.is_closed();
        let mut rng = Lcg(seed * 7919 + 1);
        for step in 0..1500 {
            let e = random_live_edge(&m, &mut rng);
            let op = rng.below(4);
            let r = match op {
                0 => m.split_edge(e).map(|_| ()),
                1 => m.flip_edge(e).map(|_| ()),
                2 => {
                    let [a, b] = m.edge_vertices(e);
                    let mid = (m.vertex(a) + m.vertex(b)) * 0.5;
                    m.collapse_edge(a, b).map(|_| m.set_vertex(a, mid))
                }
                _ => {
                    let t = m.edge_triangles(e)[0];
                    m.poke_triangle(t, DVec3::splat(1.0 / 3.0)).map(|_| ())
                }
            };
            let _ = r;
            if let Err(msg) = m.check_validity() {
                panic!("seed {seed} step {step} op {op}: {msg}");
            }
            if m.triangle_count() < 30 {
                break;
            }
        }
        assert!(m.is_consistently_oriented(), "seed {seed}");
        assert_eq!(m.is_closed(), closed);
        if closed {
            // Euler characteristic of a sphere.
            let chi = m.vertex_count() as i64 - m.edge_count() as i64 + m.triangle_count() as i64;
            assert_eq!(chi, 2, "seed {seed}");
        }
    }
}

#[test]
fn heavy_collapse_to_minimum() {
    let mut m = primitives::icosphere(1.0, 3);
    let mut rng = Lcg(99);
    let mut fails = 0;
    while m.triangle_count() > 8 && fails < 20000 {
        let e = random_live_edge(&m, &mut rng);
        let [a, b] = m.edge_vertices(e);
        if m.collapse_edge(a, b).is_err() {
            fails += 1;
        }
    }
    m.check_validity().unwrap();
    let chi = m.vertex_count() as i64 - m.edge_count() as i64 + m.triangle_count() as i64;
    assert_eq!(chi, 2);
}

#[test]
fn nonmanifold_input_is_split() {
    // Three triangles sharing edge (0,1).
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.5, 1.0, 0.0),
        DVec3::new(0.5, -1.0, 0.0),
        DVec3::new(0.5, 0.0, 1.0),
    ];
    let tris = [[0, 1, 2], [1, 0, 3], [0, 1, 4]];
    let (m, rep) = Mesh::from_triangles(&pos, &tris, None);
    assert_eq!(rep.nonmanifold_split, 1);
    assert_eq!(m.triangle_count(), 3);
    m.check_validity().unwrap();
}

#[test]
fn remove_and_reuse_slots() {
    let mut m = primitives::icosphere(1.0, 1);
    let max_t = m.max_tid();
    let tris: Vec<u32> = m.triangle_ids().take(5).collect();
    m.remove_triangles(tris);
    m.check_validity().unwrap();
    assert!(!m.is_closed());
    let e = m.edge_ids().find(|&e| !m.is_boundary_edge(e)).unwrap();
    m.split_edge(e).unwrap();
    assert_eq!(m.max_tid(), max_t, "freed slots should be reused");
    m.check_validity().unwrap();
}

#[test]
fn reverse_orientation_roundtrip() {
    let mut m = primitives::cube(1.0, 2);
    let v0 = crate::measure::volume(&m);
    m.reverse_orientation();
    m.check_validity().unwrap();
    assert!(m.is_consistently_oriented());
    assert!((crate::measure::volume(&m) + v0).abs() < 1e-9);
}

#[test]
fn bvh_queries() {
    use crate::bvh::Bvh;
    use crate::geom::Ray;
    let m = primitives::icosphere(2.0, 3);
    let bvh = Bvh::build(&m);
    let hit = bvh.raycast(&m, &Ray::new(DVec3::new(10.0, 0.0, 0.0), -DVec3::X)).unwrap();
    assert!((hit.point.x - 2.0).abs() < 0.05);
    let n = bvh.nearest(&m, DVec3::new(0.0, 5.0, 0.0), f64::INFINITY).unwrap();
    assert!((n.point.y - 2.0).abs() < 0.05);
    assert!(bvh.is_inside(&m, DVec3::new(0.1, 0.2, 0.3)));
    assert!(!bvh.is_inside(&m, DVec3::new(3.0, 0.0, 0.0)));
    // Brute-force nearest agrees.
    let p = DVec3::new(0.3, -2.7, 1.1);
    let n = bvh.nearest(&m, p, f64::INFINITY).unwrap();
    let brute = m
        .triangle_ids()
        .map(|t| {
            let [a, b, c] = m.tri_positions(t);
            crate::geom::closest_point_on_triangle(p, a, b, c).0.distance_squared(p)
        })
        .fold(f64::INFINITY, f64::min);
    assert!((n.dist_sq - brute).abs() < 1e-12);
}

#[test]
fn split_bowtie_and_partial_fan() {
    // Two triangles sharing only vertex 0.
    let pos = vec![
        DVec3::ZERO,
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 1.0, 0.0),
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, -1.0, 0.0),
    ];
    let (mut m, _) = Mesh::from_triangles(&pos, &[[0, 1, 2], [0, 3, 4]], None);
    assert!(m.is_bowtie_vertex(0));
    assert_eq!(m.split_bowtie_vertices(), 1);
    m.check_validity().unwrap();
    assert!(m.vertex_ids().all(|v| !m.is_bowtie_vertex(v)));
    assert_eq!(m.vertex_count(), 6);

    // Detaching part of a manifold fan opens a slit.
    let mut s = primitives::icosphere(1.0, 1);
    let v = s.vertex_ids().next().unwrap();
    let tris = s.vertex_triangles(v);
    let half: Vec<u32> = s.triangle_fan(v, tris[0]).into_iter().take(2).collect();
    s.split_vertex(v, &half).unwrap();
    s.check_validity().unwrap();
    assert!(!s.is_closed());
}

#[test]
fn curvature_and_triangle_quality() {
    let s = crate::primitives::icosphere(2.0, 4);
    let k = crate::measure::gaussian_curvature(&s);
    let h = crate::measure::mean_curvature(&s);
    for v in s.vertex_ids() {
        assert!((k[v as usize] - 0.25).abs() < 0.03, "K {}", k[v as usize]);
        assert!((h[v as usize] - 0.5).abs() < 0.03, "H {}", h[v as usize]);
    }
    // Total Gaussian curvature of a closed genus-0 surface is 4π.
    let areas = crate::measure::mixed_areas(&s);
    let total: f64 = s.vertex_ids().map(|v| k[v as usize] * areas[v as usize]).sum();
    assert!((total - 4.0 * std::f64::consts::PI).abs() < 1e-9);
    let mut eq = Mesh::new();
    let a = eq.append_vertex(DVec3::ZERO);
    let b = eq.append_vertex(DVec3::X);
    let c = eq.append_vertex(DVec3::new(0.5, 3f64.sqrt() / 2.0, 0.0));
    let t = eq.append_triangle([a, b, c], 0).unwrap();
    assert!((crate::measure::triangle_aspect_ratio(&eq, t) - 1.0).abs() < 1e-12);
    assert!(crate::measure::triangle_angles(&eq, t).iter().all(|x| (x.to_degrees() - 60.0).abs() < 1e-9));
}

#[test]
fn clearance_between_spheres() {
    let a = crate::primitives::icosphere(1.0, 3);
    let mut b = crate::primitives::icosphere(1.0, 3);
    b.transform(|p| p + DVec3::new(2.5, 0.0, 0.0));
    let (ba, bb) = (crate::Bvh::build(&a), crate::Bvh::build(&b));
    let c = crate::measure::clearance(&a, &ba, &b, &bb, true);
    assert!(!c.intersecting);
    assert!((c.min - 0.5).abs() < 0.01, "{}", c.min);
    assert!(c.closest.0.x > 0.95 && c.closest.1.x < 1.55);
    assert!(c.dist_b.is_some());
    let mut near = crate::primitives::icosphere(1.0, 3);
    near.transform(|p| p + DVec3::new(1.5, 0.0, 0.0));
    let c = crate::measure::clearance(&a, &ba, &near, &crate::Bvh::build(&near), false);
    assert!(c.intersecting && c.min == 0.0);
}

#[test]
fn vertex_colors_follow_edits() {
    let mut m = crate::primitives::plane_grid(2.0, 2.0, 2, 2);
    assert!(!m.has_vertex_colors());
    for v in m.vertex_ids().collect::<Vec<_>>() {
        let p = m.vertex(v);
        m.set_vertex_color(v, [(p.x as f32 + 1.0) / 2.0, 0.0, 0.0]);
    }
    // Splitting an edge interpolates.
    let e = m.edge_ids().find(|&e| {
        let [a, b] = m.edge_vertices(e);
        (m.vertex(a).x - m.vertex(b).x).abs() > 0.5 && (m.vertex(a).y - m.vertex(b).y).abs() < 1e-9
    }).unwrap();
    let [a, b] = m.edge_vertices(e);
    let want = (m.vertex_color(a).unwrap()[0] + m.vertex_color(b).unwrap()[0]) / 2.0;
    let s = m.split_edge(e).unwrap();
    assert!((m.vertex_color(s.v_new).unwrap()[0] - want).abs() < 1e-6);
    // Poking a triangle blends its corners.
    let t = m.triangle_ids().next().unwrap();
    let corners = m.triangle(t).map(|v| m.vertex_color(v).unwrap()[0]);
    let p = m.poke_triangle(t, DVec3::splat(1.0 / 3.0)).unwrap();
    assert!((m.vertex_color(p.v_new).unwrap()[0] - corners.iter().sum::<f32>() / 3.0).abs() < 1e-6);
    // Copies keep colours.
    let (c, map) = m.compacted();
    for v in m.vertex_ids() {
        assert_eq!(c.vertex_color(map[v as usize]), m.vertex_color(v));
    }
    let mut host = crate::primitives::cube(1.0, 1);
    let vm = host.append_mesh(&m, 1);
    assert!(host.has_vertex_colors());
    assert_eq!(host.vertex_color(vm[a as usize]), m.vertex_color(a));
    assert_eq!(host.vertex_color(0), Some(DEFAULT_VERTEX_COLOR));
}
