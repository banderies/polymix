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

/// The bounds as `Mesh::bounds` computed them before they were cached: every corner of every
/// live triangle, in triangle order.
fn bounds_by_triangles(m: &Mesh) -> crate::Aabb {
    let mut b = crate::Aabb::EMPTY;
    for t in m.triangle_ids() {
        for p in m.tri_positions(t) {
            b.include(p);
        }
    }
    b
}

fn bits(b: crate::Aabb) -> [u64; 6] {
    let (lo, hi) = (b.min.to_array(), b.max.to_array());
    [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]].map(f64::to_bits)
}

/// Checks the bounds bit for bit against the triangle walk: computed, then from the cache.
fn assert_bounds(m: &Mesh, what: &str) {
    let want = bits(bounds_by_triangles(m));
    assert_eq!(bits(m.bounds()), want, "{what}: computed");
    assert_eq!(m.cached_bounds().map(bits), Some(want), "{what}: cached");
    assert_eq!(bits(m.bounds()), want, "{what}: from the cache");
}

#[test]
fn cached_bounds_follow_every_edit() {
    for seed in 0..12u64 {
        let mut m = match seed % 4 {
            0 => primitives::icosphere(1.0, 2),
            // Flat at z = 0: the zeros there get mixed with -0.0 below.
            1 => primitives::plane_grid(2.0, 2.0, 6, 6),
            2 => primitives::cube(1.5, 3),
            _ => primitives::cylinder(0.7, 2.0, 12, 4, false),
        };
        let mut rng = Lcg(seed * 104_729 + 3);
        assert_bounds(&m, "start");
        for step in 0..1500 {
            // Ask first, so that each edit has a cached box to reset.
            let _ = m.bounds();
            let e = random_live_edge(&m, &mut rng);
            let v = m.edge_vertices(e)[rng.below(2)];
            let t = m.edge_triangles(e)[0];
            let op = rng.below(11);
            match op {
                0 => drop(m.split_edge(e)),
                1 => drop(m.flip_edge(e)),
                2 => {
                    let [a, b] = m.edge_vertices(e);
                    let mid = (m.vertex(a) + m.vertex(b)) * 0.5;
                    let _ = m.collapse_edge(a, b).map(|_| m.set_vertex(a, mid));
                }
                3 => drop(m.poke_triangle(t, DVec3::new(0.2, 0.3, 0.5))),
                4 => drop(m.remove_triangle(t, rng.below(2) == 0)),
                // Move a vertex out (a new extreme) or in.
                5 => m.set_vertex(v, m.vertex(v) * if rng.below(2) == 0 { 1.7 } else { 0.4 }),
                // Written without bumping the version, as some callers do.
                6 => m.positions_mut()[v as usize] *= 2.5,
                7 => {
                    m.positions_mut()[v as usize].x -= 3.0;
                    m.bump_shape_version();
                }
                // Signed zeros, on axes where they can be the extreme.
                8 => {
                    let mut p = m.vertex(v);
                    p[rng.below(3)] = if rng.below(2) == 0 { 0.0 } else { -0.0 };
                    m.set_vertex(v, p);
                }
                // NaN coordinates (from a broken file, say): the triangle walk's order decides.
                9 if seed >= 8 => {
                    let mut p = m.vertex(v);
                    p[rng.below(3)] = f64::NAN;
                    m.set_vertex(v, p);
                }
                9 => m.reverse_triangle(t),
                _ => {
                    m.remove_isolated_vertices();
                }
            }
            assert_bounds(&m, &format!("seed {seed} step {step} op {op}"));
            if step % 97 == 0 {
                // A copy carries the cache; editing it leaves the original's alone.
                let mut c = m.clone();
                assert_eq!(c.cached_bounds().map(bits), m.cached_bounds().map(bits));
                c.transform(|p| p * 3.0 + DVec3::X);
                assert_bounds(&c, "transformed copy");
                assert_bounds(&m, "original after the copy's edit");
                let mut host = primitives::cube(0.5, 1);
                let _ = host.bounds();
                host.append_mesh(&m, 1);
                assert_bounds(&host, "appended");
            }
            if m.triangle_count() < 20 {
                break;
            }
        }
        m.check_validity().unwrap();
    }
}

#[test]
fn bounds_with_signed_zeros_and_nan_match_any_triangle_order() {
    // x is +0.0 at some corners and -0.0 at others, and both are the minimum; so is z.
    let pos = vec![
        DVec3::new(0.0, 0.0, 0.0),
        DVec3::new(-0.0, 1.0, 0.0),
        DVec3::new(1.0, 0.0, -0.0),
        DVec3::new(1.0, 1.0, 0.5),
        DVec3::new(-0.0, 2.0, 0.25),
        DVec3::new(0.0, 3.0, 1.0),
    ];
    let tris = [[0, 2, 1], [1, 2, 3], [1, 3, 4], [4, 3, 5]];
    for rot in 0..tris.len() {
        let mut order = tris.to_vec();
        order.rotate_left(rot);
        let (mut m, _) = Mesh::from_triangles(&pos, &order, None);
        assert_bounds(&m, &format!("signed zeros, rotation {rot}"));
        let v = m.vertex_ids().nth(rot).unwrap();
        let mut p = m.vertex(v);
        p.y = f64::NAN;
        m.set_vertex(v, p);
        assert_bounds(&m, &format!("NaN, rotation {rot}"));
    }
    assert_bounds(&Mesh::new(), "empty");
}

/// One random local edit, the same on any mesh in the same state with the same `rng`.
fn random_edit(m: &mut Mesh, rng: &mut Lcg) {
    let e = random_live_edge(m, rng);
    match rng.below(6) {
        0 => drop(m.split_edge(e)),
        1 => drop(m.flip_edge(e)),
        2 => {
            let [a, b] = m.edge_vertices(e);
            let mid = (m.vertex(a) + m.vertex(b)) * 0.5;
            let _ = m.collapse_edge(a, b).map(|_| m.set_vertex(a, mid));
        }
        3 => drop(m.poke_triangle(m.edge_triangles(e)[0], DVec3::new(0.2, 0.3, 0.5))),
        4 => drop(m.remove_triangle(m.edge_triangles(e)[0], true)),
        _ => {
            let t = m.edge_triangles(e)[0];
            m.set_tri_group(t, 7);
        }
    }
}

#[test]
fn clones_copy_every_field() {
    // Copies after edits that leave free lists, groups, colours and a cached box behind. The
    // capped cylinders have cap centres with 40 and 300 edges, so some edge lists live on the
    // heap; one of them is then cut back to a few edges while staying there.
    for (sub, colors) in [(2, false), (3, true), (7, false), (7, true), (40, false), (300, true)] {
        let mut m = if sub < 10 { primitives::icosphere(1.0, sub) } else { primitives::cylinder(1.0, 2.0, sub, 300, true) };
        let mut rng = Lcg(sub as u64 * 31 + colors as u64);
        if colors {
            m.set_vertex_color(5, [0.1, 0.2, 0.3]);
        }
        if sub == 40 {
            let hub = m.vertex_ids().max_by_key(|&v| m.vertex_valence(v)).unwrap();
            let fan = m.vertex_triangles(hub);
            m.remove_triangles(fan.iter().copied().skip(5));
            assert!(m.vertex_valence(hub) <= 8 && m.edge_list_storage()[hub as usize].1);
        }
        for _ in 0..400 {
            random_edit(&mut m, &mut rng);
        }
        let _ = m.bounds();
        let what = format!("mesh {sub} colours {colors}");
        let derived = m.clone_derived();
        m.assert_same_state(&derived, &format!("{what}: derived clone"));
        let copy = m.clone();
        copy.assert_same_state(&derived, &format!("{what}: clone"));
        assert_eq!(copy.edge_list_storage(), derived.edge_list_storage(), "{what}: edge list storage");
        assert!(sub < 10 || copy.edge_list_storage().iter().any(|&(_, heap)| heap), "{what}: no list on the heap");
        let roomy = m.clone_with_headroom(1000, 2000);
        m.assert_same_state(&roomy, &format!("{what}: headroom"));
        let [sv, st, se] = roomy.spare_capacity();
        assert!(sv >= 1000 && st >= 2000 && se >= 3000, "{what}: spare {sv} {st} {se}");
        // The copies then take the same edits the same way: new elements land in the same
        // slots, so the free lists came across in order.
        let (mut a, mut b) = (m.clone(), roomy);
        let seed = rng.next();
        for (x, copy) in [(&mut m, 0), (&mut a, 1), (&mut b, 2)] {
            let mut r = Lcg(seed);
            for _ in 0..300 {
                random_edit(x, &mut r);
            }
            assert!(x.check_validity().is_ok(), "{what}: copy {copy}");
        }
        // Each edit takes a fresh version, so the versions differ now; compare the rest.
        a.assert_same_content(&m, &format!("{what}: clone after edits"));
        b.assert_same_content(&m, &format!("{what}: roomy clone after edits"));
    }
    Mesh::new().assert_same_state(&Mesh::new().clone(), "empty");
}

#[test]
fn checked_collapses_match_collapse_edge() {
    for seed in 0..8u64 {
        let base = if seed % 2 == 0 { primitives::icosphere(1.0, 3) } else { primitives::plane_grid(2.0, 2.0, 16, 16) };
        // `plain` collapses with collapse_edge; `tokened` checks first, then collapses with
        // the token, sometimes after an edit elsewhere has made the token stale.
        let mut plain = base.clone();
        let mut tokened = base.clone();
        let mut rng = Lcg(seed * 7 + 5);
        let mut stale_seen = 0;
        for step in 0..600 {
            let e = random_live_edge(&plain, &mut rng);
            let [a, b] = plain.edge_vertices(e);
            let (keep, rem) = if rng.below(2) == 0 { (a, b) } else { (b, a) };
            let checked = tokened.check_collapse(keep, rem);
            assert_eq!(checked.map(|_| ()), plain.can_collapse(keep, rem), "seed {seed} step {step}: check");
            let Ok(token) = checked else { continue };
            assert_eq!((token.v_keep(), token.v_remove()), (keep, rem));
            match rng.below(4) {
                // A position change keeps the token's topology.
                0 => {
                    let p = plain.vertex(keep) * 1.01;
                    plain.set_vertex(keep, p);
                    tokened.set_vertex(keep, p);
                }
                // A topology change elsewhere (or nearby) makes it stale.
                1 => {
                    let f = random_live_edge(&plain, &mut rng);
                    let (x, y) = (plain.flip_edge(f).is_ok(), tokened.flip_edge(f).is_ok());
                    assert_eq!(x, y);
                    stale_seen += x as usize;
                }
                2 => {
                    let f = random_live_edge(&plain, &mut rng);
                    let (x, y) = (plain.split_edge(f).map(|i| i.v_new).ok(), tokened.split_edge(f).map(|i| i.v_new).ok());
                    assert_eq!(x, y);
                    stale_seen += 1;
                }
                _ => {}
            }
            let x = plain.collapse_edge(keep, rem).map(|i| (i.v_kept, i.v_removed, i.is_boundary, i.e_collapsed, i.t_removed, i.e_removed, i.e_kept)).map_err(|e| e.to_string());
            let y = tokened.collapse_edge_checked(token).map(|i| (i.v_kept, i.v_removed, i.is_boundary, i.e_collapsed, i.t_removed, i.e_removed, i.e_kept)).map_err(|e| e.to_string());
            assert_eq!(x, y, "seed {seed} step {step}: collapse");
            if plain.triangle_count() < 40 {
                break;
            }
        }
        assert!(stale_seen > 10);
        plain.check_validity().unwrap();
        tokened.assert_same_content(&plain, &format!("seed {seed}"));
        // A token is good on a copy that still has the same topology.
        let e = plain.edge_ids().find(|&e| {
            let [a, b] = plain.edge_vertices(e);
            plain.can_collapse(a, b).is_ok()
        });
        if let Some(e) = e {
            let [a, b] = plain.edge_vertices(e);
            let token = plain.check_collapse(a, b).unwrap();
            let mut copy = plain.clone();
            let x = copy.collapse_edge_checked(token).map(|i| i.t_removed).ok();
            assert_eq!(x, plain.collapse_edge(a, b).map(|i| i.t_removed).ok());
            copy.assert_same_content(&plain, "copy");
        }
    }
}
