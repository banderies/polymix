//! Exact mesh booleans: volumes obey inclusion–exclusion exactly, results are closed and
//! manifold, and untouched input triangles survive unchanged.

use super::*;
use crate::*;

fn moved(mut m: Mesh, d: DVec3) -> Mesh {
    m.transform(|p| p + d);
    m
}

fn run(a: &Mesh, b: &Mesh, op: BooleanOp) -> Mesh {
    let r = mesh_boolean(a, b, op).unwrap_or_else(|e| panic!("{op:?}: {e}"));
    assert_valid(&r.mesh);
    assert!(r.mesh.is_closed(), "{op:?}: open result");
    r.mesh
}

/// Check the three operations against inclusion–exclusion; returns (union, intersection).
fn check_volumes(a: &Mesh, b: &Mesh, tol: f64) -> (Mesh, Mesh) {
    let (va, vb) = (volume(a), volume(b));
    let u = run(a, b, BooleanOp::Union);
    let i = run(a, b, BooleanOp::Intersection);
    let d = run(a, b, BooleanOp::Difference);
    let (vu, vi, vd) = (volume(&u), volume(&i), volume(&d));
    assert!((vu + vi - va - vb).abs() < tol, "union {vu} + intersection {vi} != {va} + {vb}");
    assert!((vd - (va - vi)).abs() < tol, "difference {vd} != {va} - {vi}");
    (u, i)
}

#[test]
fn overlapping_spheres() {
    let a = primitives::icosphere(1.0, 3);
    let c = DVec3::new(1.0, 0.13, 0.07);
    let b = moved(primitives::icosphere(1.0, 3), c);
    let (u, i) = check_volumes(&a, &b, 1e-9);
    // Close to the analytic lens volume π(4r + d)(2r − d)²/12 for r = 1.
    let d = c.length();
    let lens = PI * (4.0 + d) * (2.0 - d).powi(2) / 12.0;
    assert!((volume(&i) - lens).abs() < 0.05 * lens, "{} vs {lens}", volume(&i));
    // Away from the seam the union keeps the input vertices exactly.
    let outside: Vec<DVec3> = a.vertex_ids().map(|v| a.vertex(v)).filter(|p| p.distance(c) > 1.05).collect();
    let have: std::collections::HashSet<[u64; 3]> = u.vertex_ids().map(|v| u.vertex(v).to_array().map(f64::to_bits)).collect();
    assert!(outside.iter().all(|p| have.contains(&p.to_array().map(f64::to_bits))));
}

#[test]
fn coplanar_cubes() {
    // Shared face planes on four sides force the degenerate-configuration retry.
    let a = primitives::cube(1.0, 3);
    let b = moved(primitives::cube(1.0, 3), DVec3::new(0.5, 0.0, 0.0));
    for (op, want) in [(BooleanOp::Union, 1.5), (BooleanOp::Intersection, 0.5), (BooleanOp::Difference, 0.5)] {
        let v = volume(&run(&a, &b, op));
        assert!((v - want).abs() < 1e-5, "{op:?}: {v}");
    }
}

#[test]
fn curve_loop_inside_one_triangle() {
    // A coarse box (two triangles per side) pierced by a small sphere: the intersection curve
    // is a closed loop inside a single box triangle.
    let a = primitives::cube(4.0, 1);
    let b = moved(primitives::icosphere(0.3, 3), DVec3::new(0.7, 0.4, 2.0));
    check_volumes(&a, &b, 1e-9);
}

#[test]
fn disjoint_and_nested() {
    let a = primitives::icosphere(1.0, 2);
    let far = moved(primitives::icosphere(0.5, 2), DVec3::new(5.0, 0.0, 0.0));
    let u = run(&a, &far, BooleanOp::Union);
    assert_eq!(u.triangle_count(), a.triangle_count() + far.triangle_count());
    assert_eq!(run(&a, &far, BooleanOp::Intersection).triangle_count(), 0);
    assert_eq!(run(&a, &far, BooleanOp::Difference).triangle_count(), a.triangle_count());
    let inner = primitives::icosphere(0.5, 2);
    let cavity = run(&a, &inner, BooleanOp::Difference);
    assert!((volume(&cavity) - (volume(&a) - volume(&inner))).abs() < 1e-9);
    assert_eq!(run(&a, &inner, BooleanOp::Union).triangle_count(), a.triangle_count());
}

#[test]
fn groups_and_open_input() {
    let a = primitives::icosphere(1.0, 2);
    let b = moved(primitives::icosphere(1.0, 2), DVec3::new(1.0, 0.1, 0.0));
    let r = mesh_boolean(&a, &b, BooleanOp::Union).unwrap();
    let groups: std::collections::BTreeSet<i32> = r.mesh.triangle_ids().map(|t| r.mesh.tri_group(t)).collect();
    assert_eq!(groups.len(), 2, "each operand keeps its own group");
    assert!(!r.seam_vertices.is_empty());
    let mut open = primitives::icosphere(1.0, 2);
    let t = open.triangle_ids().next().unwrap();
    open.remove_triangles(vec![t]);
    assert!(mesh_boolean(&open, &b, BooleanOp::Union).is_err());
}

#[test]
fn many_random_configurations() {
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    let shapes = [primitives::icosphere(1.0, 2), primitives::cube(1.6, 2), primitives::icosphere(0.6, 3)];
    for k in 0..60 {
        let a = &shapes[k % 3];
        let q = mm_core::DQuat::from_euler(glam::EulerRot::XYZ, rnd() * 6.3, rnd() * 6.3, rnd() * 6.3);
        let d = DVec3::new(rnd() - 0.5, rnd() - 0.5, rnd() - 0.5) * 2.0;
        let mut b = shapes[(k / 3) % 3].clone();
        b.transform(|p| q * p + d);
        let (va, vb) = (volume(a), volume(&b));
        let u = run(a, &b, BooleanOp::Union);
        let i = run(a, &b, BooleanOp::Intersection);
        let tol = 1e-9 * (va + vb);
        assert!((volume(&u) + volume(&i) - va - vb).abs() < tol, "case {k}: {} + {} vs {va} + {vb}", volume(&u), volume(&i));
    }
}

#[test]
fn identical_and_flush_operands() {
    // Every face coplanar with its twin.
    let a = primitives::cube(1.0, 2);
    let u = run(&a, &a.clone(), BooleanOp::Union);
    assert!((volume(&u) - 1.0).abs() < 1e-5, "{}", volume(&u));
    // A capped cylinder standing flush on the box's top face.
    let cyl = moved(primitives::cylinder(0.3, 1.0, 24, 4, true), DVec3::new(0.1, 0.0, 0.5));
    let box_ = primitives::cube(1.0, 2);
    check_volumes(&box_, &cyl, 1e-6);
}
