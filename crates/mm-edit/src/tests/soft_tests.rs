//! Soft Transform: handle / band split, the four band solvers, fixed boundaries.

use super::*;
use crate::soft::*;

fn grid() -> (Mesh, Vec<u32>) {
    let m = primitives::plane_grid(6.0, 6.0, 30, 30);
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).truncate().length() < 0.8).collect();
    (m, sel)
}

fn apply(m: &Mesh, moves: &[(u32, DVec3)]) -> Mesh {
    let mut out = m.clone();
    for &(v, p) in moves {
        out.set_vertex(v, p);
    }
    out
}

#[test]
fn region_split_and_weights() {
    let (m, sel) = grid();
    let r = soft_region(&m, &sel, 1.0, false, false, true);
    let sel_verts: std::collections::HashSet<u32> = sel.iter().flat_map(|&t| m.triangle(t)).collect();
    assert_eq!(r.handle.len(), sel_verts.len());
    assert!(!r.band.is_empty());
    assert!(r.band.iter().all(|v| !sel_verts.contains(v)));
    assert!(r.band_weight.iter().all(|&w| w > 0.0 && w < 1.0));
    // Weights fall off with distance from the selection.
    let (near, far) = r.band.iter().zip(&r.band_weight).fold((0.0f64, 1.0f64), |(n, f), (&v, &w)| {
        let d = m.vertex(v).truncate().length();
        if d < 1.1 { (n.max(w), f) } else if d > 1.6 { (n, f.min(w)) } else { (n, f) }
    });
    assert!(near > far, "{near} > {far}");
}

#[test]
fn every_kind_lifts_the_handle_smoothly() {
    let (m, sel) = grid();
    let r = soft_region(&m, &sel, 1.2, false, false, true);
    let lift = |_: u32, p: DVec3| p + DVec3::Z;
    for kind in [SoftKind::Linear, SoftKind::Nonlinear, SoftKind::Flat, SoftKind::Stretchy] {
        for cont in [Continuity::Boundaries, Continuity::SoftPosition, Continuity::SoftTangent] {
            let out = apply(&m, &soft_deform(&m, &r, &lift, kind, cont));
            out.check_validity().unwrap();
            for v in m.vertex_ids() {
                let (z, d) = (out.vertex(v).z, m.vertex(v).truncate().length());
                assert!(z.is_finite() && (-0.05..=1.05).contains(&z), "{kind:?}/{cont:?}: z {z} at {d}");
                if r.handle.contains(&v) {
                    assert!((z - 1.0).abs() < 1e-12);
                }
                if d > 0.8 + 1.2 + 0.25 {
                    assert!(z.abs() < 1e-12, "{kind:?}: far vertex moved");
                }
            }
        }
    }
}

#[test]
fn nonlinear_keeps_the_band_rigid_under_rotation() {
    let (m, sel) = grid();
    let r = soft_region(&m, &sel, 1.2, false, false, true);
    let q = mm_core::DQuat::from_rotation_x(0.9);
    let turn = |_: u32, p: DVec3| q * p + DVec3::Z * 0.5;
    let band: std::collections::HashSet<u32> = r.band.iter().copied().collect();
    let distortion = |kind| {
        let out = apply(&m, &soft_deform(&m, &r, &turn, kind, Continuity::SoftPosition));
        let (mut sum, mut n) = (0.0, 0);
        for e in m.edge_ids() {
            let [a, b] = m.edge_vertices(e);
            if band.contains(&a) && band.contains(&b) {
                let l0 = m.vertex(a).distance(m.vertex(b));
                sum += (out.vertex(a).distance(out.vertex(b)) - l0).abs() / l0;
                n += 1;
            }
        }
        sum / n as f64
    };
    let (lin, arap) = (distortion(SoftKind::Linear), distortion(SoftKind::Nonlinear));
    assert!(arap < lin, "ARAP stretch {arap} < linear {lin}");
}

#[test]
fn fixed_boundary_pins_the_selection_border() {
    let (m, sel) = grid();
    let r = soft_region(&m, &sel, 0.4, true, false, true);
    let out = apply(&m, &soft_deform(&m, &r, &|_, p| p + DVec3::Z, SoftKind::Linear, Continuity::SoftPosition));
    let selected: std::collections::HashSet<u32> = sel.iter().copied().collect();
    for v in m.vertex_ids() {
        let inside = m.vertex_triangles(v).iter().all(|t| selected.contains(t));
        if !inside {
            assert_eq!(out.vertex(v), m.vertex(v), "border and outside stay");
        }
    }
    assert!(!r.handle.is_empty() && r.handle.iter().all(|&v| (out.vertex(v).z - 1.0).abs() < 1e-12));
}

#[test]
fn warp_reaches_handles_and_keeps_the_border() {
    let m = primitives::plane_grid(4.0, 4.0, 24, 24);
    let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).truncate().length() < 1.5).collect();
    let centre = m.vertex_ids().min_by(|&a, &b| m.vertex(a).length().total_cmp(&m.vertex(b).length())).unwrap();
    let target = m.vertex(centre) + DVec3::new(0.2, 0.0, 0.8);
    for rigid in [false, true] {
        let out = apply(&m, &warp(&m, &sel, &[(centre, target)], rigid));
        out.check_validity().unwrap();
        assert_eq!(out.vertex(centre), target);
        let selected: std::collections::HashSet<u32> = sel.iter().copied().collect();
        for v in m.vertex_ids() {
            let inside = m.vertex_triangles(v).iter().all(|t| selected.contains(t));
            if !inside {
                assert_eq!(out.vertex(v), m.vertex(v));
            } else {
                let z = out.vertex(v).z;
                assert!(z.is_finite() && z > -0.1 && z < 0.9, "rigid {rigid}: z {z}");
            }
        }
    }
    assert!(warp(&m, &sel, &[], true).is_empty());
}

