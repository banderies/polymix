//! Tests for mm-edit. Every test checks `check_validity()` and consistent orientation.

mod boolean_tests;
mod bridge_tests;
mod cut_tests;
mod extrude_tests;
mod flow_tests;
mod local_cut_tests;
mod misc_tests;
mod perf;
mod soft_tests;
mod target_tests;

use mm_core::topology::{boundary_loops, connected_components};
use mm_core::{measure, primitives, DVec3, Mesh};
use std::f64::consts::{PI, TAU};

pub(crate) fn assert_valid(m: &Mesh) {
    m.check_validity().unwrap();
    assert!(m.is_consistently_oriented(), "mesh is not consistently oriented");
}

pub(crate) fn volume(m: &Mesh) -> f64 {
    measure::volume(m)
}

/// Torus around the Z axis (major radius `big`, minor radius `small`).
pub(crate) fn torus(big: f64, small: f64, nu: usize, nv: usize) -> Mesh {
    let mut pos = Vec::new();
    for i in 0..nu {
        let u = TAU * i as f64 / nu as f64;
        for j in 0..nv {
            let v = TAU * j as f64 / nv as f64;
            let r = big + small * v.cos();
            pos.push(DVec3::new(r * u.cos(), r * u.sin(), small * v.sin()));
        }
    }
    let idx = |i: usize, j: usize| ((i % nu) * nv + (j % nv)) as u32;
    let mut tris = Vec::new();
    for i in 0..nu {
        for j in 0..nv {
            let (a, b, c, d) = (idx(i, j), idx(i + 1, j), idx(i + 1, j + 1), idx(i, j + 1));
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    let (mut m, _) = Mesh::from_triangles(&pos, &tris, None);
    if volume(&m) < 0.0 {
        m.reverse_orientation();
    }
    m
}

/// Open U-shaped tube: up the leg at x=0, a half circle over, down the leg at x=`span`.
pub(crate) fn u_tube(span: f64, height: f64, radius: f64, slices: usize) -> Mesh {
    let mut path = Vec::new();
    let legn = 12;
    for i in 0..=legn {
        path.push(DVec3::new(0.0, 0.0, height * i as f64 / legn as f64));
    }
    let arcn = 24;
    let rr = span * 0.5;
    for i in 1..arcn {
        let a = PI * i as f64 / arcn as f64;
        path.push(DVec3::new(rr - rr * a.cos(), 0.0, height + rr * a.sin()));
    }
    for i in 0..=legn {
        path.push(DVec3::new(span, 0.0, height * (legn - i) as f64 / legn as f64));
    }
    let radii = vec![radius; path.len()];
    primitives::tube_along(&path, &radii, slices)
}

pub(crate) fn n_loops(m: &Mesh) -> usize {
    boundary_loops(m).len()
}

pub(crate) fn n_components(m: &Mesh) -> usize {
    connected_components(m).len()
}
