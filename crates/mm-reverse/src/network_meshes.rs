// SPDX-License-Identifier: AGPL-3.0-only
//! The synthetic scans the patch network's tests draw on: a grid of (u, v) mapped into space
//! (a sheet, a strip of a cylinder, a saddle) with noise along the normal and holes left
//! out, and a noisy sphere. And a network written out as nodes and curves through grips.
//!
//! Ported from ScanRuler (`plugins/reverse/tests/networkMeshes.ts`, and `icosphere` and
//! `gaussian` from `tests/helpers.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use std::collections::{BTreeMap, HashMap};

use glam::DVec3;

use crate::fit2d::Mulberry32;
use crate::network_feature::{NetworkCell, NetworkCurve, NetworkFeature, NetworkFitParams, NetworkNode};

/// A scan as typed arrays: x, y, z per vertex and three vertices a triangle.
#[derive(Debug, Clone)]
pub struct TestMesh {
    pub positions: Vec<f32>,
    pub indices: Vec<u32>,
}

/// Normal noise from the generator, as `networkMeshes.ts` draws it (a draw of zero held at
/// 1e-12).
pub fn gaussian(rand: &mut Mulberry32) -> f64 {
    let u = rand.next_f64().max(1e-12);
    (-2.0 * u.ln()).sqrt() * (2.0 * std::f64::consts::PI * rand.next_f64()).cos()
}

/// How a grid scan is made: `wrap` closes u round; `hole` leaves out the cells it says;
/// `noise` moves each vertex along the grid's own normal there.
#[derive(Default)]
pub struct GridOptions<'a> {
    pub wrap: bool,
    pub hole: Option<&'a dyn Fn(f64, f64) -> bool>,
    pub noise: f64,
    pub seed: Option<u32>,
}

/// A grid of nu × nv cells over [0, 1]² mapped by `at`, wound so its normals point the way
/// `at`'s du × dv does (see [`GridOptions`]).
pub fn grid_scan(nu: usize, nv: usize, at: &dyn Fn(f64, f64) -> DVec3, opts: &GridOptions<'_>) -> TestMesh {
    let cols = if opts.wrap { nu } else { nu + 1 };
    let mut rand = Mulberry32::new(opts.seed.unwrap_or(5));
    let mut positions: Vec<f32> = Vec::with_capacity((nv + 1) * cols * 3);
    for j in 0..=nv {
        for i in 0..cols {
            let u = i as f64 / nu as f64;
            let v = j as f64 / nv as f64;
            let mut p = at(u, v);
            if opts.noise != 0.0 {
                let e = 1e-4;
                let pu = at(u + e, v);
                let pv = at(u, v + e);
                let du = pu - p;
                let dv = pv - p;
                let n = DVec3::new(du.y * dv.z - du.z * dv.y, du.z * dv.x - du.x * dv.z, du.x * dv.y - du.y * dv.x);
                let l = n.length();
                let l = if l != 0.0 && !l.is_nan() { l } else { 1.0 };
                let g = gaussian(&mut rand) * opts.noise;
                p.x += (n.x / l) * g;
                p.y += (n.y / l) * g;
                p.z += (n.z / l) * g;
            }
            positions.extend_from_slice(&[p.x as f32, p.y as f32, p.z as f32]);
        }
    }
    let mut indices: Vec<u32> = Vec::new();
    let id = |i: usize, j: usize| (j * cols + if opts.wrap { i % nu } else { i }) as u32;
    for j in 0..nv {
        for i in 0..nu {
            if let Some(hole) = opts.hole {
                if hole((i as f64 + 0.5) / nu as f64, (j as f64 + 0.5) / nv as f64) {
                    continue;
                }
            }
            indices.extend_from_slice(&[id(i, j), id(i + 1, j), id(i + 1, j + 1), id(i, j), id(i + 1, j + 1), id(i, j + 1)]);
        }
    }
    TestMesh { positions, indices }
}

/// An icosahedron subdivided `subdivisions` times onto a sphere of `radius`, each vertex
/// moved out by normal noise of `noise` (ScanRuler's tests' `icosphere`; its noise drawn by
/// [`Mulberry32::gaussian`] from `seed`).
pub fn icosphere(subdivisions: usize, radius: f64, noise: f64, seed: u32) -> TestMesh {
    let t = (1.0 + 5.0f64.sqrt()) / 2.0;
    let normalize = |v: [f64; 3]| {
        let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / l, v[1] / l, v[2] / l]
    };
    let mut verts: Vec<[f64; 3]> = [
        [-1.0, t, 0.0],
        [1.0, t, 0.0],
        [-1.0, -t, 0.0],
        [1.0, -t, 0.0],
        [0.0, -1.0, t],
        [0.0, 1.0, t],
        [0.0, -1.0, -t],
        [0.0, 1.0, -t],
        [t, 0.0, -1.0],
        [t, 0.0, 1.0],
        [-t, 0.0, -1.0],
        [-t, 0.0, 1.0],
    ]
    .into_iter()
    .map(normalize)
    .collect();
    let mut faces: Vec<[usize; 3]> = vec![
        [0, 11, 5],
        [0, 5, 1],
        [0, 1, 7],
        [0, 7, 10],
        [0, 10, 11],
        [1, 5, 9],
        [5, 11, 4],
        [11, 10, 2],
        [10, 7, 6],
        [7, 1, 8],
        [3, 9, 4],
        [3, 4, 2],
        [3, 2, 6],
        [3, 6, 8],
        [3, 8, 9],
        [4, 9, 5],
        [2, 4, 11],
        [6, 2, 10],
        [8, 6, 7],
        [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut cache: HashMap<(usize, usize), usize> = HashMap::new();
        let mut midpoint = |a: usize, b: usize, verts: &mut Vec<[f64; 3]>| -> usize {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&hit) = cache.get(&key) {
                return hit;
            }
            let m = normalize([(verts[a][0] + verts[b][0]) / 2.0, (verts[a][1] + verts[b][1]) / 2.0, (verts[a][2] + verts[b][2]) / 2.0]);
            verts.push(m);
            cache.insert(key, verts.len() - 1);
            verts.len() - 1
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for &[a, b, c] in &faces {
            let ab = midpoint(a, b, &mut verts);
            let bc = midpoint(b, c, &mut verts);
            let ca = midpoint(c, a, &mut verts);
            next.extend_from_slice(&[[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    let mut rand = Mulberry32::new(seed);
    let mut positions = Vec::with_capacity(verts.len() * 3);
    for v in &verts {
        let r = radius + if noise > 0.0 { rand.gaussian() * noise } else { 0.0 };
        positions.extend_from_slice(&[(v[0] * r) as f32, (v[1] * r) as f32, (v[2] * r) as f32]);
    }
    let indices = faces.iter().flat_map(|f| f.iter().map(|&v| v as u32)).collect();
    TestMesh { positions, indices }
}

/// A sphere of radius r about the origin with noise, closed.
pub fn sphere_scan(r: f64, noise: f64, subdivisions: usize) -> TestMesh {
    icosphere(subdivisions, r, noise, 3)
}

/// A curve of a network written out: its nodes, its grips and whether it is a crease.
pub struct CurveSpec {
    pub from: u32,
    pub to: u32,
    pub via: Vec<DVec3>,
    pub crease: bool,
}

/// A curve from one node to another, through no grips.
pub fn curve(from: u32, to: u32) -> CurveSpec {
    CurveSpec { from, to, via: Vec::new(), crease: false }
}

/// A curve through grips.
pub fn curve_via(from: u32, to: u32, via: Vec<DVec3>) -> CurveSpec {
    CurveSpec { from, to, via, crease: false }
}

/// A network written out: nodes by id at their points (in the order of their ids, as a
/// JavaScript object's integer keys come), curves as from, to, grips and a crease flag, seeds
/// for the cells.
pub fn network(nodes: &[(u32, DVec3)], curves: &[CurveSpec], seeds: &[DVec3]) -> NetworkFeature {
    let mut ordered: Vec<(u32, DVec3)> = nodes.to_vec();
    ordered.sort_by_key(|n| n.0);
    NetworkFeature {
        nodes: ordered.into_iter().map(|(id, at)| NetworkNode { id, at, on: None, corner: None }).collect(),
        curves: curves
            .iter()
            .enumerate()
            .map(|(i, c)| NetworkCurve { id: i as u32 + 1, from: c.from, to: c.to, via: c.via.clone(), crease: if c.crease { Some(true) } else { None }, along: None, side: None })
            .collect(),
        cells: seeds.iter().map(|&seed| NetworkCell { seed, centre: None, corners: None }).collect(),
        spans: BTreeMap::new(),
        fit: NetworkFitParams { smoothness: 0.001, robust: None, tolerance: None },
    }
}

/// A rows × cols grid network over a surface given by `at` on [0, 1]²: its nodes on the
/// grid's crossings, a curve along each line between two, with `grips` grips evenly along it,
/// and a seed in every cell; within `bounds` (u0, u1, v0, v1) of the square.
pub fn grid_network(rows: usize, cols: usize, at: &dyn Fn(f64, f64) -> DVec3, grips: usize, bounds: [f64; 4]) -> NetworkFeature {
    let [u0, u1, v0, v1] = bounds;
    let uu = |i: usize| u0 + ((u1 - u0) * i as f64) / cols as f64;
    let vv = |j: usize| v0 + ((v1 - v0) * j as f64) / rows as f64;
    let nid = |i: usize, j: usize| (j * (cols + 1) + i + 1) as u32;
    let mut nodes: Vec<(u32, DVec3)> = Vec::new();
    for j in 0..=rows {
        for i in 0..=cols {
            nodes.push((nid(i, j), at(uu(i), vv(j))));
        }
    }
    let via = |a: [f64; 2], b: [f64; 2]| -> Vec<DVec3> {
        (0..grips).map(|k| at(a[0] + ((b[0] - a[0]) * (k + 1) as f64) / (grips + 1) as f64, a[1] + ((b[1] - a[1]) * (k + 1) as f64) / (grips + 1) as f64)).collect()
    };
    let mut curves: Vec<CurveSpec> = Vec::new();
    for j in 0..=rows {
        for i in 0..cols {
            curves.push(curve_via(nid(i, j), nid(i + 1, j), via([uu(i), vv(j)], [uu(i + 1), vv(j)])));
        }
    }
    for j in 0..rows {
        for i in 0..=cols {
            curves.push(curve_via(nid(i, j), nid(i, j + 1), via([uu(i), vv(j)], [uu(i), vv(j + 1)])));
        }
    }
    let mut seeds = Vec::new();
    for j in 0..rows {
        for i in 0..cols {
            seeds.push(at((uu(i) + uu(i + 1)) / 2.0, (vv(j) + vv(j + 1)) / 2.0));
        }
    }
    network(&nodes, &curves, &seeds)
}
