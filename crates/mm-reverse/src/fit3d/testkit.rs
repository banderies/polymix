// SPDX-License-Identifier: AGPL-3.0-only
//! Synthetic scans for the fit tests: points on a sphere, a cylinder, a cone or a plane with
//! Gaussian noise, an icosphere, index lists, and the closed meshes the seeded fits are
//! tried on (a cylinder, a frustum and a box as triangle soups) with the graphs built from
//! them. Ported from ScanRuler (`tests/helpers.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.
//!
//! ScanRuler keeps positions in a `Float32Array`, so every coordinate a helper writes is
//! rounded to single precision here too: the fits then see the very numbers they see there.

use glam::DVec3;

use crate::fit2d::Mulberry32;
use crate::mesh_graph::{build_mesh_graph, MeshGraph, ParsedMesh};

/// A coordinate as a `Float32Array` stores it.
pub fn f32r(x: f64) -> f64 {
    x as f32 as f64
}

/// A point as a `Float32Array` stores it.
pub fn p32(x: f64, y: f64, z: f64) -> DVec3 {
    DVec3::new(f32r(x), f32r(y), f32r(z))
}

/// A vector rounded to single precision.
pub fn v32(v: DVec3) -> DVec3 {
    p32(v.x, v.y, v.z)
}

/// `(deg * Math.PI) / 180`, as the tests write it.
pub fn rad(deg: f64) -> f64 {
    (deg * std::f64::consts::PI) / 180.0
}

pub fn all_indices(n: usize) -> Vec<u32> {
    (0..n as u32).collect()
}

/// A vector scaled to unit length (`Math.hypot`).
pub fn unit(v: [f64; 3]) -> DVec3 {
    let l = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    DVec3::new(v[0] / l, v[1] / l, v[2] / l)
}

fn normalize(v: DVec3) -> DVec3 {
    let l = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    DVec3::new(v.x / l, v.y / l, v.z / l)
}

/// The helpers' frame across a direction: `u = d × helper`, `v = d × u`.
pub fn helper_basis(d: DVec3) -> (DVec3, DVec3) {
    let helper = if d.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
    let u = normalize(d.cross(helper));
    let v = normalize(d.cross(u));
    (u, v)
}

/// Random points on a sphere surface with radial Gaussian noise. When `cap_angle` (radians,
/// half-angle around +Z) is given, only that cap is sampled — mimicking a partially visible
/// sphere.
pub fn sample_sphere(n: usize, center: [f64; 3], radius: f64, noise: f64, seed: u32, cap_angle: Option<f64>) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(seed);
    let cos_cap = cap_angle.map_or(-1.0, f64::cos);
    (0..n)
        .map(|_| {
            let z = cos_cap + rand.next_f64() * (1.0 - cos_cap);
            let phi = rand.next_f64() * 2.0 * std::f64::consts::PI;
            let s = (1.0 - z * z).max(0.0).sqrt();
            let r = radius + rand.gaussian() * noise;
            p32(center[0] + r * s * phi.cos(), center[1] + r * s * phi.sin(), center[2] + r * z)
        })
        .collect()
}

/// Random points on a cylinder wall about the given axis point and direction, with radial
/// Gaussian noise. `arc` (radians) limits how far around the axis the points wrap — a
/// partially visible cylinder.
#[allow(clippy::too_many_arguments)]
pub fn sample_cylinder(n: usize, point: [f64; 3], axis: [f64; 3], radius: f64, length: f64, noise: f64, seed: u32, arc: f64) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(seed);
    let d = normalize(DVec3::from_array(axis));
    let (u, v) = helper_basis(d);
    let p = DVec3::from_array(point);
    (0..n)
        .map(|_| {
            let ang = (rand.next_f64() - 0.5) * arc;
            let t = (rand.next_f64() - 0.5) * length;
            let r = radius + rand.gaussian() * noise;
            let cu = ang.cos() * r;
            let cv = ang.sin() * r;
            p32(
                p.x + cu * u.x + cv * v.x + t * d.x,
                p.y + cu * u.y + cv * v.y + t * d.y,
                p.z + cu * u.z + cv * v.z + t * d.z,
            )
        })
        .collect()
}

/// Random points on a cone wall about the given axis point and direction, with Gaussian
/// noise along the exact surface normal — so the expected orthogonal residual of a perfect
/// fit is the noise itself. `radius` is the surface radius at `point`; the radius grows
/// along +axis by tan(half_angle). `arc` (radians) limits how far around the axis the points
/// wrap.
#[allow(clippy::too_many_arguments)]
pub fn sample_cone(n: usize, point: [f64; 3], axis: [f64; 3], radius: f64, half_angle: f64, length: f64, noise: f64, seed: u32, arc: f64) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(seed);
    let d = normalize(DVec3::from_array(axis));
    let (u, v) = helper_basis(d);
    let p = DVec3::from_array(point);
    let (cos, sin, tan) = (half_angle.cos(), half_angle.sin(), half_angle.tan());
    (0..n)
        .map(|_| {
            let ang = (rand.next_f64() - 0.5) * arc;
            let t = (rand.next_f64() - 0.5) * length;
            let r = radius + t * tan;
            let wu = ang.cos();
            let wv = ang.sin();
            let h = rand.gaussian() * noise;
            // Surface point plus offset along the outward normal cos·ŵ − sin·â.
            let cu = wu * (r + h * cos);
            let cv = wv * (r + h * cos);
            let ct = t - h * sin;
            p32(
                p.x + cu * u.x + cv * v.x + ct * d.x,
                p.y + cu * u.y + cv * v.y + ct * d.y,
                p.z + cu * u.z + cv * v.z + ct * d.z,
            )
        })
        .collect()
}

/// Random points on a plane patch with Gaussian noise along the normal.
pub fn sample_plane(n: usize, point: [f64; 3], normal: [f64; 3], size: f64, noise: f64, seed: u32) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(seed);
    let nrm = normalize(DVec3::from_array(normal));
    let (u, v) = helper_basis(nrm);
    let p = DVec3::from_array(point);
    (0..n)
        .map(|_| {
            let a = (rand.next_f64() - 0.5) * size;
            let b = (rand.next_f64() - 0.5) * size;
            let h = rand.gaussian() * noise;
            p32(
                p.x + a * u.x + b * v.x + h * nrm.x,
                p.y + a * u.y + b * v.y + h * nrm.y,
                p.z + a * u.z + b * v.z + h * nrm.z,
            )
        })
        .collect()
}

/// Subdivided icosahedron on a sphere of the given radius, with optional radial noise — a
/// small synthetic scan of a ball. Positions and triangle indices.
pub fn icosphere(subdivisions: usize, radius: f64, noise: f64, seed: u32) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let raw = [
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
    ];
    let mut verts: Vec<DVec3> = raw.iter().map(|v| normalize(DVec3::from_array(*v))).collect();
    let mut faces: Vec<[u32; 3]> = vec![
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
        let mut mid_cache = std::collections::HashMap::<(u32, u32), u32>::new();
        let mut midpoint = |verts: &mut Vec<DVec3>, a: u32, b: u32| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            if let Some(&hit) = mid_cache.get(&key) {
                return hit;
            }
            let (va, vb) = (verts[a as usize], verts[b as usize]);
            verts.push(normalize(DVec3::new((va.x + vb.x) / 2.0, (va.y + vb.y) / 2.0, (va.z + vb.z) / 2.0)));
            let id = verts.len() as u32 - 1;
            mid_cache.insert(key, id);
            id
        };
        let mut next = Vec::with_capacity(faces.len() * 4);
        for [a, b, c] in faces {
            let ab = midpoint(&mut verts, a, b);
            let bc = midpoint(&mut verts, b, c);
            let ca = midpoint(&mut verts, c, a);
            next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
        }
        faces = next;
    }
    let mut rand = Mulberry32::new(seed);
    let positions = verts
        .iter()
        .map(|v| {
            let r = radius + if noise > 0.0 { rand.gaussian() * noise } else { 0.0 };
            p32(v.x * r, v.y * r, v.z * r)
        })
        .collect();
    (positions, faces)
}

type P3 = [f64; 3];

/// Emit a quad as two triangles into a soup. Corner coordinates are shared between
/// neighbouring quads bit for bit, so welding rebuilds the topology.
pub fn quad(tris: &mut Vec<DVec3>, a: P3, b: P3, c: P3, d: P3) {
    for p in [a, b, c, a, c, d] {
        tris.push(p32(p[0], p[1], p[2]));
    }
}

/// The wall grid of a surface of revolution about +Z with flat end caps, as a soup: rings of
/// radius `r_at(j)` at `axial + 1` heights, radial noise off the rims. Shared by
/// `cylinder_mesh` and `cone_mesh`.
fn closed_revolution(r_at: impl Fn(usize) -> f64, length: f64, radial: usize, axial: usize, noise: f64, seed: u32) -> (Vec<DVec3>, f64) {
    let mut rand = Mulberry32::new(seed);
    let z0 = -length / 2.0;
    // Precompute the vertex grid: every triangle then references these exact values rather
    // than re-rolling the noise per triangle.
    let mut grid: Vec<Vec<P3>> = Vec::with_capacity(axial + 1);
    for j in 0..=axial {
        let z = z0 + (length * j as f64) / axial as f64;
        let rz = r_at(j);
        let row = (0..radial)
            .map(|k| {
                let a = (k as f64 / radial as f64) * 2.0 * std::f64::consts::PI;
                let rim = j == 0 || j == axial;
                let r = rz + if noise > 0.0 && !rim { rand.gaussian() * noise } else { 0.0 };
                [a.cos() * r, a.sin() * r, z]
            })
            .collect();
        grid.push(row);
    }
    let mut tris = Vec::new();
    for j in 0..axial {
        for k in 0..radial {
            let k2 = (k + 1) % radial;
            quad(&mut tris, grid[j][k], grid[j][k2], grid[j + 1][k2], grid[j + 1][k]);
        }
    }
    let wall_triangles = tris.len() / 3;
    for j in [0, axial] {
        let z = grid[j][0][2];
        let hub = [0.0, 0.0, z];
        for k in 0..radial {
            let k2 = (k + 1) % radial;
            let (a, b) = if j == 0 { (grid[j][k2], grid[j][k]) } else { (grid[j][k], grid[j][k2]) };
            for p in [hub, a, b] {
                tris.push(p32(p[0], p[1], p[2]));
            }
        }
    }
    let wall_fraction = wall_triangles as f64 / (tris.len() / 3) as f64;
    (tris, wall_fraction)
}

/// Triangle soup of a closed cylinder about +Z — wall plus flat end caps, so a fit of the
/// wall has neighbouring surfaces it must not leak onto. Wall vertices carry radial noise,
/// except on the rims, which stay exact so they weld to the caps. Returns the soup and the
/// share of it that is wall.
pub fn cylinder_mesh(radius: f64, length: f64, radial: usize, axial: usize, noise: f64, seed: u32) -> (Vec<DVec3>, f64) {
    closed_revolution(|_| radius, length, radial, axial, noise, seed)
}

/// Triangle soup of a closed frustum about +Z — the cone tool's counterpart of
/// `cylinder_mesh`: tapered wall from `radius1` at −length/2 up to `radius2` at +length/2,
/// plus flat end caps, with the rims kept exact so they weld.
pub fn cone_mesh(radius1: f64, radius2: f64, length: f64, radial: usize, axial: usize, noise: f64, seed: u32) -> (Vec<DVec3>, f64) {
    closed_revolution(|j| radius1 + ((radius2 - radius1) * j as f64) / axial as f64, length, radial, axial, noise, seed)
}

/// Triangle soup of an axis-aligned box, each face a grid so region growing has somewhere to
/// grow. Noise is applied along each face's own normal, but not on the face borders — those
/// stay exact so the six faces weld into one closed mesh and a plane fit has edges it must
/// stop at.
pub fn box_mesh(size: f64, grid: usize, noise: f64, seed: u32) -> Vec<DVec3> {
    let mut rand = Mulberry32::new(seed);
    let h = size / 2.0;
    let mut tris = Vec::new();
    // Each face as (origin, edge u, edge v), wound outward — the face normal is
    // cross(u, v), so the order of the two edges is what points a face out of the solid.
    let faces: [(P3, P3, P3); 6] = [
        ([-h, -h, h], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        ([-h, -h, -h], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]),
        ([h, -h, -h], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        ([-h, -h, -h], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]),
        ([-h, h, -h], [0.0, 0.0, 1.0], [1.0, 0.0, 0.0]),
        ([-h, -h, -h], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    for (o, eu, ev) in faces {
        let nrm = normalize(DVec3::from_array(eu).cross(DVec3::from_array(ev))).to_array();
        let mut verts: Vec<Vec<P3>> = Vec::with_capacity(grid + 1);
        for i in 0..=grid {
            let mut row = Vec::with_capacity(grid + 1);
            for j in 0..=grid {
                let a = (i as f64 / grid as f64) * size;
                let b = (j as f64 / grid as f64) * size;
                let border = i == 0 || j == 0 || i == grid || j == grid;
                let d = if noise > 0.0 && !border { rand.gaussian() * noise } else { 0.0 };
                row.push([0, 1, 2].map(|k| o[k] + a * eu[k] + b * ev[k] + d * nrm[k]));
            }
            verts.push(row);
        }
        for i in 0..grid {
            for j in 0..grid {
                quad(&mut tris, verts[i][j], verts[i + 1][j], verts[i + 1][j + 1], verts[i][j + 1]);
            }
        }
    }
    tris
}

/// The graph of a soup, as `buildMeshGraph({ kind: 'soup', positions })` makes it.
pub fn graph_soup(soup: Vec<DVec3>) -> MeshGraph {
    build_mesh_graph(ParsedMesh::soup(soup)).expect("a test soup builds")
}

/// The graph of positions and triangles, as `buildMeshGraph({ kind: 'indexed', ... })`.
pub fn graph_indexed((positions, faces): (Vec<DVec3>, Vec<[u32; 3]>)) -> MeshGraph {
    let indices = faces.iter().flatten().copied().collect();
    build_mesh_graph(ParsedMesh::indexed(positions, indices)).expect("a test mesh builds")
}

/// The vertex nearest a point — stands in for the user's click.
pub fn seed_near(g: &MeshGraph, p: [f64; 3]) -> u32 {
    let mut best = 0u32;
    let mut best_d = f64::INFINITY;
    for (v, q) in g.positions.iter().enumerate() {
        let d = (q.x - p[0]).powi(2) + (q.y - p[1]).powi(2) + (q.z - p[2]).powi(2);
        if d < best_d {
            best_d = d;
            best = v as u32;
        }
    }
    best
}
