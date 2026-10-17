//! Procedural mesh generators (used by tools and heavily by tests).

use crate::mesh::Mesh;
use glam::DVec3;
use std::f64::consts::PI;

/// Build a mesh from positions and triangles that are known to be manifold.
fn build(pos: Vec<DVec3>, tris: Vec<[u32; 3]>) -> Mesh {
    let (m, rep) = Mesh::from_triangles(&pos, &tris, None);
    debug_assert_eq!(rep.nonmanifold_split, 0, "generator produced non-manifold output");
    m
}

/// Axis-aligned box centred at the origin, each face split into `n x n` quads.
pub fn cube(size: f64, n: usize) -> Mesh {
    let n = n.max(1);
    let h = size * 0.5;
    let mut pos = Vec::new();
    let mut tris = Vec::new();
    let mut index = std::collections::HashMap::new();
    let mut vid = |p: DVec3, pos: &mut Vec<DVec3>| -> u32 {
        // Quantize to merge shared face-edge vertices.
        let key = ((p.x * 1e6).round() as i64, (p.y * 1e6).round() as i64, (p.z * 1e6).round() as i64);
        *index.entry(key).or_insert_with(|| {
            pos.push(p);
            (pos.len() - 1) as u32
        })
    };
    // (normal axis, sign)
    for axis in 0..3 {
        for &s in &[-1.0f64, 1.0] {
            let u_axis = (axis + 1) % 3;
            let v_axis = (axis + 2) % 3;
            let mut grid = vec![vec![0u32; n + 1]; n + 1];
            for (i, row) in grid.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    let mut p = DVec3::ZERO;
                    p[axis] = s * h;
                    p[u_axis] = -h + size * i as f64 / n as f64;
                    p[v_axis] = -h + size * j as f64 / n as f64;
                    *cell = vid(p, &mut pos);
                }
            }
            for i in 0..n {
                for j in 0..n {
                    let a = grid[i][j];
                    let b = grid[i + 1][j];
                    let c = grid[i + 1][j + 1];
                    let d = grid[i][j + 1];
                    // u x v = +axis, so (a,b,c) is CCW seen from +axis.
                    if s > 0.0 {
                        tris.push([a, b, c]);
                        tris.push([a, c, d]);
                    } else {
                        tris.push([a, c, b]);
                        tris.push([a, d, c]);
                    }
                }
            }
        }
    }
    build(pos, tris)
}

/// UV sphere with `slices` around and `stacks` from pole to pole.
pub fn uv_sphere(radius: f64, slices: usize, stacks: usize) -> Mesh {
    let slices = slices.max(3);
    let stacks = stacks.max(2);
    let mut pos = vec![DVec3::new(0.0, 0.0, radius)];
    for i in 1..stacks {
        let phi = PI * i as f64 / stacks as f64;
        for j in 0..slices {
            let th = 2.0 * PI * j as f64 / slices as f64;
            pos.push(DVec3::new(phi.sin() * th.cos(), phi.sin() * th.sin(), phi.cos()) * radius);
        }
    }
    pos.push(DVec3::new(0.0, 0.0, -radius));
    let south = (pos.len() - 1) as u32;
    let ring = |i: usize, j: usize| (1 + (i - 1) * slices + (j % slices)) as u32;
    let mut tris = Vec::new();
    for j in 0..slices {
        tris.push([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..stacks - 1 {
        for j in 0..slices {
            let a = ring(i, j);
            let b = ring(i + 1, j);
            let c = ring(i + 1, j + 1);
            let d = ring(i, j + 1);
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    for j in 0..slices {
        tris.push([south, ring(stacks - 1, j + 1), ring(stacks - 1, j)]);
    }
    build(pos, tris)
}

/// Icosphere: subdivided icosahedron projected onto a sphere (near-uniform triangles).
pub fn icosphere(radius: f64, subdivisions: usize) -> Mesh {
    let t = (1.0 + 5f64.sqrt()) / 2.0;
    let mut pos: Vec<DVec3> = [
        [-1.0, t, 0.0], [1.0, t, 0.0], [-1.0, -t, 0.0], [1.0, -t, 0.0],
        [0.0, -1.0, t], [0.0, 1.0, t], [0.0, -1.0, -t], [0.0, 1.0, -t],
        [t, 0.0, -1.0], [t, 0.0, 1.0], [-t, 0.0, -1.0], [-t, 0.0, 1.0],
    ]
    .iter()
    .map(|p| DVec3::from_array(*p).normalize())
    .collect();
    let mut tris: Vec<[u32; 3]> = vec![
        [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
        [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
        [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
        [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
    ];
    for _ in 0..subdivisions {
        let mut mid = std::collections::HashMap::new();
        let mut get = |a: u32, b: u32, pos: &mut Vec<DVec3>| -> u32 {
            let key = if a < b { (a, b) } else { (b, a) };
            *mid.entry(key).or_insert_with(|| {
                pos.push(((pos[a as usize] + pos[b as usize]) * 0.5).normalize());
                (pos.len() - 1) as u32
            })
        };
        let mut nt = Vec::with_capacity(tris.len() * 4);
        for &[a, b, c] in &tris {
            let ab = get(a, b, &mut pos);
            let bc = get(b, c, &mut pos);
            let ca = get(c, a, &mut pos);
            nt.push([a, ab, ca]);
            nt.push([b, bc, ab]);
            nt.push([c, ca, bc]);
            nt.push([ab, bc, ca]);
        }
        tris = nt;
    }
    for p in pos.iter_mut() {
        *p *= radius;
    }
    build(pos, tris)
}

/// Open cylinder (tube) along +Z from z=0 to z=height, optionally capped.
pub fn cylinder(radius: f64, height: f64, slices: usize, stacks: usize, capped: bool) -> Mesh {
    let slices = slices.max(3);
    let stacks = stacks.max(1);
    let mut pos = Vec::new();
    for i in 0..=stacks {
        let z = height * i as f64 / stacks as f64;
        for j in 0..slices {
            let th = 2.0 * PI * j as f64 / slices as f64;
            pos.push(DVec3::new(radius * th.cos(), radius * th.sin(), z));
        }
    }
    let idx = |i: usize, j: usize| (i * slices + (j % slices)) as u32;
    let mut tris = Vec::new();
    for i in 0..stacks {
        for j in 0..slices {
            let a = idx(i, j);
            let b = idx(i, j + 1);
            let c = idx(i + 1, j + 1);
            let d = idx(i + 1, j);
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    if capped {
        let bot = pos.len() as u32;
        pos.push(DVec3::ZERO);
        let top = pos.len() as u32;
        pos.push(DVec3::new(0.0, 0.0, height));
        for j in 0..slices {
            tris.push([bot, idx(0, j + 1), idx(0, j)]);
            tris.push([top, idx(stacks, j), idx(stacks, j + 1)]);
        }
    }
    build(pos, tris)
}

/// Flat rectangular grid in the XY plane centred at the origin (an open surface).
pub fn plane_grid(width: f64, height: f64, nx: usize, ny: usize) -> Mesh {
    let nx = nx.max(1);
    let ny = ny.max(1);
    let mut pos = Vec::new();
    for j in 0..=ny {
        for i in 0..=nx {
            pos.push(DVec3::new(
                -width / 2.0 + width * i as f64 / nx as f64,
                -height / 2.0 + height * j as f64 / ny as f64,
                0.0,
            ));
        }
    }
    let idx = |i: usize, j: usize| (j * (nx + 1) + i) as u32;
    let mut tris = Vec::new();
    for j in 0..ny {
        for i in 0..nx {
            tris.push([idx(i, j), idx(i + 1, j), idx(i + 1, j + 1)]);
            tris.push([idx(i, j), idx(i + 1, j + 1), idx(i, j + 1)]);
        }
    }
    build(pos, tris)
}

/// Swept tube along a polyline with per-point radius (open ends). Useful for building
/// synthetic vessel-like test geometry.
pub fn tube_along(path: &[DVec3], radii: &[f64], slices: usize) -> Mesh {
    assert!(path.len() >= 2 && radii.len() == path.len());
    let slices = slices.max(3);
    let mut pos = Vec::new();
    // Parallel-transport frames.
    let tangent = |i: usize| -> DVec3 {
        let a = path[i.saturating_sub(1)];
        let b = path[(i + 1).min(path.len() - 1)];
        (b - a).normalize()
    };
    let t0 = tangent(0);
    let (mut u, _) = crate::geom::orthonormal_basis(t0);
    let mut prev_t = t0;
    for i in 0..path.len() {
        let t = tangent(i);
        // Rotate u from prev_t to t.
        let axis = prev_t.cross(t);
        if axis.length() > 1e-12 {
            let ang = prev_t.angle_between(t);
            let q = glam::DQuat::from_axis_angle(axis.normalize(), ang);
            u = q * u;
        }
        u = (u - t * u.dot(t)).normalize();
        let v = t.cross(u);
        prev_t = t;
        for j in 0..slices {
            let th = 2.0 * PI * j as f64 / slices as f64;
            pos.push(path[i] + (u * th.cos() + v * th.sin()) * radii[i]);
        }
    }
    let idx = |i: usize, j: usize| (i * slices + (j % slices)) as u32;
    let mut tris = Vec::new();
    for i in 0..path.len() - 1 {
        for j in 0..slices {
            let a = idx(i, j);
            let b = idx(i, j + 1);
            let c = idx(i + 1, j + 1);
            let d = idx(i + 1, j);
            tris.push([a, b, c]);
            tris.push([a, c, d]);
        }
    }
    build(pos, tris)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn primitives_are_valid() {
        for m in [cube(2.0, 3), uv_sphere(1.0, 16, 8), icosphere(1.0, 2), cylinder(1.0, 2.0, 12, 3, true)] {
            m.check_validity().unwrap();
            assert!(m.is_closed());
            assert!(m.is_consistently_oriented());
            assert!(crate::measure::volume(&m) > 0.0, "outward orientation");
        }
        let g = plane_grid(1.0, 1.0, 4, 4);
        g.check_validity().unwrap();
        assert!(!g.is_closed());
        let tube = tube_along(&[DVec3::ZERO, DVec3::Z, DVec3::new(0.5, 0.0, 2.0)], &[0.2, 0.2, 0.15], 10);
        tube.check_validity().unwrap();
    }
}
