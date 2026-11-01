// SPDX-License-Identifier: AGPL-3.0-only
//! The outline a round face shows from where it is looked at. Ported from ScanRuler
//! (`plugins/reverse/core/silhouette.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A body is drawn by its edges, and a cylinder has none along its sides, a ball none at all:
//! drawn solid in the body's colour they have no rim, and a boss on a plate reads as a disc.
//! What a drawing puts there is the silhouette — the line on the surface where it turns from
//! facing the eye to facing away.
//!
//! On a mesh with a normal at every vertex that line is where normal · view is zero, and it is
//! found a triangle at a time with nothing known of the triangle's neighbours: the value at
//! the three corners, and where it changes sign along two of the sides, a segment between the
//! two crossings. Taken off the triangles' own flat normals the line would run along mesh
//! edges in a zigzag; off the vertex normals it cuts across the triangles and is as smooth as
//! the surface was meshed.

use glam::DVec3;
use serde::{Deserialize, Serialize};

/// Where the body is looked at from: along one direction (an orthographic camera) or from a
/// point. In the mesh's own coordinates.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SilhouetteView {
    Dir(DVec3),
    Eye(DVec3),
}

/// The silhouette of the given triangles — `ranges` as (first triangle, count), the round
/// faces of a body; a flat face has one normal and no silhouette of its own — as segments,
/// XYZ XYZ per segment. `positions` and `normals` are XYZ per vertex, `indices` three per
/// triangle.
pub fn silhouette_segments(positions: &[f32], normals: &[f32], indices: &[u32], ranges: &[(usize, usize)], view: SilhouetteView) -> Vec<f32> {
    let mut out: Vec<f32> = Vec::new();
    let mut g = [0.0f64; 3];
    let mut at = [0usize; 3];
    let p = |i: usize| positions.get(i).copied().unwrap_or(0.0) as f64;
    let nm = |i: usize| normals.get(i).copied().unwrap_or(0.0) as f64;
    for &(first, count) in ranges {
        for t in first..first + count {
            if t * 3 + 2 >= indices.len() {
                break;
            }
            for k in 0..3 {
                let v = indices[t * 3 + k] as usize * 3;
                at[k] = v;
                let d = match view {
                    SilhouetteView::Dir(dir) => dir,
                    SilhouetteView::Eye(eye) => DVec3::new(p(v) - eye.x, p(v + 1) - eye.y, p(v + 2) - eye.z),
                };
                g[k] = nm(v) * d.x + nm(v + 1) * d.y + nm(v + 2) * d.z;
            }
            // A corner dead on the line counts as facing: a sign is wanted, and a zero taken
            // either way leaves no gap, only a segment a hair shorter.
            let (s0, s1, s2) = (g[0] >= 0.0, g[1] >= 0.0, g[2] >= 0.0);
            if s0 == s1 && s1 == s2 {
                continue;
            }
            // The corner on its own side, and the two sides that leave it.
            let lone = if s0 != s1 && s0 != s2 {
                0
            } else if s1 != s0 && s1 != s2 {
                1
            } else {
                2
            };
            for other in [(lone + 1) % 3, (lone + 2) % 3] {
                let f = g[lone] / (g[lone] - g[other]);
                let (a, b) = (at[lone], at[other]);
                for k in 0..3 {
                    out.push((p(a + k) + (p(b + k) - p(a + k)) * f) as f32);
                }
            }
        }
    }
    out
}

#[cfg(test)]
pub(crate) mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reSilhouette.test.ts`.
    use super::*;
    use std::collections::HashMap;
    use std::f64::consts::PI;

    /// The wall of a cylinder about Z, radius `r`, with radial normals.
    fn tube(r: f64, length: f64, around: usize, along: usize) -> (Vec<f32>, Vec<f32>, Vec<u32>, usize) {
        let mut positions = Vec::new();
        let mut normals = Vec::new();
        let mut indices: Vec<u32> = Vec::new();
        for j in 0..=along {
            for k in 0..around {
                let a = (k as f64 / around as f64) * 2.0 * PI;
                positions.extend([(r * a.cos()) as f32, (r * a.sin()) as f32, ((j as f64 / along as f64) * length) as f32]);
                normals.extend([a.cos() as f32, a.sin() as f32, 0.0]);
            }
        }
        for j in 0..along {
            for k in 0..around {
                let a = (j * around + k) as u32;
                let b = (j * around + (k + 1) % around) as u32;
                let c = a + around as u32;
                let d = b + around as u32;
                indices.extend([a, b, d, a, d, c]);
            }
        }
        let tris = indices.len() / 3;
        (positions, normals, indices, tris)
    }

    /// The icosphere of ScanRuler's `tests/helpers.ts`: an icosahedron subdivided, its
    /// vertices pushed out to the radius (no noise).
    pub(crate) fn icosphere(subdivisions: usize, radius: f64) -> (Vec<f32>, Vec<u32>) {
        let t = (1.0 + 5f64.sqrt()) / 2.0;
        let norm = |v: [f64; 3]| {
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
        .map(norm)
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
            let mut mid = |a: usize, b: usize, verts: &mut Vec<[f64; 3]>| -> usize {
                let key = if a < b { (a, b) } else { (b, a) };
                if let Some(&i) = cache.get(&key) {
                    return i;
                }
                let m = norm([(verts[a][0] + verts[b][0]) / 2.0, (verts[a][1] + verts[b][1]) / 2.0, (verts[a][2] + verts[b][2]) / 2.0]);
                verts.push(m);
                cache.insert(key, verts.len() - 1);
                verts.len() - 1
            };
            let mut next = Vec::new();
            for [a, b, c] in faces {
                let ab = mid(a, b, &mut verts);
                let bc = mid(b, c, &mut verts);
                let ca = mid(c, a, &mut verts);
                next.extend([[a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]]);
            }
            faces = next;
        }
        let positions = verts.iter().flat_map(|v| [(v[0] * radius) as f32, (v[1] * radius) as f32, (v[2] * radius) as f32]).collect();
        let indices = faces.iter().flat_map(|f| [f[0] as u32, f[1] as u32, f[2] as u32]).collect();
        (positions, indices)
    }

    #[test]
    fn is_the_two_lines_down_a_cylinders_sides_seen_square_to_its_axis() {
        let (pos, nrm, idx, tris) = tube(10.0, 30.0, 48, 6);
        // Looked at along +X — turned a hair, so that no vertex sits dead on the line: the
        // sides are at y = ±10.
        let a = 0.01f64;
        let seg = silhouette_segments(&pos, &nrm, &idx, &[(0, tris)], SilhouetteView::Dir(DVec3::new(a.cos(), a.sin(), 0.0)));
        assert_eq!(seg.len() % 6, 0);
        assert!(!seg.is_empty());
        let (mut left, mut right) = (0, 0);
        for i in (0..seg.len()).step_by(3) {
            let (x, y) = (seg[i] as f64, seg[i + 1] as f64);
            // On the cylinder to within the chord's sag, where the normal is square to the view.
            assert!(x.hypot(y) > 10.0 * (PI / 48.0).cos() - 1e-6);
            assert!((x * a.cos() + y * a.sin()).abs() < 0.05);
            if y > 0.0 {
                left += 1;
            } else {
                right += 1;
            }
        }
        assert_eq!(left, right);
        // Each side runs the cylinder's whole length.
        let zs: Vec<f64> = seg.iter().skip(2).step_by(3).map(|z| *z as f64).collect();
        assert!(zs.iter().cloned().fold(f64::INFINITY, f64::min).abs() < 5e-7);
        assert!((zs.iter().cloned().fold(f64::NEG_INFINITY, f64::max) - 30.0).abs() < 5e-7);
    }

    #[test]
    fn is_nothing_seen_down_the_axis_and_nothing_of_a_flat_face() {
        let (pos, nrm, idx, tris) = tube(10.0, 30.0, 48, 6);
        assert!(silhouette_segments(&pos, &nrm, &idx, &[(0, tris)], SilhouetteView::Dir(DVec3::Z)).is_empty());
        let flat_pos = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let flat_nrm = [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0];
        assert!(silhouette_segments(&flat_pos, &flat_nrm, &[0, 1, 2], &[(0, 1)], SilhouetteView::Dir(DVec3::X)).is_empty());
    }

    #[test]
    fn is_a_balls_great_circle_square_to_the_view_and_from_an_eye_close_by_a_smaller_circle_nearer_to_it() {
        let (pos, idx) = icosphere(4, 10.0);
        let normals: Vec<f32> = pos.iter().map(|v| ((*v as f64) / 10.0) as f32).collect();
        let tris = idx.len() / 3;
        let dir = DVec3::new(0.3, -0.5, 0.81);
        let l = dir.length();
        let seg = silhouette_segments(&pos, &normals, &idx, &[(0, tris)], SilhouetteView::Dir(dir / l));
        assert!(seg.len() > 60);
        for i in (0..seg.len()).step_by(3) {
            let q = DVec3::new(seg[i] as f64, seg[i + 1] as f64, seg[i + 2] as f64);
            assert!(q.length() > 9.9);
            assert!((q.dot(dir) / l).abs() < 0.05);
        }
        // From an eye 25 mm out along Z the rim is the circle of tangency: at z = R²/d = 4,
        // not at the equator.
        let near = silhouette_segments(&pos, &normals, &idx, &[(0, tris)], SilhouetteView::Eye(DVec3::new(0.0, 0.0, 25.0)));
        assert!(!near.is_empty());
        for i in (0..near.len()).step_by(3) {
            assert!((near[i + 2] as f64 - 4.0).abs() < 0.5);
        }
    }
}
