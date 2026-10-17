//! Generate Face Groups (Meshmixer *Edit > Generate Face Groups*): segment a mesh into face
//! groups by crease angle or by normal similarity, then merge groups that are too small.

use mm_core::{Mesh, INVALID};
use std::collections::HashMap;

/// How regions are grown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaceGroupMode {
    /// Regions end at edges whose dihedral angle exceeds the threshold.
    EdgeAngle,
    /// Each region holds faces within the threshold of its first (largest) face's normal:
    /// near-planar patches.
    NormalAngle,
}

/// Replace every face group: grow regions (see [`FaceGroupMode`]) with the given angle
/// threshold (degrees), then merge regions with fewer than `min_size` triangles into the
/// neighbour they share the longest border with. Groups are numbered from 0. Returns the
/// number of groups.
pub fn generate_face_groups(mesh: &mut Mesh, mode: FaceGroupMode, angle_deg: f64, min_size: usize) -> usize {
    let c = angle_deg.to_radians().cos();
    let mut label = vec![u32::MAX; mesh.max_tid()];
    let mut sizes: Vec<usize> = Vec::new(); // triangles per region
    // Seeds in order of decreasing area, so regions start from big faces.
    let mut order: Vec<u32> = mesh.triangle_ids().collect();
    order.sort_by(|&a, &b| mesh.tri_area(b).total_cmp(&mesh.tri_area(a)));
    for seed in order {
        if label[seed as usize] != u32::MAX {
            continue;
        }
        let id = sizes.len() as u32;
        let n0 = mesh.tri_normal(seed);
        let mut stack = vec![seed];
        label[seed as usize] = id;
        let mut n = 0;
        while let Some(t) = stack.pop() {
            n += 1;
            let nt = mesh.tri_normal(t);
            for nb in mesh.tri_neighbors(t) {
                if nb == INVALID || label[nb as usize] != u32::MAX {
                    continue;
                }
                let nn = mesh.tri_normal(nb);
                let ok = match mode {
                    FaceGroupMode::EdgeAngle => nt.dot(nn) >= c,
                    FaceGroupMode::NormalAngle => n0.dot(nn) >= c,
                };
                if ok {
                    label[nb as usize] = id;
                    stack.push(nb);
                }
            }
        }
        sizes.push(n);
    }
    // Merge small regions into their longest-bordering neighbour.
    if min_size > 1 {
        let mut members: Vec<Vec<u32>> = vec![Vec::new(); sizes.len()];
        for t in mesh.triangle_ids() {
            members[label[t as usize] as usize].push(t);
        }
        for _ in 0..8 {
            let mut changed = false;
            let mut small: Vec<usize> = (0..members.len()).filter(|&g| !members[g].is_empty() && members[g].len() < min_size).collect();
            small.sort_by_key(|&g| members[g].len());
            for g in small {
                if members[g].is_empty() || members[g].len() >= min_size {
                    continue;
                }
                let mut border: HashMap<u32, f64> = HashMap::new();
                for &t in &members[g] {
                    for e in mesh.tri_edges(t) {
                        let o = mesh.edge_other_triangle(e, t);
                        if o != INVALID && label[o as usize] != g as u32 {
                            *border.entry(label[o as usize]).or_insert(0.0) += mesh.edge_length(e);
                        }
                    }
                }
                if let Some((&to, _)) = border.iter().max_by(|a, b| a.1.total_cmp(b.1)) {
                    let moved = std::mem::take(&mut members[g]);
                    for &t in &moved {
                        label[t as usize] = to;
                    }
                    members[to as usize].extend(moved);
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
    }
    // Compact the ids.
    let mut remap: HashMap<u32, i32> = HashMap::new();
    for t in mesh.triangle_ids().collect::<Vec<_>>() {
        let next = remap.len() as i32;
        let g = *remap.entry(label[t as usize]).or_insert(next);
        mesh.set_tri_group(t, g);
    }
    remap.len()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn cube_faces_become_groups() {
        let mut m = primitives::cube(2.0, 4);
        assert_eq!(generate_face_groups(&mut m, FaceGroupMode::EdgeAngle, 30.0, 0), 6);
        let mut m = primitives::cube(2.0, 4);
        assert_eq!(generate_face_groups(&mut m, FaceGroupMode::NormalAngle, 30.0, 0), 6);
        // Each group is planar.
        for g in 0..6 {
            let ns: Vec<_> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).map(|t| m.tri_normal(t)).collect();
            assert!(ns.iter().all(|n| n.dot(ns[0]) > 0.999));
        }
    }

    #[test]
    fn normal_angle_splits_a_sphere_and_small_groups_merge() {
        let mut m = primitives::icosphere(1.0, 3);
        let n = generate_face_groups(&mut m, FaceGroupMode::NormalAngle, 40.0, 0);
        assert!((6..60).contains(&n), "{n} patches");
        // The smooth sphere is one region by edge angle.
        let mut s = primitives::icosphere(1.0, 3);
        assert_eq!(generate_face_groups(&mut s, FaceGroupMode::EdgeAngle, 30.0, 0), 1);
        // Merging: no group smaller than the threshold survives.
        let mut k = primitives::icosphere(1.0, 3);
        let merged = generate_face_groups(&mut k, FaceGroupMode::NormalAngle, 40.0, 60);
        assert!(merged < n);
        for g in 0..merged as i32 {
            let size = k.triangle_ids().filter(|&t| k.tri_group(t) == g).count();
            assert!(size >= 60 || merged == 1, "group {g} has {size}");
        }
    }
}
