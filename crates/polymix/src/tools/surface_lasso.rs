//! Meshmixer's Lasso Mode: curves drawn on the surface (the view may turn between them) are
//! joined into a closed path along mesh edges, and the faces it encloses are selected.

use mm_core::{Mesh, INVALID};
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(PartialEq)]
struct Item(f64, u32);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        o.0.total_cmp(&self.0)
    }
}

/// The corner of triangle `t` nearest to `p`.
pub fn nearest_corner(m: &Mesh, t: u32, p: mm_core::DVec3) -> u32 {
    m.triangle(t).into_iter().min_by(|&a, &b| m.vertex(a).distance(p).total_cmp(&m.vertex(b).distance(p))).unwrap()
}

/// Shortest vertex path from `a` to `b` along mesh edges (inclusive), or `None` if they are
/// not connected.
pub fn shortest_path(m: &Mesh, a: u32, b: u32) -> Option<Vec<u32>> {
    if a == b {
        return Some(vec![a]);
    }
    let mut dist: HashMap<u32, f64> = HashMap::from([(a, 0.0)]);
    let mut prev: HashMap<u32, u32> = HashMap::new();
    let mut heap = BinaryHeap::from([Item(0.0, a)]);
    let goal = m.vertex(b);
    while let Some(Item(_, v)) = heap.pop() {
        if v == b {
            let mut path = vec![b];
            let mut x = b;
            while let Some(&p) = prev.get(&x) {
                path.push(p);
                x = p;
            }
            path.reverse();
            return Some(path);
        }
        let d = dist[&v];
        let pv = m.vertex(v);
        for u in m.vertex_neighbors(v) {
            let nd = d + pv.distance(m.vertex(u));
            if nd < dist.get(&u).copied().unwrap_or(f64::INFINITY) {
                dist.insert(u, nd);
                prev.insert(u, v);
                // A* toward the goal keeps long hops cheap.
                heap.push(Item(nd + m.vertex(u).distance(goal), u));
            }
        }
    }
    None
}

/// Join snapped curve vertices (in drawing order) into a closed vertex loop along edges.
pub fn close_loop(m: &Mesh, waypoints: &[u32]) -> Option<Vec<u32>> {
    let mut pts: Vec<u32> = waypoints.to_vec();
    pts.dedup();
    if pts.len() < 3 {
        return None;
    }
    let mut out: Vec<u32> = Vec::new();
    for i in 0..pts.len() {
        let seg = shortest_path(m, pts[i], pts[(i + 1) % pts.len()])?;
        out.extend_from_slice(&seg[..seg.len() - 1]);
    }
    Some(out)
}

/// Faces enclosed by a closed vertex loop: the loop's edges cut the surface, and the
/// smaller (by area) of the pieces next to the cut is returned. Empty if the loop does not
/// separate the surface.
pub fn enclosed_faces(m: &Mesh, lp: &[u32]) -> Vec<u32> {
    let mut cut: HashSet<u32> = HashSet::new();
    for i in 0..lp.len() {
        if let Some(e) = m.find_edge(lp[i], lp[(i + 1) % lp.len()]) {
            cut.insert(e);
        }
    }
    if cut.is_empty() {
        return Vec::new();
    }
    // Pieces touching the cut, flooded without crossing it.
    let mut label: HashMap<u32, usize> = HashMap::new();
    let mut pieces: Vec<(Vec<u32>, f64)> = Vec::new();
    for &e in &cut {
        for t in m.edge_triangles(e) {
            if t == INVALID || label.contains_key(&t) {
                continue;
            }
            let id = pieces.len();
            let mut tris = Vec::new();
            let mut area = 0.0;
            let mut stack = vec![t];
            label.insert(t, id);
            while let Some(x) = stack.pop() {
                tris.push(x);
                area += m.tri_area(x);
                for e2 in m.tri_edges(x) {
                    if cut.contains(&e2) {
                        continue;
                    }
                    let n = m.edge_other_triangle(e2, x);
                    if n != INVALID && !label.contains_key(&n) {
                        label.insert(n, id);
                        stack.push(n);
                    }
                }
            }
            pieces.push((tris, area));
        }
    }
    if pieces.len() < 2 {
        return Vec::new();
    }
    pieces.into_iter().min_by(|a, b| a.1.total_cmp(&b.1)).map(|p| p.0).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::{primitives, Bvh, DVec3, Ray};

    #[test]
    fn circle_on_a_sphere_encloses_the_cap() {
        let m = primitives::icosphere(1.0, 4);
        let bvh = Bvh::build(&m);
        // A circle at z = 0.6 drawn as three separate curves (as if turning the view).
        let mut waypoints = Vec::new();
        for k in 0..36 {
            let a = k as f64 / 36.0 * std::f64::consts::TAU;
            let dir = DVec3::new(0.8 * a.cos(), 0.8 * a.sin(), 0.6);
            let hit = bvh.raycast(&m, &Ray::new(dir * 3.0, -dir)).unwrap();
            waypoints.push(nearest_corner(&m, hit.tid, hit.point));
        }
        let lp = close_loop(&m, &waypoints).unwrap();
        let cap = enclosed_faces(&m, &lp);
        assert!(!cap.is_empty());
        assert!(cap.iter().all(|&t| m.tri_centroid(t).z > 0.45), "only the cap");
        let expected = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.7).count();
        assert!(cap.len() >= expected, "{} >= {expected}", cap.len());
    }

    #[test]
    fn open_path_encloses_nothing() {
        let m = primitives::plane_grid(4.0, 4.0, 8, 8);
        let near = |p: DVec3| m.vertex_ids().min_by(|&x, &y| m.vertex(x).distance(p).total_cmp(&m.vertex(y).distance(p))).unwrap();
        // A path inside the plane that goes there and back cuts nothing off…
        let path = shortest_path(&m, near(DVec3::new(-1.0, 0.0, 0.0)), near(DVec3::new(1.0, 0.0, 0.0))).unwrap();
        assert!(path.len() > 3);
        let mut back = path.clone();
        back.extend(path.iter().rev().skip(1).take(path.len() - 2));
        assert!(enclosed_faces(&m, &back).is_empty());
        // …while one from edge to edge splits the plane, and the smaller side is returned.
        let across = shortest_path(&m, near(DVec3::new(-2.0, 0.5, 0.0)), near(DVec3::new(2.0, 0.5, 0.0))).unwrap();
        let mut cut = across.clone();
        cut.extend(across.iter().rev().skip(1).take(across.len() - 2));
        let side = enclosed_faces(&m, &cut);
        assert!(!side.is_empty() && side.iter().all(|&t| m.tri_centroid(t).y > 0.5));
    }
}
