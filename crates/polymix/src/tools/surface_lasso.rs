//! Meshmixer's Lasso Mode: curves drawn on the surface (the view may turn between them) are
//! joined into a closed path along mesh edges, and the faces it encloses are selected.

use mm_core::hash::FastMap;
use mm_core::{Mesh, INVALID};
use rayon::prelude::*;
use std::collections::BinaryHeap;

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
    // Looked up by vertex only (never walked), so the fast hash changes nothing.
    let mut dist: FastMap<u32, f64> = FastMap::from_iter([(a, 0.0)]);
    let mut prev: FastMap<u32, u32> = FastMap::default();
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

/// The loop's edges, in loop order and each once, and marks over the edge ids to test them
/// by (every triangle a flood meets asks about its three edges).
fn cut_edges(m: &Mesh, lp: &[u32]) -> (Vec<u32>, Vec<bool>) {
    let mut cut = vec![false; m.max_eid()];
    let mut edges: Vec<u32> = Vec::new();
    for i in 0..lp.len() {
        if let Some(e) = m.find_edge(lp[i], lp[(i + 1) % lp.len()]) {
            if !std::mem::replace(&mut cut[e as usize], true) {
                edges.push(e);
            }
        }
    }
    (edges, cut)
}

/// Faces enclosed by a closed vertex loop: the loop's edges cut the surface, and the
/// smaller (by area) of the pieces next to the cut is returned. Empty if the loop does not
/// separate the surface.
///
/// The pieces are found from the cut's edges in loop order and each is flooded from its
/// first triangle, summing its area in flood order; the first of the smallest wins. A small
/// loop on a large model would flood the whole rest of the model that way, so the pieces are
/// first told apart by flooding them all side by side, and only those that may be the
/// smallest are then flooded as before (see [`PieceSizes`]): the answer is the very same.
pub fn enclosed_faces(m: &Mesh, lp: &[u32]) -> Vec<u32> {
    let (edges, cut) = cut_edges(m, lp);
    if edges.is_empty() {
        return Vec::new();
    }
    // Sizing pieces up compares sums of the same areas taken in different orders, which
    // differ only by rounding while every area is finite (no huge or NaN coordinates).
    let tame = m.positions().par_iter().all(|p| p.abs().cmplt(mm_core::DVec3::splat(1e100)).all());
    if tame {
        PieceSizes::new(m, &edges, &cut).smallest()
    } else {
        enclosed_faces_in_turn(m, &edges, &cut)
    }
}

/// One piece flooded from `t` without crossing the cut: its triangles in flood order and its
/// area summed in that order. `mark` marks a triangle, saying whether it was unmarked.
fn flood_piece(m: &Mesh, cut: &[bool], t: u32, mut mark: impl FnMut(u32) -> bool) -> (Vec<u32>, f64) {
    let mut tris = Vec::new();
    let mut area = 0.0;
    let mut stack = vec![t];
    mark(t);
    while let Some(x) = stack.pop() {
        tris.push(x);
        area += m.tri_area(x);
        for e2 in m.tri_edges(x) {
            if cut[e2 as usize] {
                continue;
            }
            let n = m.edge_other_triangle(e2, x);
            if n != INVALID && mark(n) {
                stack.push(n);
            }
        }
    }
    (tris, area)
}

/// [`enclosed_faces`] flooding each piece in full, one after the other.
fn enclosed_faces_in_turn(m: &Mesh, edges: &[u32], cut: &[bool]) -> Vec<u32> {
    // The flood reaches every triangle of every piece: its marks are an array over the ids.
    let mut labelled = vec![false; m.max_tid()];
    let mut pieces: Vec<(Vec<u32>, f64)> = Vec::new();
    for &e in edges {
        for t in m.edge_triangles(e) {
            if t == INVALID || labelled[t as usize] {
                continue;
            }
            pieces.push(flood_piece(m, cut, t, |n| !std::mem::replace(&mut labelled[n as usize], true)));
        }
    }
    if pieces.len() < 2 {
        return Vec::new();
    }
    pieces.into_iter().min_by(|a, b| a.1.total_cmp(&b.1)).map(|p| p.0).unwrap_or_default()
}

/// Triangles a flood takes per turn.
const FLOOD_TURN: usize = 256;
/// A piece is out once its area so far passes a whole piece's by this fraction: sums of up
/// to 10^9 non-negative areas in any two orders differ by far less (about 10^-7).
const AREA_MARGIN: f64 = 1e-6;

/// The pieces next to the cut, sized up by flooding them side by side.
///
/// Every triangle next to the cut starts a flood, numbered in the order the pieces are found
/// in, and all floods take turns. Floods that meet are in the same piece and are joined (a
/// union-find whose root is the earliest flood, so the piece's first triangle). A piece is
/// whole when all its floods have run out; one whose area so far passes a whole piece's by
/// the margin is larger than it however its area is summed, so it is out, and its floods
/// stop. The whole pieces left near the smallest are then flooded as one after the other
/// does, from their first triangle, and the first of the smallest wins.
struct PieceSizes<'a> {
    m: &'a Mesh,
    cut: &'a [bool],
    /// Per triangle: 1 + the flood that reached it, or 0.
    label: Vec<u32>,
    /// Per flood: its first triangle and what is left to flood.
    seed: Vec<u32>,
    stack: Vec<Vec<u32>>,
    /// Union-find over the floods; at a root, the piece's area so far, its floods still
    /// running, and whether it is out.
    parent: Vec<u32>,
    area: Vec<f64>,
    running: Vec<u32>,
    out: Vec<bool>,
    /// The least area of a whole piece so far.
    least: f64,
}

impl<'a> PieceSizes<'a> {
    fn new(m: &'a Mesh, edges: &[u32], cut: &'a [bool]) -> Self {
        let mut s = Self { m, cut, label: vec![0; m.max_tid()], seed: Vec::new(), stack: Vec::new(), parent: Vec::new(), area: Vec::new(), running: Vec::new(), out: Vec::new(), least: f64::INFINITY };
        for &e in edges {
            for t in m.edge_triangles(e) {
                if t != INVALID && s.label[t as usize] == 0 {
                    let k = s.seed.len() as u32;
                    s.label[t as usize] = k + 1;
                    s.seed.push(t);
                    s.stack.push(vec![t]);
                    s.parent.push(k);
                    s.area.push(0.0);
                    s.running.push(1);
                    s.out.push(false);
                }
            }
        }
        s
    }

    fn find(&mut self, k: u32) -> u32 {
        let mut r = k;
        while self.parent[r as usize] != r {
            r = self.parent[r as usize];
        }
        let mut k = k;
        while self.parent[k as usize] != r {
            let next = self.parent[k as usize];
            self.parent[k as usize] = r;
            k = next;
        }
        r
    }

    /// Join the pieces of floods `a` and `b`; the earlier root stays the root.
    fn join(&mut self, a: u32, b: u32) {
        let (a, b) = (self.find(a), self.find(b));
        if a == b {
            return;
        }
        let (root, child) = (a.min(b) as usize, a.max(b) as usize);
        self.parent[child] = root as u32;
        self.area[root] += self.area[child];
        self.running[root] += self.running[child];
        self.out[root] |= self.out[child];
    }

    fn smallest(mut self) -> Vec<u32> {
        let (m, cut) = (self.m, self.cut);
        let mut active: Vec<u32> = (0..self.seed.len() as u32).collect();
        while !active.is_empty() {
            for &k in &active {
                let r = self.find(k) as usize;
                if self.out[r] {
                    continue;
                }
                for _ in 0..FLOOD_TURN {
                    let Some(x) = self.stack[k as usize].pop() else { break };
                    let r = self.find(k) as usize;
                    self.area[r] += m.tri_area(x);
                    for e2 in m.tri_edges(x) {
                        if cut[e2 as usize] {
                            continue;
                        }
                        let n = m.edge_other_triangle(e2, x);
                        if n == INVALID {
                            continue;
                        }
                        match self.label[n as usize] {
                            0 => {
                                self.label[n as usize] = k + 1;
                                self.stack[k as usize].push(n);
                            }
                            j => self.join(k, j - 1),
                        }
                    }
                }
                let r = self.find(k) as usize;
                if self.stack[k as usize].is_empty() {
                    self.running[r] -= 1;
                    if self.running[r] == 0 && !self.out[r] {
                        self.least = self.least.min(self.area[r]);
                    }
                }
            }
            // Pieces now clearly larger than a whole one are out (their floods stop).
            let least = self.least * (1.0 + AREA_MARGIN);
            for &k in &active {
                let r = self.find(k) as usize;
                if self.running[r] > 0 && self.area[r] > least {
                    self.out[r] = true;
                }
            }
            let mut keep = Vec::with_capacity(active.len());
            for &k in &active {
                let r = self.find(k) as usize;
                if !self.stack[k as usize].is_empty() && !self.out[r] {
                    keep.push(k);
                }
            }
            active = keep;
        }
        // Every piece is now whole or out; one that is out is larger than a whole one, so
        // there are two pieces or more as soon as a piece is out.
        let mut whole: Vec<u32> = Vec::new();
        let mut any_out = false;
        for k in 0..self.seed.len() as u32 {
            if self.find(k) == k {
                if self.out[k as usize] {
                    any_out = true;
                } else {
                    whole.push(k);
                }
            }
        }
        if whole.len() + usize::from(any_out) < 2 {
            return Vec::new();
        }
        // The whole pieces that may be the smallest, flooded as one after the other floods
        // them (each from its first triangle, which its root flood started from).
        let near = self.least * (1.0 + AREA_MARGIN);
        let mut best: Option<(Vec<u32>, f64)> = None;
        for r in whole {
            if self.area[r as usize] > near {
                continue;
            }
            let label = &mut self.label;
            let piece = flood_piece(m, cut, self.seed[r as usize], |n| label[n as usize] != u32::MAX && {
                label[n as usize] = u32::MAX;
                true
            });
            // In piece order, so the first of equals stays.
            if best.as_ref().is_none_or(|b| piece.1.total_cmp(&b.1).is_lt()) {
                best = Some(piece);
            }
        }
        best.map(|b| b.0).unwrap_or_default()
    }
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

    /// A loop of edges around `axis` at height `h` on a sphere-like mesh.
    fn ring(m: &Mesh, center: DVec3, axis: DVec3, h: f64, r: f64) -> Option<Vec<u32>> {
        let bvh = Bvh::build(m);
        let (u, v) = mm_core::geom::orthonormal_basis(axis);
        let rr = (r * r - h * h).max(0.0).sqrt();
        let waypoints: Vec<u32> = (0..48)
            .filter_map(|k| {
                let a = k as f64 / 48.0 * std::f64::consts::TAU;
                let dir = (u * a.cos() * rr + v * a.sin() * rr + axis * h).normalize();
                let hit = bvh.raycast(m, &Ray::new(center + dir * 1.5 * r, -dir))?;
                Some(nearest_corner(m, hit.tid, hit.point))
            })
            .collect();
        close_loop(m, &waypoints)
    }

    /// The pieces as they were found before: in the order a hash set gave the cut's edges,
    /// with hash-map marks.
    fn enclosed_faces_hashed(m: &Mesh, lp: &[u32]) -> Vec<u32> {
        let cut: std::collections::HashSet<u32> = (0..lp.len()).filter_map(|i| m.find_edge(lp[i], lp[(i + 1) % lp.len()])).collect();
        let mut label = std::collections::HashMap::new();
        let mut pieces: Vec<(Vec<u32>, f64)> = Vec::new();
        for &e in &cut {
            for t in m.edge_triangles(e) {
                if t == INVALID || label.contains_key(&t) {
                    continue;
                }
                let (mut tris, mut area, mut stack) = (Vec::new(), 0.0, vec![t]);
                label.insert(t, pieces.len());
                while let Some(x) = stack.pop() {
                    tris.push(x);
                    area += m.tri_area(x);
                    for e2 in m.tri_edges(x) {
                        let n = m.edge_other_triangle(e2, x);
                        if !cut.contains(&e2) && n != INVALID && !label.contains_key(&n) {
                            label.insert(n, pieces.len());
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

    #[test]
    fn side_by_side_floods_pick_the_same_piece() {
        // Two spheres (the loop is on one of them; the other is never reached), a lumpy one,
        // and an open sheet.
        let mut two = primitives::icosphere(1.0, 4);
        let mut other = primitives::icosphere(0.5, 3);
        other.transform(|p| p + DVec3::new(3.0, 0.0, 0.0));
        two.append_mesh(&other, 1);
        let mut lumpy = primitives::icosphere(1.0, 4);
        lumpy.transform(|p| p * (1.0 + 0.05 * (p.x * 7.0).sin() * (p.y * 5.0).cos()));
        let sheet = primitives::plane_grid(4.0, 4.0, 40, 40);
        let mut loops = 0;
        for (m, label) in [(&two, "two spheres"), (&lumpy, "lumpy")] {
            for axis in [DVec3::Z, DVec3::new(0.3, -0.5, 0.8).normalize(), DVec3::X] {
                // Small caps, half, large caps (the inside is then the larger piece).
                for h in [0.95, 0.7, 0.3, 0.0, -0.4, -0.9] {
                    let Some(lp) = ring(m, DVec3::ZERO, axis, h, 1.0) else { continue };
                    let (edges, cut) = cut_edges(m, &lp);
                    let in_turn = enclosed_faces_in_turn(m, &edges, &cut);
                    let side_by_side = PieceSizes::new(m, &edges, &cut).smallest();
                    assert_eq!(side_by_side, in_turn, "{label}, axis {axis:?}, height {h}");
                    assert_eq!(enclosed_faces(m, &lp), in_turn);
                    assert!(!in_turn.is_empty(), "{label}, axis {axis:?}, height {h}");
                    // As before (the same faces; an exact tie could have gone either way).
                    let mut before = enclosed_faces_hashed(m, &lp);
                    let mut now = in_turn.clone();
                    before.sort_unstable();
                    now.sort_unstable();
                    if h != 0.0 {
                        assert_eq!(now, before, "{label}, axis {axis:?}, height {h}");
                    }
                    loops += 1;
                }
            }
        }
        assert!(loops >= 30, "{loops}");
        // A loop that goes there and back cuts nothing off; one across the sheet does.
        let near = |p: DVec3| sheet.vertex_ids().min_by(|&x, &y| sheet.vertex(x).distance(p).total_cmp(&sheet.vertex(y).distance(p))).unwrap();
        for (a, b, splits) in [(DVec3::new(-1.0, 0.0, 0.0), DVec3::new(1.0, 0.3, 0.0), false), (DVec3::new(-2.0, 0.5, 0.0), DVec3::new(2.0, -0.7, 0.0), true)] {
            let path = shortest_path(&sheet, near(a), near(b)).unwrap();
            let mut lp = path.clone();
            lp.extend(path.iter().rev().skip(1).take(path.len() - 2));
            let (edges, cut) = cut_edges(&sheet, &lp);
            let side_by_side = PieceSizes::new(&sheet, &edges, &cut).smallest();
            assert_eq!(side_by_side, enclosed_faces_in_turn(&sheet, &edges, &cut));
            assert_eq!(!side_by_side.is_empty(), splits);
        }
        // A loop around a band across the sheet (partly along its border) leaves three
        // pieces; at y = ±1 the two outer ones are the same size.
        for (y0, y1) in [(0.5, -0.7), (1.0, -1.0), (1.5, 0.2)] {
            let corners = [DVec3::new(-2.0, y0, 0.0), DVec3::new(2.0, y0, 0.0), DVec3::new(2.0, y1, 0.0), DVec3::new(-2.0, y1, 0.0)];
            let mut lp = Vec::new();
            for i in 0..4 {
                let side = shortest_path(&sheet, near(corners[i]), near(corners[(i + 1) % 4])).unwrap();
                lp.extend_from_slice(&side[..side.len() - 1]);
            }
            let (edges, cut) = cut_edges(&sheet, &lp);
            let want = enclosed_faces_in_turn(&sheet, &edges, &cut);
            assert!(!want.is_empty());
            assert_eq!(PieceSizes::new(&sheet, &edges, &cut).smallest(), want, "band {y0} {y1}");
        }
        // Huge coordinates (areas that overflow) take the one-after-the-other flood.
        let mut huge = primitives::icosphere(1.0, 3);
        let lp = ring(&huge, DVec3::ZERO, DVec3::Z, 0.6, 1.0).unwrap();
        let far = huge.vertex_ids().find(|&v| huge.vertex(v).z < -0.9).unwrap();
        huge.set_vertex(far, DVec3::new(0.0, 0.0, -1e200));
        let (edges, cut) = cut_edges(&huge, &lp);
        let got = enclosed_faces(&huge, &lp);
        assert_eq!(got, enclosed_faces_in_turn(&huge, &edges, &cut));
        assert!(!got.is_empty() && got.iter().all(|&t| huge.tri_centroid(t).z > 0.3), "the cap");
    }
}
