// SPDX-License-Identifier: AGPL-3.0-only
//! Cutting the scan with a plane: the polylines where its triangles cross it. Ported from
//! ScanRuler (`src/core/section/slice.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Every triangle that straddles the plane contributes one segment, with an end on each of
//! the two edges the plane passes through. Neighbouring triangles share an edge and so share
//! the end on it — the point is computed once per mesh edge and referenced by both, which
//! lets the segments be linked into chains by identity rather than by hunting for nearby
//! ends. On a welded, manifold mesh the chains come out as simple paths (open where the scan
//! has a hole) and loops; a loop is written with its first point repeated at the end.
//!
//! Every point also carries the side the material is on: the triangles are wound so their
//! normals point out of the material, so each segment takes the in-plane direction square to
//! it that its triangle faces, and a point the mean of its segments'.

use std::collections::HashMap;

use glam::{DVec2, DVec3};

use crate::frame::SectionFrame;

/// The chains a plane cuts through a mesh.
#[derive(Debug, Clone, Default)]
pub struct SectionCut {
    /// Chain by chain, in the mesh's own coordinates.
    pub chains: Vec<Vec<DVec3>>,
    /// Per point, the unit in-plane direction square to the chain that points out of the
    /// material; zero where the segments disagree (a flipped triangle).
    pub normals: Vec<Vec<DVec3>>,
}

impl SectionCut {
    pub fn point_count(&self) -> usize {
        self.chains.iter().map(|c| c.len()).sum()
    }

    /// The cut laid flat in a frame's sheet, chain by chain.
    pub fn project(&self, frame: &SectionFrame) -> Vec<Vec<DVec2>> {
        self.chains.iter().map(|c| c.iter().map(|p| frame.to_sheet(*p)).collect()).collect()
    }
}

/// Cut an indexed mesh with the plane through `origin` with normal `normal`. Chains shorter
/// than `min_length` millimetres of path are dropped — the specks a noisy scan cuts into.
///
/// A vertex is on the positive side when its signed distance is zero or more, so a triangle
/// either lies wholly on one side or has exactly two edges crossing. A vertex sitting exactly
/// on the plane is one point rather than two coincident ones.
pub fn slice_mesh(positions: &[DVec3], triangles: &[[u32; 3]], origin: DVec3, normal: DVec3, min_length: f64) -> SectionCut {
    let Some(n) = normal.try_normalize() else { return SectionCut::default() };
    if positions.is_empty() || triangles.is_empty() {
        return SectionCut::default();
    }
    let d0 = n.dot(origin);
    let dist: Vec<f64> = positions.iter().map(|p| n.dot(*p) - d0).collect();

    let mut pts: Vec<DVec3> = Vec::new();
    let mut edge_point: HashMap<(u32, u32), usize> = HashMap::new();
    let mut vertex_point: HashMap<u32, usize> = HashMap::new();
    let mut on_vertex = |i: u32, pts: &mut Vec<DVec3>| -> usize {
        *vertex_point.entry(i).or_insert_with(|| {
            pts.push(positions[i as usize]);
            pts.len() - 1
        })
    };
    let mut on_edge = |i: u32, j: u32, pts: &mut Vec<DVec3>, on_vertex: &mut dyn FnMut(u32, &mut Vec<DVec3>) -> usize| -> usize {
        let (di, dj) = (dist[i as usize], dist[j as usize]);
        if di == 0.0 {
            return on_vertex(i, pts);
        }
        if dj == 0.0 {
            return on_vertex(j, pts);
        }
        let key = if i < j { (i, j) } else { (j, i) };
        *edge_point.entry(key).or_insert_with(|| {
            // Opposite signs here, so t lies strictly inside (0, 1).
            let t = di / (di - dj);
            let (pi, pj) = (positions[i as usize], positions[j as usize]);
            pts.push(pi + (pj - pi) * t);
            pts.len() - 1
        })
    };

    let mut segs: Vec<(usize, usize)> = Vec::new();
    let mut seg_tri: Vec<usize> = Vec::new();
    for (t, tri) in triangles.iter().enumerate() {
        let [a, b, c] = *tri;
        let (sa, sb, sc) = (dist[a as usize] >= 0.0, dist[b as usize] >= 0.0, dist[c as usize] >= 0.0);
        if sa == sb && sb == sc {
            continue;
        }
        let mut p = None;
        let mut q = None;
        for (i, j, crosses) in [(a, b, sa != sb), (b, c, sb != sc), (c, a, sc != sa)] {
            if !crosses {
                continue;
            }
            let r = on_edge(i, j, &mut pts, &mut on_vertex);
            if p.is_none() {
                p = Some(r);
            } else {
                q = Some(r);
            }
        }
        // Both ends on the same vertex is a triangle touching the plane at a corner.
        if let (Some(p), Some(q)) = (p, q) {
            if p != q {
                segs.push((p, q));
                seg_tri.push(t);
            }
        }
    }
    if segs.is_empty() {
        return SectionCut::default();
    }

    // The outward direction at each point: per segment, the in-plane direction square to it,
    // turned to the side its triangle's normal faces, summed at both ends.
    let mut side = vec![DVec3::ZERO; pts.len()];
    for (s, &(p, q)) in segs.iter().enumerate() {
        let mut m = n.cross(pts[q] - pts[p]);
        let [a, b, c] = triangles[seg_tri[s]];
        let (pa, pb, pc) = (positions[a as usize], positions[b as usize], positions[c as usize]);
        if m.dot((pb - pa).cross(pc - pa)) < 0.0 {
            m = -m;
        }
        side[p] += m;
        side[q] += m;
    }

    let mut adj: Vec<Vec<usize>> = vec![Vec::new(); pts.len()];
    for (s, &(p, q)) in segs.iter().enumerate() {
        adj[p].push(s);
        adj[q].push(s);
    }
    let mut used = vec![false; segs.len()];
    let mut chains: Vec<Vec<usize>> = Vec::new();
    let mut walk = |start: usize, first: usize, used: &mut Vec<bool>| {
        let mut chain = vec![start];
        let mut cur = start;
        let mut s = first;
        loop {
            used[s] = true;
            let other = if segs[s].0 == cur { segs[s].1 } else { segs[s].0 };
            chain.push(other);
            cur = other;
            match adj[cur].iter().copied().find(|&cand| !used[cand]) {
                Some(next) => s = next,
                None => break,
            }
        }
        chains.push(chain);
    };
    // Open runs first, from their ends, so a whole path comes out as one walk; whatever is
    // left is loops, which a walk from anywhere closes on its own.
    for p in 0..pts.len() {
        if adj[p].len() == 1 && !used[adj[p][0]] {
            walk(p, adj[p][0], &mut used);
        }
    }
    for s in 0..segs.len() {
        if !used[s] {
            walk(segs[s].0, s, &mut used);
        }
    }

    let mut out = SectionCut::default();
    for chain in chains {
        if min_length > 0.0 {
            let l: f64 = chain.windows(2).map(|w| (pts[w[1]] - pts[w[0]]).length()).sum();
            if l < min_length {
                continue;
            }
        }
        out.chains.push(chain.iter().map(|&p| pts[p]).collect());
        out.normals.push(chain.iter().map(|&p| side[p].try_normalize().unwrap_or(DVec3::ZERO)).collect());
    }
    out
}

/// Chains shorter than this are dropped from a section: what a noisy scan cuts into specks.
pub const EDGE_MIN_FEATURE_MM: f64 = 1.0;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cube_cut_across_is_one_closed_square() {
        let m = mm_core::primitives::cube(10.0, 3);
        let (pos, tris, _) = m.to_arrays();
        let cut = slice_mesh(&pos, &tris, DVec3::new(0.0, 0.0, 0.1), DVec3::Z, 1.0);
        assert_eq!(cut.chains.len(), 1);
        let c = &cut.chains[0];
        assert!((c[0] - c[c.len() - 1]).length() < 1e-12, "a loop repeats its first point");
        let length: f64 = c.windows(2).map(|w| (w[1] - w[0]).length()).sum();
        assert!((length - 40.0).abs() < 1e-6, "{length}");
        // The outward normals point away from the centre.
        for (p, n) in c.iter().zip(&cut.normals[0]) {
            assert!(DVec2::new(p.x, p.y).dot(DVec2::new(n.x, n.y)) > 0.0);
        }
    }
}
