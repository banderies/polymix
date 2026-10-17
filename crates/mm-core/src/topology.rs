//! Topological queries: connected components, boundary loops, non-manifold detection.

use crate::mesh::{tri_edge_index, Mesh, INVALID};
use glam::DVec3;

/// Edge-connected components of the triangles. Each component is a list of triangle ids.
pub fn connected_components(m: &Mesh) -> Vec<Vec<u32>> {
    let mut comp = vec![u32::MAX; m.max_tid()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for seed in m.triangle_ids() {
        if comp[seed as usize] != u32::MAX {
            continue;
        }
        let id = out.len() as u32;
        let mut tris = Vec::new();
        comp[seed as usize] = id;
        stack.push(seed);
        while let Some(t) = stack.pop() {
            tris.push(t);
            for n in m.tri_neighbors(t) {
                if n != INVALID && comp[n as usize] == u32::MAX {
                    comp[n as usize] = id;
                    stack.push(n);
                }
            }
        }
        out.push(tris);
    }
    out
}

/// Components connected through shared vertices (bowtie-connected parts are merged).
pub fn vertex_connected_components(m: &Mesh) -> Vec<Vec<u32>> {
    let mut comp = vec![u32::MAX; m.max_tid()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for seed in m.triangle_ids() {
        if comp[seed as usize] != u32::MAX {
            continue;
        }
        let id = out.len() as u32;
        let mut tris = Vec::new();
        comp[seed as usize] = id;
        stack.push(seed);
        while let Some(t) = stack.pop() {
            tris.push(t);
            for v in m.triangle(t) {
                for n in m.vertex_triangles(v) {
                    if comp[n as usize] == u32::MAX {
                        comp[n as usize] = id;
                        stack.push(n);
                    }
                }
            }
        }
        out.push(tris);
    }
    out
}

/// Connected component (edge-adjacent) containing `seed`, optionally restricted by `allow`.
pub fn flood_fill_triangles(m: &Mesh, seeds: &[u32], allow: impl Fn(u32) -> bool) -> Vec<u32> {
    let mut seen = vec![false; m.max_tid()];
    let mut out = Vec::new();
    let mut stack: Vec<u32> = Vec::new();
    for &s in seeds {
        if m.is_triangle(s) && !seen[s as usize] && allow(s) {
            seen[s as usize] = true;
            stack.push(s);
        }
    }
    while let Some(t) = stack.pop() {
        out.push(t);
        for n in m.tri_neighbors(t) {
            if n != INVALID && !seen[n as usize] && allow(n) {
                seen[n as usize] = true;
                stack.push(n);
            }
        }
    }
    out
}

/// A closed loop of boundary edges.
#[derive(Clone, Debug, Default)]
pub struct EdgeLoop {
    /// Vertices in order. Edge i connects `vertices[i]` and `vertices[(i+1) % n]`,
    /// oriented the same way as in the adjacent triangle (so a filling triangle must
    /// use the reverse orientation).
    pub vertices: Vec<u32>,
    pub edges: Vec<u32>,
}

impl EdgeLoop {
    pub fn len(&self) -> usize {
        self.vertices.len()
    }
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
    pub fn positions(&self, m: &Mesh) -> Vec<DVec3> {
        self.vertices.iter().map(|&v| m.vertex(v)).collect()
    }
    pub fn centroid(&self, m: &Mesh) -> DVec3 {
        let n = self.vertices.len().max(1) as f64;
        self.vertices.iter().map(|&v| m.vertex(v)).sum::<DVec3>() / n
    }
    pub fn perimeter(&self, m: &Mesh) -> f64 {
        self.edges.iter().map(|&e| m.edge_length(e)).sum()
    }
}

/// From boundary edge `e` arriving at `q`, rotate around `q` through its triangle fan
/// (crossing edges incident to `q`, independent of triangle winding) to the next boundary
/// edge.
fn next_boundary_edge(m: &Mesh, e: u32, q: u32) -> Option<u32> {
    let mut t = m.edge_triangles(e)[0];
    let mut cur = e;
    for _ in 0..=m.vertex_valence(q) {
        let f = m.tri_edges(t).into_iter().find(|&x| x != cur && m.edge_vertices(x).contains(&q))?;
        let n = m.edge_other_triangle(f, t);
        if n == INVALID {
            return Some(f);
        }
        cur = f;
        t = n;
    }
    None
}

/// Extract all boundary loops. The walk does not depend on triangle winding, so loops are
/// also found on inconsistently oriented meshes; each loop follows the winding of its
/// first edge's triangle. Bowtie vertices are handled per fan, so a vertex may appear in
/// several loops.
pub fn boundary_loops(m: &Mesh) -> Vec<EdgeLoop> {
    let mut used = vec![false; m.max_eid()];
    let mut loops = Vec::new();
    for e0 in m.boundary_edge_ids().collect::<Vec<_>>() {
        if used[e0 as usize] {
            continue;
        }
        let [p, q] = m.oriented_edge_vertices(e0);
        let mut lp = EdgeLoop { vertices: vec![p], edges: vec![e0] };
        used[e0 as usize] = true;
        let (mut e, mut cur) = (e0, q);
        let mut closed = false;
        for _ in 0..=m.edge_count() {
            if cur == p {
                closed = true;
                break;
            }
            let Some(ne) = next_boundary_edge(m, e, cur) else { break };
            if used[ne as usize] {
                break;
            }
            used[ne as usize] = true;
            lp.vertices.push(cur);
            lp.edges.push(ne);
            cur = m.edge_other_vertex(ne, cur);
            e = ne;
        }
        if closed {
            loops.push(lp);
        }
    }
    loops
}

/// Summary of topological defects (used by the Inspector).
#[derive(Clone, Debug, Default)]
pub struct DefectReport {
    pub boundary_loops: Vec<EdgeLoop>,
    pub bowtie_vertices: Vec<u32>,
    pub components: Vec<Vec<u32>>,
    pub degenerate_triangles: Vec<u32>,
    pub inconsistent_edges: Vec<u32>,
}

pub fn find_defects(m: &Mesh, degenerate_area: f64) -> DefectReport {
    let boundary_loops = boundary_loops(m);
    let bowtie_vertices = m.vertex_ids().filter(|&v| m.is_bowtie_vertex(v)).collect();
    let components = connected_components(m);
    let degenerate_triangles = m.triangle_ids().filter(|&t| m.tri_area(t) <= degenerate_area).collect();
    let inconsistent_edges = m
        .edge_ids()
        .filter(|&e| {
            let [t0, t1] = m.edge_triangles(e);
            if t1 == INVALID {
                return false;
            }
            let [a, b] = m.edge_vertices(e);
            let tv0 = m.triangle(t0);
            let tv1 = m.triangle(t1);
            let j0 = tri_edge_index(tv0, a, b).unwrap();
            let j1 = tri_edge_index(tv1, a, b).unwrap();
            tv0[j0] != tv1[(j1 + 1) % 3]
        })
        .collect();
    DefectReport { boundary_loops, bowtie_vertices, components, degenerate_triangles, inconsistent_edges }
}

/// Make triangle orientation consistent within each connected component by flood fill,
/// then orient each closed component outward (positive volume).
/// Returns the number of triangles flipped.
pub fn orient_consistently(m: &mut Mesh) -> usize {
    let mut flipped = 0;
    let mut visited = vec![false; m.max_tid()];
    let comps = connected_components(m);
    for comp in comps {
        let seed = comp[0];
        visited[seed as usize] = true;
        let mut stack = vec![seed];
        while let Some(t) = stack.pop() {
            for e in m.tri_edges(t) {
                let n = m.edge_other_triangle(e, t);
                if n == INVALID || visited[n as usize] {
                    continue;
                }
                let [a, b] = m.edge_vertices(e);
                let tv0 = m.triangle(t);
                let tv1 = m.triangle(n);
                let j0 = tri_edge_index(tv0, a, b).unwrap();
                let j1 = tri_edge_index(tv1, a, b).unwrap();
                if tv0[j0] == tv1[j1] {
                    m.reverse_triangle(n);
                    flipped += 1;
                }
                visited[n as usize] = true;
                stack.push(n);
            }
        }
        // Outward orientation for each component: positive signed volume.
        let vol: f64 = comp
            .iter()
            .map(|&t| {
                let [a, b, c] = m.tri_positions(t);
                a.dot(b.cross(c))
            })
            .sum();
        if vol < 0.0 {
            for &t in &comp {
                m.reverse_triangle(t);
            }
            flipped += comp.len();
        }
    }
    flipped
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives;

    #[test]
    fn cylinder_has_two_loops() {
        let m = primitives::cylinder(1.0, 2.0, 16, 4, false);
        let loops = boundary_loops(&m);
        assert_eq!(loops.len(), 2);
        for l in &loops {
            assert_eq!(l.len(), 16);
        }
        assert_eq!(connected_components(&m).len(), 1);
    }

    #[test]
    fn components_of_two_spheres() {
        let mut a = primitives::icosphere(1.0, 1);
        let mut b = primitives::icosphere(1.0, 1);
        b.transform(|p| p + DVec3::X * 3.0);
        a.append_mesh(&b, 1);
        assert_eq!(connected_components(&a).len(), 2);
        assert!(boundary_loops(&a).is_empty());
    }

    #[test]
    fn bowtie_loops() {
        // Two triangles sharing only a vertex.
        let pos = vec![
            DVec3::ZERO,
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 1.0, 0.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, -1.0, 0.0),
        ];
        let (m, _) = Mesh::from_triangles(&pos, &[[0, 1, 2], [0, 3, 4]], None);
        assert!(m.is_bowtie_vertex(0));
        let loops = boundary_loops(&m);
        assert_eq!(loops.len(), 2);
        assert!(loops.iter().all(|l| l.len() == 3));
    }

    #[test]
    fn loops_on_inconsistently_oriented_mesh() {
        let mut m = primitives::cylinder(1.0, 2.0, 16, 4, false);
        let flip: Vec<u32> = m.triangle_ids().filter(|t| t % 3 == 0).collect();
        for t in flip {
            m.reverse_triangle(t);
        }
        assert!(!m.is_consistently_oriented());
        let loops = boundary_loops(&m);
        assert_eq!(loops.len(), 2);
        assert!(loops.iter().all(|l| l.len() == 16));
    }

    #[test]
    fn orient_fixes_flipped_triangles() {
        let mut m = primitives::icosphere(1.0, 1);
        let ids: Vec<u32> = m.triangle_ids().collect();
        // Rebuild with some triangles reversed.
        let (pos, mut tris, _) = m.to_arrays();
        for (i, t) in tris.iter_mut().enumerate() {
            if i % 3 == 0 {
                t.swap(1, 2);
            }
        }
        let _ = ids;
        let (mut m2, _) = Mesh::from_triangles(&pos, &tris, None);
        assert!(!m2.is_consistently_oriented());
        orient_consistently(&mut m2);
        assert!(m2.is_consistently_oriented());
        assert!(crate::measure::volume(&m2) > 0.0);
        m = m2;
        m.check_validity().unwrap();
    }
}
