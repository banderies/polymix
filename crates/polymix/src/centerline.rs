//! Vessel centerlines from geodesic level sets.
//!
//! Geodesic distance from the inlet is computed over the surface (Dijkstra on the edge
//! graph, multi-source from the inlet boundary/cap). The surface is cut into bands of
//! equal distance; every connected ring of vertices in a band becomes a centerline node
//! (area-weighted centroid, mean radius), and nodes of consecutive bands are linked when
//! their rings touch. On a branching tube this yields a tree whose nodes lie on the
//! vessel axes; bifurcations appear where one ring splits into several. The tree is split
//! into branches, short spurs from surface noise are pruned, and branches are smoothed.

use glam::DVec3;
use mm_core::{Mesh, INVALID};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};

#[derive(Clone, Copy, Debug)]
pub struct CenterlineNode {
    pub position: DVec3,
    pub radius: f64,
    /// Geodesic distance band index (0 at the inlet).
    pub band: usize,
}

#[derive(Clone, Debug)]
pub struct Branch {
    /// Node indices from the proximal (inlet side) end to the distal end.
    pub nodes: Vec<usize>,
    pub length: f64,
    pub mean_radius: f64,
    pub min_radius: f64,
    pub max_radius: f64,
    /// Index of the parent branch (None for the root).
    pub parent: Option<usize>,
}

#[derive(Clone, Debug, Default)]
pub struct Centerline {
    pub nodes: Vec<CenterlineNode>,
    pub edges: Vec<(usize, usize)>,
    pub branches: Vec<Branch>,
}

#[derive(Clone, Copy, Debug)]
pub struct CenterlineParams {
    /// Band width (geodesic distance between rings). <= 0: automatic (half the inlet radius).
    pub step: f64,
    /// Spurs shorter than this multiple of their radius are pruned.
    pub prune_factor: f64,
    pub smoothing_iterations: usize,
}

impl Default for CenterlineParams {
    fn default() -> Self {
        Self { step: 0.0, prune_factor: 2.5, smoothing_iterations: 3 }
    }
}

#[derive(PartialEq)]
struct Item(f64, u32);
impl Eq for Item {}
impl PartialOrd for Item {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for Item {
    fn cmp(&self, o: &Self) -> Ordering {
        o.0.total_cmp(&self.0).then(o.1.cmp(&self.1))
    }
}

/// Multi-source Dijkstra distances over mesh edges (indexed by vertex id; infinity if
/// unreachable).
pub fn geodesic_distance(mesh: &Mesh, sources: &[u32]) -> Vec<f64> {
    let mut dist = vec![f64::INFINITY; mesh.max_vid()];
    let mut heap = BinaryHeap::new();
    for &s in sources {
        if mesh.is_vertex(s) {
            dist[s as usize] = 0.0;
            heap.push(Item(0.0, s));
        }
    }
    while let Some(Item(d, v)) = heap.pop() {
        if d > dist[v as usize] {
            continue;
        }
        let p = mesh.vertex(v);
        for u in mesh.vertex_neighbors(v) {
            let nd = d + p.distance(mesh.vertex(u));
            if nd < dist[u as usize] {
                dist[u as usize] = nd;
                heap.push(Item(nd, u));
            }
        }
    }
    dist
}

fn find(parent: &mut [u32], mut x: u32) -> u32 {
    while parent[x as usize] != x {
        parent[x as usize] = parent[parent[x as usize] as usize];
        x = parent[x as usize];
    }
    x
}

/// Compute the centerline tree. `sources` are the inlet vertices (e.g. a boundary loop or
/// the vertices of an inlet cap); `inlet_radius` sets the automatic band width.
pub fn compute(mesh: &Mesh, sources: &[u32], inlet_radius: f64, params: &CenterlineParams) -> Centerline {
    let dist = geodesic_distance(mesh, sources);
    let step = if params.step > 0.0 { params.step } else { (inlet_radius * 0.5).max(mesh.average_edge_length() * 2.0) };
    let band = |v: u32| -> usize {
        let d = dist[v as usize];
        if d.is_finite() {
            (d / step) as usize
        } else {
            usize::MAX
        }
    };
    // Vertex areas for weighted centroids.
    let mut area = vec![0.0; mesh.max_vid()];
    for t in mesh.triangle_ids() {
        let a = mesh.tri_area(t) / 3.0;
        for v in mesh.triangle(t) {
            area[v as usize] += a;
        }
    }
    // Union-find of vertices within the same band.
    let mut parent: Vec<u32> = (0..mesh.max_vid() as u32).collect();
    for e in mesh.edge_ids() {
        let [a, b] = mesh.edge_vertices(e);
        if band(a) == band(b) && band(a) != usize::MAX {
            let (ra, rb) = (find(&mut parent, a), find(&mut parent, b));
            if ra != rb {
                parent[ra as usize] = rb;
            }
        }
    }
    // Components -> nodes.
    let mut comp_of_root: HashMap<u32, usize> = HashMap::new();
    let mut sums: Vec<(DVec3, f64, usize)> = Vec::new();
    let mut vcomp = vec![usize::MAX; mesh.max_vid()];
    for v in mesh.vertex_ids() {
        if band(v) == usize::MAX {
            continue;
        }
        let r = find(&mut parent, v);
        let c = *comp_of_root.entry(r).or_insert_with(|| {
            sums.push((DVec3::ZERO, 0.0, band(v)));
            sums.len() - 1
        });
        vcomp[v as usize] = c;
        let w = area[v as usize].max(1e-30);
        sums[c].0 += mesh.vertex(v) * w;
        sums[c].1 += w;
    }
    let mut nodes: Vec<CenterlineNode> = sums.iter().map(|&(s, w, b)| CenterlineNode { position: s / w, radius: 0.0, band: b }).collect();
    let mut rsum = vec![(0.0, 0.0); nodes.len()];
    for v in mesh.vertex_ids() {
        let c = vcomp[v as usize];
        if c != usize::MAX {
            let w = area[v as usize].max(1e-30);
            rsum[c].0 += mesh.vertex(v).distance(nodes[c].position) * w;
            rsum[c].1 += w;
        }
    }
    for (n, (s, w)) in nodes.iter_mut().zip(&rsum) {
        n.radius = if *w > 0.0 { s / w } else { 0.0 };
    }
    // Links between rings of consecutive bands.
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for e in mesh.edge_ids() {
        let [a, b] = mesh.edge_vertices(e);
        let (ca, cb) = (vcomp[a as usize], vcomp[b as usize]);
        if ca == usize::MAX || cb == usize::MAX || ca == cb {
            continue;
        }
        let (lo, hi) = if nodes[ca].band < nodes[cb].band { (ca, cb) } else { (cb, ca) };
        if nodes[hi].band == nodes[lo].band + 1 {
            edges.push((lo, hi));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    let mut cl = Centerline { nodes, edges, branches: Vec::new() };
    build_branches(&mut cl, params);
    cl
}

fn build_branches(cl: &mut Centerline, params: &CenterlineParams) {
    let n = cl.nodes.len();
    if n == 0 {
        return;
    }
    // Directed tree: children are nodes one band further.
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut has_parent = vec![false; n];
    for &(a, b) in &cl.edges {
        if !has_parent[b] {
            children[a].push(b);
            has_parent[b] = true;
        }
    }
    // Root: band-0 node with the most descendants.
    let roots: Vec<usize> = (0..n).filter(|&i| !has_parent[i]).collect();
    let count = |r: usize, children: &Vec<Vec<usize>>| {
        let mut stack = vec![r];
        let mut k = 0;
        while let Some(x) = stack.pop() {
            k += 1;
            stack.extend(children[x].iter().copied());
        }
        k
    };
    let Some(&root) = roots.iter().max_by_key(|&&r| count(r, &children)) else { return };
    // Prune short spurs repeatedly.
    loop {
        let mut pruned = false;
        for i in 0..n {
            if children[i].len() < 2 {
                continue;
            }
            let kids = children[i].clone();
            for k in kids {
                // Walk the chain below k until a leaf or a junction.
                let mut chain = vec![k];
                let mut cur = k;
                while children[cur].len() == 1 {
                    cur = children[cur][0];
                    chain.push(cur);
                }
                if !children[cur].is_empty() {
                    continue;
                }
                let len: f64 = std::iter::once(i).chain(chain.iter().copied()).collect::<Vec<_>>().windows(2).map(|w| cl.nodes[w[0]].position.distance(cl.nodes[w[1]].position)).sum();
                let r = cl.nodes[i].radius.max(1e-12);
                if len < params.prune_factor * r {
                    children[i].retain(|&c| c != k);
                    pruned = true;
                }
            }
        }
        if !pruned {
            break;
        }
    }
    // Branches: chains between root/junctions and junctions/leaves.
    let mut branches: Vec<Branch> = Vec::new();
    let mut stack: Vec<(usize, Option<usize>)> = vec![(root, None)];
    while let Some((start, parent)) = stack.pop() {
        let mut nodes = vec![start];
        let mut cur = start;
        while children[cur].len() == 1 {
            cur = children[cur][0];
            nodes.push(cur);
        }
        let id = branches.len();
        branches.push(Branch { nodes, length: 0.0, mean_radius: 0.0, min_radius: 0.0, max_radius: 0.0, parent });
        for &c in &children[cur] {
            // Child branches start at the junction node so they connect.
            stack.push((c, Some(id)));
        }
        if let Some(p) = parent {
            let junction = *branches[p].nodes.last().unwrap();
            branches[id].nodes.insert(0, junction);
        }
    }
    // Smooth node positions along branches (endpoints fixed).
    for _ in 0..params.smoothing_iterations {
        let mut newpos: Vec<DVec3> = cl.nodes.iter().map(|n| n.position).collect();
        for b in &branches {
            for w in b.nodes.windows(3) {
                newpos[w[1]] = (cl.nodes[w[0]].position + cl.nodes[w[1]].position * 2.0 + cl.nodes[w[2]].position) / 4.0;
            }
        }
        for (n, p) in cl.nodes.iter_mut().zip(newpos) {
            n.position = p;
        }
    }
    for b in &mut branches {
        let pts: Vec<&CenterlineNode> = b.nodes.iter().map(|&i| &cl.nodes[i]).collect();
        b.length = pts.windows(2).map(|w| w[0].position.distance(w[1].position)).sum();
        // Radius stats exclude the shared junction node, whose ring spans several branches.
        let interior: Vec<f64> = pts.iter().skip(if b.parent.is_some() { 1 } else { 0 }).map(|n| n.radius).collect();
        let rs = if interior.is_empty() { vec![pts[0].radius] } else { interior };
        b.mean_radius = rs.iter().sum::<f64>() / rs.len() as f64;
        b.min_radius = rs.iter().copied().fold(f64::INFINITY, f64::min);
        b.max_radius = rs.iter().copied().fold(0.0, f64::max);
    }
    cl.branches = branches;
    let _ = INVALID;
}

impl Centerline {
    /// CSV: branch, node index along branch, x, y, z, radius, distance along branch.
    pub fn to_csv(&self) -> String {
        let mut s = String::from("branch,point,x,y,z,radius,arc_length\n");
        for (bi, b) in self.branches.iter().enumerate() {
            let mut arc = 0.0;
            for (k, &i) in b.nodes.iter().enumerate() {
                if k > 0 {
                    arc += self.nodes[b.nodes[k - 1]].position.distance(self.nodes[i].position);
                }
                let p = self.nodes[i].position;
                s += &format!("{bi},{k},{:.6},{:.6},{:.6},{:.6},{:.6}\n", p.x, p.y, p.z, self.nodes[i].radius, arc);
            }
        }
        s
    }

    /// Legacy ASCII VTK polydata (one polyline per branch, point data "Radius"),
    /// readable by ParaView and VMTK.
    pub fn to_vtk(&self) -> String {
        let mut pts: Vec<(DVec3, f64)> = Vec::new();
        let mut lines: Vec<Vec<usize>> = Vec::new();
        for b in &self.branches {
            let mut l = Vec::new();
            for &i in &b.nodes {
                l.push(pts.len());
                pts.push((self.nodes[i].position, self.nodes[i].radius));
            }
            lines.push(l);
        }
        let mut s = String::from("# vtk DataFile Version 3.0\nPolymix centerline\nASCII\nDATASET POLYDATA\n");
        s += &format!("POINTS {} double\n", pts.len());
        for (p, _) in &pts {
            s += &format!("{} {} {}\n", p.x, p.y, p.z);
        }
        let size: usize = lines.iter().map(|l| l.len() + 1).sum();
        s += &format!("LINES {} {}\n", lines.len(), size);
        for l in &lines {
            s += &l.len().to_string();
            for i in l {
                s += &format!(" {i}");
            }
            s += "\n";
        }
        s += &format!("POINT_DATA {}\nSCALARS Radius double 1\nLOOKUP_TABLE default\n", pts.len());
        for (_, r) in &pts {
            s += &format!("{r}\n");
        }
        s
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn straight_tube_centerline() {
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(0.0, 0.0, i as f64 * 0.5)).collect();
        let tube = primitives::tube_along(&path, &vec![1.0; path.len()], 32);
        let lp = mm_core::topology::boundary_loops(&tube).into_iter().min_by(|a, b| a.centroid(&tube).z.partial_cmp(&b.centroid(&tube).z).unwrap()).unwrap();
        let cl = compute(&tube, &lp.vertices, 1.0, &CenterlineParams::default());
        assert_eq!(cl.branches.len(), 1, "{:?}", cl.branches);
        let b = &cl.branches[0];
        assert!((b.length - 20.0).abs() < 2.0, "length {}", b.length);
        assert!((b.mean_radius - 1.0).abs() < 0.1, "radius {}", b.mean_radius);
        for &i in &b.nodes {
            let p = cl.nodes[i].position;
            assert!((p.x * p.x + p.y * p.y).sqrt() < 0.15, "off axis {p:?}");
        }
    }

    #[test]
    fn bifurcation_has_three_branches() {
        // Y made of three overlapping tubes, merged into one surface.
        let mut raw = Mesh::new();
        let trunk: Vec<DVec3> = (0..=20).map(|i| DVec3::new(0.0, 0.0, i as f64)).collect();
        let left: Vec<DVec3> = (0..=20).map(|i| DVec3::new(-(i as f64) * 0.6, 0.0, 20.0 + i as f64 * 0.8)).collect();
        let right: Vec<DVec3> = (0..=20).map(|i| DVec3::new(i as f64 * 0.6, 0.0, 20.0 + i as f64 * 0.8)).collect();
        raw.append_mesh(&primitives::tube_along(&trunk, &[2.0; 21], 32), 0);
        raw.append_mesh(&primitives::tube_along(&left, &[1.5; 21], 32), 0);
        raw.append_mesh(&primitives::tube_along(&right, &[1.5; 21], 32), 0);
        let params = mm_solid::MakeSolidParams::from_sliders(&raw, 0.45, 0.5);
        let mut solid = mm_solid::make_solid(&raw, &params, None).unwrap();
        // Open the inlet: cut the trunk near its start.
        let plane = mm_core::Plane::from_point_normal(DVec3::new(0.0, 0.0, 1.0), -DVec3::Z);
        mm_edit::plane_cut(&mut solid, plane, &mm_edit::PlaneCutParams { keep: mm_edit::KeepSide::Below, fill: mm_edit::CutFill::None, ..Default::default() }).unwrap();
        let lp = mm_core::topology::boundary_loops(&solid).into_iter().next().unwrap();
        let cl = compute(&solid, &lp.vertices, 2.0, &CenterlineParams::default());
        assert_eq!(cl.branches.len(), 3, "branches: {:?}", cl.branches.iter().map(|b| (b.length, b.mean_radius)).collect::<Vec<_>>());
        let mut radii: Vec<f64> = cl.branches.iter().map(|b| b.mean_radius).collect();
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((radii[2] - 2.0).abs() < 0.35, "trunk radius {radii:?}");
        assert!((radii[0] - 1.5).abs() < 0.35, "branch radius {radii:?}");
        let vtk = cl.to_vtk();
        assert!(vtk.contains("LINES 3"));
    }
}
