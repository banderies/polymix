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
use mm_core::{Bvh, Mesh, INVALID};
use std::cmp::Ordering;
use rayon::prelude::*;
use std::collections::BinaryHeap;
use std::sync::{Arc, Condvar, Mutex, MutexGuard, PoisonError};
use std::time::Duration;

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
    /// The node (ring) each vertex belongs to, by vertex id (`usize::MAX`: unreached).
    pub vertex_node: Vec<usize>,
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
    compute_with(mesh, sources, inlet_radius, params, None).expect("only a progress callback stops it")
}

/// [`compute`], reporting progress (0..1) to `progress` between its steps; `None` once
/// `progress` returns false (the work then stops there).
pub fn compute_with(mesh: &Mesh, sources: &[u32], inlet_radius: f64, params: &CenterlineParams, progress: Option<&dyn Fn(f32) -> bool>) -> Option<Centerline> {
    let going = |x: f32| progress.is_none_or(|p| p(x));
    let dist = geodesic_distance(mesh, sources);
    if !going(0.4) {
        return None;
    }
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
    if !going(0.6) {
        return None;
    }
    // Components -> nodes (numbered in the order their first vertex comes up).
    let mut comp_of_root = vec![u32::MAX; mesh.max_vid()];
    let mut sums: Vec<(DVec3, f64, usize)> = Vec::new();
    let mut vcomp = vec![usize::MAX; mesh.max_vid()];
    for v in mesh.vertex_ids() {
        if band(v) == usize::MAX {
            continue;
        }
        let r = find(&mut parent, v);
        if comp_of_root[r as usize] == u32::MAX {
            comp_of_root[r as usize] = sums.len() as u32;
            sums.push((DVec3::ZERO, 0.0, band(v)));
        }
        let c = comp_of_root[r as usize] as usize;
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
    if !going(0.8) {
        return None;
    }
    // Links between rings of later bands (consecutive ones, or further apart where an edge
    // is longer than a band on a coarse mesh).
    let mut edges: Vec<(usize, usize)> = Vec::new();
    for e in mesh.edge_ids() {
        let [a, b] = mesh.edge_vertices(e);
        let (ca, cb) = (vcomp[a as usize], vcomp[b as usize]);
        if ca == usize::MAX || cb == usize::MAX || ca == cb {
            continue;
        }
        let (lo, hi) = if nodes[ca].band < nodes[cb].band { (ca, cb) } else { (cb, ca) };
        if nodes[hi].band > nodes[lo].band {
            edges.push((lo, hi));
        }
    }
    edges.sort_unstable();
    edges.dedup();
    if !going(0.9) {
        return None;
    }
    let mut cl = Centerline { nodes, edges, branches: Vec::new(), vertex_node: vcomp };
    build_branches(&mut cl, params);
    Some(cl)
}

fn build_branches(cl: &mut Centerline, params: &CenterlineParams) {
    let n = cl.nodes.len();
    if n == 0 {
        return;
    }
    // Directed tree: children are nodes one band further (or further on, where a coarse
    // mesh skips bands and there is no nearer parent).
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];
    let mut has_parent = vec![false; n];
    let mut links = cl.edges.clone();
    links.sort_by_key(|&(a, b)| (cl.nodes[b].band - cl.nodes[a].band, a, b));
    for &(a, b) in &links {
        if !has_parent[b] {
            children[a].push(b);
            has_parent[b] = true;
        }
    }
    // Root: band-0 node with the most descendants.
    let roots: Vec<usize> = match (0..n).filter(|&i| !has_parent[i] && cl.nodes[i].band == 0).collect::<Vec<_>>() {
        r if !r.is_empty() => r,
        _ => (0..n).filter(|&i| !has_parent[i]).collect(),
    };
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

/// A centerline through the centres of the largest spheres inside the lumen, with their radii.
#[derive(Clone, Debug, Default)]
pub struct InscribedPath {
    pub points: Vec<DVec3>,
    pub radii: Vec<f64>,
}

/// The lumen's centerline from near `a` to near `b` through the centres of the largest spheres
/// inside it, as VMTK's Voronoi centerlines run, with those spheres' radii. Geodesic rings'
/// centroids can lie millimetres off the lumen's centre, even outside it, where a vessel bends
/// or narrows eccentrically; these centres cannot.
///
/// Found on a voxel grid: the lumen is voxelized around `guide` (a rough course from `a` to
/// `b`, such as a traced centerline, with the vessel's radius along it in `radii`), each voxel
/// gets its distance to the wall, and the path is the cheapest between the deepest voxels near
/// `a` and `b`, each step costing 1 / distance², so it keeps to the middle. It is smoothed, and
/// its radii are the exact distances to the wall. Only voxels within a few radii of the guide
/// are searched, so the path cannot wander into another vessel. `None` where the lumen does not
/// connect `a` to `b` at the grid's resolution.
///
/// Caps a copy of the mesh and builds its tree each time: for several paths through one mesh,
/// see [`inscribed_path_with`].
#[cfg_attr(not(test), allow(dead_code))]
pub fn inscribed_path(mesh: &Mesh, guide: &[DVec3], radii: &[f64], a: DVec3, b: DVec3) -> Option<InscribedPath> {
    inscribed_path_with(mesh, &CappedCache::default(), guide, radii, a, b, None)
}

/// The surface with its open ends capped and a tree over it, as [`inscribed_path`] casts its
/// rays and measures its depths against.
struct Capped {
    mesh: Mesh,
    bvh: Bvh,
}

/// A single-entry cache of [`inscribed_path`]'s capped surface: the capping and its tree, the
/// slow part of a path through a big mesh, are built once per shape of the mesh (its shape
/// version, which no other shape shares) and kept for the next path through it, or the
/// failure to cap it. Clones share the entry, so a background job can fill it for the tool
/// that keeps it. It holds a copy of the mesh (hundreds of MB for millions of triangles):
/// keep it only while paths are being traced.
#[derive(Clone, Default)]
pub struct CappedCache(Arc<CappedShared>);

#[derive(Default)]
struct CappedShared {
    slot: Mutex<CappedSlot>,
    /// Woken when a build ends (kept, cancelled or failed).
    built: Condvar,
}

#[derive(Default)]
struct CappedSlot {
    entry: CappedEntry,
    /// Builds started so far, to tell whose build the entry waits for.
    claims: u64,
}

#[derive(Default)]
enum CappedEntry {
    #[default]
    Empty,
    /// Being built for a shape version, by the caller holding `claim`.
    Building { version: u64, claim: u64 },
    /// Built for a shape version (`None`: its open ends could not be capped).
    Ready { version: u64, capped: Option<Arc<Capped>> },
}

impl CappedCache {
    fn lock(&self) -> MutexGuard<'_, CappedSlot> {
        self.0.slot.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The capped surface of `mesh` (`None` where its open ends cannot be capped, or once
    /// `going` returns false). Built now if the entry is for another shape, outside the lock:
    /// a caller for another shape does not wait for it (the build it replaces is not kept),
    /// and callers for the same shape wait for this one build. `going` (0..1 of the build)
    /// is asked between the build's steps and while waiting, so a caller no longer
    /// interested stops early.
    fn get(&self, mesh: &Mesh, going: &dyn Fn(f32) -> bool) -> Option<Arc<Capped>> {
        let version = mesh.shape_version();
        let mut slot = self.lock();
        loop {
            match &slot.entry {
                CappedEntry::Ready { version: v, capped } if *v == version => return capped.clone(),
                CappedEntry::Building { version: v, .. } if *v == version => {
                    if !going(0.0) {
                        return None;
                    }
                    slot = self.0.built.wait_timeout(slot, Duration::from_millis(20)).unwrap_or_else(PoisonError::into_inner).0;
                }
                _ => break,
            }
        }
        slot.claims += 1;
        let claim = slot.claims;
        let old = std::mem::replace(&mut slot.entry, CappedEntry::Building { version, claim });
        drop(slot);
        // The old surface is let go before this one is built, so two are not held at once
        // (unless another caller still holds it).
        drop(old);
        let unfinished = Unfinished { cache: self, claim };
        let built = Self::build(mesh, going)?;
        let capped = built.map(Arc::new);
        let mut slot = self.lock();
        if matches!(slot.entry, CappedEntry::Building { claim: c, .. } if c == claim) {
            slot.entry = CappedEntry::Ready { version, capped: capped.clone() };
        }
        drop(slot);
        drop(unfinished);
        capped
    }

    /// `mesh` capped, with its tree (`Some(None)`: it cannot be capped); `None` once `going`
    /// returns false.
    fn build(mesh: &Mesh, going: &dyn Fn(f32) -> bool) -> Option<Option<Capped>> {
        if !going(0.0) {
            return None;
        }
        let mut closed = mesh.clone();
        let loops: Vec<Vec<u32>> = mm_core::topology::boundary_loops(&closed).into_iter().map(|l| l.vertices).collect();
        if !loops.is_empty() && mm_edit::cap_planar_loops(&mut closed, &loops, None, mm_edit::CutFill::Flat).is_err() {
            return Some(None);
        }
        if !going(0.3) {
            return None;
        }
        let bvh = Bvh::build(&closed);
        Some(Some(Capped { mesh: closed, bvh }))
    }
}

/// A claim on the cache's entry while its build runs: when it ends without the build kept
/// (cancelled, or a panic), the entry is emptied if it still waits for this build, and the
/// callers waiting for it are woken either way.
struct Unfinished<'a> {
    cache: &'a CappedCache,
    claim: u64,
}

impl Drop for Unfinished<'_> {
    fn drop(&mut self) {
        let mut slot = self.cache.lock();
        if matches!(slot.entry, CappedEntry::Building { claim, .. } if claim == self.claim) {
            slot.entry = CappedEntry::Empty;
        }
        drop(slot);
        self.cache.0.built.notify_all();
    }
}

/// [`inscribed_path`], taking the capped surface from `cache` (or filling it), and reporting
/// progress (0..1) to `progress` between its steps: `None` once that returns false.
pub fn inscribed_path_with(mesh: &Mesh, cache: &CappedCache, guide: &[DVec3], radii: &[f64], a: DVec3, b: DVec3, progress: Option<&dyn Fn(f32) -> bool>) -> Option<InscribedPath> {
    use mm_core::Ray;
    let going = |x: f32| progress.is_none_or(|p| p(x));
    if guide.len() < 2 || radii.len() != guide.len() {
        return None;
    }
    // The searched tube around the guide, and the voxel size: fine enough for the narrowest
    // stretch, coarse enough for the grid to stay small.
    let reach: Vec<f64> = radii.iter().map(|&r| 2.0 * r + 2.0).collect();
    let rmin = radii.iter().copied().fold(f64::INFINITY, f64::min).max(1e-3);
    let (lo, hi) = guide.iter().zip(&reach).fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(l, h), (&p, &d)| (l.min(p - DVec3::splat(d)), h.max(p + DVec3::splat(d))));
    let size = hi - lo;
    let mut v = (rmin / 8.0).clamp(0.05, 0.25);
    while (size.x / v) * (size.y / v) * (size.z / v) > 8.0e6 {
        v *= 1.2;
    }
    let dims = [(size.x / v).ceil() as usize + 1, (size.y / v).ceil() as usize + 1, (size.z / v).ceil() as usize + 1];
    let [nx, ny, nz] = dims;
    let n = nx * ny * nz;
    let at = |i: usize, j: usize, k: usize| lo + DVec3::new(i as f64 + 0.5, j as f64 + 0.5, k as f64 + 0.5) * v;
    let index = |i: usize, j: usize, k: usize| (k * ny + j) * nx + i;
    // The tube: voxels within reach of the guide.
    let mut tube = vec![false; n];
    for w in 0..guide.len() - 1 {
        let (p, q, d) = (guide[w], guide[w + 1], reach[w].max(reach[w + 1]));
        let (bl, bh) = ((p.min(q) - DVec3::splat(d) - lo) / v, (p.max(q) + DVec3::splat(d) - lo) / v);
        let range = |a: f64, b: f64, m: usize| (a.floor().max(0.0) as usize)..((b.ceil() as usize).min(m - 1) + 1);
        for k in range(bl.z, bh.z, nz) {
            for j in range(bl.y, bh.y, ny) {
                for i in range(bl.x, bh.x, nx) {
                    let c = at(i, j, k);
                    let pq = q - p;
                    let t = ((c - p).dot(pq) / pq.length_squared().max(1e-300)).clamp(0.0, 1.0);
                    if c.distance(p + pq * t) < d {
                        tube[index(i, j, k)] = true;
                    }
                }
            }
        }
    }
    // Inside the lumen, by the parity of crossings along x from outside the mesh (its open
    // ends capped, so rays cannot slip out through them).
    // (Building the capped surface, when it is not kept yet, takes most of the time.)
    let capped = cache.get(mesh, &|x| going(0.5 * x))?;
    let (closed, bvh) = (&capped.mesh, &capped.bvh);
    if !going(0.5) {
        return None;
    }
    let start_x = closed.bounds().min.x - 1.0;
    let rows: Vec<(usize, usize)> = (0..nz).flat_map(|k| (0..ny).map(move |j| (j, k))).filter(|&(j, k)| (0..nx).any(|i| tube[index(i, j, k)])).collect();
    let inside_rows: Vec<Vec<(f64, f64)>> = rows
        .par_iter()
        .map(|&(j, k)| {
            // Nudged off the grid, so a ray does not run exactly along an edge.
            let c = at(0, j, k) + DVec3::new(0.0, 0.000_37 * v, 0.000_61 * v);
            let mut hits = Vec::new();
            bvh.raycast_all(closed, &Ray::new(DVec3::new(start_x, c.y, c.z), DVec3::X), &mut hits);
            let mut ts: Vec<f64> = hits.iter().map(|h| start_x + h.t).collect();
            ts.sort_by(f64::total_cmp);
            ts.dedup_by(|x, y| (*x - *y).abs() < 1e-9);
            ts.as_chunks::<2>().0.iter().map(|&[a, b]| (a, b)).collect()
        })
        .collect();
    let mut outside = vec![1u8; n];
    for (&(j, k), spans) in rows.iter().zip(&inside_rows) {
        for &(x0, x1) in spans {
            let (i0, i1) = (((x0 - lo.x) / v - 0.5).ceil().max(0.0) as usize, ((x1 - lo.x) / v - 0.5).floor());
            if i1 < 0.0 {
                continue;
            }
            for i in i0..=(i1 as usize).min(nx - 1) {
                outside[index(i, j, k)] = 0;
            }
        }
    }
    if !going(0.7) {
        return None;
    }
    let depth: Vec<f32> = mm_segment::ops::distance_sq(dims, &outside, [v; 3]).into_iter().map(|d| d.sqrt() as f32).collect();
    if !going(0.75) {
        return None;
    }
    // The deepest lumen voxel near a point.
    let deepest = |p: DVec3, r: f64| -> Option<usize> {
        let (bl, bh) = ((p - DVec3::splat(r) - lo) / v, (p + DVec3::splat(r) - lo) / v);
        let range = |a: f64, b: f64, m: usize| (a.floor().max(0.0) as usize)..((b.ceil().max(0.0) as usize).min(m - 1) + 1);
        let mut best: Option<(f32, usize)> = None;
        for k in range(bl.z, bh.z, nz) {
            for j in range(bl.y, bh.y, ny) {
                for i in range(bl.x, bh.x, nx) {
                    let x = index(i, j, k);
                    if outside[x] == 0 && tube[x] && at(i, j, k).distance(p) <= r && best.is_none_or(|b| depth[x] > b.0) {
                        best = Some((depth[x], x));
                    }
                }
            }
        }
        best.map(|b| b.1)
    };
    let near = |p: DVec3| {
        let (i, _) = guide.iter().enumerate().min_by(|x, y| x.1.distance(p).total_cmp(&y.1.distance(p)))?;
        let r = radii[i];
        deepest(p, (0.5 * r).max(2.0 * v)).or_else(|| deepest(p, 2.0 * r + v))
    };
    let (from, to) = (near(a)?, near(b)?);
    // Cheapest path, each step costing its length over the squared depth.
    let mut cost = vec![f32::INFINITY; n];
    let mut prev = vec![u32::MAX; n];
    let mut heap = BinaryHeap::new();
    cost[from] = 0.0;
    heap.push(Item(0.0, from as u32));
    let steps: Vec<(isize, isize, isize, f32)> = (-1..=1).flat_map(|dk| (-1..=1).flat_map(move |dj| (-1..=1).map(move |di| (di, dj, dk)))).filter(|&d| d != (0, 0, 0)).map(|(di, dj, dk)| (di, dj, dk, ((di * di + dj * dj + dk * dk) as f32).sqrt() * v as f32)).collect();
    let floor = (0.25 * v) as f32;
    while let Some(Item(c, x)) = heap.pop() {
        let x = x as usize;
        if c as f32 > cost[x] {
            continue;
        }
        if x == to {
            break;
        }
        let (i, j, k) = (x % nx, (x / nx) % ny, x / (nx * ny));
        for &(di, dj, dk, len) in &steps {
            let (a, b, cc) = (i as isize + di, j as isize + dj, k as isize + dk);
            if a < 0 || b < 0 || cc < 0 || a >= nx as isize || b >= ny as isize || cc >= nz as isize {
                continue;
            }
            let y = index(a as usize, b as usize, cc as usize);
            if outside[y] != 0 || !tube[y] {
                continue;
            }
            let d = depth[y].max(floor);
            let nc = cost[x] + len / (d * d);
            if nc < cost[y] {
                cost[y] = nc;
                prev[y] = x as u32;
                heap.push(Item(nc as f64, y as u32));
            }
        }
    }
    if (prev[to] == u32::MAX && to != from) || !going(0.85) {
        return None;
    }
    let mut cells = vec![to];
    while *cells.last().unwrap() != from {
        cells.push(prev[*cells.last().unwrap()] as usize);
    }
    cells.reverse();
    let raw: Vec<DVec3> = cells.iter().map(|&x| at(x % nx, (x / nx) % ny, x / (nx * ny))).collect();
    let deep: Vec<f64> = cells.iter().map(|&x| depth[x] as f64).collect();
    // Smoothed over about half the local depth (the voxel path is a staircase), unless that
    // takes a point much nearer the wall.
    let wall = |p: DVec3| bvh.nearest(closed, p, f64::INFINITY).map_or(0.0, |h| h.dist_sq.sqrt());
    let m = raw.len();
    let mut pts = raw.clone();
    for _ in 0..3 {
        let old = pts.clone();
        for i in 1..m.saturating_sub(1) {
            let half = ((0.5 * deep[i] / v).round() as usize).clamp(1, 10).min(i).min(m - 1 - i);
            pts[i] = old[i - half..=i + half].iter().sum::<DVec3>() / (2 * half + 1) as f64;
        }
    }
    let pts: Vec<DVec3> = (0..m).into_par_iter().map(|i| if wall(pts[i]) < 0.6 * deep[i] { raw[i] } else { pts[i] }).collect();
    // Each point then climbs, within its cross-section, to where the wall is farthest (steps
    // halving from a voxel), so the centring is finer than the grid.
    let nearest = |p: DVec3| bvh.nearest(closed, p, f64::INFINITY).map(|h| h.point);
    let points: Vec<DVec3> = (0..m)
        .into_par_iter()
        .map(|i| {
            let mut p = pts[i];
            if i == 0 || i + 1 == m {
                return p;
            }
            let t = (pts[i + 1] - pts[i - 1]).normalize_or_zero();
            let mut d = wall(p);
            let mut step = v;
            for _ in 0..24 {
                let Some(q) = nearest(p) else { break };
                let away = p - q;
                let g = (away - t * away.dot(t)).normalize_or_zero();
                let trial = p + g * step;
                let dt = wall(trial);
                if g != DVec3::ZERO && dt > d {
                    (p, d) = (trial, dt);
                } else {
                    step *= 0.5;
                    if step < 0.01 * v {
                        break;
                    }
                }
            }
            p
        })
        .collect();
    let radii: Vec<f64> = points.par_iter().map(|&p| wall(p)).collect();
    Some(InscribedPath { points, radii })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn inscribed_path_finds_an_eccentric_narrowing() {
        // The lumen narrows to radius 0.8 at z = 30 while its centre moves 2.5 to the side; the
        // guide runs straight along z, outside the lumen there.
        let bump = |z: f64| (-((z - 30.0) / 4.0).powi(2)).exp();
        let centre = move |z: f64| DVec3::new(2.5 * bump(z), 0.0, z);
        let radius = move |z: f64| 3.0 - 2.2 * bump(z);
        let zs: Vec<f64> = (0..=240).map(|k| 0.25 * k as f64).collect();
        let m = mm_core::primitives::tube_along(&zs.iter().map(|&z| centre(z)).collect::<Vec<_>>(), &zs.iter().map(|&z| radius(z)).collect::<Vec<_>>(), 64);
        let guide: Vec<DVec3> = (8..=52).map(|z| DVec3::new(0.0, 0.0, z as f64)).collect();
        let path = inscribed_path(&m, &guide, &vec![3.0; guide.len()], centre(16.0), centre(44.0)).expect("found");
        assert!(path.points.len() > 20);
        for (p, &r) in path.points.iter().zip(&path.radii) {
            if (20.0..40.0).contains(&p.z) {
                let off = p.truncate().distance(centre(p.z).truncate());
                assert!(off < 0.25 * radius(p.z), "{off:.2} off the centre at z = {:.1} (radius {:.2})", p.z, radius(p.z));
                // The sphere fits the section, and in a narrowing, bending lumen touches the
                // narrower wall nearby, so it is somewhat smaller.
                assert!(r < radius(p.z) + 0.05 && r > 0.7 * radius(p.z), "radius {r:.2} at z = {:.1} (section radius {:.2})", p.z, radius(p.z));
            }
        }
        let narrowest = path.radii.iter().copied().fold(f64::INFINITY, f64::min);
        assert!((narrowest - 0.8).abs() < 0.1, "narrowest {narrowest:.2}");
    }

    #[test]
    fn a_kept_capping_gives_the_same_path() {
        let bump = |z: f64| (-((z - 30.0) / 4.0).powi(2)).exp();
        let centre = move |z: f64| DVec3::new(2.5 * bump(z), 0.0, z);
        let zs: Vec<f64> = (0..=240).map(|k| 0.25 * k as f64).collect();
        let m = mm_core::primitives::tube_along(&zs.iter().map(|&z| centre(z)).collect::<Vec<_>>(), &zs.iter().map(|&z| 3.0 - 2.2 * bump(z)).collect::<Vec<_>>(), 48);
        let guide: Vec<DVec3> = (8..=52).map(|z| DVec3::new(0.0, 0.0, z as f64)).collect();
        let radii = vec![3.0; guide.len()];
        let bits = |p: &InscribedPath| (p.points.iter().map(|q| q.to_array().map(f64::to_bits)).collect::<Vec<_>>(), p.radii.iter().map(|r| r.to_bits()).collect::<Vec<_>>());
        let cache = CappedCache::default();
        // Filled by the first path, used by the next (other ends too); each the uncached path.
        for (a, b) in [(16.0, 44.0), (20.0, 40.0), (16.0, 44.0)] {
            let kept = inscribed_path_with(&m, &cache, &guide, &radii, centre(a), centre(b), None).unwrap();
            assert_eq!(bits(&kept), bits(&inscribed_path(&m, &guide, &radii, centre(a), centre(b)).unwrap()));
        }
        let always = |_: f32| true;
        let first = cache.get(&m, &always).unwrap();
        assert!(Arc::ptr_eq(&first, &cache.get(&m, &always).unwrap()), "kept, not built again");
        // Another shape takes the entry's place.
        let mut moved = m.clone();
        moved.transform(|p| p + DVec3::X);
        let shifted: Vec<DVec3> = guide.iter().map(|&p| p + DVec3::X).collect();
        let kept = inscribed_path_with(&moved, &cache, &shifted, &radii, centre(16.0) + DVec3::X, centre(44.0) + DVec3::X, None).unwrap();
        assert_eq!(bits(&kept), bits(&inscribed_path(&moved, &shifted, &radii, centre(16.0) + DVec3::X, centre(44.0) + DVec3::X).unwrap()));
        assert!(!Arc::ptr_eq(&first, &cache.get(&moved, &always).unwrap()));
        // A path no longer wanted stops (at each step, building or not) and keeps nothing.
        // (The capped surface is built over 0..0.15, and kept once built.)
        for stop_at in [0.0, 0.1, 0.4, 0.6, 0.72, 0.8] {
            let fresh = CappedCache::default();
            let going = |x: f32| x < stop_at;
            assert!(inscribed_path_with(&m, &fresh, &guide, &radii, centre(16.0), centre(44.0), Some(&going)).is_none(), "stopped at {stop_at}");
            let entry = &fresh.lock().entry;
            assert!(if stop_at < 0.15 { matches!(entry, CappedEntry::Empty) } else { matches!(entry, CappedEntry::Ready { capped: Some(_), .. }) });
        }
    }

    #[test]
    fn the_capped_cache_is_built_outside_its_lock_once_per_shape() {
        let m = primitives::tube_along(&[DVec3::ZERO, DVec3::Z * 10.0], &[2.0, 2.0], 24);
        let mut moved = m.clone();
        moved.transform(|p| p + DVec3::X);
        let always = |_: f32| true;
        let cache = CappedCache::default();
        let kept = |mesh: &Mesh| matches!(&cache.lock().entry, CappedEntry::Ready { version, .. } if *version == mesh.shape_version());
        type Held<'s> = (std::thread::ScopedJoinHandle<'s, Option<Arc<Capped>>>, std::sync::mpsc::Sender<()>);
        // A build of `mesh` held part way, on a thread of its own: released by sending on the
        // channel returned.
        fn held<'s>(s: &'s std::thread::Scope<'s, '_>, cache: &CappedCache, mesh: &Mesh) -> Held<'s> {
            let (at_tx, at_rx) = std::sync::mpsc::channel::<()>();
            let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
            let mesh = mesh.clone();
            let cache = cache.clone();
            let h = s.spawn(move || {
                let hold = |x: f32| {
                    if x > 0.0 {
                        let _ = at_tx.send(());
                        let _ = go_rx.recv();
                    }
                    true
                };
                cache.get(&mesh, &hold)
            });
            at_rx.recv().unwrap();
            (h, go_tx)
        }
        std::thread::scope(|s| {
            // A caller for the same shape waits for the build under way and gets what it built.
            let (a, go) = held(s, &cache, &m);
            let waiter = s.spawn(|| cache.get(&m, &always));
            std::thread::sleep(Duration::from_millis(50));
            go.send(()).unwrap();
            let built = a.join().unwrap().unwrap();
            assert!(Arc::ptr_eq(&built, &waiter.join().unwrap().unwrap()));
            assert!(kept(&m));
            // A caller for another shape does not wait for a build under way (held until after
            // it), and the build it replaced is not kept.
            let (b, go) = held(s, &cache, &moved);
            let other = cache.get(&m, &always).unwrap();
            assert!(!Arc::ptr_eq(&other, &built));
            go.send(()).unwrap();
            assert!(b.join().unwrap().is_some());
            assert!(kept(&m) && Arc::ptr_eq(&other, &cache.get(&m, &always).unwrap()));
        });
        // A cancelled build leaves the entry empty; the next caller builds it.
        assert!(cache.get(&moved, &|x: f32| x < 0.2).is_none());
        assert!(matches!(cache.lock().entry, CappedEntry::Empty));
        assert!(cache.get(&moved, &always).is_some() && kept(&moved));
    }

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
