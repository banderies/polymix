//! Soft Transform: move a selection with a smooth transition into the surrounding surface.
//!
//! [`soft_region`] splits the vertices into the *handle* (moved exactly by the user's
//! transform), the *transition band* within a geodesic falloff distance, and the rest
//! (fixed). Without a fixed boundary the handle is the selection and the band lies outside
//! it; with a fixed boundary the selection's border stays put and the band lies inside the
//! selection. [`soft_deform`] then fills in the band:
//!
//! * [`SoftKind::Linear`] blends each band vertex's displacement by its falloff weight,
//!   eased according to [`Continuity`];
//! * [`SoftKind::Flat`] re-solves the band as a membrane (or thin plate for the soft
//!   continuities) between the moved handle and the fixed surface, so it comes out smooth;
//! * [`SoftKind::Stretchy`] solves a harmonic (or biharmonic) displacement field instead,
//!   so the band's surface detail is kept and stretched;
//! * [`SoftKind::Nonlinear`] solves for an as-rigid-as-possible band (local rotations fitted
//!   to each one-ring, alternated with a global Laplacian solve), which keeps detail
//!   undistorted when the handle turns.

use mm_core::linalg::closest_rotation;
use mm_core::{DMat3, DVec3, Mesh, INVALID};
use mm_repair::sparse::{solve_pcg, CsrMatrix, PcgParams};
use mm_repair::LaplacianWeights;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// How the transition band is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SoftKind {
    /// Blend the displacement by the falloff weight.
    #[default]
    Linear,
    /// As-rigid-as-possible band.
    Nonlinear,
    /// Smooth membrane / thin-plate band (surface detail in the band is smoothed away).
    Flat,
    /// Harmonic / biharmonic displacement (surface detail is kept and stretched).
    Stretchy,
}

/// Smoothness where the band meets the fixed surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Continuity {
    /// Position only (a crease may form at the band's edge).
    Boundaries,
    /// Smooth start: the band leaves the fixed surface tangentially.
    #[default]
    SoftPosition,
    /// Smoother still (zero curvature change at the band's edges).
    SoftTangent,
}

/// Vertex roles for a soft transform.
#[derive(Clone, Debug, Default)]
pub struct SoftRegion {
    /// Vertices moved exactly by their piece's transform.
    pub handle: Vec<u32>,
    pub handle_piece: Vec<u32>,
    /// Transition vertices, their falloff weight (1 at the handle, 0 at the fixed side) and
    /// the piece whose transform they follow.
    pub band: Vec<u32>,
    pub band_weight: Vec<f64>,
    pub band_piece: Vec<u32>,
}

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

/// Is edge `e` a face-group border?
fn group_border(m: &Mesh, e: u32) -> bool {
    let [a, b] = m.edge_triangles(e);
    a != INVALID && b != INVALID && m.tri_group(a) != m.tri_group(b)
}

/// Multi-source Dijkstra along edges up to `limit`; `step(e)` vetoes edges. Returns
/// distance and source label per reached vertex.
fn geodesic(m: &Mesh, sources: &[(u32, u32)], limit: f64, step: impl Fn(u32, u32) -> bool) -> HashMap<u32, (f64, u32)> {
    let mut best: HashMap<u32, (f64, u32)> = HashMap::new();
    let mut heap = BinaryHeap::new();
    for &(v, label) in sources {
        best.insert(v, (0.0, label));
        heap.push(Item(0.0, v));
    }
    while let Some(Item(d, v)) = heap.pop() {
        let (dv, label) = best[&v];
        if d > dv {
            continue;
        }
        let pv = m.vertex(v);
        for &e in m.vertex_edges(v) {
            let [a, b] = m.edge_vertices(e);
            let u = if a == v { b } else { a };
            if !step(e, u) {
                continue;
            }
            let nd = d + pv.distance(m.vertex(u));
            if nd <= limit && nd < best.get(&u).map_or(f64::INFINITY, |x| x.0) {
                best.insert(u, (nd, label));
                heap.push(Item(nd, u));
            }
        }
    }
    best
}

/// Split `selection` (triangles) into handle and transition band. `falloff` is the band's
/// geodesic width. With `shared_frame == false` each connected piece of the selection gets
/// its own label (so it can follow its own transform); `ignore_groups == false` keeps the
/// band from spreading across face-group borders.
pub fn soft_region(mesh: &Mesh, selection: &[u32], falloff: f64, fixed_boundary: bool, ignore_groups: bool, shared_frame: bool) -> SoftRegion {
    let selected: HashSet<u32> = selection.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
    let mut region = SoftRegion::default();
    if selected.is_empty() {
        return region;
    }
    // Piece label per selected vertex.
    let mut piece: HashMap<u32, u32> = HashMap::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut label = 0;
    let mut order: Vec<u32> = selected.iter().copied().collect();
    order.sort_unstable();
    for &t0 in &order {
        if !seen.insert(t0) {
            continue;
        }
        let mut stack = vec![t0];
        while let Some(t) = stack.pop() {
            for v in mesh.triangle(t) {
                piece.entry(v).or_insert(label);
            }
            for n in mesh.tri_neighbors(t) {
                if n != INVALID && selected.contains(&n) && seen.insert(n) {
                    stack.push(n);
                }
            }
        }
        if !shared_frame {
            label += 1;
        }
    }
    let in_selection = |v: u32| mesh.vertex_triangles(v).iter().all(|t| selected.contains(t));
    let falloff = falloff.max(0.0);
    if !fixed_boundary {
        let mut handle: Vec<(u32, u32)> = piece.iter().map(|(&v, &p)| (v, p)).collect();
        handle.sort_unstable();
        let reach = geodesic(mesh, &handle, falloff, |e, u| !piece.contains_key(&u) && (ignore_groups || !group_border(mesh, e)));
        for &(v, p) in &handle {
            region.handle.push(v);
            region.handle_piece.push(p);
        }
        let mut band: Vec<(u32, f64, u32)> = reach.into_iter().filter(|(v, _)| !piece.contains_key(v)).map(|(v, (d, p))| (v, d, p)).collect();
        band.sort_unstable_by_key(|b| b.0);
        for (v, d, p) in band {
            if d < falloff {
                region.band.push(v);
                region.band_weight.push(1.0 - d / falloff.max(1e-300));
                region.band_piece.push(p);
            }
        }
    } else {
        // Distance from the selection's border, inside the selection.
        let mut border: Vec<(u32, u32)> = piece.iter().filter(|(&v, _)| !in_selection(v)).map(|(&v, &p)| (v, p)).collect();
        border.sort_unstable();
        let inside = geodesic(mesh, &border, f64::INFINITY, |e, u| {
            let [a, b] = mesh.edge_triangles(e);
            piece.contains_key(&u) && (selected.contains(&a) || (b != INVALID && selected.contains(&b)))
        });
        let deepest = inside.values().map(|x| x.0).fold(0.0, f64::max);
        let width = falloff.min(0.999 * deepest).max(1e-300);
        let mut verts: Vec<(&u32, &(f64, u32))> = inside.iter().collect();
        verts.sort_unstable_by_key(|x| *x.0);
        for (&v, &(d, _)) in verts {
            let p = piece[&v];
            if d >= width {
                region.handle.push(v);
                region.handle_piece.push(p);
            } else if d > 0.0 {
                region.band.push(v);
                region.band_weight.push(d / width);
                region.band_piece.push(p);
            }
        }
    }
    region
}

fn ease(w: f64, c: Continuity) -> f64 {
    let w = w.clamp(0.0, 1.0);
    match c {
        Continuity::Boundaries => w,
        Continuity::SoftPosition => w * w * (3.0 - 2.0 * w),
        Continuity::SoftTangent => w * w * w * (w * (6.0 * w - 15.0) + 10.0),
    }
}

/// New positions for the handle and band vertices of `region` on `mesh` (unchanged source
/// positions), where `transform(piece, p)` moves a point with that piece's transform.
pub fn soft_deform(mesh: &Mesh, region: &SoftRegion, transform: &dyn Fn(u32, DVec3) -> DVec3, kind: SoftKind, continuity: Continuity) -> Vec<(u32, DVec3)> {
    let mut out: Vec<(u32, DVec3)> = region.handle.iter().zip(&region.handle_piece).map(|(&v, &p)| (v, transform(p, mesh.vertex(v)))).collect();
    // Linear blend: also the starting point of the other kinds.
    let linear: Vec<(u32, DVec3)> = region
        .band
        .iter()
        .zip(region.band_weight.iter().zip(&region.band_piece))
        .map(|(&v, (&w, &p))| {
            let x = mesh.vertex(v);
            (v, x + (transform(p, x) - x) * ease(w, continuity))
        })
        .collect();
    if region.band.is_empty() || kind == SoftKind::Linear {
        out.extend(linear);
        return out;
    }
    let order = if continuity == Continuity::Boundaries { 1 } else { 2 };
    let band: Vec<u32> = region.band.clone();
    match kind {
        SoftKind::Flat => {
            let mut m = mesh.clone();
            for &(v, p) in &out {
                m.set_vertex(v, p);
            }
            for &(v, p) in &linear {
                m.set_vertex(v, p);
            }
            if mm_repair::fair_region(&mut m, &band, order, LaplacianWeights::Cotangent).is_err() {
                out.extend(linear);
                return out;
            }
            out.extend(band.iter().map(|&v| (v, m.vertex(v))));
        }
        SoftKind::Stretchy => {
            // Fair the displacement field: positions of a scratch copy hold displacements.
            let mut m = mesh.clone();
            for v in mesh.vertex_ids() {
                m.set_vertex(v, DVec3::ZERO);
            }
            for &(v, p) in &out {
                m.set_vertex(v, p - mesh.vertex(v));
            }
            if mm_repair::fair_region(&mut m, &band, order, LaplacianWeights::Uniform).is_err() {
                out.extend(linear);
                return out;
            }
            out.extend(band.iter().map(|&v| (v, mesh.vertex(v) + m.vertex(v))));
        }
        SoftKind::Nonlinear => {
            let mut pos: HashMap<u32, DVec3> = out.iter().copied().chain(linear.iter().copied()).collect();
            arap(mesh, &band, &mut pos, 8);
            out.extend(band.iter().map(|&v| (v, pos[&v])));
        }
        SoftKind::Linear => unreachable!(),
    }
    out
}

/// Clamped cotangent weight of edge `e` on the source mesh.
fn edge_weight(m: &Mesh, e: u32) -> f64 {
    let [a, b] = m.edge_vertices(e);
    let (pa, pb) = (m.vertex(a), m.vertex(b));
    let mut w = 0.0;
    for o in m.edge_opposite_vertices(e) {
        if o == INVALID {
            continue;
        }
        let po = m.vertex(o);
        let (u, v) = (pa - po, pb - po);
        w += 0.5 * u.dot(v) / u.cross(v).length().max(1e-300);
    }
    w.max(1e-3)
}

/// As-rigid-as-possible solve for the `free` vertices; every other vertex keeps the
/// position in `pos` (or its source position).
fn arap(m: &Mesh, free: &[u32], pos: &mut HashMap<u32, DVec3>, iterations: usize) {
    let index: HashMap<u32, usize> = free.iter().enumerate().map(|(i, &v)| (v, i)).collect();
    let n = free.len();
    // Vertices whose rotations matter: the free ones and their neighbours.
    let mut involved: Vec<u32> = free.to_vec();
    for &v in free {
        involved.extend(m.vertex_neighbors(v).filter(|u| !index.contains_key(u)));
    }
    involved.sort_unstable();
    involved.dedup();
    let current = |pos: &HashMap<u32, DVec3>, v: u32| pos.get(&v).copied().unwrap_or_else(|| m.vertex(v));
    let mut weights: HashMap<u32, f64> = HashMap::new();
    let mut w_of = |e: u32| *weights.entry(e).or_insert_with(|| edge_weight(m, e));
    // System matrix on the free vertices.
    let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    let mut edges_of: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
    for (i, &v) in free.iter().enumerate() {
        let mut diag = 0.0;
        for &e in m.vertex_edges(v) {
            let [a, b] = m.edge_vertices(e);
            let u = if a == v { b } else { a };
            let w = w_of(e);
            diag += w;
            edges_of[i].push((u, w));
            if let Some(&j) = index.get(&u) {
                rows[i].push((j as u32, -w));
            }
        }
        rows[i].push((i as u32, diag));
    }
    let a = CsrMatrix::from_rows(n, n, rows);
    let mut rot: HashMap<u32, DMat3> = HashMap::new();
    let mut x: Vec<DVec3> = free.iter().map(|&v| current(pos, v)).collect();
    for _ in 0..iterations {
        for &v in &involved {
            let pv = m.vertex(v);
            let qv = current(pos, v);
            let mut s = DMat3::ZERO;
            for &e in m.vertex_edges(v) {
                let [a, b] = m.edge_vertices(e);
                let u = if a == v { b } else { a };
                let w = w_of(e);
                let e0 = pv - m.vertex(u);
                let e1 = qv - current(pos, u);
                s += DMat3::from_cols(e1 * e0.x, e1 * e0.y, e1 * e0.z) * w;
            }
            rot.insert(v, closest_rotation(s));
        }
        let mut b = vec![DVec3::ZERO; n];
        for (i, &v) in free.iter().enumerate() {
            let pv = m.vertex(v);
            let rv = rot[&v];
            for &(u, w) in &edges_of[i] {
                let ru = rot.get(&u).copied().unwrap_or(rv);
                b[i] += (rv + ru) * (pv - m.vertex(u)) * (0.5 * w);
                if !index.contains_key(&u) {
                    b[i] += current(pos, u) * w;
                }
            }
        }
        solve_pcg(&a, &b, &mut x, &PcgParams { tolerance: 1e-7, max_iterations: 2000 });
        for (i, &v) in free.iter().enumerate() {
            pos.insert(v, x[i]);
        }
    }
}

/// Warp: deform the `selection` (triangles) so that each handle vertex reaches its target,
/// keeping the selection's border and everything outside fixed. The region is solved as a
/// smooth (biharmonic) displacement; `rigid` refines it as-rigid-as-possible so surface
/// detail keeps its shape. Returns new positions for the moved vertices.
pub fn warp(mesh: &Mesh, selection: &[u32], handles: &[(u32, DVec3)], rigid: bool) -> Vec<(u32, DVec3)> {
    let selected: HashSet<u32> = selection.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
    let pinned: HashSet<u32> = handles.iter().map(|h| h.0).collect();
    let mut free: Vec<u32> = selected
        .iter()
        .flat_map(|&t| mesh.triangle(t))
        .filter(|&v| !pinned.contains(&v) && mesh.vertex_triangles(v).iter().all(|t| selected.contains(t)))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut out: Vec<(u32, DVec3)> = handles.to_vec();
    if handles.is_empty() || free.is_empty() {
        return out;
    }
    // Smooth start: biharmonic displacement (positions of a scratch copy hold displacements).
    let mut d = mesh.clone();
    for v in mesh.vertex_ids() {
        d.set_vertex(v, DVec3::ZERO);
    }
    for &(v, p) in handles {
        d.set_vertex(v, p - mesh.vertex(v));
    }
    let _ = mm_repair::fair_region(&mut d, &free, 2, LaplacianWeights::Uniform);
    let mut pos: HashMap<u32, DVec3> = free.iter().map(|&v| (v, mesh.vertex(v) + d.vertex(v))).collect();
    pos.extend(handles.iter().copied());
    if rigid {
        arap(mesh, &free, &mut pos, 6);
    }
    out.extend(free.iter().map(|&v| (v, pos[&v])));
    out
}
