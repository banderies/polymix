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
//!
//! [`soft_deform_refined`] is the "Refine Mesh" option: it splits the edges the deformation
//! stretches, so the moved region keeps its triangle size.

use mm_core::hash::{FastMap, FastSet};
use mm_core::linalg::closest_rotation;
use mm_core::{DMat3, DVec3, Mesh, INVALID};
use mm_repair::laplacian::{fair_positions, FairSolver};
use mm_repair::sparse::{solve_pcg_preconditioned, CsrMatrix, Jacobi, PcgParams};
use mm_repair::LaplacianWeights;
use rayon::prelude::*;
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
///
/// For many transforms on the same region (the moves of a drag), [`SoftSolver`] gives the
/// same results without redoing the work that does not depend on the transform.
pub fn soft_deform(mesh: &Mesh, region: &SoftRegion, transform: &dyn Fn(u32, DVec3) -> DVec3, kind: SoftKind, continuity: Continuity) -> Vec<(u32, DVec3)> {
    SoftSolver::new(mesh, region, kind, continuity).deform(mesh, region, transform)
}

/// Solver settings of the fairing solves: those of [`mm_repair::fair_region`].
fn fair_pcg() -> PcgParams {
    let d = mm_repair::FairParams::default();
    PcgParams { tolerance: d.tolerance, max_iterations: d.max_iterations }
}

/// Where a vertex of a [`SoftRegion`] sits in its lists.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Slot {
    Handle(u32),
    Band(u32),
}

/// Vertex → its (last) place in the handle or band list; a band entry wins over a handle
/// entry of the same vertex, as the band's positions are written after the handle's.
fn region_slots(region: &SoftRegion) -> FastMap<u32, Slot> {
    let mut slots = mm_core::hash::fast_map_with_capacity(region.handle.len() + region.band.len());
    for (i, &v) in region.handle.iter().enumerate() {
        slots.insert(v, Slot::Handle(i as u32));
    }
    for (i, &v) in region.band.iter().enumerate() {
        slots.insert(v, Slot::Band(i as u32));
    }
    slots
}

/// [`soft_deform`] prepared for one mesh, region, kind and continuity, to be run for any
/// number of transforms: the moves of a drag. What does not depend on the transform is
/// worked out once: the as-rigid-as-possible system with its weights and index maps
/// ([`SoftKind::Nonlinear`]), the displacement field's matrix and preconditioner
/// ([`SoftKind::Stretchy`]), and the region's vertex lookup. Each [`SoftSolver::deform`]
/// returns exactly what [`soft_deform`] returns for the same arguments.
pub struct SoftSolver {
    kind: SoftKind,
    continuity: Continuity,
    /// Fairing order of the Flat and Stretchy kinds.
    order: u32,
    slots: FastMap<u32, Slot>,
    solved: Solved,
}

enum Solved {
    /// Linear blend only (also when the band is empty).
    Blend,
    Nonlinear(Arap),
    /// Cotangent weights follow the moved positions: assembled on every move.
    Flat,
    Stretchy {
        /// The displacement field's system, unless the region's handle and band overlap
        /// (then the band's start values differ from move to move and it is assembled every
        /// time).
        system: Option<FairSolver>,
        /// Handle vertex → its (last) place in the handle list.
        handles: FastMap<u32, u32>,
    },
}

impl SoftSolver {
    /// Prepare for [`SoftSolver::deform`] on `mesh` (unchanged source positions) and `region`.
    pub fn new(mesh: &Mesh, region: &SoftRegion, kind: SoftKind, continuity: Continuity) -> Self {
        let order = if continuity == Continuity::Boundaries { 1 } else { 2 };
        let slots = region_slots(region);
        let solved = if region.band.is_empty() {
            Solved::Blend
        } else {
            match kind {
                SoftKind::Linear => Solved::Blend,
                SoftKind::Nonlinear => Solved::Nonlinear(Arap::new(mesh, &region.band)),
                SoftKind::Flat => Solved::Flat,
                SoftKind::Stretchy => {
                    let handles: FastMap<u32, u32> = region.handle.iter().enumerate().map(|(i, &v)| (v, i as u32)).collect();
                    // The band starts at zero displacement on every move (unless a band vertex is
                    // also a handle vertex), so the system and its near-null space stay the same.
                    let overlap = region.band.iter().any(|v| handles.contains_key(v));
                    let system = if overlap { None } else { FairSolver::new(mesh, &region.band, order, LaplacianWeights::Uniform, &|_| DVec3::ZERO).ok() };
                    Solved::Stretchy { system, handles }
                }
            }
        };
        Self { kind, continuity, order, slots, solved }
    }

    /// The kind and continuity this solver was prepared for.
    pub fn settings(&self) -> (SoftKind, Continuity) {
        (self.kind, self.continuity)
    }

    /// [`soft_deform`] with the prepared kind and continuity; `mesh` and `region` must be the
    /// ones given to [`SoftSolver::new`].
    pub fn deform(&self, mesh: &Mesh, region: &SoftRegion, transform: &dyn Fn(u32, DVec3) -> DVec3) -> Vec<(u32, DVec3)> {
        let continuity = self.continuity;
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
        if region.band.is_empty() || self.kind == SoftKind::Linear {
            out.extend(linear);
            return out;
        }
        // The positions the band is solved against: moved handle, blended band, the rest as is.
        let at = |v: u32| match self.slots.get(&v) {
            Some(Slot::Handle(i)) => out[*i as usize].1,
            Some(Slot::Band(i)) => linear[*i as usize].1,
            None => mesh.vertex(v),
        };
        let band = &region.band;
        let new_band: Vec<(u32, DVec3)> = match &self.solved {
            Solved::Blend => unreachable!("the solved kinds have a band"),
            Solved::Flat => {
                let Ok(sol) = fair_positions(mesh, band, self.order, LaplacianWeights::Cotangent, &at, &fair_pcg()) else {
                    out.extend(linear);
                    return out;
                };
                // Band vertices left out of the solve keep their blended position.
                let solved = self.band_values(band.len(), &sol.free, sol.positions);
                band.iter().map(|&v| (v, self.band_value(&solved, v).unwrap_or_else(|| at(v)))).collect()
            }
            Solved::Stretchy { system, handles } => {
                // Fair the displacement field: zero everywhere but on the moved handle.
                let disp = |v: u32| match handles.get(&v) {
                    Some(&i) => out[i as usize].1 - mesh.vertex(v),
                    None => DVec3::ZERO,
                };
                let res = match system {
                    Some(s) => s.solve(&disp, &fair_pcg()).map(|(x, _)| (s.free().to_vec(), x)),
                    None => fair_positions(mesh, band, self.order, LaplacianWeights::Uniform, &disp, &fair_pcg()).map(|s| (s.free, s.positions)),
                };
                let Ok((free, x)) = res else {
                    out.extend(linear);
                    return out;
                };
                let solved = self.band_values(band.len(), &free, x);
                band.iter().map(|&v| (v, mesh.vertex(v) + self.band_value(&solved, v).unwrap_or_else(|| disp(v)))).collect()
            }
            Solved::Nonlinear(arap) => {
                let x = arap.solve(&at, 8);
                band.iter().copied().zip(x).collect()
            }
        };
        out.extend(new_band);
        out
    }

    /// Solved values by band slot (`free` is a subset of the band).
    fn band_values(&self, n: usize, free: &[u32], x: Vec<DVec3>) -> Vec<Option<DVec3>> {
        let mut solved = vec![None; n];
        for (&v, p) in free.iter().zip(x) {
            if let Some(Slot::Band(i)) = self.slots.get(&v) {
                solved[*i as usize] = Some(p);
            }
        }
        solved
    }

    /// The solved value of band vertex `v`, if it was solved for.
    fn band_value(&self, solved: &[Option<DVec3>], v: u32) -> Option<DVec3> {
        match self.slots.get(&v) {
            Some(Slot::Band(i)) => solved[*i as usize],
            _ => None,
        }
    }
}

/// How [`soft_deform_refined`] solves and refines.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RefineOptions {
    pub kind: SoftKind,
    pub continuity: Continuity,
    /// The region was made with a fixed boundary (see [`soft_region`]).
    pub fixed_boundary: bool,
    /// Stretched edges longer than this are split (see [`refine_length`]).
    pub max_edge: f64,
}

/// Result of [`soft_deform_refined`].
#[derive(Clone, Debug)]
pub struct RefinedSoft {
    /// The refined mesh, deformed.
    pub mesh: Mesh,
    /// The selection on the refined mesh (both halves of a split triangle keep its membership).
    pub selection: Vec<u32>,
    /// Handle and band on the refined mesh (rest pose), with the original pieces' labels.
    pub region: SoftRegion,
    /// The vertex budget stopped the refinement before every stretched edge was split.
    pub capped: bool,
}

/// Splitting rounds: each halves the edges still too long, so stretches up to about 2⁸ are
/// refined.
const REFINE_ROUNDS: usize = 8;

/// Further splitting rounds when solving the new band vertices stretches edges the plan left
/// alone.
const REFINE_CHECKS: usize = 3;

/// Edges longer than this many times the region's mean edge length are split while
/// refining (so they come out between about 2/3 and 4/3 of it).
pub const REFINE_FACTOR: f64 = 4.0 / 3.0;

/// The edge length above which Refine Mesh splits a stretched edge: [`REFINE_FACTOR`] times the
/// mean length of the edges of `region` on `mesh` (at rest), or of the whole mesh if the region
/// is empty.
pub fn refine_length(mesh: &Mesh, region: &SoftRegion) -> f64 {
    let verts: HashSet<u32> = region.handle.iter().chain(&region.band).copied().collect();
    let edges: HashSet<u32> = verts.iter().flat_map(|&v| mesh.vertex_edges(v).iter().copied()).collect();
    let mean = if edges.is_empty() {
        mesh.average_edge_length()
    } else {
        edges.iter().map(|&e| mesh.edge_length(e)).sum::<f64>() / edges.len() as f64
    };
    REFINE_FACTOR * mean
}

/// A vertex's part in a soft transform: handle (weight 1) or band vertex, and its piece.
#[derive(Clone, Copy, Debug)]
struct Role {
    handle: bool,
    weight: f64,
    piece: u32,
}

/// The moving vertices of a refined soft transform with their parts and deformed positions,
/// in dense slots (vertex ids index `slot`, which grows with the mesh).
#[derive(Default)]
struct Moving {
    slot: Vec<u32>,
    keys: Vec<u32>,
    roles: Vec<Role>,
    pos: Vec<DVec3>,
}

impl Moving {
    #[inline]
    fn get(&self, v: u32) -> Option<usize> {
        self.slot.get(v as usize).copied().filter(|&s| s != INVALID).map(|s| s as usize)
    }
    #[inline]
    fn pos(&self, v: u32) -> Option<DVec3> {
        self.get(v).map(|s| self.pos[s])
    }
    #[inline]
    fn role(&self, v: u32) -> Option<Role> {
        self.get(v).map(|s| self.roles[s])
    }
    #[inline]
    fn contains(&self, v: u32) -> bool {
        self.get(v).is_some()
    }
    fn len(&self) -> usize {
        self.keys.len()
    }
    /// Give `v` a part and a position (replacing what it had).
    fn set(&mut self, v: u32, role: Role, p: DVec3) {
        if let Some(s) = self.get(v) {
            self.roles[s] = role;
            self.pos[s] = p;
            return;
        }
        if self.slot.len() <= v as usize {
            self.slot.resize(v as usize + 1, INVALID);
        }
        self.slot[v as usize] = self.keys.len() as u32;
        self.keys.push(v);
        self.roles.push(role);
        self.pos.push(p);
    }
    /// Move `v`, which must already have a part.
    fn set_pos(&mut self, v: u32, p: DVec3) {
        let s = self.get(v).expect("a moving vertex");
        self.pos[s] = p;
    }
}

fn region_from_roles(moving: &Moving) -> SoftRegion {
    let mut verts: Vec<u32> = moving.keys.clone();
    verts.sort_unstable();
    let mut r = SoftRegion::default();
    for v in verts {
        let role = moving.role(v).unwrap();
        if role.handle {
            r.handle.push(v);
            r.handle_piece.push(role.piece);
        } else {
            r.band.push(v);
            r.band_weight.push(role.weight);
            r.band_piece.push(role.piece);
        }
    }
    r
}

/// Soft Transform with Meshmixer's "Refine Mesh": the deformation [`soft_deform`] gives
/// `region` (from [`soft_region`] on `mesh` and `selection`), on a copy of `mesh` whose edges
/// are split (at their rest-pose midpoints) wherever the deformation stretches them past
/// `opts.max_edge`, so the triangles of the moved region stay roughly the size they were.
/// Edges the deformation does not lengthen are left alone.
///
/// The original vertices end exactly where [`soft_deform`] puts them, so turning refinement
/// on changes the shape only between them; fixed boundaries, group borders and separate
/// pieces behave as without it. A new vertex takes its part from the split edge: it joins the
/// handle where the selection covers it (and moves with its piece), otherwise the band with
/// the mean of its ends' weights (a fixed boundary's border stays fixed). New band vertices
/// are blended like the rest for [`SoftKind::Linear`]; for the solved kinds the splits are
/// planned on the interpolated displacement and the new band vertices are then solved between
/// the original ones (as rigidly as possible for [`SoftKind::Nonlinear`], as a smooth
/// displacement for the others), with a few re-checks. Splits keep face groups,
/// and both halves of a selected triangle stay selected.
pub fn soft_deform_refined(mesh: &Mesh, region: &SoftRegion, selection: &[u32], transform: &dyn Fn(u32, DVec3) -> DVec3, opts: &RefineOptions) -> RefinedSoft {
    soft_deform_refined_in(mesh.clone(), mesh, region, selection, transform, opts, None)
}

/// [`soft_deform_refined`] refining `rest`, a copy of `mesh` the caller made (so it can come
/// with room to grow, or be a copy the caller had anyway), with the unrefined deformation
/// worked out by `solver` when one prepared for `mesh`, `region` and the options' kind and
/// continuity is given. The result is the same.
pub fn soft_deform_refined_in(
    mut rest: Mesh,
    mesh: &Mesh,
    region: &SoftRegion,
    selection: &[u32],
    transform: &dyn Fn(u32, DVec3) -> DVec3,
    opts: &RefineOptions,
    solver: Option<&SoftSolver>,
) -> RefinedSoft {
    // Deformed positions of the moving vertices, starting with the unrefined result (its
    // handle entries, then its band entries, like the region's lists).
    let unrefined = match solver {
        Some(s) if s.settings() == (opts.kind, opts.continuity) => s.deform(mesh, region, transform),
        _ => soft_deform(mesh, region, transform, opts.kind, opts.continuity),
    };
    let mut moving = Moving { slot: vec![INVALID; mesh.max_vid()], ..Default::default() };
    let nh = region.handle.len();
    for (i, (&v, &piece)) in region.handle.iter().zip(&region.handle_piece).enumerate() {
        moving.set(v, Role { handle: true, weight: 1.0, piece }, unrefined[i].1);
    }
    for (i, (&v, (&weight, &piece))) in region.band.iter().zip(region.band_weight.iter().zip(&region.band_piece)).enumerate() {
        moving.set(v, Role { handle: false, weight, piece }, unrefined[nh + i].1);
    }
    let budget = (16 * moving.len() + 1000).min(2_000_000);
    let mut selected: Vec<u32> = selection.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
    selected.sort_unstable();
    selected.dedup();
    let mut in_sel: FastSet<u32> = selected.iter().copied().collect();
    // New band vertices whose positions the solved kinds work out (all at once, after planning).
    let mut new_band: Vec<u32> = Vec::new();
    let linear = opts.kind == SoftKind::Linear;
    let (mut rounds, mut checks, mut added, mut capped, mut solved, mut fills) = (0, 0, 0usize, false, true, 0);
    let refine = opts.max_edge.is_finite() && opts.max_edge > 0.0;
    loop {
        // Checking a solve of the new band vertices (not the plan)?
        let check = solved && !linear && fills > 0;
        let more = refine && !capped && if check { checks < REFINE_CHECKS } else { rounds < REFINE_ROUNDS };
        let long = if more { stretched_edges(&rest, &moving, opts.max_edge) } else { Vec::new() };
        if long.is_empty() {
            if !solved {
                solve_new_band(&rest, &new_band, &mut moving, opts);
                solved = true;
                fills += 1;
                continue;
            }
            break;
        }
        if check {
            checks += 1;
        } else {
            rounds += 1;
        }
        let mut fresh: Vec<u32> = Vec::new();
        for e in long {
            if added >= budget {
                capped = true;
                break;
            }
            let [a, b] = rest.edge_vertices(e);
            let shift = |v: u32| moving.pos(v).map_or(DVec3::ZERO, |p| p - rest.vertex(v));
            let moved = 0.5 * (shift(a) + shift(b));
            let Ok(info) = rest.split_edge(e) else { continue };
            added += 1;
            fresh.push(info.v_new);
            let tris: Vec<u32> = info.t_orig.into_iter().filter(|&t| t != INVALID).collect();
            let covered = tris.iter().filter(|t| in_sel.contains(t)).count();
            for k in 0..2 {
                if info.t_new[k] != INVALID && in_sel.contains(&info.t_orig[k]) {
                    in_sel.insert(info.t_new[k]);
                    selected.push(info.t_new[k]);
                }
            }
            let (ra, rb) = (moving.role(info.v_a), moving.role(info.v_b));
            let w = |r: Option<Role>| r.map_or(0.0, |r| r.weight);
            let both_handle = matches!((ra, rb), (Some(x), Some(y)) if x.handle && y.handle && x.piece == y.piece);
            // Without a fixed boundary the handle is every vertex of a selected triangle; with
            // one, the selection's interior away from its border.
            let (handle, may_move) = if opts.fixed_boundary { (both_handle && covered == tris.len(), covered == tris.len()) } else { (covered > 0 && both_handle, true) };
            let weight = 0.5 * (w(ra) + w(rb));
            let piece = match (ra, rb) {
                (Some(x), Some(y)) if y.weight > x.weight => y.piece,
                (Some(x), _) | (None, Some(x)) => x.piece,
                (None, None) => 0,
            };
            let (f, x) = (info.v_new, rest.vertex(info.v_new));
            if handle {
                moving.set(f, Role { handle: true, weight: 1.0, piece }, transform(piece, x));
            } else if may_move && weight > 0.0 {
                let weight = weight.min(1.0);
                let role = Role { handle: false, weight, piece };
                if linear {
                    moving.set(f, role, x + (transform(piece, x) - x) * ease(weight, opts.continuity));
                } else {
                    // Planned on the interpolated displacement; solved once the splits are done.
                    moving.set(f, role, x + moved);
                    new_band.push(f);
                    solved = false;
                }
            }
        }
        delaunay_flips(&mut rest, &fresh, &in_sel, &|v| moving.contains(v));
    }
    if moving.len() > 0 {
        let pos = rest.positions_mut();
        for (&v, &p) in moving.keys.iter().zip(&moving.pos) {
            pos[v as usize] = p;
        }
        rest.bump_shape_version();
    }
    selected.sort_unstable();
    RefinedSoft { mesh: rest, selection: selected, region: region_from_roles(&moving), capped }
}

/// Work out the positions of the new band vertices `free` of a refined mesh (rest pose
/// `rest`), every other vertex staying where `pos` (or `rest`) has it: as-rigid-as-possible for
/// [`SoftKind::Nonlinear`], otherwise a smooth (biharmonic) displacement between the original
/// vertices' solved positions. (Re-solving a membrane on the stretched, refined band itself
/// would weigh its slivers by their cotangents and could overshoot.)
fn solve_new_band(rest: &Mesh, free: &[u32], moving: &mut Moving, opts: &RefineOptions) {
    if free.is_empty() {
        return;
    }
    match opts.kind {
        SoftKind::Linear => {}
        SoftKind::Nonlinear => {
            let x = Arap::new(rest, free).solve(&|v| moving.pos(v).unwrap_or_else(|| rest.vertex(v)), 8);
            for (&v, p) in free.iter().zip(x) {
                moving.set_pos(v, p);
            }
        }
        SoftKind::Flat | SoftKind::Stretchy => {
            // The displacement field: what the moving vertices moved, zero elsewhere.
            let disp = |v: u32| moving.pos(v).map_or(DVec3::ZERO, |p| p - rest.vertex(v));
            if let Ok(sol) = fair_positions(rest, free, 2, LaplacianWeights::Uniform, &disp, &fair_pcg()) {
                // Free vertices left out of the solve keep their displacement.
                let solved: FastMap<u32, DVec3> = sol.free.iter().copied().zip(sol.positions).collect();
                let new: Vec<DVec3> = free.iter().map(|&v| rest.vertex(v) + solved.get(&v).copied().unwrap_or_else(|| disp(v))).collect();
                for (&v, p) in free.iter().zip(new) {
                    moving.set_pos(v, p);
                }
            }
        }
    }
}

/// Cotangent of the angle between `u` and `v`.
fn cot(u: DVec3, v: DVec3) -> f64 {
    u.dot(v) / u.cross(v).length().max(1e-300)
}

/// Flip the edges around the new vertices `fresh` that break the Delaunay condition (so the
/// split fans do not leave slivers whose clamped weights would soak up a solved band's
/// stretch). Only edges with a `moving` end, in nearly flat quads whose two triangles share
/// face group and selection state, are flipped: the surface that stays put, the group
/// borders and the selection stay as they were.
fn delaunay_flips(m: &mut Mesh, fresh: &[u32], in_sel: &FastSet<u32>, moving: &dyn Fn(u32) -> bool) {
    let mut queue: Vec<u32> = fresh.iter().filter(|&&v| m.is_vertex(v)).flat_map(|&v| m.vertex_triangles(v)).flat_map(|t| m.tri_edges(t)).collect();
    queue.sort_unstable();
    queue.dedup();
    for _ in 0..4 {
        let mut next = Vec::new();
        for &e in &queue {
            if !m.is_edge(e) {
                continue;
            }
            let [t0, t1] = m.edge_triangles(e);
            if t1 == INVALID || m.tri_group(t0) != m.tri_group(t1) || in_sel.contains(&t0) != in_sel.contains(&t1) {
                continue;
            }
            let [a, b] = m.edge_vertices(e);
            let [c, d] = m.edge_opposite_vertices(e);
            if c == INVALID || d == INVALID || c == d || !(moving(a) || moving(b)) {
                continue;
            }
            let (pa, pb, pc, pd) = (m.vertex(a), m.vertex(b), m.vertex(c), m.vertex(d));
            if cot(pa - pc, pb - pc) + cot(pa - pd, pb - pd) >= -1e-9 {
                continue;
            }
            // Nearly flat, and the flipped pair keeps facing the same way.
            let (n0, n1) = (m.tri_normal(t0), m.tri_normal(t1));
            let (f0, f1) = ((pa - pc).cross(pd - pc).normalize_or_zero(), (pb - pd).cross(pc - pd).normalize_or_zero());
            let n = (n0 + n1).normalize_or_zero();
            if n0.dot(n1) < 0.94 || f0.dot(n).abs() < 0.9 || f1.dot(n).abs() < 0.9 || f0.dot(f1) < 0.9 {
                continue;
            }
            if let Ok(info) = m.flip_edge(e) {
                next.extend(info.t.iter().flat_map(|&t| m.tri_edges(t)).filter(|&x| x != e));
            }
        }
        if next.is_empty() {
            break;
        }
        next.sort_unstable();
        next.dedup();
        queue = next;
    }
}

/// Edges with a moved end (in `moving`) that the deformation makes longer than `max_edge` and
/// longer than at rest, longest first.
fn stretched_edges(m: &Mesh, moving: &Moving, max_edge: f64) -> Vec<u32> {
    let at = |v: u32| moving.pos(v).unwrap_or_else(|| m.vertex(v));
    let mut long: Vec<(f64, u32)> = Vec::new();
    for &v in &moving.keys {
        for &e in m.vertex_edges(v) {
            let [a, b] = m.edge_vertices(e);
            // Each edge once: from its first end that moves.
            if a != v && moving.contains(a) {
                continue;
            }
            let len = at(a).distance(at(b));
            // 1% slack, so rigidly moved edges (unchanged up to round-off) are never split.
            if len > max_edge && len > 1.01 * m.edge_length(e) {
                long.push((len, e));
            }
        }
    }
    long.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
    long.into_iter().map(|x| x.1).collect()
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

/// Lists at least this long are worked on in parallel by [`Arap::solve`].
const ARAP_PAR_MIN: usize = 1024;

/// An as-rigid-as-possible system on the rest pose of a mesh: local rotations fitted to each
/// one-ring, alternated with a global Laplacian solve for the free vertices. Everything that
/// depends only on the rest pose (index maps, clamped cotangent weights, rest edge vectors,
/// the matrix and its Jacobi preconditioner) is built once; [`Arap::solve`] runs the
/// iterations for given positions of the other vertices.
///
/// The vertices it touches get dense slots: first the "involved" ones (the free vertices and
/// their neighbours, whose rotations are fitted), in ascending id, then the neighbours of
/// those, whose positions are only read.
struct Arap {
    free: Vec<u32>,
    /// Vertex of each slot.
    vertex: Vec<u32>,
    /// Slots `0..involved` have rotations.
    involved: usize,
    /// Slot of each free vertex (in `free` order).
    free_slot: Vec<u32>,
    /// Whether a slot's vertex is free.
    is_free: Vec<bool>,
    /// Per involved slot, its edges in `vertex_edges` order: (neighbour slot, clamped
    /// cotangent weight, rest edge vector from the neighbour to the vertex).
    edge_ptr: Vec<usize>,
    edges: Vec<(u32, f64, DVec3)>,
    matrix: CsrMatrix,
    jacobi: Jacobi,
}

impl Arap {
    fn new(m: &Mesh, free: &[u32]) -> Self {
        // Free index of each free vertex (the last one if listed twice).
        let index: FastMap<u32, usize> = free.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let n = free.len();
        // Vertices whose rotations matter: the free ones and their neighbours.
        let mut involved: Vec<u32> = free.to_vec();
        for &v in free {
            involved.extend(m.vertex_neighbors(v).filter(|u| !index.contains_key(u)));
        }
        involved.sort_unstable();
        involved.dedup();
        let mut slot: FastMap<u32, u32> = mm_core::hash::fast_map_with_capacity(involved.len() * 2);
        let mut vertex = involved.clone();
        for (s, &v) in involved.iter().enumerate() {
            slot.insert(v, s as u32);
        }
        for &v in &involved {
            for u in m.vertex_neighbors(v) {
                if let std::collections::hash_map::Entry::Vacant(e) = slot.entry(u) {
                    e.insert(vertex.len() as u32);
                    vertex.push(u);
                }
            }
        }
        let mut weights: FastMap<u32, f64> = mm_core::hash::fast_map_with_capacity(involved.len() * 3);
        let mut edge_ptr = Vec::with_capacity(involved.len() + 1);
        edge_ptr.push(0);
        let mut edges = Vec::with_capacity(involved.len() * 6);
        for &v in &involved {
            let pv = m.vertex(v);
            for &e in m.vertex_edges(v) {
                let [a, b] = m.edge_vertices(e);
                let u = if a == v { b } else { a };
                let w = *weights.entry(e).or_insert_with(|| edge_weight(m, e));
                edges.push((slot[&u], w, pv - m.vertex(u)));
            }
            edge_ptr.push(edges.len());
        }
        let free_slot: Vec<u32> = free.iter().map(|v| slot[v]).collect();
        let mut is_free = vec![false; vertex.len()];
        for &s in &free_slot {
            is_free[s as usize] = true;
        }
        // System matrix on the free vertices.
        let mut rows: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n];
        for (i, &s) in free_slot.iter().enumerate() {
            let mut diag = 0.0;
            for &(u, w, _) in &edges[edge_ptr[s as usize]..edge_ptr[s as usize + 1]] {
                diag += w;
                if let Some(&j) = index.get(&vertex[u as usize]) {
                    rows[i].push((j as u32, -w));
                }
            }
            rows[i].push((i as u32, diag));
        }
        let matrix = CsrMatrix::from_rows(n, n, rows);
        let jacobi = Jacobi::new(&matrix);
        Self { free: free.to_vec(), vertex, involved: involved.len(), free_slot, is_free, edge_ptr, edges, matrix, jacobi }
    }

    /// Positions of the free vertices after `iterations` rounds, every other vertex `v` staying
    /// at `at(v)` (which also gives the free vertices' starting positions).
    fn solve(&self, at: &(dyn Fn(u32) -> DVec3 + Sync), iterations: usize) -> Vec<DVec3> {
        let n = self.free.len();
        let par = self.vertex.len() >= ARAP_PAR_MIN;
        // Current positions per slot.
        let mut q: Vec<DVec3> = if par { self.vertex.par_iter().map(|&v| at(v)).collect() } else { self.vertex.iter().map(|&v| at(v)).collect() };
        let mut x: Vec<DVec3> = self.free.iter().map(|&v| at(v)).collect();
        let mut rot = vec![DMat3::ZERO; self.involved];
        let mut b = vec![DVec3::ZERO; n];
        let params = PcgParams { tolerance: 1e-7, max_iterations: 2000 };
        for _ in 0..iterations {
            let q_ref: &[DVec3] = &q;
            let fit = |(s, r): (usize, &mut DMat3)| {
                let qv = q_ref[s];
                let mut sm = DMat3::ZERO;
                for &(u, w, e0) in &self.edges[self.edge_ptr[s]..self.edge_ptr[s + 1]] {
                    let e1 = qv - q_ref[u as usize];
                    sm += DMat3::from_cols(e1 * e0.x, e1 * e0.y, e1 * e0.z) * w;
                }
                *r = closest_rotation(sm);
            };
            if par {
                rot.par_iter_mut().enumerate().for_each(fit);
            } else {
                rot.iter_mut().enumerate().for_each(fit);
            }
            let rot_ref: &[DMat3] = &rot;
            let gather = |(i, bi): (usize, &mut DVec3)| {
                let s = self.free_slot[i] as usize;
                let rv = rot_ref[s];
                let mut acc = DVec3::ZERO;
                for &(u, w, e0) in &self.edges[self.edge_ptr[s]..self.edge_ptr[s + 1]] {
                    let ru = rot_ref.get(u as usize).copied().unwrap_or(rv);
                    acc += (rv + ru) * e0 * (0.5 * w);
                    if !self.is_free[u as usize] {
                        acc += q_ref[u as usize] * w;
                    }
                }
                *bi = acc;
            };
            if par {
                b.par_iter_mut().enumerate().for_each(gather);
            } else {
                b.iter_mut().enumerate().for_each(gather);
            }
            solve_pcg_preconditioned(&self.matrix, &b, &mut x, &params, &self.jacobi, None);
            for (&s, p) in self.free_slot.iter().zip(&x) {
                q[s as usize] = *p;
            }
        }
        self.free_slot.iter().map(|&s| q[s as usize]).collect()
    }
}

/// Warp: deform the `selection` (triangles) so that each handle vertex reaches its target,
/// keeping the selection's border and everything outside fixed. The region is solved as a
/// smooth (biharmonic) displacement; `rigid` refines it as-rigid-as-possible so surface
/// detail keeps its shape. Returns new positions for the moved vertices.
pub fn warp(mesh: &Mesh, selection: &[u32], handles: &[(u32, DVec3)], rigid: bool) -> Vec<(u32, DVec3)> {
    let selected: FastSet<u32> = selection.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
    // Handle targets (the last one where a vertex is given twice).
    let target: FastMap<u32, DVec3> = handles.iter().copied().collect();
    let mut free: Vec<u32> = selected
        .iter()
        .flat_map(|&t| mesh.triangle(t))
        .filter(|&v| !target.contains_key(&v) && mesh.vertex_triangles(v).iter().all(|t| selected.contains(t)))
        .collect();
    free.sort_unstable();
    free.dedup();
    let mut out: Vec<(u32, DVec3)> = handles.to_vec();
    if handles.is_empty() || free.is_empty() {
        return out;
    }
    // Smooth start: biharmonic displacement, zero but on the handles.
    let disp = |v: u32| target.get(&v).map_or(DVec3::ZERO, |p| *p - mesh.vertex(v));
    let mut start: Vec<DVec3> = free.iter().map(|&v| mesh.vertex(v) + disp(v)).collect();
    if let Ok(sol) = fair_positions(mesh, &free, 2, LaplacianWeights::Uniform, &disp, &fair_pcg()) {
        // The solved vertices are some of `free`, in its (ascending) order.
        let mut k = 0;
        for (&v, d) in sol.free.iter().zip(sol.positions) {
            while free[k] != v {
                k += 1;
            }
            start[k] = mesh.vertex(v) + d;
        }
    }
    if rigid {
        let at = |v: u32| match target.get(&v) {
            Some(p) => *p,
            None => free.binary_search(&v).map_or_else(|_| mesh.vertex(v), |i| start[i]),
        };
        let x = Arap::new(mesh, &free).solve(&at, 6);
        out.extend(free.iter().copied().zip(x));
    } else {
        out.extend(free.iter().copied().zip(start));
    }
    out
}

#[cfg(test)]
mod refine_tests {
    use super::*;
    use mm_core::primitives;

    /// A 6 × 6 grid of 0.2 squares and a disc of radius 0.8 selected in its middle.
    fn grid() -> (Mesh, Vec<u32>) {
        let m = primitives::plane_grid(6.0, 6.0, 30, 30);
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).truncate().length() < 0.8).collect();
        (m, sel)
    }

    fn apply(m: &Mesh, moves: &[(u32, DVec3)]) -> Mesh {
        let mut out = m.clone();
        for &(v, p) in moves {
            out.set_vertex(v, p);
        }
        out
    }

    fn longest_edge(m: &Mesh) -> f64 {
        m.edge_ids().map(|e| m.edge_length(e)).fold(0.0, f64::max)
    }

    #[test]
    fn refine_bounds_stretched_edges_and_keeps_the_shape() {
        let (m, sel) = grid();
        let lift = |_: u32, p: DVec3| p + DVec3::Z * 2.0;
        for kind in [SoftKind::Linear, SoftKind::Nonlinear, SoftKind::Flat, SoftKind::Stretchy] {
            for fixed_boundary in [false, true] {
                // (A fixed boundary's band lies inside the selection: a narrow one leaves a
                // handle wide enough not to be a point pull, which no refinement makes smooth.)
                let falloff = if fixed_boundary { 0.4 } else { 1.2 };
                let r = soft_region(&m, &sel, falloff, fixed_boundary, false, true);
                let max_edge = refine_length(&m, &r);
                assert!(max_edge > 0.2 * 2f64.sqrt(), "the grid's diagonals are not split at rest: {max_edge}");
                let plain = apply(&m, &soft_deform(&m, &r, &lift, kind, Continuity::SoftPosition));
                assert!(longest_edge(&plain) > 1.5 * max_edge, "{kind:?}: the lift stretches the band");
                let opts = RefineOptions { kind, continuity: Continuity::SoftPosition, fixed_boundary, max_edge };
                let out = soft_deform_refined(&m, &r, &sel, &lift, &opts);
                out.mesh.check_validity().unwrap();
                assert!(out.mesh.is_consistently_oriented());
                assert!(!out.capped && out.mesh.vertex_count() > m.vertex_count());
                // The linear blend is refined to the bound; a solved band is planned on the
                // interpolated displacement and then re-solved, which can stretch a few edges a
                // little past it after the capped re-checks.
                let longest = longest_edge(&out.mesh);
                let bound = if kind == SoftKind::Linear { 1.0 + 1e-9 } else { 1.25 };
                assert!(longest <= max_edge * bound, "{kind:?} fixed {fixed_boundary}: edge {longest} > {max_edge}");
                // The original vertices land exactly where they do without refinement.
                let tol = 1e-12;
                let worst = m.vertex_ids().map(|v| out.mesh.vertex(v).distance(plain.vertex(v))).fold(0.0, f64::max);
                assert!(worst <= tol, "{kind:?} fixed {fixed_boundary}: original vertices moved {worst} from the unrefined result");
                // Vertices outside the region (new ones included) stay in the grid's plane.
                let moving: HashSet<u32> = out.region.handle.iter().chain(&out.region.band).copied().collect();
                assert!(out.mesh.vertex_ids().filter(|v| !moving.contains(v)).all(|v| out.mesh.vertex(v).z == 0.0));
                // The selection keeps its split triangles (with a fixed boundary the band, which
                // stretches, lies inside it).
                assert!(out.selection.len() >= sel.len() && (out.selection.len() > sel.len()) == fixed_boundary);
                assert!(out.selection.iter().all(|&t| out.mesh.is_triangle(t)));
                if fixed_boundary {
                    // The selection's border (and everything outside it) stays put.
                    let inside: HashSet<u32> = out.selection.iter().copied().collect();
                    for v in out.mesh.vertex_ids() {
                        if out.mesh.vertex_triangles(v).iter().any(|t| !inside.contains(t)) {
                            assert_eq!(out.mesh.vertex(v).z, 0.0, "border vertex {v} moved");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn refine_follows_pieces_groups_and_scaling() {
        // Two separate pieces, each scaled up about its own centre (so the handles stretch
        // too), next to a face-group border the band must not cross.
        let mut m = primitives::plane_grid(8.0, 4.0, 40, 20);
        for t in m.triangle_ids().collect::<Vec<_>>() {
            if m.tri_centroid(t).y > 1.0 {
                m.set_tri_group(t, 7);
            }
        }
        let centres = [DVec3::new(-2.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0)];
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| centres.iter().any(|c| m.tri_centroid(t).distance(*c) < 0.6)).collect();
        let r = soft_region(&m, &sel, 0.8, false, false, false);
        let mut pivots = [DVec3::ZERO; 2];
        for (&v, &p) in r.handle.iter().zip(&r.handle_piece) {
            pivots[p as usize] = if m.vertex(v).x < 0.0 { centres[0] } else { centres[1] };
        }
        let grow = move |piece: u32, p: DVec3| {
            let c = pivots[piece as usize];
            c + (p - c) * 2.5 + DVec3::Z * if piece == 0 { 0.5 } else { -0.5 }
        };
        let plain = apply(&m, &soft_deform(&m, &r, &grow, SoftKind::Linear, Continuity::SoftPosition));
        let max_edge = refine_length(&m, &r);
        let opts = RefineOptions { kind: SoftKind::Linear, continuity: Continuity::SoftPosition, fixed_boundary: false, max_edge };
        let out = soft_deform_refined(&m, &r, &sel, &grow, &opts);
        out.mesh.check_validity().unwrap();
        assert!(longest_edge(&plain) > 2.0 * max_edge);
        assert!(longest_edge(&out.mesh) <= max_edge * (1.0 + 1e-9));
        assert!(m.vertex_ids().all(|v| out.mesh.vertex(v).distance(plain.vertex(v)) < 1e-12), "each piece keeps its transform");
        // Split triangles keep their face group, and the band stops at the group border
        // exactly as it does unrefined.
        let groups = |mm: &Mesh| mm.triangle_ids().map(|t| mm.tri_group(t)).collect::<HashSet<i32>>();
        assert_eq!(groups(&out.mesh), groups(&m));
        let beyond = |mm: &Mesh| m.vertex_ids().filter(|&v| m.vertex_triangles(v).iter().all(|&t| m.tri_group(t) == 7) && mm.vertex(v) != m.vertex(v)).count();
        assert_eq!(beyond(&out.mesh), beyond(&plain));
        // Moving without stretching splits nothing.
        let shift = |_: u32, p: DVec3| p + DVec3::new(0.05, 0.0, 0.0);
        let same = soft_deform_refined(&m, &r, &sel, &shift, &opts);
        assert_eq!(same.mesh.vertex_count(), m.vertex_count());
    }
}

/// The solvers checked against the way they used to work (hash maps, a copy of the mesh per
/// fairing solve, the matrix rebuilt on every call), bit for bit: below the sparse solver's
/// parallel sizes the arithmetic is the same.
#[cfg(test)]
mod previous_tests {
    use super::*;
    use mm_core::primitives;

    fn old_soft_deform(mesh: &Mesh, region: &SoftRegion, transform: &dyn Fn(u32, DVec3) -> DVec3, kind: SoftKind, continuity: Continuity) -> Vec<(u32, DVec3)> {
        let mut out: Vec<(u32, DVec3)> = region.handle.iter().zip(&region.handle_piece).map(|(&v, &p)| (v, transform(p, mesh.vertex(v)))).collect();
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
                old_arap(mesh, &band, &mut pos, 8);
                out.extend(band.iter().map(|&v| (v, pos[&v])));
            }
            SoftKind::Linear => unreachable!(),
        }
        out
    }

    fn old_arap(m: &Mesh, free: &[u32], pos: &mut HashMap<u32, DVec3>, iterations: usize) {
        let index: HashMap<u32, usize> = free.iter().enumerate().map(|(i, &v)| (v, i)).collect();
        let n = free.len();
        let mut involved: Vec<u32> = free.to_vec();
        for &v in free {
            involved.extend(m.vertex_neighbors(v).filter(|u| !index.contains_key(u)));
        }
        involved.sort_unstable();
        involved.dedup();
        let current = |pos: &HashMap<u32, DVec3>, v: u32| pos.get(&v).copied().unwrap_or_else(|| m.vertex(v));
        let mut weights: HashMap<u32, f64> = HashMap::new();
        let mut w_of = |e: u32| *weights.entry(e).or_insert_with(|| edge_weight(m, e));
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
            mm_repair::sparse::solve_pcg(&a, &b, &mut x, &PcgParams { tolerance: 1e-7, max_iterations: 2000 });
            for (i, &v) in free.iter().enumerate() {
                pos.insert(v, x[i]);
            }
        }
    }

    fn old_warp(mesh: &Mesh, selection: &[u32], handles: &[(u32, DVec3)], rigid: bool) -> Vec<(u32, DVec3)> {
        let selected: HashSet<u32> = selection.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
        let pinned: HashSet<u32> = handles.iter().map(|h| h.0).collect();
        let mut free: Vec<u32> = selected.iter().flat_map(|&t| mesh.triangle(t)).filter(|&v| !pinned.contains(&v) && mesh.vertex_triangles(v).iter().all(|t| selected.contains(t))).collect();
        free.sort_unstable();
        free.dedup();
        let mut out: Vec<(u32, DVec3)> = handles.to_vec();
        if handles.is_empty() || free.is_empty() {
            return out;
        }
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
            old_arap(mesh, &free, &mut pos, 6);
        }
        out.extend(free.iter().map(|&v| (v, pos[&v])));
        out
    }

    fn old_region_from_roles(roles: &HashMap<u32, Role>) -> SoftRegion {
        let mut verts: Vec<(&u32, &Role)> = roles.iter().collect();
        verts.sort_unstable_by_key(|x| *x.0);
        let mut r = SoftRegion::default();
        for (&v, role) in verts {
            if role.handle {
                r.handle.push(v);
                r.handle_piece.push(role.piece);
            } else {
                r.band.push(v);
                r.band_weight.push(role.weight);
                r.band_piece.push(role.piece);
            }
        }
        r
    }

    fn old_stretched_edges(m: &Mesh, pos: &HashMap<u32, DVec3>, max_edge: f64) -> Vec<u32> {
        let mut moved: Vec<Option<DVec3>> = vec![None; m.max_vid()];
        for (&v, &p) in pos {
            moved[v as usize] = Some(p);
        }
        let at = |v: u32| moved[v as usize].unwrap_or_else(|| m.vertex(v));
        let mut seen = vec![false; m.max_eid()];
        let mut long: Vec<(f64, u32)> = Vec::new();
        for &v in pos.keys() {
            for &e in m.vertex_edges(v) {
                if std::mem::replace(&mut seen[e as usize], true) {
                    continue;
                }
                let [a, b] = m.edge_vertices(e);
                let len = at(a).distance(at(b));
                if len > max_edge && len > 1.01 * m.edge_length(e) {
                    long.push((len, e));
                }
            }
        }
        long.sort_unstable_by(|x, y| y.0.total_cmp(&x.0).then(x.1.cmp(&y.1)));
        long.into_iter().map(|x| x.1).collect()
    }

    fn old_solve_new_band(rest: &Mesh, free: &[u32], pos: &mut HashMap<u32, DVec3>, opts: &RefineOptions) {
        if free.is_empty() {
            return;
        }
        match opts.kind {
            SoftKind::Linear => {}
            SoftKind::Nonlinear => old_arap(rest, free, pos, 8),
            SoftKind::Flat | SoftKind::Stretchy => {
                let mut m = rest.clone();
                for v in rest.vertex_ids() {
                    m.set_vertex(v, DVec3::ZERO);
                }
                for (&v, &p) in pos.iter() {
                    m.set_vertex(v, p - rest.vertex(v));
                }
                if mm_repair::fair_region(&mut m, free, 2, LaplacianWeights::Uniform).is_ok() {
                    for &v in free {
                        pos.insert(v, rest.vertex(v) + m.vertex(v));
                    }
                }
            }
        }
    }

    fn old_soft_deform_refined(mesh: &Mesh, region: &SoftRegion, selection: &[u32], transform: &dyn Fn(u32, DVec3) -> DVec3, opts: &RefineOptions) -> RefinedSoft {
        let mut roles: HashMap<u32, Role> = HashMap::new();
        for (&v, &piece) in region.handle.iter().zip(&region.handle_piece) {
            roles.insert(v, Role { handle: true, weight: 1.0, piece });
        }
        for (&v, (&weight, &piece)) in region.band.iter().zip(region.band_weight.iter().zip(&region.band_piece)) {
            roles.insert(v, Role { handle: false, weight, piece });
        }
        let mut pos: HashMap<u32, DVec3> = old_soft_deform(mesh, region, transform, opts.kind, opts.continuity).into_iter().collect();
        let budget = (16 * roles.len() + 1000).min(2_000_000);
        let mut rest = mesh.clone();
        let mut selected: Vec<u32> = selection.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
        selected.sort_unstable();
        selected.dedup();
        let mut in_sel: FastSet<u32> = selected.iter().copied().collect();
        let mut new_band: Vec<u32> = Vec::new();
        let linear = opts.kind == SoftKind::Linear;
        let (mut rounds, mut checks, mut added, mut capped, mut solved, mut fills) = (0, 0, 0usize, false, true, 0);
        let refine = opts.max_edge.is_finite() && opts.max_edge > 0.0;
        loop {
            let check = solved && !linear && fills > 0;
            let more = refine && !capped && if check { checks < REFINE_CHECKS } else { rounds < REFINE_ROUNDS };
            let long = if more { old_stretched_edges(&rest, &pos, opts.max_edge) } else { Vec::new() };
            if long.is_empty() {
                if !solved {
                    old_solve_new_band(&rest, &new_band, &mut pos, opts);
                    solved = true;
                    fills += 1;
                    continue;
                }
                break;
            }
            if check {
                checks += 1;
            } else {
                rounds += 1;
            }
            let mut fresh: Vec<u32> = Vec::new();
            for e in long {
                if added >= budget {
                    capped = true;
                    break;
                }
                let [a, b] = rest.edge_vertices(e);
                let shift = |v: u32| pos.get(&v).map_or(DVec3::ZERO, |p| *p - rest.vertex(v));
                let moved = 0.5 * (shift(a) + shift(b));
                let Ok(info) = rest.split_edge(e) else { continue };
                added += 1;
                fresh.push(info.v_new);
                let tris: Vec<u32> = info.t_orig.into_iter().filter(|&t| t != INVALID).collect();
                let covered = tris.iter().filter(|t| in_sel.contains(t)).count();
                for k in 0..2 {
                    if info.t_new[k] != INVALID && in_sel.contains(&info.t_orig[k]) {
                        in_sel.insert(info.t_new[k]);
                        selected.push(info.t_new[k]);
                    }
                }
                let (ra, rb) = (roles.get(&info.v_a).copied(), roles.get(&info.v_b).copied());
                let w = |r: Option<Role>| r.map_or(0.0, |r| r.weight);
                let both_handle = matches!((ra, rb), (Some(x), Some(y)) if x.handle && y.handle && x.piece == y.piece);
                let (handle, may_move) = if opts.fixed_boundary { (both_handle && covered == tris.len(), covered == tris.len()) } else { (covered > 0 && both_handle, true) };
                let weight = 0.5 * (w(ra) + w(rb));
                let piece = match (ra, rb) {
                    (Some(x), Some(y)) if y.weight > x.weight => y.piece,
                    (Some(x), _) | (None, Some(x)) => x.piece,
                    (None, None) => 0,
                };
                let (f, x) = (info.v_new, rest.vertex(info.v_new));
                if handle {
                    roles.insert(f, Role { handle: true, weight: 1.0, piece });
                    pos.insert(f, transform(piece, x));
                } else if may_move && weight > 0.0 {
                    let weight = weight.min(1.0);
                    roles.insert(f, Role { handle: false, weight, piece });
                    if linear {
                        pos.insert(f, x + (transform(piece, x) - x) * ease(weight, opts.continuity));
                    } else {
                        pos.insert(f, x + moved);
                        new_band.push(f);
                        solved = false;
                    }
                }
            }
            delaunay_flips(&mut rest, &fresh, &in_sel, &|v| roles.contains_key(&v));
        }
        for (&v, &p) in &pos {
            rest.set_vertex(v, p);
        }
        selected.sort_unstable();
        RefinedSoft { mesh: rest, selection: selected, region: old_region_from_roles(&roles), capped }
    }

    fn bits(p: DVec3) -> [u64; 3] {
        p.to_array().map(f64::to_bits)
    }

    fn same_moves(a: &[(u32, DVec3)], b: &[(u32, DVec3)]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.0 == y.0 && bits(x.1) == bits(y.1))
    }

    fn same_mesh(a: &Mesh, b: &Mesh) -> bool {
        a.max_vid() == b.max_vid()
            && a.max_tid() == b.max_tid()
            && (0..a.max_vid() as u32).all(|v| a.is_vertex(v) == b.is_vertex(v) && bits(a.vertex(v)) == bits(b.vertex(v)))
            && (0..a.max_tid() as u32).all(|t| a.is_triangle(t) == b.is_triangle(t) && (!a.is_triangle(t) || (a.triangle(t) == b.triangle(t) && a.tri_group(t) == b.tri_group(t))))
    }

    fn same_region(a: &SoftRegion, b: &SoftRegion) -> bool {
        a.handle == b.handle && a.handle_piece == b.handle_piece && a.band == b.band && a.band_piece == b.band_piece && a.band_weight.iter().map(|w| w.to_bits()).eq(b.band_weight.iter().map(|w| w.to_bits()))
    }

    /// A bumpy 6 × 6 grid of 0.2 squares, two discs selected (two pieces).
    fn grid() -> (Mesh, Vec<u32>) {
        let mut m = primitives::plane_grid(6.0, 6.0, 30, 30);
        for v in m.vertex_ids().collect::<Vec<_>>() {
            let p = m.vertex(v);
            m.set_vertex(v, p + DVec3::Z * (0.05 * (3.0 * p.x).sin() * (2.0 * p.y).cos()));
        }
        let centres = [DVec3::new(-1.2, 0.3, 0.0), DVec3::new(1.4, -0.5, 0.0)];
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| centres.iter().any(|c| (m.tri_centroid(t) - *c).truncate().length() < 0.7)).collect();
        (m, sel)
    }

    const KINDS: [SoftKind; 4] = [SoftKind::Linear, SoftKind::Nonlinear, SoftKind::Flat, SoftKind::Stretchy];
    const CONTINUITIES: [Continuity; 3] = [Continuity::Boundaries, Continuity::SoftPosition, Continuity::SoftTangent];

    fn moves(k: usize) -> impl Fn(u32, DVec3) -> DVec3 {
        let q = mm_core::DQuat::from_rotation_x(0.2 + 0.3 * k as f64) * mm_core::DQuat::from_rotation_z(0.1 * k as f64);
        move |piece: u32, p: DVec3| {
            let c = DVec3::new(if piece == 0 { -1.2 } else { 1.4 }, 0.0, 0.0);
            c + q * (p - c) * (1.0 + 0.1 * k as f64) + DVec3::new(0.1, 0.0, 0.4 + 0.2 * k as f64)
        }
    }

    #[test]
    fn soft_solvers_match_the_previous_ones() {
        let (m, sel) = grid();
        for (fixed, shared) in [(false, true), (false, false), (true, true)] {
            let region = soft_region(&m, &sel, if fixed { 0.4 } else { 1.0 }, fixed, false, shared);
            assert!(!region.band.is_empty() && region.band.len() < 4096);
            for kind in KINDS {
                for continuity in CONTINUITIES {
                    let solver = SoftSolver::new(&m, &region, kind, continuity);
                    for k in 0..3 {
                        let tf = moves(k);
                        let old = old_soft_deform(&m, &region, &tf, kind, continuity);
                        assert!(same_moves(&soft_deform(&m, &region, &tf, kind, continuity), &old), "{kind:?} {continuity:?} fixed {fixed} shared {shared} move {k}");
                        assert!(same_moves(&solver.deform(&m, &region, &tf), &old), "cached: {kind:?} {continuity:?} fixed {fixed} shared {shared} move {k}");
                    }
                }
            }
        }
    }

    #[test]
    fn overlapping_regions_behave_as_before() {
        // A region whose band repeats a handle vertex (soft_region never makes one): the
        // band's value wins where the old code wrote it last, the handle's where it did.
        let (m, sel) = grid();
        let mut region = soft_region(&m, &sel, 1.0, false, false, true);
        let v = region.handle[region.handle.len() / 2];
        region.band.push(v);
        region.band_weight.push(0.5);
        region.band_piece.push(0);
        for kind in KINDS {
            for continuity in [Continuity::Boundaries, Continuity::SoftPosition] {
                let tf = moves(1);
                let old = old_soft_deform(&m, &region, &tf, kind, continuity);
                assert!(same_moves(&SoftSolver::new(&m, &region, kind, continuity).deform(&m, &region, &tf), &old), "{kind:?} {continuity:?}");
            }
        }
    }

    #[test]
    fn warp_matches_the_previous_one() {
        let m = primitives::plane_grid(4.0, 4.0, 24, 24);
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).truncate().length() < 1.5).collect();
        let near = |p: DVec3| m.vertex_ids().min_by(|&a, &b| m.vertex(a).distance(p).total_cmp(&m.vertex(b).distance(p))).unwrap();
        let (a, b) = (near(DVec3::ZERO), near(DVec3::new(0.6, 0.4, 0.0)));
        let handles = [(a, m.vertex(a) + DVec3::new(0.2, 0.0, 0.8)), (b, m.vertex(b) + DVec3::new(0.0, -0.1, -0.3))];
        for rigid in [false, true] {
            for h in [&handles[..1], &handles[..]] {
                assert!(same_moves(&warp(&m, &sel, h, rigid), &old_warp(&m, &sel, h, rigid)), "rigid {rigid}, {} handles", h.len());
            }
        }
    }

    #[test]
    fn refining_matches_the_previous_version() {
        let (m, sel) = grid();
        for fixed in [false, true] {
            let region = soft_region(&m, &sel, if fixed { 0.4 } else { 1.0 }, fixed, false, true);
            let max_edge = refine_length(&m, &region);
            for kind in KINDS {
                let continuity = Continuity::SoftPosition;
                let opts = RefineOptions { kind, continuity, fixed_boundary: fixed, max_edge };
                let tf = moves(2);
                let old = old_soft_deform_refined(&m, &region, &sel, &tf, &opts);
                let solver = SoftSolver::new(&m, &region, kind, continuity);
                for new in [soft_deform_refined(&m, &region, &sel, &tf, &opts), soft_deform_refined_in(m.clone(), &m, &region, &sel, &tf, &opts, Some(&solver))] {
                    assert!(new.mesh.vertex_count() > m.vertex_count(), "{kind:?}: the move is refined");
                    assert!(same_mesh(&new.mesh, &old.mesh), "{kind:?} fixed {fixed}");
                    assert_eq!(new.selection, old.selection);
                    assert!(same_region(&new.region, &old.region));
                    assert_eq!(new.capped, old.capped);
                }
            }
        }
    }
}
