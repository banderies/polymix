//! Thicken a surface into a shell of a given wall thickness, for printing a vessel as a
//! hollow model (Meshmixer: select all → Extrude). The wall grows outward from the surface
//! (the lumen keeps its shape), inward (the outside keeps its shape) or both ways evenly.
//! Open ends (inlets and outlets cut flat) stay open, the wall ending in a flat rim there.
//!
//! The new side of the wall is a level set of the surface's signed distance field
//! ([`mm_solid::make_solid`] with an offset), so it lies the wall's thickness from the
//! surface everywhere and never runs into itself. A move along the normals would fold
//! where the surface curves into itself (the crotch of a bifurcation, the inside of a tight
//! bend) and, inward, close up branches narrower than twice the thickness.
//!
//! 1. Each opening is prolonged straight on along its outward direction and closed: the
//!    lumen, a closed solid, whose offset near the opening is that of a vessel running on
//!    (not rounded over a cap).
//! 2. The offset solid (or two, for a centred wall) is made from the distance field.
//! 3. Exact booleans: the wall's outer solid minus a short cylinder standing on each
//!    opening (just in front of its plane, so the lumen's prolongation crosses the
//!    cylinder's base cleanly), minus the inner solid. The cylinders' bases become the flat
//!    rims; where wall from elsewhere runs into an opening's cylinder (a tight bend, two
//!    branches cut close together) it is cut along the cylinder. The kept surface is the
//!    input itself, never cut: only its prolongation, a twentieth of the wall's thickness
//!    long, stays on in front of each opening up to its rim.

use crate::boolean::{mesh_boolean_with, mesh_boolean_with_trees, BooleanOp};
use crate::util::{loop_outward_direction, mean, positions};
use crate::{EditError, EditResult};
use mm_core::{measure, Bvh, DVec3, Mesh};
use mm_solid::field::FIELD_BETA;
use mm_solid::WindingNumber;
use rayon::prelude::*;
use std::borrow::Cow;
use std::f64::consts::TAU;
use std::sync::atomic::{AtomicBool, Ordering};

/// Which way the wall grows from the surface.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum ThickenSide {
    /// Outside the surface: the lumen keeps its shape (a flow model's wall).
    #[default]
    Outward,
    /// Inside the surface: the outside keeps its shape, the lumen narrows.
    Inward,
    /// Half each way.
    Centered,
}

/// Parameters for [`thicken`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ThickenParams {
    /// Wall thickness (model units, mm).
    pub thickness: f64,
    pub side: ThickenSide,
    /// Spacing of the distance field's grid: the new wall follows the surface to about a
    /// tenth of this. `None`: a sixth of the thickness (coarser on very large surfaces).
    pub cell_size: Option<f64>,
    /// Remesh the new wall to the surface's own edge length.
    pub remesh: bool,
}

impl Default for ThickenParams {
    fn default() -> Self {
        Self { thickness: 2.0, side: ThickenSide::Outward, cell_size: None, remesh: true }
    }
}

/// An open end of the surface, where the wall ends in a flat rim.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Opening {
    pub centre: DVec3,
    /// Pointing out of the surface (away from the lumen's inside).
    pub normal: DVec3,
    /// Mean and largest distance of its rim from its centre.
    pub radius: f64,
    pub extent: f64,
    /// How far its rim strays from its plane, over its radius (0: flat, as cut by a plane).
    pub flatness: f64,
}

/// What [`thicken`] made.
#[derive(Clone, Debug)]
pub struct ThickenResult {
    /// The shell: closed, the input surface (when kept) one side of its wall.
    pub mesh: Mesh,
    pub openings: Vec<Opening>,
    /// Face group of the new wall (the kept surface keeps its groups).
    pub wall_group: i32,
    /// Face group of each opening's rim, in the order of `openings`.
    pub rim_groups: Vec<i32>,
    /// The wall's thickness, measured from the new wall's vertices to the other side: least,
    /// mean and most.
    pub measured: [f64; 3],
    /// The grid spacing used.
    pub cell_size: f64,
    /// The surface ran into itself, so its side of the wall was rebuilt from the distance
    /// field (within a small fraction of the grid spacing of it) rather than kept exactly.
    pub rebuilt: bool,
}

/// How far a rim may stray from flat (over its radius) before it is reported.
pub const FLAT_ENOUGH: f64 = 0.1;

/// The open ends of `mesh` with their planes.
pub fn openings(mesh: &Mesh) -> Vec<(Vec<u32>, Opening)> {
    mm_core::topology::boundary_loops(mesh)
        .into_iter()
        .filter_map(|lp| {
            let pts = positions(mesh, &lp.vertices);
            let centre = mean(&pts);
            let normal = loop_outward_direction(mesh, &lp.vertices)?;
            let radius = pts.iter().map(|p| p.distance(centre)).sum::<f64>() / pts.len() as f64;
            let extent = pts.iter().map(|p| p.distance(centre)).fold(0.0, f64::max);
            let flatness = pts.iter().map(|p| (*p - centre).dot(normal).abs()).fold(0.0, f64::max) / radius.max(1e-300);
            Some((lp.vertices, Opening { centre, normal, radius, extent, flatness }))
        })
        .collect()
}

/// Thicken `mesh` into a shell (see the module documentation). `progress` gets 0..1 and
/// may return `false` to cancel.
pub fn thicken(mesh: &Mesh, params: &ThickenParams, progress: mm_solid::Progress) -> EditResult<ThickenResult> {
    let report = |x: f32| progress.is_none_or(|p| p(x));
    let shell = make_shell(mesh, params, &report)?;
    finish(shell, params.remesh, &report)
}

/// The first part of [`thicken`]: the shell with its new side smoothed, not yet remeshed
/// (a preview can show it while [`thicken_finish`] runs). `progress` gets 0..1 over this
/// part and may return `false` to cancel.
pub fn thicken_shell(mesh: &Mesh, params: &ThickenParams, progress: mm_solid::Progress) -> EditResult<ThickenShell> {
    let report = |x: f32| progress.is_none_or(|p| p(x / SHELL_DONE));
    let shell = make_shell(mesh, params, &report)?;
    report(SHELL_DONE);
    Ok(shell)
}

/// The rest of [`thicken`]: remesh the shell's new wall (when `remesh`) and check the result.
/// `thicken_finish(thicken_shell(mesh, params, ..)?, params.remesh, ..)` gives exactly
/// what [`thicken`] gives. `progress` gets 0..1 over this part and may return `false` to
/// cancel.
pub fn thicken_finish(shell: ThickenShell, remesh: bool, progress: mm_solid::Progress) -> EditResult<ThickenResult> {
    let report = |x: f32| progress.is_none_or(|p| p(((x - SHELL_DONE) / (1.0 - SHELL_DONE)).max(0.0)));
    finish(shell, remesh, &report)
}

/// The share of [`thicken`]'s progress taken by [`thicken_shell`].
const SHELL_DONE: f32 = 0.86;

/// What [`thicken_shell`] makes: the shell, its wall carved and the new side smoothed,
/// before the remesh.
#[derive(Clone, Debug)]
pub struct ThickenShell {
    /// The shell: closed, the input surface (when kept) one side of its wall.
    pub mesh: Mesh,
    pub openings: Vec<Opening>,
    /// Face groups of the new wall and of the rims (all of them, until the result gives each
    /// its own).
    pub wall_group: i32,
    pub rim_group: i32,
    /// The wall's thickness: least, mean and most (see [`ThickenResult::measured`]).
    pub measured: [f64; 3],
    /// The grid spacing used.
    pub cell_size: f64,
    /// The surface's side of the wall was rebuilt from the distance field.
    pub rebuilt: bool,
    /// Triangles of `mesh` (by id) within three rings of a rim, which the remesh leaves alone.
    near: Vec<bool>,
    /// The input surface's average edge length, which the remesh aims for.
    edge_length: f64,
}

impl ThickenShell {
    /// The shell as a finished result without the remesh (what [`thicken`] gives with
    /// `remesh` off).
    pub fn into_result(self) -> ThickenResult {
        let ThickenShell { mut mesh, openings, wall_group, rim_group, measured, cell_size, rebuilt, .. } = self;
        let rim_groups = split_rims(&mut mesh, rim_group, &openings);
        ThickenResult { mesh, openings, wall_group, rim_groups, measured, cell_size, rebuilt }
    }
}

fn cancelled() -> EditError {
    EditError::Failed("cancelled".into())
}

fn is_cancelled(e: &EditError) -> bool {
    matches!(e, EditError::Failed(m) if m == "cancelled")
}

/// Search trees of a mesh, as the booleans and Make Solid take them.
type Trees<'a> = Option<(&'a Bvh, &'a WindingNumber)>;

/// Steps 1 to 3 and the smoothing; `report` gets [`thicken`]'s progress.
fn make_shell(mesh: &Mesh, params: &ThickenParams, report: &dyn Fn(f32) -> bool) -> EditResult<ThickenShell> {
    let t = params.thickness;
    if !(t > 0.0 && t.is_finite()) {
        return Err(EditError::InvalidParameter("the wall thickness must be positive".into()));
    }
    if mesh.triangle_count() == 0 {
        return Err(EditError::InvalidParameter("the mesh is empty".into()));
    }
    let ends = openings(mesh);
    // The grid: a sixth of the thickness, coarser where the narrow band around a large
    // surface would hold more than about 40 million cells.
    let area = measure::surface_area(mesh);
    let h = params.cell_size.unwrap_or_else(|| (t / 6.0).max((area * 3.0 * t / 4e7).cbrt())).max(1e-6 * t);
    let edge_length = mesh.average_edge_length();

    // 1. The lumen: the surface, oriented outward, each opening prolonged straight on and
    // closed. The prolongations and their ends get groups of their own.
    let groups: Vec<i32> = mesh.triangle_ids().map(|tr| mesh.tri_group(tr)).collect();
    let top = groups.iter().copied().max().unwrap_or(0);
    let main = most_common(&groups);
    let (prolonged, closing, new_side, wall_group, rim_group) = (top + 1, top + 2, top + 3, top + 4, top + 5);
    // (Far enough that the offset near the opening does not see the closing cap.)
    let reach = t + 4.0 * h;
    let mut closed = mesh.clone();
    for (lp, o) in &ends {
        let n = lp.len();
        let row: Vec<u32> = lp.iter().map(|&v| closed.append_vertex(closed.vertex(v) + o.normal * reach)).collect();
        let tip = closed.append_vertex(o.centre + o.normal * reach);
        for i in 0..n {
            let (a, b, wa, wb) = (lp[i], lp[(i + 1) % n], row[i], row[(i + 1) % n]);
            for (tri, g) in [([b, a, wa], prolonged), ([b, wa, wb], prolonged), ([tip, wb, wa], closing)] {
                closed.append_triangle(tri, g).map_err(|e| EditError::Failed(format!("closing an opening: {e:?}")))?;
            }
        }
    }
    if !closed.is_closed() {
        return Err(EditError::Failed("the surface has holes besides its openings, or is not manifold: repair it first (Analysis → Inspector)".into()));
    }
    if measure::volume(&closed) < 0.0 {
        closed.reverse_orientation();
    }
    if !report(0.05) {
        return Err(cancelled());
    }
    // The lumen's search trees, built once for the offset solids, the wall's measurement,
    // the booleans that take the lumen itself and the rebuild.
    let (bvh_c, wn_c) = rayon::join(|| Bvh::build(&closed), || WindingNumber::with_beta(&closed, FIELD_BETA));
    let lumen: Trees = Some((&bvh_c, &wn_c));

    // 2. The offset solid(s), as the wall's new side.
    let offset = |d: f64, from: f32, to: f32| -> EditResult<Mesh> {
        let sub = |x: f32| report(from + (to - from) * x);
        let solid_params = mm_solid::MakeSolidParams { cell_size: h, offset: d, transfer_colors: false, ..Default::default() };
        let mut m = mm_solid::solid::make_solid_with(&closed, &solid_params, lumen, Some(&sub)).map_err(|e| EditError::Failed(format!("the offset surface: {e}")))?;
        for tr in m.triangle_ids().collect::<Vec<_>>() {
            m.set_tri_group(tr, new_side);
        }
        Ok(m)
    };
    let side = params.side;
    let (outer, inner): (Cow<Mesh>, Cow<Mesh>) = match side {
        ThickenSide::Outward => (Cow::Owned(offset(t, 0.05, 0.7)?), Cow::Borrowed(&closed)),
        ThickenSide::Inward => (Cow::Borrowed(&closed), Cow::Owned(offset(-t, 0.05, 0.7)?)),
        ThickenSide::Centered => (Cow::Owned(offset(0.5 * t, 0.05, 0.4)?), Cow::Owned(offset(-0.5 * t, 0.4, 0.7)?)),
    };
    // Each opening's rim stands this far in front of its plane.
    let ahead = 0.05 * t;
    // An inner solid that does not reach through an opening would close it.
    let inner_ids: Vec<u32> = inner.vertex_ids().collect();
    for (i, (_, o)) in ends.iter().enumerate() {
        let through = inner_ids.par_iter().map(|&v| inner.vertex(v) - o.centre).any(|q| q.dot(o.normal) > ahead + h && (q - o.normal * q.dot(o.normal)).length() < o.extent + h);
        if !through {
            return Err(EditError::Failed(format!(
                "opening {} (⌀ {:.1}) is too narrow for a wall this thick: the lumen closes up inside it; thicken outward, or thinner",
                i + 1,
                2.0 * o.radius
            )));
        }
    }
    drop(inner_ids);

    // 3. The outer solid cut back by a cylinder standing on each opening, then the inner
    // solid taken out of it. The wall is measured meanwhile.
    let carver = Carver { ends: &ends, t, h, ahead, reach, prolonged, closing, new_side, wall_group, rim_group, main, report };
    // The lumen's trees serve whichever solid is the lumen itself.
    let outer_trees = if side == ThickenSide::Inward { lumen } else { None };
    let inner_trees = if side == ThickenSide::Outward { lumen } else { None };
    let carve = || -> EditResult<(Mesh, bool)> {
        let cut = carver.cut_rims(&outer, outer_trees, 0.7, 0.78);
        let first = match &cut {
            Ok(shell) => carver.hollow(shell, &inner, inner_trees, 0.8),
            Err(e) => Err(e.clone()),
        };
        let first = match first {
            Ok(m) => return Ok((m, false)),
            Err(e) if is_cancelled(&e) || side == ThickenSide::Centered => return Err(e),
            Err(e) => e,
        };
        // The surface runs into itself (overlapping pieces merged without a union), which
        // the exact booleans cannot take: its side of the wall is rebuilt as the distance
        // field's zero set, which cannot run into itself (snapping it back onto the surface
        // could, where the surface does).
        if !report(0.8) {
            return Err(cancelled());
        }
        // Outward the same outer solid is cut by the same cylinders again, so their cut is
        // kept, and when that cut failed the rebuild cannot help.
        let kept = match (side, cut) {
            (ThickenSide::Outward, Ok(shell)) => Some(shell),
            (ThickenSide::Outward, Err(_)) => return Err(first),
            _ => None,
        };
        let sub = |x: f32| report(0.8 + 0.03 * x);
        let rebuild = mm_solid::MakeSolidParams { mode: mm_solid::SolidMode::Fast, cell_size: h, transfer_groups: true, transfer_colors: false, ..Default::default() };
        let clean = match mm_solid::solid::make_solid_with(&closed, &rebuild, lumen, Some(&sub)) {
            Ok(m) => m,
            Err(mm_solid::SolidError::Cancelled) => return Err(cancelled()),
            Err(_) => return Err(first),
        };
        let second = match kept {
            Some(shell) => carver.hollow(&shell, &clean, None, 0.85),
            None => carver.cut_rims(&clean, None, 0.83, 0.845).and_then(|shell| carver.hollow(&shell, &inner, None, 0.85)),
        };
        match second {
            Ok(m) => Ok((m, true)),
            Err(e) if is_cancelled(&e) => Err(e),
            Err(_) => Err(first),
        }
    };
    // The wall's thickness, from the new side to the solid across the wall, measured
    // alongside the carving (which reports progress, so it stays on this thread).
    let (from, to, to_bvh): (&Mesh, &Mesh, Option<&Bvh>) = match side {
        ThickenSide::Outward => (&outer, &closed, Some(&bvh_c)),
        ThickenSide::Inward => (&inner, &closed, Some(&bvh_c)),
        ThickenSide::Centered => (&outer, &inner, None),
    };
    let stop = AtomicBool::new(false);
    let mut measured = None;
    let carved = rayon::in_place_scope(|s| {
        s.spawn(|_| measured = measure_wall(from, to, to_bvh, &stop));
        let r = carve();
        if r.is_err() {
            stop.store(true, Ordering::Relaxed);
        }
        r
    });
    let (mut out, rebuilt) = carved?;
    let measured = measured.expect("the wall is measured unless stopped");
    // The solids and the lumen's trees are done with.
    drop(outer);
    drop(inner);
    drop(closed);
    drop(bvh_c);
    drop(wn_c);
    if !report(0.85) {
        return Err(cancelled());
    }

    // The new wall smoothed (here) and remeshed to the surface's edge length ([`finish`]),
    // away from the rims: three rings of triangles beside each rim stay as they are (the
    // rims, flat, too), since moving a vertex next to that corner could carry it across the
    // rim's plane.
    // - Smoothing: where the wall meets itself in a crease (outside a junction of branches)
    //   the grid leaves it stair-stepped; a few volume-keeping (Taubin) steps even it out,
    //   moving the crease outward, so the wall only thickens a little there.
    // - A remesh, or smoothing, that made the wall run into anything is undone.
    let mut near = vec![false; out.max_tid()];
    let mut ring: Vec<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == rim_group).collect();
    for &tr in &ring {
        near[tr as usize] = true;
    }
    for _ in 0..3 {
        // Each ring is what touches the one before (earlier rings' neighbours are marked).
        let mut next = Vec::new();
        for &tr in &ring {
            for v in out.triangle(tr) {
                for o in out.vertex_triangles(v) {
                    if !near[o as usize] {
                        near[o as usize] = true;
                        next.push(o);
                    }
                }
            }
        }
        ring = next;
    }
    let mut free: Vec<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == wall_group).flat_map(|tr| out.triangle(tr)).collect();
    free.sort_unstable();
    free.dedup();
    free.retain(|&v| out.vertex_triangles(v).iter().all(|&tr| out.tri_group(tr) == wall_group && !near[tr as usize]));
    // Only the free vertices move, so only their positions are kept to undo the smoothing.
    let before: Vec<DVec3> = free.iter().map(|&v| out.vertex(v)).collect();
    if !taubin(&mut out, &free, 20, &|| report(0.85)) {
        return Err(cancelled());
    }
    let mut moved: Vec<u32> = free.iter().flat_map(|&v| out.vertex_triangles(v)).collect();
    moved.sort_unstable();
    moved.dedup();
    match runs_into_anything(&out, &moved, &|| report(0.855)) {
        Crossing::Hit => {
            for (&v, &p) in free.iter().zip(&before) {
                out.set_vertex(v, p);
            }
        }
        Crossing::Clear => {}
        Crossing::Cancelled => return Err(cancelled()),
    }
    Ok(ThickenShell { mesh: out, openings: ends.into_iter().map(|(_, o)| o).collect(), wall_group, rim_group, measured, cell_size: h, rebuilt, near, edge_length })
}

/// The remesh of the shell's new wall and the result; `report` gets [`thicken`]'s progress.
fn finish(shell: ThickenShell, remesh: bool, report: &dyn Fn(f32) -> bool) -> EditResult<ThickenResult> {
    if !report(SHELL_DONE) {
        return Err(cancelled());
    }
    let ThickenShell { mesh: mut out, openings, wall_group, rim_group, measured, cell_size, rebuilt, near, edge_length } = shell;
    if remesh {
        let part: Vec<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == wall_group && !near[tr as usize]).collect();
        let rp = mm_remesh::RemeshParams {
            target_edge_length: edge_length,
            iterations: 6,
            preserve_group_boundaries: true,
            region: Some(part),
            ..mm_remesh::RemeshParams::with_target(edge_length)
        };
        let before = out.clone();
        let sub = |f: f32| report(SHELL_DONE + 0.12 * f);
        let remeshed = mm_remesh::remesh(&mut out, &rp, Some(&sub)).map_err(|e| EditError::Failed(format!("remeshing the new wall: {e}")))?;
        if remeshed.cancelled {
            return Err(cancelled());
        }
        if let Some(region) = remeshed.region_triangles {
            match runs_into_anything(&out, &region, &|| report(0.98)) {
                Crossing::Hit => out = before,
                Crossing::Clear => {}
                Crossing::Cancelled => return Err(cancelled()),
            }
        }
    }
    report(1.0);
    let rim_groups = split_rims(&mut out, rim_group, &openings);
    Ok(ThickenResult { mesh: out, openings, wall_group, rim_groups, measured, cell_size, rebuilt })
}

/// Give each opening's rim a face group of its own (a patch each, like the caps of a vessel's
/// ends): the first keeps `rim_group`, the others get new ones. Returns them in the order of
/// `openings`.
fn split_rims(mesh: &mut Mesh, rim_group: i32, openings: &[Opening]) -> Vec<i32> {
    if openings.is_empty() {
        return Vec::new();
    }
    // The rims, each a band of connected rim triangles, with the opening nearest its centre.
    let mut seen = vec![false; mesh.max_tid()];
    let mut rims: Vec<(usize, Vec<u32>)> = Vec::new();
    for start in mesh.triangle_ids() {
        if mesh.tri_group(start) != rim_group || seen[start as usize] {
            continue;
        }
        seen[start as usize] = true;
        let mut part = vec![start];
        let mut i = 0;
        while i < part.len() {
            for n in mesh.tri_neighbors(part[i]) {
                if n != mm_core::INVALID && !seen[n as usize] && mesh.tri_group(n) == rim_group {
                    seen[n as usize] = true;
                    part.push(n);
                }
            }
            i += 1;
        }
        let centre = part.iter().map(|&t| mesh.tri_centroid(t)).sum::<DVec3>() / part.len() as f64;
        let nearest = (0..openings.len()).min_by(|&a, &b| openings[a].centre.distance(centre).total_cmp(&openings[b].centre.distance(centre))).unwrap();
        rims.push((nearest, part));
    }
    rims.sort_by_key(|r| r.0);
    let mut groups: Vec<Option<i32>> = vec![None; openings.len()];
    let mut next = mesh.max_group_id().max(rim_group) + 1;
    let mut first = true;
    for (i, part) in rims {
        let g = *groups[i].get_or_insert_with(|| {
            if std::mem::take(&mut first) {
                rim_group
            } else {
                next += 1;
                next - 1
            }
        });
        for t in part {
            mesh.set_tri_group(t, g);
        }
    }
    groups.into_iter().map(|g| g.unwrap_or(rim_group)).collect()
}

/// The booleans of step 3, with what they share.
struct Carver<'a> {
    ends: &'a [(Vec<u32>, Opening)],
    t: f64,
    h: f64,
    ahead: f64,
    reach: f64,
    prolonged: i32,
    closing: i32,
    new_side: i32,
    wall_group: i32,
    rim_group: i32,
    /// The surface's commonest group.
    main: i32,
    report: &'a dyn Fn(f32) -> bool,
}

impl Carver<'_> {
    /// `outer` cut back by a cylinder standing on each opening; the cylinders' bases become
    /// the rims. (After each boolean the cylinder's triangles, whose groups follow the
    /// shell's, are told apart by group and relabelled at once.) `outer_trees` are the trees
    /// of `outer` itself. Progress goes from `from` to `to`.
    fn cut_rims<'m>(&self, outer: &'m Mesh, outer_trees: Trees, from: f32, to: f32) -> EditResult<Cow<'m, Mesh>> {
        let (t, h) = (self.t, self.h);
        let mut shell: Cow<Mesh> = Cow::Borrowed(outer);
        for (i, (_, o)) in self.ends.iter().enumerate() {
            let cyl = cylinder(o.centre + o.normal * self.ahead, o.normal, o.extent + t + 4.0 * h, self.reach + t + 4.0 * h, h);
            let before = shell.max_group_id();
            let trees = if i == 0 { outer_trees } else { None };
            let cut = mesh_boolean_with_trees(&shell, &cyl, BooleanOp::Difference, trees, None)
                .map_err(|e| EditError::Failed(format!("cutting the wall back at opening {}: {e}", i + 1)))?;
            let mut m = cut.mesh;
            for tr in m.triangle_ids().collect::<Vec<_>>() {
                if m.tri_group(tr) > before {
                    m.set_tri_group(tr, self.rim_group);
                }
            }
            shell = Cow::Owned(m);
            if !(self.report)(from + (to - from) * (i + 1) as f32 / self.ends.len() as f32) {
                return Err(cancelled());
            }
        }
        Ok(shell)
    }

    /// `inner` taken out of `shell`, with the groups settled. `inner_trees` are the trees
    /// of `inner`. Reports `at` when done.
    fn hollow(&self, shell: &Mesh, inner: &Mesh, inner_trees: Trees, at: f32) -> EditResult<Mesh> {
        let before = shell.max_group_id();
        let hollow = mesh_boolean_with(shell, inner, BooleanOp::Difference, inner_trees).map_err(|e| EditError::Failed(format!("taking the lumen out of the wall: {e}")))?;
        let from_inner = hollow.group_offset;
        let mut out = hollow.mesh;
        // Groups: the kept surface keeps its own (its prolongations take its commonest), the
        // new side and the rims one each.
        for tr in out.triangle_ids().collect::<Vec<_>>() {
            let raw = out.tri_group(tr);
            let g = if raw > before { raw - from_inner } else { raw };
            let g = if raw <= before && raw == self.rim_group {
                self.rim_group
            } else if g == self.new_side {
                self.wall_group
            } else if g == self.prolonged || g == self.closing {
                self.main
            } else {
                g
            };
            out.set_tri_group(tr, g);
        }
        if !(self.report)(at) {
            return Err(cancelled());
        }
        Ok(out)
    }
}

/// A closed cylinder standing on the disk of radius `r` about `c` square to `n`, `len` long
/// along `n`, of even triangles about `spacing` across (a fan of thin wedges would leave the
/// boolean slivers to cut where the lumen crosses it). Its base is face group 0, the rest
/// group 1.
fn cylinder(c: DVec3, n: DVec3, r: f64, len: f64, spacing: f64) -> Mesh {
    let n = n.normalize();
    let u = n.any_orthonormal_vector();
    let mut b = Builder { u, v: n.cross(u), pos: Vec::new(), tris: Vec::new(), groups: Vec::new() };
    let rings = ((r / spacing).ceil() as usize).max(1);
    let around = |k: usize| ((TAU * r * k as f64 / rings as f64 / spacing).round() as usize).max(6);
    // A disk of rings at `at`, facing `face`; its rim.
    let disk = |b: &mut Builder, at: DVec3, face: DVec3, g: i32| {
        b.pos.push(at);
        let centre = b.pos.len() as u32 - 1;
        let mut inner = b.ring(at, r / rings as f64, around(1));
        for w in 0..inner.len() {
            b.tri([centre, inner[w].1, inner[(w + 1) % inner.len()].1], face, g);
        }
        for k in 2..=rings {
            let outer = b.ring(at, r * k as f64 / rings as f64, around(k));
            b.zip(&inner, &outer, &|_| face, g);
            inner = outer;
        }
        inner
    };
    let bottom = disk(&mut b, c, -n, 0);
    let top = disk(&mut b, c + n * len, n, 1);
    // The side in rows, its rings matching the disks' rims.
    let rows = ((len / spacing).ceil() as usize).max(1);
    let radial = |m: DVec3| {
        let q = m - c;
        q - n * q.dot(n)
    };
    let mut below = bottom;
    for row in 1..=rows {
        let above = if row == rows { top.clone() } else { b.ring(c + n * (len * row as f64 / rows as f64), r, below.len()) };
        b.zip(&below, &above, &radial, 1);
        below = above;
    }
    Mesh::from_triangles(&b.pos, &b.tris, Some(&b.groups)).0
}

/// Triangles of [`cylinder`] being built.
struct Builder {
    u: DVec3,
    v: DVec3,
    pos: Vec<DVec3>,
    tris: Vec<[u32; 3]>,
    groups: Vec<i32>,
}

impl Builder {
    /// A ring of `count` points (with their angles) about `at` in the cylinder's frame,
    /// started an irrational fraction of a step round, so no regular tube's seams line up
    /// with its edges (where the boolean would meet edges crossing edges exactly).
    fn ring(&mut self, at: DVec3, radius: f64, count: usize) -> Vec<(f64, u32)> {
        (0..count)
            .map(|j| {
                let a = TAU * (j as f64 + 0.381_966) / count as f64;
                self.pos.push(at + (self.u * a.cos() + self.v * a.sin()) * radius);
                (a, self.pos.len() as u32 - 1)
            })
            .collect()
    }

    /// A triangle facing `out`, whichever way round its corners came.
    fn tri(&mut self, t: [u32; 3], out: DVec3, g: i32) {
        let [a, b, c] = t.map(|i| self.pos[i as usize]);
        self.tris.push(if (b - a).cross(c - a).dot(out) < 0.0 { [t[0], t[2], t[1]] } else { t });
        self.groups.push(g);
    }

    /// Two concentric (or stacked) rings joined by zipping round them in angle order, each
    /// triangle facing `out` at its centre.
    fn zip(&mut self, a: &[(f64, u32)], b: &[(f64, u32)], out: &dyn Fn(DVec3) -> DVec3, g: i32) {
        let next = |r: &[(f64, u32)], i: usize| if i < r.len() { r[(i + 1) % r.len()].0 + if i + 1 == r.len() { TAU } else { 0.0 } } else { f64::INFINITY };
        let (mut i, mut j) = (0, 0);
        while i < a.len() || j < b.len() {
            let t = if next(a, i) <= next(b, j) {
                i += 1;
                [a[(i - 1) % a.len()].1, a[i % a.len()].1, b[j % b.len()].1]
            } else {
                j += 1;
                [b[(j - 1) % b.len()].1, b[j % b.len()].1, a[i % a.len()].1]
            };
            let m = t.iter().map(|&k| self.pos[k as usize]).sum::<DVec3>() / 3.0;
            self.tri(t, out(m), g);
        }
    }
}

/// The wall's thickness: each vertex of the new side `from` against the solid across the
/// wall, `to` (whose BVH may come built). Least, mean and most; `None` when `stop` was set
/// before the end.
fn measure_wall(from: &Mesh, to: &Mesh, to_bvh: Option<&Bvh>, stop: &AtomicBool) -> Option<[f64; 3]> {
    let built;
    let bvh = match to_bvh {
        Some(b) => b,
        None => {
            built = Bvh::build(to);
            &built
        }
    };
    let ids: Vec<u32> = from.vertex_ids().collect();
    let mut d: Vec<f64> = Vec::with_capacity(ids.len());
    // In chunks, so that a carving that fails or is cancelled does not wait for all of it.
    for chunk in ids.chunks(1 << 16) {
        if stop.load(Ordering::Relaxed) {
            return None;
        }
        d.par_extend(chunk.par_iter().filter_map(|&v| bvh.nearest(to, from.vertex(v), f64::INFINITY).map(|h| h.dist_sq.sqrt())));
    }
    if d.is_empty() {
        return Some([0.0; 3]);
    }
    Some([d.iter().copied().fold(f64::INFINITY, f64::min), d.iter().sum::<f64>() / d.len() as f64, d.iter().copied().fold(0.0, f64::max)])
}

/// Whether a check for crossings found one.
enum Crossing {
    Hit,
    Clear,
    Cancelled,
}

/// Whether any of `tris` cuts through another triangle of `m` (neighbours sharing a vertex
/// aside). `go_on` is asked between chunks of triangles whether to carry on.
fn runs_into_anything(m: &Mesh, tris: &[u32], go_on: &dyn Fn() -> bool) -> Crossing {
    if !go_on() {
        return Crossing::Cancelled;
    }
    let bvh = Bvh::build(m);
    for chunk in tris.chunks(1 << 16) {
        if !go_on() {
            return Crossing::Cancelled;
        }
        let hit = chunk.par_iter().any(|&t| {
            let (tv, tp) = (m.triangle(t), m.tri_positions(t));
            let mut hit = false;
            bvh.query_aabb(&mm_core::Aabb::from_points(tp), |o| {
                if !hit && o != t && !m.triangle(o).iter().any(|v| tv.contains(v)) && mm_core::intersect::triangles_intersect(tp, m.tri_positions(o)) {
                    hit = true;
                }
            });
            hit
        });
        if hit {
            return Crossing::Hit;
        }
    }
    Crossing::Clear
}

/// Volume-keeping smoothing (Taubin λ|μ steps) of the vertices `free`, the rest held.
/// `go_on` is asked before each iteration; returns `false` when it said stop.
fn taubin(m: &mut Mesh, free: &[u32], iterations: usize, go_on: &dyn Fn() -> bool) -> bool {
    let rings: Vec<Vec<u32>> = free.iter().map(|&v| m.vertex_neighbors(v).collect()).collect();
    for _ in 0..iterations {
        if !go_on() {
            return false;
        }
        for step in [0.5, -0.53] {
            let moved: Vec<DVec3> = free
                .par_iter()
                .zip(&rings)
                .map(|(&v, ring)| {
                    let p = m.vertex(v);
                    let mean = ring.iter().map(|&u| m.vertex(u)).sum::<DVec3>() / ring.len().max(1) as f64;
                    p + (mean - p) * step
                })
                .collect();
            let pos = m.positions_mut();
            for (&v, p) in free.iter().zip(moved) {
                pos[v as usize] = p;
            }
            m.bump_shape_version();
        }
    }
    true
}

/// The commonest of `groups` (0 when empty).
fn most_common(groups: &[i32]) -> i32 {
    let mut count: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for &g in groups {
        *count.entry(g).or_default() += 1;
    }
    count.into_iter().max_by_key(|&(g, n)| (n, -g)).map_or(0, |(g, _)| g)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use std::sync::atomic::AtomicUsize;

    fn same(a: &ThickenResult, b: &ThickenResult) -> bool {
        let (x, y) = (&a.mesh, &b.mesh);
        x.max_vid() == y.max_vid()
            && x.max_tid() == y.max_tid()
            && x.vertex_ids().eq(y.vertex_ids())
            && x.vertex_ids().all(|v| x.vertex(v).to_array().map(f64::to_bits) == y.vertex(v).to_array().map(f64::to_bits))
            && x.triangle_ids().eq(y.triangle_ids())
            && x.triangle_ids().all(|t| x.triangle(t) == y.triangle(t) && x.tri_group(t) == y.tri_group(t))
            && a.measured.map(f64::to_bits) == b.measured.map(f64::to_bits)
            && a.openings == b.openings
            && (a.wall_group, &a.rim_groups, a.cell_size.to_bits(), a.rebuilt) == (b.wall_group, &b.rim_groups, b.cell_size.to_bits(), b.rebuilt)
    }

    fn tube(x: f64, r: f64) -> Mesh {
        let points: Vec<DVec3> = (0..=80).map(|k| DVec3::new(x + (k as f64 * 0.05).sin(), 0.0, 20.0 * k as f64 / 80.0)).collect();
        primitives::tube_along(&points, &vec![r; points.len()], 32)
    }

    /// Two tubes side by side, overlapping along their length: Thicken's rebuild path.
    fn overlapping_tubes() -> Mesh {
        let mut m = tube(0.0, 3.0);
        m.append_mesh(&tube(1.7, 2.6), 0);
        m
    }

    #[test]
    fn shell_then_finish_is_thicken() {
        let m = tube(0.0, 3.0);
        for side in [ThickenSide::Outward, ThickenSide::Inward, ThickenSide::Centered] {
            for remesh in [true, false] {
                let params = ThickenParams { thickness: 1.5, side, cell_size: Some(0.3), remesh };
                let whole = thicken(&m, &params, None).unwrap();
                let shell = thicken_shell(&m, &params, None).unwrap();
                if !remesh {
                    assert!(same(&whole, &shell.clone().into_result()), "{side:?}: shell as the result");
                }
                assert!(same(&whole, &thicken_finish(shell, remesh, None).unwrap()), "{side:?} remesh {remesh}");
            }
        }
        let params = ThickenParams { thickness: 1.0, cell_size: Some(0.25), ..Default::default() };
        let whole = thicken(&overlapping_tubes(), &params, None).unwrap();
        assert!(whole.rebuilt);
        assert!(same(&whole, &thicken_finish(thicken_shell(&overlapping_tubes(), &params, None).unwrap(), true, None).unwrap()));
    }

    #[test]
    fn rings_beside_the_rims_are_those_of_three_passes() {
        let shell = thicken_shell(&tube(0.0, 3.0), &ThickenParams { thickness: 1.5, cell_size: Some(0.3), ..Default::default() }, None).unwrap();
        let out = &shell.mesh;
        // Three times over: every triangle sharing a corner with one already near.
        let mut near: std::collections::HashSet<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == shell.rim_group).collect();
        assert!(!near.is_empty());
        for _ in 0..3 {
            let more: Vec<u32> = near.iter().flat_map(|&tr| out.triangle(tr)).flat_map(|v| out.vertex_triangles(v)).collect();
            near.extend(more);
        }
        let marked: std::collections::HashSet<u32> = out.triangle_ids().filter(|&tr| shell.near[tr as usize]).collect();
        assert_eq!(near, marked);
    }

    /// Cancelled at any progress report, Thicken stops with a cancellation, including on its
    /// rebuild path.
    #[test]
    fn cancels_at_every_report() {
        for (m, params) in [
            (tube(0.0, 3.0), ThickenParams { thickness: 1.5, cell_size: Some(0.3), ..Default::default() }),
            (overlapping_tubes(), ThickenParams { thickness: 1.0, cell_size: Some(0.25), ..Default::default() }),
        ] {
            let calls = AtomicUsize::new(0);
            let count = |_: f32| {
                calls.fetch_add(1, Ordering::Relaxed);
                true
            };
            thicken(&m, &params, Some(&count)).unwrap();
            let total = calls.load(Ordering::Relaxed);
            assert!(total > 10, "{total} reports");
            // The last report (1.0) comes after the work and is not asked to stop it.
            for stop_at in (0..total - 1).step_by((total / 12).max(1)) {
                let seen = AtomicUsize::new(0);
                let cb = |_: f32| seen.fetch_add(1, Ordering::Relaxed) < stop_at;
                let e = thicken(&m, &params, Some(&cb)).unwrap_err();
                assert!(e.to_string().contains("cancelled"), "stopped at report {stop_at} of {total}: {e}");
                // Nothing more than a check or two after the refusal.
                assert!(seen.load(Ordering::Relaxed) <= stop_at + 2, "stopped at report {stop_at}: {} reports", seen.load(Ordering::Relaxed));
            }
        }
    }
}
