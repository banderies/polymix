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

use crate::boolean::{mesh_boolean, BooleanOp};
use crate::util::{loop_outward_direction, mean, positions};
use crate::{EditError, EditResult};
use mm_core::{measure, Bvh, DVec3, Mesh};
use rayon::prelude::*;
use std::f64::consts::TAU;

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
    /// Face groups of the new wall and of the rims (the kept surface keeps its groups).
    pub wall_group: i32,
    pub rim_group: i32,
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
    let t = params.thickness;
    if !(t > 0.0 && t.is_finite()) {
        return Err(EditError::InvalidParameter("the wall thickness must be positive".into()));
    }
    if mesh.triangle_count() == 0 {
        return Err(EditError::InvalidParameter("the mesh is empty".into()));
    }
    let report = |x: f32| progress.is_none_or(|p| p(x));
    let ends = openings(mesh);
    // The grid: a sixth of the thickness, coarser where the narrow band around a large
    // surface would hold more than about 40 million cells.
    let area = measure::surface_area(mesh);
    let h = params.cell_size.unwrap_or_else(|| (t / 6.0).max((area * 3.0 * t / 4e7).cbrt())).max(1e-6 * t);

    // 1. The lumen: the surface, oriented outward, each opening prolonged straight on and
    // closed. The prolongations and their ends get groups of their own.
    let mut surface = mesh.clone();
    let groups: Vec<i32> = surface.triangle_ids().map(|tr| surface.tri_group(tr)).collect();
    let top = groups.iter().copied().max().unwrap_or(0);
    let main = most_common(&groups);
    let (prolonged, closing, new_side, wall_group, rim_group) = (top + 1, top + 2, top + 3, top + 4, top + 5);
    // (Far enough that the offset near the opening does not see the closing cap.)
    let reach = t + 4.0 * h;
    let mut closed = surface.clone();
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
        surface.reverse_orientation();
        closed.reverse_orientation();
    }
    if !report(0.05) {
        return Err(EditError::Failed("cancelled".into()));
    }

    // 2. The offset solid(s), as the wall's new side.
    let offset = |d: f64, from: f32, to: f32| -> EditResult<Mesh> {
        let sub = |x: f32| report(from + (to - from) * x);
        let solid_params = mm_solid::MakeSolidParams { cell_size: h, offset: d, transfer_colors: false, ..Default::default() };
        let mut m = mm_solid::make_solid(&closed, &solid_params, Some(&sub)).map_err(|e| EditError::Failed(format!("the offset surface: {e}")))?;
        for tr in m.triangle_ids().collect::<Vec<_>>() {
            m.set_tri_group(tr, new_side);
        }
        Ok(m)
    };
    let (outer, inner) = match params.side {
        ThickenSide::Outward => (offset(t, 0.05, 0.7)?, closed.clone()),
        ThickenSide::Inward => (closed.clone(), offset(-t, 0.05, 0.7)?),
        ThickenSide::Centered => (offset(0.5 * t, 0.05, 0.4)?, offset(-0.5 * t, 0.4, 0.7)?),
    };
    let measured = match params.side {
        ThickenSide::Outward => measure_wall(&outer, &closed),
        ThickenSide::Inward => measure_wall(&inner, &closed),
        ThickenSide::Centered => measure_wall(&outer, &inner),
    };
    // Each opening's rim stands this far in front of its plane.
    let ahead = 0.05 * t;
    // An inner solid that does not reach through an opening would close it.
    for (i, (_, o)) in ends.iter().enumerate() {
        let through = inner.vertex_ids().map(|v| inner.vertex(v) - o.centre).any(|q| q.dot(o.normal) > ahead + h && (q - o.normal * q.dot(o.normal)).length() < o.extent + h);
        if !through {
            return Err(EditError::Failed(format!(
                "opening {} (⌀ {:.1}) is too narrow for a wall this thick: the lumen closes up inside it; thicken outward, or thinner",
                i + 1,
                2.0 * o.radius
            )));
        }
    }

    // 3. The outer solid cut back by a cylinder standing on each opening, then the inner
    // solid taken out of it.
    // (After each boolean the second mesh's triangles, whose groups follow the first's,
    // are told apart by group and relabelled at once.)
    let carve = |outer: &Mesh, inner: &Mesh| -> EditResult<Mesh> {
        let mut shell = outer.clone();
        for (i, (_, o)) in ends.iter().enumerate() {
            let cyl = cylinder(o.centre + o.normal * ahead, o.normal, o.extent + t + 4.0 * h, reach + t + 4.0 * h, h);
            let before = shell.max_group_id();
            let cut = mesh_boolean(&shell, &cyl, BooleanOp::Difference).map_err(|e| EditError::Failed(format!("cutting the wall back at opening {}: {e}", i + 1)))?;
            shell = cut.mesh;
            for tr in shell.triangle_ids().collect::<Vec<_>>() {
                if shell.tri_group(tr) > before {
                    shell.set_tri_group(tr, rim_group);
                }
            }
        }
        let before = shell.max_group_id();
        let hollow = mesh_boolean(&shell, inner, BooleanOp::Difference).map_err(|e| EditError::Failed(format!("taking the lumen out of the wall: {e}")))?;
        let from_inner = hollow.group_offset;
        let mut out = hollow.mesh;
        // Groups: the kept surface keeps its own (its prolongations take its commonest), the
        // new side and the rims one each.
        for tr in out.triangle_ids().collect::<Vec<_>>() {
            let raw = out.tri_group(tr);
            let g = if raw > before { raw - from_inner } else { raw };
            let g = if raw <= before && raw == rim_group {
                rim_group
            } else if g == new_side {
                wall_group
            } else if g == prolonged || g == closing {
                main
            } else {
                g
            };
            out.set_tri_group(tr, g);
        }
        Ok(out)
    };
    let mut rebuilt = false;
    let mut out = match carve(&outer, &inner) {
        Ok(m) => m,
        Err(first) if params.side != ThickenSide::Centered => {
            // The surface runs into itself (overlapping pieces merged without a union), which
            // the exact booleans cannot take: its side of the wall is rebuilt as the distance
            // field's zero set, which cannot run into itself (snapping it back onto the
            // surface could, where the surface does).
            let rebuild = mm_solid::MakeSolidParams { mode: mm_solid::SolidMode::Fast, cell_size: h, transfer_groups: true, transfer_colors: false, ..Default::default() };
            let clean = mm_solid::make_solid(&closed, &rebuild, None).map_err(|_| first.clone())?;
            rebuilt = true;
            match params.side {
                ThickenSide::Outward => carve(&outer, &clean),
                _ => carve(&clean, &inner),
            }
            .map_err(|_| first)?
        }
        Err(e) => return Err(e),
    };
    if !report(0.85) {
        return Err(EditError::Failed("cancelled".into()));
    }

    // The new wall smoothed and remeshed to the surface's edge length, away from the rims:
    // three rings of triangles beside each rim stay as they are (the rims, flat, too), since
    // moving a vertex next to that corner could carry it across the rim's plane.
    // - Smoothing: where the wall meets itself in a crease (outside a junction of branches)
    //   the grid leaves it stair-stepped; a few volume-keeping (Taubin) steps even it out,
    //   moving the crease outward, so the wall only thickens a little there.
    // - A remesh, or smoothing, that made the wall run into anything is undone.
    let mut near: std::collections::HashSet<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == rim_group).collect();
    for _ in 0..3 {
        let more: Vec<u32> = near.iter().flat_map(|&tr| out.triangle(tr)).flat_map(|v| out.vertex_triangles(v)).collect();
        near.extend(more);
    }
    {
        let mut free: Vec<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == wall_group).flat_map(|tr| out.triangle(tr)).collect();
        free.sort_unstable();
        free.dedup();
        free.retain(|&v| out.vertex_triangles(v).iter().all(|tr| out.tri_group(*tr) == wall_group && !near.contains(tr)));
        let before = out.clone();
        taubin(&mut out, &free, 20);
        let moved: Vec<u32> = free.iter().flat_map(|&v| out.vertex_triangles(v)).collect();
        if runs_into_anything(&out, &moved) {
            out = before;
        }
    }
    if params.remesh {
        let part: Vec<u32> = out.triangle_ids().filter(|&tr| out.tri_group(tr) == wall_group && !near.contains(&tr)).collect();
        let rp = mm_remesh::RemeshParams {
            target_edge_length: mesh.average_edge_length(),
            iterations: 6,
            preserve_group_boundaries: true,
            region: Some(part),
            ..mm_remesh::RemeshParams::with_target(mesh.average_edge_length())
        };
        let before = out.clone();
        let report = mm_remesh::remesh(&mut out, &rp, None).map_err(|e| EditError::Failed(format!("remeshing the new wall: {e}")))?;
        if let Some(region) = report.region_triangles {
            if runs_into_anything(&out, &region) {
                out = before;
            }
        }
    }

    report(1.0);
    Ok(ThickenResult { mesh: out, openings: ends.into_iter().map(|(_, o)| o).collect(), wall_group, rim_group, measured, cell_size: h, rebuilt })
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
/// wall, `to`. Least, mean and most.
fn measure_wall(from: &Mesh, to: &Mesh) -> [f64; 3] {
    let bvh = Bvh::build(to);
    let ids: Vec<u32> = from.vertex_ids().collect();
    let d: Vec<f64> = ids.par_iter().filter_map(|&v| bvh.nearest(to, from.vertex(v), f64::INFINITY).map(|h| h.dist_sq.sqrt())).collect();
    if d.is_empty() {
        return [0.0; 3];
    }
    [d.iter().copied().fold(f64::INFINITY, f64::min), d.iter().sum::<f64>() / d.len() as f64, d.iter().copied().fold(0.0, f64::max)]
}

/// Whether any of `tris` cuts through another triangle of `m` (neighbours sharing a vertex
/// aside).
fn runs_into_anything(m: &Mesh, tris: &[u32]) -> bool {
    let bvh = Bvh::build(m);
    tris.par_iter().any(|&t| {
        let (tv, tp) = (m.triangle(t), m.tri_positions(t));
        let mut hit = false;
        bvh.query_aabb(&mm_core::Aabb::from_points(tp), |o| {
            if !hit && o != t && !m.triangle(o).iter().any(|v| tv.contains(v)) && mm_core::intersect::triangles_intersect(tp, m.tri_positions(o)) {
                hit = true;
            }
        });
        hit
    })
}

/// Volume-keeping smoothing (Taubin λ|μ steps) of the vertices `free`, the rest held.
fn taubin(m: &mut Mesh, free: &[u32], iterations: usize) {
    let rings: Vec<Vec<u32>> = free.iter().map(|&v| m.vertex_neighbors(v).collect()).collect();
    for _ in 0..iterations {
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
            for (&v, p) in free.iter().zip(moved) {
                m.set_vertex(v, p);
            }
        }
    }
}

/// The commonest of `groups` (0 when empty).
fn most_common(groups: &[i32]) -> i32 {
    let mut count: std::collections::HashMap<i32, usize> = std::collections::HashMap::new();
    for &g in groups {
        *count.entry(g).or_default() += 1;
    }
    count.into_iter().max_by_key(|&(g, n)| (n, -g)).map_or(0, |(g, _)| g)
}
