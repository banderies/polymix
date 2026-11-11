//! Stand: a base under an object and legs holding the object up off it, to set a printed
//! model on a bench (part of the model, unlike print supports).
//!
//! * The **base** ([`base_mesh`]) is a slab, a rounded rectangle or an ellipse, whose top face
//!   lies in the plane of a [`BaseFrame`].
//! * A **leg** runs from a point on the object (its *root*) to a point on the base's top (its
//!   *foot*): a column flaring out to a broad root and a broad foot ([`LegParams`]).
//!   [`plan_leg`] works out where it goes (cheap: previews and checks), [`leg_mesh`] builds it,
//!   [`join_stand`] joins the base and legs into one solid to unite with the object.
//!
//! A leg is a distance field meshed on a grid ([`mm_solid::mesh_implicit`]), the union of:
//!
//! 1. The root: a solid of revolution about the root's axis (the object's mean outward normal
//!    around the root point): a plug of the root's diameter below the surface, then a flare
//!    narrowing to the column's diameter. It is draped over the surface: a height field of
//!    the surface along the axis (rays cast down the axis from beyond the top of the flare)
//!    shifts the plug and the foot of the flare down onto it, fading out up the flare. So the
//!    flare meets the surface all round its footprint, and the plug's bottom follows the
//!    surface. Each ray goes on to the next surface below (out of the wall into a lumen, out
//!    the far side of a thin part, …) and the plug stops [`PLUG_SHARE`] of the way there
//!    (a quarter of the root's radius at most): it never crosses another surface. United with
//!    a hollow shell (a printed vessel), a leg changes only its outer surface, never what
//!    lies under the wall (the lumen).
//! 2. The column: capsules along a Hermite curve that leaves the root along its axis and
//!    comes down square to the base.
//! 3. The foot: a flare onto the base like the root's, its plug sunk halfway into the slab.
//!
//! [`plan_leg`] reports what would make a leg cut into the object instead: the object rising
//! over the root's flare ([`LegPlan::buried`]: a crotch, another branch close by) or the
//! column or the foot running into it ([`LegPlan::clearance`] below zero). Such legs should
//! not be made (Polymix's Stand tool will not accept them).

use crate::boolean::{mesh_boolean, BooleanOp};
use crate::{EditError, EditResult};
use mm_core::geom::orthonormal_basis;
use mm_core::{measure, Aabb, Bvh, DVec2, DVec3, Mesh, Ray};
use rayon::prelude::*;
use std::f64::consts::{FRAC_PI_2, TAU};

/// Where a base sits: the centre of its top face, the way up (toward the object), and the
/// direction its width runs (in the top's plane). Its depth runs along `up × width_axis`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseFrame {
    pub center: DVec3,
    pub up: DVec3,
    pub width_axis: DVec3,
}

impl BaseFrame {
    /// The frame with `up` made unit and `width` turned into the top's plane (any direction in
    /// it when `width` is along `up`).
    pub fn new(center: DVec3, up: DVec3, width: DVec3) -> Self {
        let up = up.try_normalize().unwrap_or(DVec3::Z);
        let width_axis = (width - up * width.dot(up)).try_normalize().unwrap_or_else(|| orthonormal_basis(up).0);
        Self { center, up, width_axis }
    }

    pub fn depth_axis(&self) -> DVec3 {
        self.up.cross(self.width_axis)
    }

    /// `p` as (along the width, along the depth, height above the top).
    pub fn local(&self, p: DVec3) -> DVec3 {
        let d = p - self.center;
        DVec3::new(d.dot(self.width_axis), d.dot(self.depth_axis()), d.dot(self.up))
    }

    /// The point at local coordinates `q` (see [`BaseFrame::local`]).
    pub fn world(&self, q: DVec3) -> DVec3 {
        self.center + self.width_axis * q.x + self.depth_axis() * q.y + self.up * q.z
    }

    /// `p` dropped onto the top's plane.
    pub fn project(&self, p: DVec3) -> DVec3 {
        p - self.up * (p - self.center).dot(self.up)
    }
}

/// The base's outline.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum BaseShape {
    /// A rectangle with rounded corners ([`BaseParams::corner_radius`]).
    #[default]
    Rectangle,
    Ellipse,
}

/// The base's size (model units, mm). Its top is the frame's plane; it is `thickness` deep
/// below it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BaseParams {
    pub shape: BaseShape,
    pub width: f64,
    pub depth: f64,
    pub thickness: f64,
    /// Rectangle: radius of the rounded corners (0: square corners).
    pub corner_radius: f64,
}

impl Default for BaseParams {
    fn default() -> Self {
        Self { shape: BaseShape::Rectangle, width: 100.0, depth: 70.0, thickness: 5.0, corner_radius: 8.0 }
    }
}

impl BaseParams {
    fn check(&self) -> EditResult<()> {
        for (v, what) in [(self.width, "width"), (self.depth, "depth"), (self.thickness, "thickness")] {
            if !(v > 0.0 && v.is_finite()) {
                return Err(EditError::InvalidParameter(format!("the base's {what} must be positive (it is {v})")));
            }
        }
        Ok(())
    }

    /// The corner radius as used: no more than just under half the shorter side.
    pub fn corner(&self) -> f64 {
        if self.corner_radius.is_finite() {
            self.corner_radius.clamp(0.0, 0.49 * self.width.min(self.depth))
        } else {
            0.0
        }
    }

    /// How far a point of the top's plane (width, depth) lies outside the outline (negative
    /// inside). Exact for the rectangle; for the ellipse the sign is exact and the size
    /// about right.
    pub fn outline_distance(&self, q: DVec2) -> f64 {
        match self.shape {
            BaseShape::Rectangle => {
                let rc = self.corner();
                let d = q.abs() - (DVec2::new(self.width, self.depth) * 0.5 - DVec2::splat(rc));
                d.max(DVec2::ZERO).length() + d.x.max(d.y).min(0.0) - rc
            }
            BaseShape::Ellipse => {
                let (a, b) = (0.5 * self.width, 0.5 * self.depth);
                (DVec2::new(q.x / a, q.y / b).length() - 1.0) * a.min(b)
            }
        }
    }

    /// Whether a point of the top's plane (width, depth) is on the base.
    pub fn contains(&self, q: DVec2) -> bool {
        self.outline_distance(q) <= 0.0
    }

    /// The outline in the top's plane (width, depth), counter-clockwise seen from above, its
    /// curves sampled about `spacing` apart (a rectangle's straight sides are single edges).
    pub fn outline(&self, spacing: f64) -> Vec<DVec2> {
        let spacing = if spacing > 0.0 && spacing.is_finite() { spacing } else { 0.02 * self.width.min(self.depth) };
        match self.shape {
            BaseShape::Rectangle => {
                let rc = self.corner();
                let c = DVec2::new(0.5 * self.width - rc, 0.5 * self.depth - rc);
                let arc = if rc > 1e-9 * self.width.max(self.depth) { ((FRAC_PI_2 * rc / spacing).ceil() as usize).clamp(2, 64) } else { 0 };
                let mut out = Vec::new();
                for (k, s) in [DVec2::new(1.0, 1.0), DVec2::new(-1.0, 1.0), DVec2::new(-1.0, -1.0), DVec2::new(1.0, -1.0)].into_iter().enumerate() {
                    let centre = c * s;
                    if arc == 0 {
                        out.push(centre);
                        continue;
                    }
                    let start = k as f64 * FRAC_PI_2;
                    for i in 0..=arc {
                        let a = start + FRAC_PI_2 * i as f64 / arc as f64;
                        out.push(centre + DVec2::new(a.cos(), a.sin()) * rc);
                    }
                }
                out
            }
            BaseShape::Ellipse => {
                let (a, b) = (0.5 * self.width, 0.5 * self.depth);
                // Ramanujan's perimeter.
                let perimeter = std::f64::consts::PI * (3.0 * (a + b) - ((3.0 * a + b) * (a + 3.0 * b)).sqrt());
                let n = ((perimeter / spacing).ceil() as usize).clamp(24, 512);
                (0..n).map(|i| TAU * i as f64 / n as f64).map(|t| DVec2::new(a * t.cos(), b * t.sin())).collect()
            }
        }
    }
}

/// The base: a closed slab, its top face in the frame's plane (the outline of `p` about the
/// frame's centre) and `p.thickness` deep below it, remeshed to edges about `edge` long with
/// its rims kept sharp. One face group (0).
pub fn base_mesh(frame: &BaseFrame, p: &BaseParams, edge: f64) -> EditResult<Mesh> {
    p.check()?;
    let edge = if edge > 0.0 && edge.is_finite() { edge } else { p.width.min(p.depth) / 25.0 };
    let outline = p.outline(edge);
    let m = outline.len() as u32;
    let mut pos = Vec::with_capacity(2 * m as usize + 2);
    for z in [0.0, -p.thickness] {
        pos.extend(outline.iter().map(|q| frame.world(DVec3::new(q.x, q.y, z))));
    }
    pos.push(frame.world(DVec3::ZERO));
    pos.push(frame.world(DVec3::new(0.0, 0.0, -p.thickness)));
    let (top, bottom) = (2 * m, 2 * m + 1);
    let mut tris = Vec::with_capacity(4 * m as usize);
    for i in 0..m {
        let j = (i + 1) % m;
        // Seen from above the outline runs counter-clockwise: the top's fans face up, the
        // bottom's down, the sides out.
        tris.extend([[top, i, j], [bottom, m + j, m + i], [i, m + i, m + j], [i, m + j, j]]);
    }
    let (mut mesh, _) = Mesh::from_triangles(&pos, &tris, None);
    if measure::volume(&mesh) < 0.0 {
        mesh.reverse_orientation();
    }
    let rp = mm_remesh::RemeshParams {
        target_edge_length: edge,
        iterations: 8,
        preserve_sharp_edges: Some(45.0),
        project_to_original: true,
        ..Default::default()
    };
    mm_remesh::remesh(&mut mesh, &rp, None).map_err(|e| EditError::Failed(format!("remeshing the base: {e}")))?;
    Ok(mesh.compacted().0)
}

/// A leg's shape (model units, mm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LegParams {
    /// The column's diameter.
    pub diameter: f64,
    /// Where the leg meets the object it flares out to this diameter.
    pub root_diameter: f64,
    /// Where it meets the base it flares out to this diameter.
    pub foot_diameter: f64,
    /// How far up the leg the root's flare runs (shortened on a short leg).
    pub root_flare: f64,
    /// How far up the leg the foot's flare runs.
    pub foot_flare: f64,
    /// 0..1: how far the leg keeps to its root's direction before it turns down to the base
    /// (0: straight between the flares).
    pub bend: f64,
}

impl Default for LegParams {
    fn default() -> Self {
        Self { diameter: 3.0, root_diameter: 6.0, foot_diameter: 6.0, root_flare: 4.5, foot_flare: 4.5, bend: 0.5 }
    }
}

impl LegParams {
    fn check(&self) -> EditResult<()> {
        let sizes = [(self.diameter, "diameter"), (self.root_diameter, "root diameter"), (self.foot_diameter, "foot diameter"), (self.root_flare, "root flare"), (self.foot_flare, "foot flare")];
        for (v, what) in sizes {
            if !(v > 0.0 && v.is_finite()) {
                return Err(EditError::InvalidParameter(format!("the leg's {what} must be positive (it is {v})")));
            }
        }
        if !self.bend.is_finite() {
            return Err(EditError::InvalidParameter("the leg's bend must be a number".into()));
        }
        Ok(())
    }
}

/// The surface under a root on a square grid about the root's axis, bilinear in between: its
/// depth below the root's plane (along the axis), and how deep the root's plug may go below
/// it there.
#[derive(Clone, Debug)]
struct HeightField {
    n: usize,
    half: f64,
    step: f64,
    depth: Vec<f64>,
    plug: Vec<f64>,
}

impl HeightField {
    /// The cell holding `x` and where in it (clamped to the grid).
    fn cell(&self, x: f64) -> (usize, f64) {
        let s = ((x + self.half) / self.step).clamp(0.0, (self.n - 1) as f64);
        let i = (s.floor() as usize).min(self.n - 2);
        (i, s - i as f64)
    }

    /// `values` at (x, y) and its gradient (flat beyond the grid).
    fn sample(&self, values: &[f64], x: f64, y: f64) -> (f64, f64, f64) {
        let ((i, fx), (j, fy)) = (self.cell(x), self.cell(y));
        let at = |i: usize, j: usize| values[j * self.n + i];
        let (a, b, c, d) = (at(i, j), at(i + 1, j), at(i, j + 1), at(i + 1, j + 1));
        let h = a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) + c * (1.0 - fx) * fy + d * fx * fy;
        let gx = if x.abs() < self.half { ((b - a) * (1.0 - fy) + (d - c) * fy) / self.step } else { 0.0 };
        let gy = if y.abs() < self.half { ((c - a) * (1.0 - fx) + (d - b) * fx) / self.step } else { 0.0 };
        (h, gx, gy)
    }
}

/// What a leg will be, worked out from the object, the base and its ends ([`plan_leg`]).
#[derive(Clone, Debug)]
pub struct LegPlan {
    pub params: LegParams,
    /// The root's point on the object's surface.
    pub root: DVec3,
    /// The root's axis: the object's mean outward normal around the root.
    pub axis: DVec3,
    /// The foot's point on the base's top.
    pub foot: DVec3,
    /// The base's up.
    pub up: DVec3,
    /// The column's centreline, from the top of the root's flare to the top of the foot's.
    pub path: Vec<DVec3>,
    /// The flares' lengths as used (shortened on a short leg).
    pub root_flare: f64,
    pub foot_flare: f64,
    /// How deep the root sinks into the object at its centre (below the surface, along the
    /// axis). All over its footprint it stops short of the next surface along the axis (at
    /// most [`PLUG_SHARE`] of the way there), so it never reaches through a wall.
    pub embed: f64,
    /// The object's thickness under the root's centre, along the axis (`None`: no far side
    /// found near it).
    pub wall: Option<f64>,
    /// Share of the root's footprint with no surface under it (0..1): the root is wider than
    /// what it stands on there.
    pub overhang: f64,
    /// The object rises over the root's flare somewhere around it (in a crotch, or another
    /// branch close by): the leg would cut into it there, so it should not be made.
    pub buried: bool,
    /// The closest the column and the foot's flare come to the object, less their radius
    /// (negative where they run into it: the leg would cut into it there). The stretch next
    /// to the root is left out.
    pub clearance: f64,
    /// How deep the foot sinks into the base.
    pub foot_embed: f64,
    side: (DVec3, DVec3),
    field: HeightField,
}

/// Samples along a leg's column.
const PATH_SAMPLES: usize = 48;
/// How far a root's plug may go toward the next surface below the one it stands on.
pub const PLUG_SHARE: f64 = 0.6;
/// Height field samples along each side.
const FIELD_SAMPLES: usize = 41;

fn hermite(p0: DVec3, m0: DVec3, p1: DVec3, m1: DVec3, t: f64) -> DVec3 {
    let (t2, t3) = (t * t, t * t * t);
    p0 * (2.0 * t3 - 3.0 * t2 + 1.0) + m0 * (t3 - 2.0 * t2 + t) + p1 * (-2.0 * t3 + 3.0 * t2) + m1 * (t3 - t2)
}

/// The object's outward direction around `p`: the area-weighted mean normal of the triangles
/// within `radius` that face within 60° of `n` (the far side of a thin part is left out).
fn root_axis(mesh: &Mesh, bvh: &Bvh, p: DVec3, n: DVec3, radius: f64) -> DVec3 {
    let mut sum = DVec3::ZERO;
    bvh.query_sphere(mesh, p, radius, |t| {
        let c = mesh.tri_cross(t);
        if c.normalize_or_zero().dot(n) > 0.5 {
            sum += c;
        }
    });
    sum.try_normalize().unwrap_or(n)
}

/// Where a leg's foot goes when it is not placed by hand: on the base's top below the root,
/// stepped out the way the root faces (by half the root's height) when that is still on the
/// base, so a leg leaving the side of the object splays out.
pub fn default_foot(mesh: &Mesh, bvh: &Bvh, root: DVec3, base: &BaseFrame, base_params: &BaseParams, root_diameter: f64) -> DVec3 {
    let Some(near) = bvh.nearest(mesh, root, f64::INFINITY) else { return base.project(root) };
    let axis = root_axis(mesh, bvh, near.point, mesh.tri_normal(near.tid), 0.5 * root_diameter);
    let height = (near.point - base.center).dot(base.up).max(0.0);
    let out = axis - base.up * axis.dot(base.up);
    let stepped = base.project(near.point + out * 0.5 * height);
    let q = base.local(stepped);
    if base_params.contains(DVec2::new(q.x, q.y)) {
        stepped
    } else {
        base.project(near.point)
    }
}

/// Work out a leg from `root` (moved onto the surface of `mesh`) to `foot` (dropped onto the
/// base's top). Fails when the root is not above the base.
pub fn plan_leg(mesh: &Mesh, bvh: &Bvh, root: DVec3, foot: DVec3, base: &BaseFrame, base_thickness: f64, p: &LegParams) -> EditResult<LegPlan> {
    p.check()?;
    if !(base_thickness > 0.0 && base_thickness.is_finite()) {
        return Err(EditError::InvalidParameter(format!("the base's thickness must be positive (it is {base_thickness})")));
    }
    let (r, rr) = (0.5 * p.diameter, 0.5 * p.root_diameter);
    let near = bvh.nearest(mesh, root, f64::INFINITY).ok_or_else(|| EditError::Failed("the object has no surface".into()))?;
    let root = near.point;
    let axis = root_axis(mesh, bvh, root, mesh.tri_normal(near.tid), rr);
    let up = base.up;
    let foot = base.project(foot);
    let height = (root - base.center).dot(up);
    if height <= r {
        return Err(EditError::InvalidParameter("the leg's root is not above the base's top".into()));
    }
    // Flares that would meet (a short leg, its root facing down) are shortened, leaving the
    // column a fifth of the height.
    let drop = (-axis.dot(up)).max(0.0);
    let reach = p.root_flare * drop + p.foot_flare;
    let s = if reach > 0.8 * height { 0.8 * height / reach } else { 1.0 };
    let (fr, ff) = (p.root_flare * s, p.foot_flare * s);

    // The surface under the root: rays down the axis from above the top of the flare.
    let (e1, e2) = orthonormal_basis(axis);
    let half = 1.15 * rr.max(r);
    let n = FIELD_SAMPLES;
    let step = 2.0 * half / (n - 1) as f64;
    let top = fr + 0.5 * rr.max(r);
    let below = 2.0 * half;
    let wall_reach = 6.0 * half;
    // The plug's depth where nothing stops it (a solid part).
    let cap = 0.25 * rr.max(r);
    enum Hit {
        Surface {
            depth: f64,
            /// To the next surface down the axis (out of the wall, into a cavity, …).
            through: Option<f64>,
        },
        /// The ray started inside the object: it rises over the flare here.
        Buried,
        Miss,
    }
    let hits: Vec<Hit> = (0..n * n)
        .into_par_iter()
        .map(|k| {
            let (x, y) = (-half + (k % n) as f64 * step, -half + (k / n) as f64 * step);
            let ray = Ray::new(root + e1 * x + e2 * y + axis * top, -axis);
            let Some(hit) = bvh.raycast_filtered(mesh, &ray, top + below, |_| true) else { return Hit::Miss };
            if mesh.tri_normal(hit.tid).dot(axis) <= 0.0 {
                return Hit::Buried;
            }
            // On down the axis to the next surface, whatever it is. The hit triangle's
            // neighbours are left out: one met at their shared corner is the same surface.
            let corners = mesh.triangle(hit.tid);
            let on = Ray::new(hit.point, -axis);
            let next = bvh.raycast_range(mesh, &on, 1e-9 * (half + top), wall_reach, |t| !mesh.triangle(t).iter().any(|v| corners.contains(v)));
            Hit::Surface { depth: hit.t - top, through: next.map(|h| h.t) }
        })
        .collect();
    let in_footprint = |k: usize| {
        let (x, y) = (-half + (k % n) as f64 * step, -half + (k / n) as f64 * step);
        x * x + y * y <= rr * rr
    };
    let lowest = hits.iter().filter_map(|h| if let Hit::Surface { depth, .. } = h { Some(*depth) } else { None }).fold(0.0, f64::max);
    let (mut footprint, mut missed, mut buried) = (0usize, 0usize, false);
    let floor = -0.9 * fr;
    let (mut depth, mut plug) = (Vec::with_capacity(n * n), Vec::with_capacity(n * n));
    for (k, h) in hits.iter().enumerate() {
        let mine = in_footprint(k);
        footprint += mine as usize;
        let (d, p) = match *h {
            Hit::Surface { depth: d, through } => {
                buried |= mine && d < floor;
                // A little at least, so the root's underside never lies on the surface (where
                // the ray only grazes the edge of a part, the next surface is the same one).
                (d.max(floor), through.map_or(cap, |t| (PLUG_SHARE * t).clamp(0.05 * cap, cap)))
            }
            Hit::Buried => {
                buried |= mine;
                (floor, 0.0)
            }
            Hit::Miss => {
                missed += mine as usize;
                (lowest, cap)
            }
        };
        depth.push(d);
        plug.push(p);
    }
    let centre = (n / 2) * n + n / 2;
    let wall = match hits[centre] {
        Hit::Surface { through, .. } => through,
        _ => None,
    };
    let embed = plug[centre].max(1e-6 * cap);
    let field = HeightField { n, half, step, depth, plug };
    let overhang = missed as f64 / footprint.max(1) as f64;

    let (q0, q1) = (root + axis * fr, foot + up * ff);
    let k = 2.0 * p.bend.clamp(0.0, 1.0) * q0.distance(q1);
    let path: Vec<DVec3> = (0..=PATH_SAMPLES).map(|i| hermite(q0, axis * k, q1, -up * k, i as f64 / PATH_SAMPLES as f64)).collect();
    // How near the column and the foot's flare come to the object, away from the root it
    // stands on.
    let skip = rr + r + fr;
    let rf = 0.5 * p.foot_diameter;
    let flare: Vec<(DVec3, f64)> = (0..=8).map(|k| k as f64 / 8.0).map(|u| (foot + up * (u * ff), r + (rf - r) * (1.0 - u).powi(2))).collect();
    let clearance = path
        .iter()
        .map(|&q| (q, r))
        .chain(flare)
        .collect::<Vec<_>>()
        .par_iter()
        .filter(|(q, _)| q.distance(root) > skip)
        .filter_map(|&(q, radius)| bvh.nearest(mesh, q, 2.0 * radius).map(|h| h.dist_sq.sqrt() - radius))
        .reduce(|| f64::INFINITY, f64::min);
    Ok(LegPlan {
        params: *p,
        root,
        axis,
        foot,
        up,
        path,
        root_flare: fr,
        foot_flare: ff,
        embed,
        wall,
        overhang,
        buried,
        clearance,
        foot_embed: 0.5 * base_thickness,
        side: (e1, e2),
        field,
    })
}

impl LegPlan {
    /// The root's outline on the object's surface (where its flare meets it), `n` points.
    pub fn footprint(&self, n: usize) -> Vec<DVec3> {
        let rr = 0.5 * self.params.root_diameter;
        let (e1, e2) = self.side;
        (0..n)
            .map(|i| {
                let a = TAU * i as f64 / n as f64;
                let (x, y) = (rr * a.cos(), rr * a.sin());
                self.root + e1 * x + e2 * y - self.axis * self.field.sample(&self.field.depth, x, y).0
            })
            .collect()
    }

    /// The foot's outline on the base's top, `n` points.
    pub fn foot_ring(&self, n: usize) -> Vec<DVec3> {
        let (e1, e2) = orthonormal_basis(self.up);
        let rf = 0.5 * self.params.foot_diameter;
        (0..n).map(|i| TAU * i as f64 / n as f64).map(|a| self.foot + (e1 * a.cos() + e2 * a.sin()) * rf).collect()
    }

    /// The leg's centreline from the root to the foot (through the flares and the column).
    pub fn centreline(&self) -> Vec<DVec3> {
        let mut out = vec![self.root];
        out.extend_from_slice(&self.path);
        out.push(self.foot);
        out
    }

    /// The grid spacing [`leg_mesh`] meshes the leg with: a fifth of the column's radius, no
    /// more than half the root's depth at its centre (so the plug comes out as deep as it
    /// should), and no less than a twelfth of the radius.
    pub fn cell_size(&self) -> f64 {
        let p = &self.params;
        let r = 0.5 * p.diameter;
        let fine = (0.2 * r).min(0.5 * self.embed).min(p.root_diameter / 16.0).min(p.foot_diameter / 16.0);
        fine.max(r / 12.0)
    }

    fn bounds(&self) -> Aabb {
        let p = &self.params;
        let (r, rr, rf) = (0.5 * p.diameter, 0.5 * p.root_diameter, 0.5 * p.foot_diameter);
        let mut b = Aabb::from_points(self.path.iter().copied()).expanded(r);
        let (e1, e2) = self.side;
        let h = &self.field;
        let deepest = h.depth.iter().zip(&h.plug).map(|(d, p)| d + p).fold(0.0, f64::max);
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            for z in [-deepest, self.root_flare] {
                b.include(self.root + e1 * (x * h.half) + e2 * (y * h.half) + self.axis * z);
            }
        }
        let (f1, f2) = orthonormal_basis(self.up);
        for (x, y) in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0), (1.0, 1.0)] {
            for z in [-self.foot_embed, self.foot_flare] {
                b.include(self.foot + (f1 * x + f2 * y) * rf + self.up * z);
            }
        }
        b.expanded(0.05 * rr.max(rf))
    }
}

/// A flare's half-profile mirrored into a closed polygon in (ρ, z): a plug of radius `wide`
/// from z = −`plug` up to 0, then a flare narrowing to `narrow` at z = `len` (concave, steepest
/// at its foot), closed flat there.
fn flare_polygon(wide: f64, narrow: f64, len: f64, plug: f64) -> Vec<DVec2> {
    const K: usize = 16;
    let mut right = vec![DVec2::new(wide, -plug)];
    right.extend((0..=K).map(|k| k as f64 / K as f64).map(|u| DVec2::new(narrow + (wide - narrow) * (1.0 - u).powi(2), u * len)));
    let mut poly = right.clone();
    poly.extend(right.iter().rev().map(|p| DVec2::new(-p.x, p.y)));
    poly
}

/// Signed distance from a simple polygon (negative inside).
fn polygon_distance(poly: &[DVec2], p: DVec2) -> f64 {
    let mut d = (p - poly[0]).length_squared();
    let mut inside = false;
    let mut j = poly.len() - 1;
    for i in 0..poly.len() {
        let (a, b) = (poly[i], poly[j]);
        let (e, w) = (b - a, p - a);
        let q = w - e * (w.dot(e) / e.length_squared()).clamp(0.0, 1.0);
        d = d.min(q.length_squared());
        let (c1, c2, c3) = (p.y >= a.y, p.y < b.y, e.x * w.y > e.y * w.x);
        if (c1 && c2 && c3) || (!c1 && !c2 && !c3) {
            inside = !inside;
        }
        j = i;
    }
    if inside {
        -d.sqrt()
    } else {
        d.sqrt()
    }
}

fn segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Build the leg `plan` describes: a closed, manifold mesh, one face group (0). Cancelled when
/// `progress` returns false.
pub fn leg_mesh(plan: &LegPlan, progress: mm_solid::Progress) -> EditResult<Mesh> {
    let p = &plan.params;
    let (r, rr, rf) = (0.5 * p.diameter, 0.5 * p.root_diameter, 0.5 * p.foot_diameter);
    let (fr, ff) = (plan.root_flare, plan.foot_flare);
    let cap = plan.field.plug.iter().copied().fold(0.0, f64::max);
    let root_poly = flare_polygon(rr, r, fr, cap + plan.cell_size());
    let foot_poly = flare_polygon(rf, r, ff, plan.foot_embed);
    let (e1, e2) = plan.side;
    let (f1, f2) = orthonormal_basis(plan.up);
    let field = &plan.field;
    let segments: Vec<(DVec3, DVec3)> = plan.path.windows(2).map(|w| (w[0], w[1])).collect();
    // The root, draped: `zu` is the height in the undraped solid of the point at height `z`
    // over (x, y), where the surface is `h` below the root's plane. Below the surface the
    // solid is shifted down by `h`; up the flare the shift fades out to nothing at its top.
    // The plug ends where it may go there, its bottom following the surface.
    let root = |q: DVec3| {
        let d = q - plan.root;
        let (x, y, z) = (d.dot(e1), d.dot(e2), d.dot(plan.axis));
        let (h, hx, hy) = field.sample(&field.depth, x, y);
        let (deep, dx, dy) = field.sample(&field.plug, x, y);
        let (zu, g) = if z < -h {
            (z + h, DVec3::new(hx, hy, 1.0))
        } else if z <= fr {
            let s = 1.0 + h / fr;
            let fade = (1.0 - z / fr) / (s * s);
            ((z + h) / s, DVec3::new(hx * fade, hy * fade, 1.0 / s))
        } else {
            (z, DVec3::Z)
        };
        let sd = polygon_distance(&root_poly, DVec2::new((x * x + y * y).sqrt(), zu)).max(-deep - zu);
        // The draping stretches distances: scaled back, the field stays about a distance.
        sd / ((1.0 + g.length_squared() + dx * dx + dy * dy) * 0.5).sqrt().max(1.0)
    };
    let foot = |q: DVec3| {
        let d = q - plan.foot;
        let (x, y) = (d.dot(f1), d.dot(f2));
        polygon_distance(&foot_poly, DVec2::new((x * x + y * y).sqrt(), d.dot(plan.up)))
    };
    let column = |q: DVec3| segments.iter().map(|&(a, b)| segment_distance(q, a, b)).fold(f64::INFINITY, f64::min) - r;
    let f = |q: DVec3| root(q).min(column(q)).min(foot(q));
    let mesh = mm_solid::mesh_implicit(f, 2.0, &plan.bounds(), plan.cell_size(), progress).map_err(|e| match e {
        mm_solid::SolidError::Cancelled => EditError::Failed("cancelled".into()),
        e => EditError::Failed(format!("building the leg: {e}")),
    })?;
    if mesh.triangle_count() == 0 {
        return Err(EditError::Failed("the leg came out empty".into()));
    }
    Ok(mesh)
}

/// The base and the legs joined into one solid (exact unions): the base face group 0, leg
/// `k` group `k + 1`.
pub fn join_stand(base: &Mesh, legs: &[&Mesh]) -> EditResult<Mesh> {
    let mut stand = base.clone();
    for leg in legs {
        stand = mesh_boolean(&stand, leg, BooleanOp::Union)?.mesh;
    }
    Ok(stand)
}
