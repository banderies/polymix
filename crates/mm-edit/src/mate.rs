//! Mating features for posing kit parts (the Mesh workspace's Assemble tools).
//!
//! Miniature kits join their parts with a few kinds of joint surface: a ball in a socket, a
//! round pin in a hole, and a flat-ended peg (often tapered and rectangular, so it fits only
//! one way round) in a matching recess. [`recognize`] finds such a surface from one click: it
//! fits a sphere, a cylinder and a plane to a small patch around the clicked triangle, grows
//! each over the neighbouring triangles that lie on the fitted shape (refitting as it grows),
//! and keeps the shape that explains the most surface. [`fit_region`] fits a chosen set of
//! triangles instead (a selection). [`seat`] is the rigid move that puts one part's feature on
//! another's, and [`suggest_joint`] the joint the two make. Sculpted joints are seldom true
//! spheres: [`contact_fit`] then moves one surface onto the other, so they touch rather than
//! share a centre.

use crate::fit::{fit_circle, fit_sphere};
use mm_core::hash::{fast_set_with_capacity, FastSet};
use mm_core::linalg::{add_outer, sym_eigen3};
use mm_core::{Bvh, DMat3, DQuat, DVec2, DVec3, Mesh, INVALID};
use std::f64::consts::{FRAC_PI_2, PI};

/// The kinds of surface a joint is made of.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum ShapeKind {
    /// A ball or a socket.
    Sphere,
    /// A pin or a hole.
    Cylinder,
    /// A flat face (a peg's end, the floor of a recess).
    Plane,
}

impl ShapeKind {
    pub const ALL: [ShapeKind; 3] = [ShapeKind::Sphere, ShapeKind::Cylinder, ShapeKind::Plane];

    pub fn label(self) -> &'static str {
        match self {
            ShapeKind::Sphere => "Ball / socket",
            ShapeKind::Cylinder => "Pin / hole",
            ShapeKind::Plane => "Flat face",
        }
    }
}

/// A fitted shape, in world coordinates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Shape {
    Sphere {
        centre: DVec3,
        radius: f64,
    },
    /// `centre` is on the axis, halfway along the surface's extent (`length`) along it. The
    /// axis points out of the part where that is clear (from a pin's body towards its tip, out
    /// of a hole's opening).
    Cylinder { centre: DVec3, axis: DVec3, radius: f64, length: f64 },
    /// `normal` faces out of the part. The face's outline fits in the rectangle centred at
    /// `centre` with half sizes `half` along `u` (the long side) and `normal × u`. `round`: the
    /// outline fills about π/4 of that rectangle, as a disc does.
    Plane { centre: DVec3, normal: DVec3, u: DVec3, half: DVec2, round: bool },
}

impl Shape {
    pub fn kind(&self) -> ShapeKind {
        match self {
            Shape::Sphere { .. } => ShapeKind::Sphere,
            Shape::Cylinder { .. } => ShapeKind::Cylinder,
            Shape::Plane { .. } => ShapeKind::Plane,
        }
    }

    /// Where the shape is: the sphere's centre, the middle of the cylinder's axis, the centre
    /// of the flat face.
    pub fn origin(&self) -> DVec3 {
        match *self {
            Shape::Sphere { centre, .. } | Shape::Cylinder { centre, .. } | Shape::Plane { centre, .. } => centre,
        }
    }

    /// The cylinder's axis or the face's normal (none for a sphere).
    pub fn axis(&self) -> Option<DVec3> {
        match *self {
            Shape::Sphere { .. } => None,
            Shape::Cylinder { axis, .. } => Some(axis),
            Shape::Plane { normal, .. } => Some(normal),
        }
    }

    /// The radius, or half the face's diagonal.
    pub fn size(&self) -> f64 {
        match *self {
            Shape::Sphere { radius, .. } | Shape::Cylinder { radius, .. } => radius,
            Shape::Plane { half, .. } => half.length(),
        }
    }

    /// Signed distance of `p` from the surface: positive outside the sphere or cylinder, above
    /// the plane.
    pub fn distance(&self, p: DVec3) -> f64 {
        match *self {
            Shape::Sphere { centre, radius } => (p - centre).length() - radius,
            Shape::Cylinder { centre, axis, radius, .. } => radial(p, centre, axis).length() - radius,
            Shape::Plane { centre, normal, .. } => (p - centre).dot(normal),
        }
    }

    /// The unit direction away from the centre or axis at `p` (the normal of a plane).
    pub fn outward(&self, p: DVec3) -> DVec3 {
        match *self {
            Shape::Sphere { centre, .. } => (p - centre).normalize_or_zero(),
            Shape::Cylinder { centre, axis, .. } => radial(p, centre, axis).normalize_or_zero(),
            Shape::Plane { normal, .. } => normal,
        }
    }

    /// The shape moved by the rigid motion `x ↦ rot x + shift`.
    pub fn moved(&self, rot: DQuat, shift: DVec3) -> Shape {
        let at = |p: DVec3| rot * p + shift;
        match *self {
            Shape::Sphere { centre, radius } => Shape::Sphere { centre: at(centre), radius },
            Shape::Cylinder { centre, axis, radius, length } => Shape::Cylinder { centre: at(centre), axis: rot * axis, radius, length },
            Shape::Plane { centre, normal, u, half, round } => Shape::Plane { centre: at(centre), normal: rot * normal, u: rot * u, half, round },
        }
    }
}

/// `p − centre` without its part along `axis`.
fn radial(p: DVec3, centre: DVec3, axis: DVec3) -> DVec3 {
    let d = p - centre;
    d - axis * d.dot(axis)
}

/// A recognized joint surface.
#[derive(Clone, Debug)]
pub struct Feature {
    pub shape: Shape,
    /// It bulges out of the part (a ball, a pin) rather than into it (a socket, a hole). Flat
    /// faces count as convex.
    pub convex: bool,
    /// The triangles it is made of.
    pub tris: Vec<u32>,
    pub area: f64,
    /// Root mean square distance of its vertices from the shape.
    pub rms: f64,
    /// The way from the feature to the rest of its part (unit), if there is a clear one: from a
    /// ball along its limb, into the part from a socket or a hole, behind a flat face.
    pub body: Option<DVec3>,
}

impl Feature {
    /// What the surface is: ball, socket, pin, hole or flat face.
    pub fn name(&self) -> &'static str {
        match (self.shape.kind(), self.convex) {
            (ShapeKind::Sphere, true) => "ball",
            (ShapeKind::Sphere, false) => "socket",
            (ShapeKind::Cylinder, true) => "pin",
            (ShapeKind::Cylinder, false) => "hole",
            (ShapeKind::Plane, _) => "flat face",
        }
    }

    /// e.g. "ball r 2.31", "pin r 1.00 × 4.20", "flat face 3.00 × 2.00".
    pub fn describe(&self) -> String {
        match self.shape {
            Shape::Sphere { radius, .. } => format!("{} r {radius:.2}", self.name()),
            Shape::Cylinder { radius, length, .. } => format!("{} r {radius:.2} × {length:.2}", self.name()),
            Shape::Plane { half, round: true, .. } => format!("round {}, diameter {:.2}", self.name(), half.x + half.y),
            Shape::Plane { half, .. } => format!("{} {:.2} × {:.2}", self.name(), 2.0 * half.x, 2.0 * half.y),
        }
    }

    /// The feature moved by the rigid motion `x ↦ rot x + shift` (its triangles stay).
    pub fn moved(&self, rot: DQuat, shift: DVec3) -> Feature {
        Feature { shape: self.shape.moved(rot, shift), body: self.body.map(|b| rot * b), ..self.clone() }
    }
}

/// How [`recognize`] looks for a joint surface.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct RecognizeOptions {
    /// The shape to look for (none: whichever explains the most surface).
    pub kind: Option<ShapeKind>,
    /// How far the surface may stray from a sphere or cylinder, as a fraction of its radius.
    pub tolerance: f64,
    /// How far it may stray from a plane, as a fraction of the part's size (its bounding-box
    /// diagonal).
    pub flat_tolerance: f64,
    /// Largest angle between a triangle and a curved shape's surface (degrees); flat faces
    /// allow a third of it.
    pub max_angle_deg: f64,
}

impl Default for RecognizeOptions {
    fn default() -> Self {
        Self { kind: None, tolerance: 0.06, flat_tolerance: 0.004, max_angle_deg: 30.0 }
    }
}

/// The joint surface around triangle `seed`: a sphere, cylinder or plane (or `opts.kind`)
/// grown over the triangles connected to it that lie on the shape. None when no shape fits
/// even the triangles around the seed.
pub fn recognize(mesh: &Mesh, seed: u32, opts: &RecognizeOptions) -> Option<Feature> {
    if !mesh.is_triangle(seed) {
        return None;
    }
    let size = mesh.bounds().diagonal();
    if !(size > 0.0 && size.is_finite()) {
        return None;
    }
    let kinds: Vec<ShapeKind> = opts.kind.map_or_else(|| ShapeKind::ALL.to_vec(), |k| vec![k]);
    let mut found = Vec::new();
    for kind in kinds {
        // Seed patches from a few percent of the part down to the triangles touching the seed
        // and the seed alone, for features smaller than the first.
        for reach in [0.02 * size, 0.007 * size, 0.0, -1.0] {
            let patch = if reach < 0.0 { vec![seed] } else { seed_patch(mesh, seed, reach) };
            if let Some(f) = grow(mesh, seed, &patch, kind, opts, size) {
                found.push(f);
                break;
            }
        }
    }
    choose(found)
}

/// Fit `kind` (or the best of the three) to the triangles `tris` as they are, without growing
/// them (for a selection).
pub fn fit_region(mesh: &Mesh, tris: &[u32], kind: Option<ShapeKind>) -> Option<Feature> {
    let tris: Vec<u32> = tris.iter().copied().filter(|&t| mesh.is_triangle(t)).collect();
    if tris.is_empty() {
        return None;
    }
    let size = mesh.bounds().diagonal();
    let kinds: Vec<ShapeKind> = kind.map_or_else(|| ShapeKind::ALL.to_vec(), |k| vec![k]);
    let mut best: Option<(f64, Feature)> = None;
    for k in kinds {
        let Some((shape, convex)) = fit(mesh, &tris, k) else { continue };
        if !plausible(&shape, size) {
            continue;
        }
        let f = finish(mesh, tris.clone(), shape, convex, mesh.bounds().center());
        // Relative error, a plane's measured against the region's size; a plane is preferred
        // unless a curved shape fits clearly better.
        let scale = match f.shape {
            Shape::Plane { half, .. } => half.length().max(1e-12) * 0.25,
            s => s.size().max(1e-12),
        };
        let score = f.rms / scale * if k == ShapeKind::Plane { 0.5 } else { 1.0 };
        if best.as_ref().is_none_or(|(s, _)| score < *s) {
            best = Some((score, f));
        }
    }
    best.map(|b| b.1)
}

/// The plane unless a curved shape explains clearly more surface; between the curved ones the
/// larger.
fn choose(found: Vec<Feature>) -> Option<Feature> {
    let mut plane = None;
    let mut curved: Option<Feature> = None;
    for f in found {
        if f.shape.kind() == ShapeKind::Plane {
            plane = Some(f);
        } else if curved.as_ref().is_none_or(|c| f.area > c.area) {
            curved = Some(f);
        }
    }
    match (plane, curved) {
        (Some(p), Some(c)) => Some(if c.area > 1.25 * p.area { c } else { p }),
        (p, c) => p.or(c),
    }
}

/// Triangles connected to `seed` whose centroids are within `reach` of the seed's, and the
/// seed's neighbours whatever their distance, without crossing a crease (neighbours turned
/// more than 50° from each other).
fn seed_patch(mesh: &Mesh, seed: u32, reach: f64) -> Vec<u32> {
    const MAX: usize = 4000;
    let cos_crease = 50f64.to_radians().cos();
    let c0 = mesh.tri_centroid(seed);
    let mut seen = fast_set_with_capacity(64);
    seen.insert(seed);
    let mut out = vec![seed];
    let mut i = 0;
    while i < out.len() && out.len() < MAX {
        let t = out[i];
        i += 1;
        for n in mesh.tri_neighbors(t) {
            if n == INVALID || seen.contains(&n) || mesh.tri_normal(n).dot(mesh.tri_normal(t)) < cos_crease {
                continue;
            }
            seen.insert(n);
            if t == seed || (mesh.tri_centroid(n) - c0).length() <= reach {
                out.push(n);
            }
        }
    }
    out
}

/// Grow `kind` from `patch` over the triangles connected to `seed` that lie on it.
fn grow(mesh: &Mesh, seed: u32, patch: &[u32], kind: ShapeKind, opts: &RecognizeOptions, size: f64) -> Option<Feature> {
    let (mut shape, mut convex) = fit(mesh, patch, kind)?;
    let max_angle = if kind == ShapeKind::Plane { opts.max_angle_deg / 3.0 } else { opts.max_angle_deg };
    let cos_max = max_angle.to_radians().cos();
    let mut region: Vec<u32> = Vec::new();
    for _ in 0..8 {
        if !plausible(&shape, size) {
            return None;
        }
        let tol = tolerance(&shape, opts, size);
        let next = flood(mesh, seed, |t| on_shape(mesh, t, &shape, convex, tol, cos_max));
        if next.is_empty() {
            // The seed itself is off the shape fitted around it.
            return None;
        }
        let settled = next.len() == region.len();
        region = next;
        (shape, convex) = fit(mesh, &region, kind)?;
        if settled {
            break;
        }
    }
    if !plausible(&shape, size) || region.len() < 2 {
        return None;
    }
    Some(finish(mesh, region, shape, convex, mesh.bounds().center()))
}

/// Shapes a kit's joint can be: curved ones smaller than the part.
fn plausible(shape: &Shape, size: f64) -> bool {
    match *shape {
        Shape::Sphere { radius, .. } | Shape::Cylinder { radius, .. } => radius.is_finite() && radius > 1e-9 * size && radius < 0.6 * size,
        Shape::Plane { centre, normal, .. } => centre.is_finite() && normal.is_finite(),
    }
}

/// How far a vertex may be from `shape`.
fn tolerance(shape: &Shape, opts: &RecognizeOptions, size: f64) -> f64 {
    let flat = opts.flat_tolerance * size;
    match shape {
        Shape::Plane { .. } => flat,
        // A huge sphere fitted to a nearly flat patch must not get a huge tolerance.
        s => (opts.tolerance * s.size()).min(2.0 * flat).max(1e-6 * size),
    }
}

/// Triangle `t` lies on the shape: its corners are within `tol` and it faces the way the
/// shape does there.
fn on_shape(mesh: &Mesh, t: u32, shape: &Shape, convex: bool, tol: f64, cos_max: f64) -> bool {
    let n = mesh.tri_normal(t);
    let sign = if convex { 1.0 } else { -1.0 };
    if sign * n.dot(shape.outward(mesh.tri_centroid(t))) < cos_max {
        return false;
    }
    mesh.tri_positions(t).iter().all(|&p| shape.distance(p).abs() <= tol)
}

/// The triangles connected to `seed` through triangles that pass `keep` (none if the seed
/// does not).
fn flood(mesh: &Mesh, seed: u32, keep: impl Fn(u32) -> bool) -> Vec<u32> {
    if !keep(seed) {
        return Vec::new();
    }
    let mut seen: FastSet<u32> = fast_set_with_capacity(256);
    seen.insert(seed);
    let mut out = vec![seed];
    let mut i = 0;
    while i < out.len() {
        let t = out[i];
        i += 1;
        for n in mesh.tri_neighbors(t) {
            if n != INVALID && seen.insert(n) && keep(n) {
                out.push(n);
            }
        }
    }
    out
}

/// Vertices and (centroid, area, unit normal) of triangles.
fn gather(mesh: &Mesh, tris: &[u32]) -> (Vec<DVec3>, Vec<(DVec3, f64, DVec3)>) {
    let mut vs: Vec<u32> = tris.iter().flat_map(|&t| mesh.triangle(t)).collect();
    vs.sort_unstable();
    vs.dedup();
    let pts = vs.iter().map(|&v| mesh.vertex(v)).collect();
    let faces = tris.iter().map(|&t| (mesh.tri_centroid(t), mesh.tri_area(t), mesh.tri_normal(t))).collect();
    (pts, faces)
}

/// Fit `kind` to triangles `tris`: the shape (a plane's outline not yet measured) and whether
/// the surface bulges out.
fn fit(mesh: &Mesh, tris: &[u32], kind: ShapeKind) -> Option<(Shape, bool)> {
    let (pts, faces) = gather(mesh, tris);
    if pts.len() < 3 {
        return None;
    }
    let area: f64 = faces.iter().map(|f| f.1).sum();
    let centroid = if area > 0.0 { faces.iter().map(|f| f.0 * f.1).sum::<DVec3>() / area } else { pts.iter().sum::<DVec3>() / pts.len() as f64 };
    let mean_normal = faces.iter().map(|f| f.2 * f.1).sum::<DVec3>();
    // Faces pointing away from the shape's centre / axis, by area.
    let bulges = |shape: &Shape| faces.iter().map(|f| f.1 * f.2.dot(shape.outward(f.0))).sum::<f64>() >= 0.0;
    let shape = match kind {
        ShapeKind::Sphere => {
            if pts.len() < 4 {
                return None;
            }
            let (centre, radius) = fit_sphere(&pts);
            Shape::Sphere { centre, radius }
        }
        ShapeKind::Cylinder => {
            if pts.len() < 4 {
                return None;
            }
            // The axis is the direction the normals are perpendicular to.
            let mut a = [[0.0; 3]; 3];
            for f in &faces {
                add_outer(&mut a, f.2, f.1);
            }
            let (_, vecs) = sym_eigen3(a);
            let axis = vecs[0].normalize_or_zero();
            if axis == DVec3::ZERO {
                return None;
            }
            let (u, v) = mm_core::geom::orthonormal_basis(axis);
            let p2: Vec<DVec2> = pts.iter().map(|&p| DVec2::new((p - centroid).dot(u), (p - centroid).dot(v))).collect();
            let (cc, radius) = fit_circle(&p2);
            let on_axis = centroid + u * cc.x + v * cc.y;
            let (z0, z1) = pts.iter().map(|&p| (p - on_axis).dot(axis)).fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), z| (lo.min(z), hi.max(z)));
            Shape::Cylinder { centre: on_axis + axis * (0.5 * (z0 + z1)), axis, radius, length: (z1 - z0).max(0.0) }
        }
        ShapeKind::Plane => {
            // Covariance of the triangles, each by its area (its corners and its centre), so
            // finely tessellated places do not pull the plane.
            let mut a = [[0.0; 3]; 3];
            for &t in tris.iter().filter(|&&t| mesh.is_triangle(t)) {
                let w = mesh.tri_area(t) / 3.0;
                for p in mesh.tri_positions(t) {
                    add_outer(&mut a, p - centroid, w);
                }
            }
            let (vals, vecs) = sym_eigen3(a);
            let mut normal = vecs[0].normalize_or_zero();
            // Points nearly on a line leave the plane undecided: the faces' normal decides.
            if normal == DVec3::ZERO || vals[1] <= 1e-12 * vals[2].max(1e-300) {
                normal = mean_normal.normalize_or_zero();
            }
            if normal == DVec3::ZERO {
                return None;
            }
            if normal.dot(mean_normal) < 0.0 {
                normal = -normal;
            }
            let u = mm_core::geom::orthonormal_basis(normal).0;
            Shape::Plane { centre: centroid, normal, u, half: DVec2::ZERO, round: false }
        }
    };
    let convex = match shape {
        Shape::Plane { .. } => true,
        ref s => bulges(s),
    };
    Some((shape, convex))
}

/// The feature made of `tris` on `shape` (on a part centred at `part`): a plane's outline
/// measured, its fit error, area and the way to its part's body.
fn finish(mesh: &Mesh, tris: Vec<u32>, mut shape: Shape, convex: bool, part: DVec3) -> Feature {
    let (pts, faces) = gather(mesh, &tris);
    let area: f64 = faces.iter().map(|f| f.1).sum();
    let mean_normal = faces.iter().map(|f| f.2 * f.1).sum::<DVec3>();
    if let Shape::Plane { centre, normal, .. } = shape {
        let (u0, v0) = mm_core::geom::orthonormal_basis(normal);
        let p2: Vec<DVec2> = pts.iter().map(|&p| DVec2::new((p - centre).dot(u0), (p - centre).dot(v0))).collect();
        let (angle, lo, hi) = min_area_rect(&p2);
        let (s, c) = angle.sin_cos();
        let (mut u, mut v) = (u0 * c + v0 * s, v0 * c - u0 * s);
        let mut half = (hi - lo) * 0.5;
        let mid = (lo + hi) * 0.5;
        let centre = centre + u * mid.x + v * mid.y;
        if half.y > half.x {
            half = DVec2::new(half.y, half.x);
            (u, v) = (v, -u);
        }
        let _ = v;
        let fill = area / (4.0 * half.x * half.y).max(1e-300);
        let round = half.y > 0.85 * half.x && (0.70..0.86).contains(&fill);
        shape = Shape::Plane { centre, normal, u, half, round };
    }
    let rms = (pts.iter().map(|&p| shape.distance(p).powi(2)).sum::<f64>() / pts.len().max(1) as f64).sqrt();
    let towards_part = (part - shape.origin()).normalize_or_zero();
    let away = (part - shape.origin()).length();
    let body = match shape {
        Shape::Plane { normal, .. } => Some(-normal),
        // A cup's surface faces out of its opening.
        Shape::Sphere { .. } if !convex && mean_normal.length() > 0.1 * area => Some(-mean_normal.normalize()),
        Shape::Sphere { radius, .. } => (away > 0.25 * radius).then_some(towards_part),
        Shape::Cylinder { axis, length, .. } => {
            let along = (part - shape.origin()).dot(axis);
            (along.abs() > 0.1 * length.max(1e-12)).then(|| axis * along.signum())
        }
    };
    // A cylinder's axis out of the part (its fit gives either way).
    if let (Shape::Cylinder { centre, axis, radius, length }, Some(b)) = (shape, body) {
        if axis.dot(b) > 0.0 {
            shape = Shape::Cylinder { centre, axis: -axis, radius, length };
        }
    }
    Feature { shape, convex, tris, area, rms, body }
}

/// The smallest-area rectangle around 2D points: its turn (radians, in [0, π/2)) and the
/// bounds of the points turned back by it.
fn min_area_rect(pts: &[DVec2]) -> (f64, DVec2, DVec2) {
    let bounds_at = |a: f64| {
        let (s, c) = a.sin_cos();
        pts.iter().fold((DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY)), |(lo, hi), p| {
            // Coordinates along (c, s) and (−s, c).
            let q = DVec2::new(p.x * c + p.y * s, p.y * c - p.x * s);
            (lo.min(q), hi.max(q))
        })
    };
    let area = |a: f64| {
        let (lo, hi) = bounds_at(a);
        (hi - lo).x * (hi - lo).y
    };
    let mut best = (0.0, area(0.0));
    for i in 1..90 {
        let a = (i as f64).to_radians();
        let ar = area(a);
        if ar < best.1 {
            best = (a, ar);
        }
    }
    // Refine within a degree either side.
    let step = 1f64.to_radians() / 20.0;
    let centre = best.0;
    for i in -20..=20 {
        let a = centre + i as f64 * step;
        let ar = area(a);
        if ar < best.1 {
            best = (a, ar);
        }
    }
    let (lo, hi) = bounds_at(best.0);
    (best.0, lo, hi)
}

/// The joints two parts can make.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub enum JointKind {
    /// Glued: no motion (a keyed peg in its recess).
    Fixed,
    /// Turns any way about a point (a ball in a socket).
    Ball,
    /// Turns about an axis (a pin in a hole).
    Hinge,
    /// Slides along an axis.
    Slider,
    /// Turns about and slides along an axis.
    Cylindrical,
    /// Slides on a plane and turns about its normal.
    Planar,
}

impl JointKind {
    pub const ALL: [JointKind; 6] = [JointKind::Ball, JointKind::Hinge, JointKind::Fixed, JointKind::Cylindrical, JointKind::Slider, JointKind::Planar];

    pub fn label(self) -> &'static str {
        match self {
            JointKind::Fixed => "Fixed",
            JointKind::Ball => "Ball",
            JointKind::Hinge => "Hinge",
            JointKind::Slider => "Slider",
            JointKind::Cylindrical => "Cylindrical",
            JointKind::Planar => "Planar",
        }
    }

    pub fn about(self) -> &'static str {
        match self {
            JointKind::Fixed => "Glued: the parts move as one",
            JointKind::Ball => "Turns any way about the joint's centre (drag to swing, Shift+drag to twist)",
            JointKind::Hinge => "Turns about the joint's axis",
            JointKind::Slider => "Slides along the joint's axis",
            JointKind::Cylindrical => "Turns about the axis (drag) and slides along it (Shift+drag)",
            JointKind::Planar => "Slides over the face (drag) and turns on it (Shift+drag)",
        }
    }

    /// It turns about the joint's axis.
    pub fn turns(self) -> bool {
        matches!(self, JointKind::Hinge | JointKind::Cylindrical | JointKind::Planar)
    }
}

/// The joint two features make: a ball anywhere makes a ball joint, a pin a hinge, two flat
/// faces glue (a keyed peg), unless both are round (a round peg turns).
pub fn suggest_joint(a: &Feature, b: &Feature) -> JointKind {
    use ShapeKind::*;
    match (a.shape.kind(), b.shape.kind()) {
        (Sphere, _) | (_, Sphere) => JointKind::Ball,
        (Cylinder, _) | (_, Cylinder) => JointKind::Hinge,
        (Plane, Plane) => match (a.shape, b.shape) {
            (Shape::Plane { round: true, .. }, Shape::Plane { round: true, .. }) => JointKind::Hinge,
            _ => JointKind::Fixed,
        },
    }
}

/// The gap between a ball and its socket or a pin and its hole (socket radius − ball radius;
/// negative: the ball is too big), if the two are such a pair.
pub fn clearance(a: &Feature, b: &Feature) -> Option<f64> {
    if a.shape.kind() != b.shape.kind() || a.shape.kind() == ShapeKind::Plane || a.convex == b.convex {
        return None;
    }
    let (ball, socket) = if a.convex { (a, b) } else { (b, a) };
    Some(socket.shape.size() - ball.shape.size())
}

/// A rigid motion `x ↦ rot (x − from) + to`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Seat {
    pub rot: DQuat,
    pub from: DVec3,
    pub to: DVec3,
}

impl Seat {
    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.rot * (p - self.from) + self.to
    }

    /// The same motion as `x ↦ rot x + shift`: (rot, shift).
    pub fn rot_shift(&self) -> (DQuat, DVec3) {
        (self.rot, self.to - self.rot * self.from)
    }
}

/// How [`seat`] puts one feature on another.
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct SeatOptions {
    /// Turn the moving feature's axis the other way round.
    pub flip: bool,
    /// Which way round two flat faces meet: 0 the nearest, then the others in order (a
    /// rectangle fits two ways, a square four).
    pub turn: usize,
    /// Distance from the fixed feature along its axis or normal (out of the fixed part for
    /// a flat face).
    pub offset: f64,
}

/// The rigid move that seats `moving` on `fixed`: the centres meet (a ball rests on a flat
/// face), and the moving part turns so the two parts' bodies are on either side of the joint:
/// a limb comes out of its socket's opening, a pin goes into its hole, a flat face meets the
/// other face to face, its outline's long side along the other's. Without a clear way to a
/// part's body, a pin's axis turns onto the hole's the nearer way round and a ball does not
/// turn.
pub fn seat(moving: &Feature, fixed: &Feature, o: &SeatOptions) -> Seat {
    let from = moving.shape.origin();
    let mut to = fixed.shape.origin();
    let mut rot = DQuat::IDENTITY;
    let ball = moving.shape.kind() == ShapeKind::Sphere || fixed.shape.kind() == ShapeKind::Sphere;
    match (moving.body, fixed.body) {
        // Flip keeps a ball's part turned as it is.
        _ if ball && o.flip => {}
        (Some(bm), Some(bf)) => rot = DQuat::from_rotation_arc(bm, if o.flip { bf } else { -bf }),
        _ => {
            if let (Some(a), Some(b)) = (moving.shape.axis(), fixed.shape.axis()) {
                let both_flat = moving.shape.kind() == ShapeKind::Plane && fixed.shape.kind() == ShapeKind::Plane;
                let mut target = if both_flat || a.dot(b) < 0.0 { -b } else { b };
                if o.flip {
                    target = -target;
                }
                rot = DQuat::from_rotation_arc(a, target);
            }
        }
    }
    if let (Shape::Plane { u: um, half: hm, .. }, Shape::Plane { normal, u: uf, half: hf, .. }) = (moving.shape, fixed.shape) {
        // Turn about the fixed face's normal so the outlines line up.
        let square = hm.y > 0.87 * hm.x && hf.y > 0.87 * hf.x;
        let step = if square { FRAC_PI_2 } else { PI };
        let base = signed_angle(rot * um, uf, normal);
        let n = (2.0 * PI / step).round() as usize;
        let mut turns: Vec<f64> = (0..n).map(|k| wrap(base + k as f64 * step)).collect();
        turns.sort_by(|x, y| x.abs().total_cmp(&y.abs()));
        rot = DQuat::from_axis_angle(normal, turns[o.turn % turns.len()]) * rot;
    }
    // A ball rests on a flat face, and a flat face on a ball, the face touching the ball's
    // surface rather than passing through its centre.
    match (moving.shape, fixed.shape) {
        (Shape::Sphere { radius, .. }, Shape::Plane { normal, .. }) => to += normal * radius,
        (Shape::Plane { normal, .. }, Shape::Sphere { radius, .. }) => to -= (rot * normal) * radius,
        _ => {}
    }
    if let Some(b) = fixed.shape.axis() {
        to += b * o.offset;
    }
    Seat { rot: rot.normalize(), from, to }
}

/// How [`contact_fit`] moves one surface onto another.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ContactOptions {
    /// Let the part turn as well as move. Off for balls: a sphere turned about its centre fits
    /// the same, so turning would only drift.
    pub turn: bool,
    pub iterations: usize,
    /// The fraction of facing pairs kept (the closest); the rest are taken as outliers.
    pub keep: f64,
    /// Largest angle between a pair's normals and exactly facing each other (degrees).
    pub max_angle_deg: f64,
    /// Most points sampled on the moving surface.
    pub samples: usize,
}

impl Default for ContactOptions {
    fn default() -> Self {
        Self { turn: false, iterations: 50, keep: 0.8, max_angle_deg: 60.0, samples: 4000 }
    }
}

/// The result of [`contact_fit`]: the motion `x ↦ rot x + shift` of the moving part, the gap
/// between the surfaces (root mean square distance over the facing pairs kept) before and
/// after, and how deep the moving surface then goes into the fixed one at most (over the pairs
/// kept; the fit trades a little overlap for smaller gaps elsewhere).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    pub rot: DQuat,
    pub shift: DVec3,
    pub gap_before: f64,
    pub gap_after: f64,
    pub overlap: f64,
    pub pairs: usize,
}

/// Move the surface `moving_tris` of `moving`, starting from the motion `start` (rot, shift),
/// so that it lies on the surface `fixed_tris` of `fixed`: iterative closest points, each
/// point of the moving surface paired with the nearest point of the fixed one, only where the
/// two face each other, minimising the distances to the fixed surface's tangent planes. None
/// when fewer than three points face the other surface.
pub fn contact_fit(moving: &Mesh, moving_tris: &[u32], fixed: &Mesh, fixed_tris: &[u32], start: (DQuat, DVec3), o: &ContactOptions) -> Option<Contact> {
    let mt: Vec<u32> = moving_tris.iter().copied().filter(|&t| moving.is_triangle(t)).collect();
    let ft: Vec<u32> = fixed_tris.iter().copied().filter(|&t| fixed.is_triangle(t)).collect();
    if mt.is_empty() || ft.is_empty() {
        return None;
    }
    let step = (mt.len() / o.samples.max(1)).max(1);
    let samples: Vec<(DVec3, DVec3, f64)> = mt.iter().step_by(step).map(|&t| (moving.tri_centroid(t), moving.tri_normal(t), moving.tri_area(t))).filter(|s| s.2 > 0.0).collect();
    let bvh = Bvh::build_subset(fixed, &ft);
    let cos_max = o.max_angle_deg.to_radians().cos();
    // (point, fixed normal, signed distance to the fixed tangent plane, weight)
    let pairs_at = |rot: DQuat, shift: DVec3| -> Vec<(DVec3, DVec3, f64, f64)> {
        let mut v: Vec<(DVec3, DVec3, f64, f64)> = samples
            .iter()
            .filter_map(|&(c, n, w)| {
                let p = rot * c + shift;
                let h = bvh.nearest(fixed, p, f64::INFINITY)?;
                let nf = fixed.tri_normal(h.tid);
                ((rot * n).dot(nf) <= -cos_max).then(|| (p, nf, (p - h.point).dot(nf), w))
            })
            .collect();
        v.sort_by(|a, b| a.2.abs().total_cmp(&b.2.abs()));
        let keep = ((v.len() as f64 * o.keep.clamp(0.05, 1.0)).ceil() as usize).max(v.len().min(3));
        v.truncate(keep);
        v
    };
    let gap = |pairs: &[(DVec3, DVec3, f64, f64)]| {
        let w: f64 = pairs.iter().map(|p| p.3).sum();
        (pairs.iter().map(|p| p.3 * p.2 * p.2).sum::<f64>() / w.max(1e-300)).sqrt()
    };
    let first = pairs_at(start.0, start.1);
    if first.len() < 3 {
        return None;
    }
    let size = fixed.bounds().diagonal().max(moving.bounds().diagonal()).max(1e-12);
    let (mut rot, mut shift) = start;
    for _ in 0..o.iterations {
        let pairs = pairs_at(rot, shift);
        if pairs.len() < 3 {
            break;
        }
        let (dr, dt) = if o.turn { step_rigid(&pairs) } else { (DQuat::IDENTITY, step_translation(&pairs)) };
        rot = (dr * rot).normalize();
        shift = dr * shift + dt;
        if dt.length() < 1e-9 * size && dr.angle_between(DQuat::IDENTITY) < 1e-9 {
            break;
        }
    }
    let last = pairs_at(rot, shift);
    if last.len() < 3 {
        return None;
    }
    let overlap = last.iter().map(|p| -p.2).fold(0.0, f64::max);
    Some(Contact { rot, shift, gap_before: gap(&first), gap_after: gap(&last), overlap, pairs: last.len() })
}

/// The translation that best puts the points on the tangent planes (least squares; directions
/// the planes do not hold stay put).
fn step_translation(pairs: &[(DVec3, DVec3, f64, f64)]) -> DVec3 {
    let mut a = [[0.0; 3]; 3];
    let mut b = DVec3::ZERO;
    for &(_, n, d, w) in pairs {
        add_outer(&mut a, n, w);
        b -= n * (w * d);
    }
    let tr = a[0][0] + a[1][1] + a[2][2];
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += 1e-6 * tr + 1e-300;
    }
    let m = DMat3::from_cols_array_2d(&a).transpose();
    if m.determinant().abs() < 1e-300 {
        return DVec3::ZERO;
    }
    m.inverse() * b
}

/// The small turn (about the points' centre) and translation that best put the points on the
/// tangent planes (linearised; directions the planes do not hold stay put).
fn step_rigid(pairs: &[(DVec3, DVec3, f64, f64)]) -> (DQuat, DVec3) {
    let wsum: f64 = pairs.iter().map(|p| p.3).sum();
    let c = pairs.iter().map(|p| p.0 * p.3).sum::<DVec3>() / wsum.max(1e-300);
    let mut a = [[0.0f64; 6]; 6];
    let mut b = [0.0f64; 6];
    for &(p, n, d, w) in pairs {
        let r = (p - c).cross(n);
        let j = [r.x, r.y, r.z, n.x, n.y, n.z];
        for i in 0..6 {
            for k in 0..6 {
                a[i][k] += w * j[i] * j[k];
            }
            b[i] -= w * j[i] * d;
        }
    }
    let tr: f64 = (0..6).map(|i| a[i][i]).sum();
    for (i, row) in a.iter_mut().enumerate() {
        row[i] += 1e-6 * tr + 1e-300;
    }
    let Some(x) = solve6(a, b) else { return (DQuat::IDENTITY, DVec3::ZERO) };
    let mut w = DVec3::new(x[0], x[1], x[2]);
    // Small steps keep the linearisation honest.
    if w.length() > 0.2 {
        w = w.normalize() * 0.2;
    }
    let r = DQuat::from_scaled_axis(w);
    (r, c - r * c + DVec3::new(x[3], x[4], x[5]))
}

/// Solve a 6×6 system by Gaussian elimination with partial pivoting.
fn solve6(mut a: [[f64; 6]; 6], mut b: [f64; 6]) -> Option<[f64; 6]> {
    for col in 0..6 {
        let piv = (col..6).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs()))?;
        if a[piv][col].abs() < 1e-300 {
            return None;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..6 {
            let f = a[row][col] / a[col][col];
            for k in col..6 {
                a[row][k] -= f * a[col][k];
            }
            b[row] -= f * b[col];
        }
    }
    let mut x = [0.0; 6];
    for row in (0..6).rev() {
        let s: f64 = (row + 1..6).map(|k| a[row][k] * x[k]).sum();
        x[row] = (b[row] - s) / a[row][row];
    }
    Some(x)
}

/// The angle turning `a` onto `b` about `axis` (radians, in (−π, π]).
fn signed_angle(a: DVec3, b: DVec3, axis: DVec3) -> f64 {
    let a = a - axis * a.dot(axis);
    let b = b - axis * b.dot(axis);
    a.cross(b).dot(axis).atan2(a.dot(b))
}

/// `a` wrapped into (−π, π].
fn wrap(a: f64) -> f64 {
    let w = (a + PI).rem_euclid(2.0 * PI) - PI;
    if w <= -PI { w + 2.0 * PI } else { w }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol
    }

    /// The triangle whose centroid is nearest `p`.
    fn tri_near(m: &Mesh, p: DVec3) -> u32 {
        m.triangle_ids().min_by(|&a, &b| (m.tri_centroid(a) - p).length().total_cmp(&(m.tri_centroid(b) - p).length())).unwrap()
    }

    #[test]
    fn a_ball_is_recognized_as_a_sphere() {
        let mut m = primitives::icosphere(2.0, 3);
        m.transform(|p| p + DVec3::new(5.0, -1.0, 3.0));
        let f = recognize(&m, tri_near(&m, DVec3::new(7.0, -1.0, 3.0)), &RecognizeOptions::default()).expect("found");
        match f.shape {
            Shape::Sphere { centre, radius } => {
                assert!((centre - DVec3::new(5.0, -1.0, 3.0)).length() < 0.02, "{centre}");
                assert!(close(radius, 2.0, 0.02), "{radius}");
            }
            s => panic!("not a sphere: {s:?}"),
        }
        assert!(f.convex);
        assert_eq!(f.name(), "ball");
        assert!(f.tris.len() > m.triangle_count() * 9 / 10, "{} of {}", f.tris.len(), m.triangle_count());
    }

    #[test]
    fn a_ball_on_a_neck_stops_at_the_neck_and_a_socket_in_a_block_at_its_rim() {
        // A ball r 1 on a stick r 0.4 (an arm's shoulder joint), joined into one surface.
        let mut ball = primitives::icosphere(1.0, 4);
        ball.transform(|p| p + DVec3::new(0.0, 0.0, 4.6));
        let stick = primitives::cylinder(0.4, 4.4, 24, 12, true);
        let joined = crate::boolean::mesh_boolean(&ball, &stick, crate::boolean::BooleanOp::Union).unwrap().mesh;
        let f = recognize(&joined, tri_near(&joined, DVec3::new(0.0, 0.0, 5.6)), &RecognizeOptions::default()).expect("ball");
        match f.shape {
            Shape::Sphere { centre, radius } => {
                assert!((centre - DVec3::new(0.0, 0.0, 4.6)).length() < 0.02, "{centre}");
                assert!(close(radius, 1.0, 0.02), "{radius}");
            }
            s => panic!("not a sphere: {s:?}"),
        }
        assert!(f.tris.iter().all(|&t| joined.tri_centroid(t).z > 3.5), "none of the stick");
        // Clicking the stick finds the stick.
        let s = recognize(&joined, tri_near(&joined, DVec3::new(0.4, 0.0, 2.0)), &RecognizeOptions::default()).expect("stick");
        assert_eq!(s.name(), "pin");
        // A socket: a block with the ball cut out of its top.
        let mut block = primitives::cube(1.0, 6);
        block.transform(|p| p * 4.0);
        let mut cutter = primitives::icosphere(1.0, 4);
        cutter.transform(|p| p + DVec3::new(0.0, 0.0, 2.3));
        let socketed = crate::boolean::mesh_boolean(&block, &cutter, crate::boolean::BooleanOp::Difference).unwrap().mesh;
        let f = recognize(&socketed, tri_near(&socketed, DVec3::new(0.0, 0.0, 1.3)), &RecognizeOptions::default()).expect("socket");
        assert_eq!(f.name(), "socket");
        assert!(close(f.shape.size(), 1.0, 0.02));
        assert!(f.tris.iter().all(|&t| (joined_dist(&socketed, t, DVec3::new(0.0, 0.0, 2.3)) - 1.0).abs() < 0.05), "only the cup");
        assert_eq!(suggest_joint(&recognize(&joined, tri_near(&joined, DVec3::new(0.0, 0.0, 5.6)), &RecognizeOptions::default()).unwrap(), &f), JointKind::Ball);
    }

    fn joined_dist(m: &Mesh, t: u32, c: DVec3) -> f64 {
        (m.tri_centroid(t) - c).length()
    }

    #[test]
    fn a_limb_comes_out_of_its_socket_and_a_pin_goes_into_its_hole() {
        let ball = Feature { shape: Shape::Sphere { centre: DVec3::new(5.0, 5.0, 5.0), radius: 1.0 }, convex: true, tris: Vec::new(), area: 1.0, rms: 0.0, body: Some(DVec3::Z) };
        let socket = Feature { shape: Shape::Sphere { centre: DVec3::ZERO, radius: 1.05 }, convex: false, tris: Vec::new(), area: 1.0, rms: 0.0, body: Some(-DVec3::X) };
        let s = seat(&ball, &socket, &SeatOptions::default());
        assert!((s.rot * DVec3::Z - DVec3::X).length() < 1e-9, "the limb points out of the opening");
        assert!(s.apply(DVec3::new(5.0, 5.0, 5.0)).length() < 1e-9);
        assert_eq!(seat(&ball, &socket, &SeatOptions { flip: true, ..Default::default() }).rot, DQuat::IDENTITY, "flip keeps the turn");
        let pin = Feature { shape: Shape::Cylinder { centre: DVec3::ZERO, axis: DVec3::Y, radius: 0.5, length: 2.0 }, convex: true, tris: Vec::new(), area: 1.0, rms: 0.0, body: Some(-DVec3::Y) };
        let hole = Feature { shape: Shape::Cylinder { centre: DVec3::new(0.0, 0.0, 3.0), axis: DVec3::Z, radius: 0.55, length: 2.0 }, convex: false, tris: Vec::new(), area: 1.0, rms: 0.0, body: Some(-DVec3::Z) };
        let s = seat(&pin, &hole, &SeatOptions::default());
        assert!((s.rot * -DVec3::Y - DVec3::Z).length() < 1e-9, "the pin's part stays outside the hole");
    }

    #[test]
    fn recognized_features_know_the_way_to_their_part() {
        let mut ball = primitives::icosphere(1.0, 4);
        ball.transform(|p| p + DVec3::new(0.0, 0.0, 4.6));
        let stick = primitives::cylinder(0.4, 4.4, 24, 12, true);
        let arm = crate::boolean::mesh_boolean(&ball, &stick, crate::boolean::BooleanOp::Union).unwrap().mesh;
        let f = recognize(&arm, tri_near(&arm, DVec3::new(0.0, 0.0, 5.6)), &RecognizeOptions::default()).unwrap();
        assert!(f.body.unwrap().dot(-DVec3::Z) > 0.99, "{:?}", f.body);
        let mut block = primitives::cube(1.0, 6);
        block.transform(|p| p * 4.0);
        let mut cutter = primitives::icosphere(1.0, 4);
        cutter.transform(|p| p + DVec3::new(2.3, 0.0, 0.0));
        let torso = crate::boolean::mesh_boolean(&block, &cutter, crate::boolean::BooleanOp::Difference).unwrap().mesh;
        let f = recognize(&torso, tri_near(&torso, DVec3::new(1.3, 0.0, 0.0)), &RecognizeOptions::default()).unwrap();
        assert_eq!(f.name(), "socket");
        assert!(f.body.unwrap().dot(-DVec3::X) > 0.95, "into the torso: {:?}", f.body);
    }

    #[test]
    fn a_socket_is_a_concave_sphere() {
        let mut m = primitives::icosphere(1.5, 3);
        m.reverse_orientation();
        let f = recognize(&m, 7, &RecognizeOptions::default()).expect("found");
        assert_eq!(f.shape.kind(), ShapeKind::Sphere);
        assert!(!f.convex);
        assert_eq!(f.name(), "socket");
    }

    #[test]
    fn a_pin_side_is_a_cylinder_and_its_end_a_flat_face() {
        let mut m = primitives::cylinder(0.5, 4.0, 32, 16, true);
        // Lay it along x.
        m.transform(|p| DVec3::new(p.z, p.y, -p.x));
        let side = recognize(&m, tri_near(&m, DVec3::new(2.0, 0.5, 0.0)), &RecognizeOptions::default()).expect("side");
        match side.shape {
            Shape::Cylinder { centre, axis, radius, length } => {
                assert!(axis.dot(DVec3::X).abs() > 0.999, "{axis}");
                assert!(close(radius, 0.5, 0.01), "{radius}");
                assert!(close(length, 4.0, 0.05), "{length}");
                assert!((centre - DVec3::new(2.0, 0.0, 0.0)).length() < 0.05, "{centre}");
            }
            s => panic!("not a cylinder: {s:?}"),
        }
        assert_eq!(side.name(), "pin");
        let end = recognize(&m, tri_near(&m, DVec3::new(4.0, 0.1, 0.1)), &RecognizeOptions::default()).expect("end");
        match end.shape {
            Shape::Plane { normal, round, .. } => {
                assert!(normal.dot(DVec3::X) > 0.999, "{normal}");
                assert!(round, "a disc");
            }
            s => panic!("not flat: {s:?}"),
        }
    }

    #[test]
    fn a_flat_face_measures_its_rectangle() {
        let mut m = primitives::cube(1.0, 4);
        m.transform(|p| DVec3::new(p.x * 4.0, p.y * 2.0, p.z));
        let f = recognize(&m, tri_near(&m, DVec3::new(0.3, 0.2, 0.5)), &RecognizeOptions::default()).expect("found");
        match f.shape {
            Shape::Plane { centre, normal, u, half, round } => {
                assert!(normal.dot(DVec3::Z) > 0.999);
                assert!(u.dot(DVec3::X).abs() > 0.999, "long side along x: {u}");
                assert!(close(half.x, 2.0, 1e-6) && close(half.y, 1.0, 1e-6), "{half}");
                assert!((centre - DVec3::new(0.0, 0.0, 0.5)).length() < 1e-6);
                assert!(!round);
            }
            s => panic!("not flat: {s:?}"),
        }
        // Only the top face.
        assert!(f.tris.iter().all(|&t| m.tri_normal(t).z > 0.99));
    }

    #[test]
    fn a_forced_shape_and_a_selection_fit() {
        let m = primitives::icosphere(1.0, 3);
        let f = recognize(&m, 3, &RecognizeOptions { kind: Some(ShapeKind::Cylinder), ..Default::default() });
        // A band of the sphere at most.
        assert!(f.is_none_or(|f| f.tris.len() < m.triangle_count() / 2));
        let all: Vec<u32> = m.triangle_ids().collect();
        let f = fit_region(&m, &all, None).expect("fits");
        assert_eq!(f.shape.kind(), ShapeKind::Sphere);
    }

    #[test]
    fn a_ball_seats_in_its_socket() {
        let mut ball = primitives::icosphere(1.0, 3);
        ball.transform(|p| p + DVec3::new(10.0, 2.0, 0.0));
        let mut socket = primitives::icosphere(1.1, 3);
        socket.reverse_orientation();
        socket.transform(|p| p + DVec3::new(-3.0, 0.0, 1.0));
        let opts = RecognizeOptions::default();
        let b = recognize(&ball, 0, &opts).unwrap();
        let s = recognize(&socket, 0, &opts).unwrap();
        let seat = seat(&b, &s, &SeatOptions::default());
        assert!((seat.apply(DVec3::new(10.0, 2.0, 0.0)) - DVec3::new(-3.0, 0.0, 1.0)).length() < 1e-3);
        assert!(seat.rot.angle_between(DQuat::IDENTITY) < 1e-9);
        assert_eq!(suggest_joint(&b, &s), JointKind::Ball);
        assert!(close(clearance(&b, &s).unwrap(), 0.1, 0.02));
    }

    fn flat(centre: DVec3, normal: DVec3, u: DVec3, half: DVec2) -> Feature {
        Feature { shape: Shape::Plane { centre, normal, u, half, round: false }, convex: true, tris: Vec::new(), area: 4.0 * half.x * half.y, rms: 0.0, body: Some(-normal) }
    }

    #[test]
    fn flat_faces_meet_face_to_face_with_their_rectangles_lined_up() {
        let r = DQuat::from_euler(glam::EulerRot::XYZ, 0.4, -1.1, 2.3);
        let peg = flat(DVec3::new(1.0, 2.0, 3.0), r * DVec3::Z, r * DVec3::X, DVec2::new(2.0, 1.0));
        let recess = flat(DVec3::new(-5.0, 0.0, 0.0), DVec3::Y, DVec3::Z, DVec2::new(2.0, 1.0));
        for turn in 0..2 {
            let s = seat(&peg, &recess, &SeatOptions { turn, ..Default::default() });
            assert!((s.apply(DVec3::new(1.0, 2.0, 3.0)) - DVec3::new(-5.0, 0.0, 0.0)).length() < 1e-9);
            assert!((s.rot * (r * DVec3::Z) + DVec3::Y).length() < 1e-9, "face to face");
            assert!((s.rot * (r * DVec3::X)).dot(DVec3::Z).abs() > 1.0 - 1e-9, "long sides along each other");
        }
        let a = seat(&peg, &recess, &SeatOptions::default()).rot * (r * DVec3::X);
        let b = seat(&peg, &recess, &SeatOptions { turn: 1, ..Default::default() }).rot * (r * DVec3::X);
        assert!((a + b).length() < 1e-9, "the other way round");
        let gap = seat(&peg, &recess, &SeatOptions { offset: 0.5, ..Default::default() });
        assert!((gap.apply(DVec3::new(1.0, 2.0, 3.0)) - DVec3::new(-5.0, 0.5, 0.0)).length() < 1e-9);
        assert_eq!(suggest_joint(&peg, &recess), JointKind::Fixed);
    }

    #[test]
    fn a_pin_turns_onto_the_hole_axis_the_nearer_way() {
        let pin = Feature { shape: Shape::Cylinder { centre: DVec3::new(0.0, 0.0, 5.0), axis: DVec3::new(1.0, 0.2, 0.0).normalize(), radius: 0.5, length: 2.0 }, convex: true, tris: Vec::new(), area: 1.0, rms: 0.0, body: None };
        let hole = Feature { shape: Shape::Cylinder { centre: DVec3::ZERO, axis: -DVec3::X, radius: 0.55, length: 3.0 }, convex: false, tris: Vec::new(), area: 1.0, rms: 0.0, body: None };
        let s = seat(&pin, &hole, &SeatOptions::default());
        let a = s.rot * pin.shape.axis().unwrap();
        assert!((a - DVec3::X).length() < 1e-9, "{a}");
        let f = seat(&pin, &hole, &SeatOptions { flip: true, ..Default::default() });
        assert!((f.rot * pin.shape.axis().unwrap() + DVec3::X).length() < 1e-9);
        assert!(s.apply(DVec3::new(0.0, 0.0, 5.0)).length() < 1e-9);
        assert_eq!(suggest_joint(&pin, &hole), JointKind::Hinge);
        assert!(close(clearance(&pin, &hole).unwrap(), 0.05, 1e-12));
    }

    #[test]
    fn a_cup_slides_onto_its_ball() {
        // A cup cut from a sphere r 2 (facing in), off-centre from a ball r 2: moved back onto it.
        let ball = primitives::icosphere(2.0, 4);
        let mut cup = primitives::icosphere(2.0, 4);
        cup.reverse_orientation();
        let cup_tris: Vec<u32> = cup.triangle_ids().filter(|&t| cup.tri_centroid(t).z > 0.6).collect();
        let off = DVec3::new(0.2, 0.1, -0.15);
        cup.transform(|p| p + off);
        let all: Vec<u32> = ball.triangle_ids().collect();
        let c = contact_fit(&cup, &cup_tris, &ball, &all, (DQuat::IDENTITY, DVec3::ZERO), &ContactOptions::default()).unwrap();
        assert!((c.shift + off).length() < 0.01, "{}", c.shift);
        assert!(c.gap_before > 0.05 && c.gap_after < 0.005, "{} → {}", c.gap_before, c.gap_after);
    }

    #[test]
    fn a_loose_ball_moves_to_touch_its_socket() {
        // Ball r 1 centred in a socket r 1.3 cut into the side of a block (opening +x): it moves
        // towards the socket's floor, closing the gap.
        let mut block = primitives::cube(1.0, 6);
        block.transform(|p| p * 4.0);
        let mut cutter = primitives::icosphere(1.3, 4);
        cutter.transform(|p| p + DVec3::new(2.3, 0.0, 0.0));
        let torso = crate::boolean::mesh_boolean(&block, &cutter, crate::boolean::BooleanOp::Difference).unwrap().mesh;
        let socket = recognize(&torso, tri_near(&torso, DVec3::new(1.0, 0.0, 0.0)), &RecognizeOptions::default()).unwrap();
        assert_eq!(socket.name(), "socket");
        let mut ball = primitives::icosphere(1.0, 4);
        ball.transform(|p| p + DVec3::new(2.3, 0.0, 0.0));
        let all: Vec<u32> = ball.triangle_ids().collect();
        let c = contact_fit(&ball, &all, &torso, &socket.tris, (DQuat::IDENTITY, DVec3::ZERO), &ContactOptions::default()).unwrap();
        assert!(c.shift.x < -0.15 && c.shift.y.abs() < 0.05 && c.shift.z.abs() < 0.05, "{}", c.shift);
        assert!(c.gap_after < 0.6 * c.gap_before, "{} → {}", c.gap_before, c.gap_after);
    }

    #[test]
    fn a_turned_corner_turns_back() {
        // Three faces at a cube's corner hold all six ways a part can move.
        let cube = primitives::cube(2.0, 4);
        let corner = |m: &Mesh| -> Vec<u32> { m.triangle_ids().filter(|&t| (m.tri_centroid(t) - DVec3::splat(1.0)).length() < 1.2).collect() };
        let mut recess = cube.clone();
        recess.reverse_orientation();
        let tris = corner(&recess);
        let r = DQuat::from_euler(glam::EulerRot::XYZ, 0.06, -0.04, 0.05);
        let t = DVec3::new(0.08, -0.05, 0.06);
        recess.transform(|p| r * p + t);
        let all: Vec<u32> = cube.triangle_ids().collect();
        let c = contact_fit(&recess, &tris, &cube, &all, (DQuat::IDENTITY, DVec3::ZERO), &ContactOptions { turn: true, keep: 1.0, ..Default::default() }).unwrap();
        // The fit undoes the misplacement.
        for p in [DVec3::splat(1.0), DVec3::new(1.0, 0.0, 0.3), DVec3::new(0.2, 1.0, 1.0)] {
            let moved = c.rot * (r * p + t) + c.shift;
            assert!((moved - p).length() < 0.01, "{p} → {moved}");
        }
        assert!(c.gap_after < 1e-3, "{}", c.gap_after);
    }

    #[test]
    fn a_flat_face_rests_on_a_ball_whichever_moves() {
        let ball = Feature { shape: Shape::Sphere { centre: DVec3::ZERO, radius: 2.0 }, convex: true, tris: Vec::new(), area: 1.0, rms: 0.0, body: Some(-DVec3::Z) };
        let face = flat(DVec3::new(5.0, 5.0, 5.0), DVec3::X, DVec3::Y, DVec2::new(1.0, 1.0));
        let s = seat(&face, &ball, &SeatOptions::default());
        // The face's centre on the ball's surface, away from the ball's part, facing the ball.
        assert!((s.apply(DVec3::new(5.0, 5.0, 5.0)) - DVec3::new(0.0, 0.0, 2.0)).length() < 1e-9, "{}", s.apply(DVec3::new(5.0, 5.0, 5.0)));
        assert!((s.rot * DVec3::X + DVec3::Z).length() < 1e-9);
    }

    #[test]
    fn a_cylinder_axis_points_out_of_its_part() {
        let mut m = primitives::cylinder(0.5, 4.0, 32, 16, true);
        // A stub standing on a block below it: the block's bulk is under the pin.
        let mut block = primitives::cube(6.0, 2);
        block.transform(|p| p + DVec3::new(0.0, 0.0, -3.0));
        m.append_mesh(&block, 1);
        let f = recognize(&m, tri_near(&m, DVec3::new(0.5, 0.0, 2.0)), &RecognizeOptions::default()).unwrap();
        match f.shape {
            Shape::Cylinder { axis, .. } => assert!(axis.dot(DVec3::Z) > 0.99, "{axis}"),
            s => panic!("{s:?}"),
        }
        assert!(f.body.unwrap().dot(-DVec3::Z) > 0.99);
    }

    #[test]
    fn angles_wrap() {
        assert!(close(wrap(3.0 * PI / 2.0), -PI / 2.0, 1e-12));
        assert!(close(wrap(-PI), PI, 1e-12));
        assert!(close(signed_angle(DVec3::X, DVec3::Y, DVec3::Z), PI / 2.0, 1e-12));
    }
}
