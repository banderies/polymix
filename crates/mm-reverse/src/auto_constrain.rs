// SPDX-License-Identifier: AGPL-3.0-only
//! Auto Constrain: a plane or a cylinder fitted to a region of the scan, held to what the part
//! already has. Ported from ScanRuler (`plugins/reverse/core/autoConstrain.ts`, with
//! `fitCylinderOnAxis` and `refineCylinderGeometric` from `src/core/fit/cylinder.ts`),
//! © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A free fit of a wall comes out a fifth of a degree off square to the floor beside it, and a
//! part closed out of such surfaces has no right angle in it: every face a hair trapezoid,
//! every dimension in the drawing a number with decimals nobody designed. The part was
//! designed square. So a fit that lands within a tilt tolerance of parallel or square to a
//! reference — the part's coordinate planes, a measured plane or cylinder, a surface fitted
//! before it — is fitted again with that direction held, and one that lands within a distance
//! of a reference plane, or of a reference axis, is put on it. What is held is said in the
//! report, with how far the free fit was from it, and the operator can force a relation or
//! forbid them all.
//!
//! Everything here is about a direction — a plane's normal, a cylinder's axis — against
//! reference directions: a reference plane's normal, a reference axis's own. `Along` holds the
//! direction to the reference's, `Across` holds it square to it. In the part's words a plane
//! along a plane is parallel to it and across it perpendicular; a cylinder along a plane's
//! normal stands square on the plane, and so on (`relation_words`).

use std::f64::consts::PI;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::fit2d::{fit_circle_2d, solve_linear};
use crate::linalg::ortho_basis;
pub use crate::model::{ConstrainMode, Relation};

/// Within this of parallel or square, a fit is taken to be meant so.
pub const TILT_TOLERANCE_DEG: f64 = 0.5;
/// Within this of a reference plane or axis, a parallel fit is put on it.
pub const COINCIDENCE_MM: f64 = 1.0;

/// Whether a reference is a plane or an axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RefKind {
    Plane,
    Axis,
}

/// Something a fit can be held to. The key names it in a saved feature: `datum:xy`,
/// `element:12`, `surface:7`. `dir` is a plane's normal, or an axis's direction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainRef {
    pub key: String,
    pub kind: RefKind,
    pub name: String,
    pub origin: DVec3,
    pub dir: DVec3,
}

/// A relation a fit was held to.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FoundRelation {
    pub relation: Relation,
    /// The reference, by key.
    pub reference: String,
    pub name: String,
    pub ref_kind: RefKind,
    /// How far the free fit's direction was from the relation, degrees.
    pub tilt_deg: f64,
    /// For `On`: how far the free fit lay from the reference, millimetres.
    pub shift_mm: Option<f64>,
}

/// A plane fitted and held.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedPlane {
    pub origin: DVec3,
    pub normal: DVec3,
    /// RMS of the points' distances to the plane as held, and as freely fitted — what holding
    /// it cost.
    pub rms: f64,
    pub rms_free: f64,
    pub max: f64,
    pub relations: Vec<FoundRelation>,
}

/// A cylinder fitted and held.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConstrainedCylinder {
    pub origin: DVec3,
    pub axis: DVec3,
    pub radius: f64,
    pub rms: f64,
    pub rms_free: f64,
    pub max: f64,
    pub relations: Vec<FoundRelation>,
}

/// An infinite cylinder: a point on its axis, its unit axis, its radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Cylinder {
    pub point: DVec3,
    pub axis: DVec3,
    pub r: f64,
}

/// What is being fitted: a plane or a cylinder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FitShape {
    Plane,
    Cylinder,
}

const DEG: f64 = 180.0 / PI;

/// The angle between two lines, 0…90°, whichever way their vectors point.
fn line_angle_deg(a: DVec3, b: DVec3) -> f64 {
    a.dot(b).abs().min(1.0).acos() * DEG
}

/// `d`, turned to point the way `like` does.
fn facing(d: DVec3, like: DVec3) -> DVec3 {
    if d.dot(like) < 0.0 {
        d * -1.0
    } else {
        d
    }
}

/// A vector made unit, or `None` for one shorter than 1e-12.
fn normalize(a: DVec3) -> Option<DVec3> {
    let l = a.length();
    (l >= 1e-12).then(|| a / l)
}

/// A vector made unit, or +Z for none.
fn normalize_or_z(v: DVec3) -> DVec3 {
    let len = v.length();
    if !(len > 1e-20) {
        return DVec3::Z;
    }
    v / len
}

/// What a relation is called for a plane or a cylinder against a reference plane or axis — the
/// words the box and the report use.
pub fn relation_words(shape: FitShape, relation: Relation, ref_kind: RefKind) -> &'static str {
    match (shape, ref_kind) {
        (FitShape::Plane, RefKind::Plane) => match relation {
            Relation::On => "Coincident with",
            Relation::Along => "Parallel to",
            Relation::Across => "Perpendicular to",
        },
        (FitShape::Plane, RefKind::Axis) | (FitShape::Cylinder, RefKind::Plane) => {
            if relation == Relation::Across {
                "Parallel to"
            } else {
                "Perpendicular to"
            }
        }
        (FitShape::Cylinder, RefKind::Axis) => match relation {
            Relation::On => "Coaxial with",
            Relation::Along => "Parallel to",
            Relation::Across => "Perpendicular to",
        },
    }
}

/// The relations a shape can have to a reference, for the dropdown.
pub fn relations_for(shape: FitShape, ref_kind: RefKind) -> &'static [Relation] {
    match (shape, ref_kind) {
        (FitShape::Plane, RefKind::Plane) | (FitShape::Cylinder, RefKind::Axis) => &[Relation::Along, Relation::Across, Relation::On],
        _ => &[Relation::Along, Relation::Across],
    }
}

struct Moments {
    mean: DVec3,
    /// The covariance's six numbers, row-major upper triangle, not divided.
    cov: [f64; 6],
}

fn moments_of(points: &[DVec3]) -> Moments {
    let n = points.len() as f64;
    let (mut mx, mut my, mut mz) = (0.0, 0.0, 0.0);
    for p in points {
        mx += p.x;
        my += p.y;
        mz += p.z;
    }
    mx /= n;
    my /= n;
    mz /= n;
    let (mut xx, mut xy, mut xz, mut yy, mut yz, mut zz) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for p in points {
        let (x, y, z) = (p.x - mx, p.y - my, p.z - mz);
        xx += x * x;
        xy += x * y;
        xz += x * z;
        yy += y * y;
        yz += y * z;
        zz += z * z;
    }
    Moments { mean: DVec3::new(mx, my, mz), cov: [xx, xy, xz, yy, yz, zz] }
}

/// The eigen decomposition of a symmetric 3×3 matrix (row-major) by cyclic Jacobi rotations:
/// the eigenvalues ascending and their unit eigenvectors. ScanRuler's `symmetricEigen3`
/// (`src/core/fit/linalg.ts`) as it is: 32 sweeps at most, the pairs (0, 1), (0, 2), (1, 2) in
/// turn, each vector made unit (+Z for none) once.
fn symmetric_eigen3(input: [f64; 9]) -> ([f64; 3], [DVec3; 3]) {
    let mut a = input;
    let mut v = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    for _sweep in 0..32 {
        let off = a[1].abs() + a[2].abs() + a[5].abs();
        let diag = a[0].abs() + a[4].abs() + a[8].abs();
        // 1e-14 relative is as converged as float64 rotations get.
        if off <= 1e-14 * diag || off == 0.0 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            let apq = a[p * 3 + q];
            if apq == 0.0 {
                continue;
            }
            // The rotation angle that zeroes the (p, q) entry.
            let theta = (a[q * 3 + q] - a[p * 3 + p]) / (2.0 * apq);
            let sign = if theta >= 0.0 { 1.0 } else { -1.0 };
            let t = sign / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k * 3 + p], a[k * 3 + q]);
                a[k * 3 + p] = c * akp - s * akq;
                a[k * 3 + q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p * 3 + k], a[q * 3 + k]);
                a[p * 3 + k] = c * apk - s * aqk;
                a[q * 3 + k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k * 3 + p], v[k * 3 + q]);
                v[k * 3 + p] = c * vkp - s * vkq;
                v[k * 3 + q] = s * vkp + c * vkq;
            }
        }
    }
    // Ascending, a stable sort as JavaScript's is (a negative zero read as zero).
    let mut order = [0usize, 1, 2];
    order.sort_by(|&i, &j| (a[i * 3 + i] + 0.0).total_cmp(&(a[j * 3 + j] + 0.0)));
    let values = order.map(|i| a[i * 3 + i]);
    let vectors = order.map(|i| normalize_or_z(DVec3::new(v[i], v[3 + i], v[6 + i])));
    (values, vectors)
}

/// The eigenvalues (ascending) and unit eigenvectors of the moments' covariance.
fn eigen_of(m: &Moments) -> ([f64; 3], [DVec3; 3]) {
    let c = m.cov;
    symmetric_eigen3([c[0], c[1], c[2], c[1], c[3], c[4], c[2], c[4], c[5]])
}

/// vᵀ C v for the covariance.
fn spread(m: &Moments, v: DVec3) -> f64 {
    let [xx, xy, xz, yy, yz, zz] = m.cov;
    v.x * (xx * v.x + xy * v.y + xz * v.z) + v.y * (xy * v.x + yy * v.y + yz * v.z) + v.z * (xz * v.x + yz * v.y + zz * v.z)
}

/// The direction the points scatter least along among those square to `r`: the least-squares
/// plane normal with that one condition on it. The covariance cut down to the two directions
/// square to `r`, and the smaller eigenvector of that 2 × 2.
fn least_spread_across(m: &Moments, r: DVec3) -> DVec3 {
    let (u, v) = ortho_basis(r);
    let a = spread(m, u);
    let c = spread(m, v);
    let [xx, xy, xz, yy, yz, zz] = m.cov;
    let b = u.x * (xx * v.x + xy * v.y + xz * v.z) + u.y * (xy * v.x + yy * v.y + yz * v.z) + u.z * (xz * v.x + yz * v.y + zz * v.z);
    // The smaller eigenvalue of [[a, b], [b, c]] and its vector.
    let low = (a + c) / 2.0 - ((a - c) / 2.0).hypot(b);
    let e = if b.abs() > 1e-300 {
        (b, low - a)
    } else if a <= c {
        (1.0, 0.0)
    } else {
        (0.0, 1.0)
    };
    let l = e.0.hypot(e.1);
    DVec3::new((u.x * e.0 + v.x * e.1) / l, (u.y * e.0 + v.y * e.1) / l, (u.z * e.0 + v.z * e.1) / l)
}

/// A direction held by what it is near, the relations that hold it, and the references it is
/// the direction of — the ones the fit may lie on.
struct HeldDirection<'a> {
    dir: DVec3,
    relations: Vec<FoundRelation>,
    same: Vec<&'a ConstrainRef>,
}

fn found(relation: Relation, r: &ConstrainRef, tilt_deg: f64) -> FoundRelation {
    FoundRelation { relation, reference: r.key.clone(), name: r.name.clone(), ref_kind: r.kind, tilt_deg, shift_mm: None }
}

/// A direction held by what it is near. `Across` with one reference leaves a turn about it
/// free, which `solve_across` spends on the points; with two that are not the same direction
/// nothing is left, and it is their cross product. `same` are the references the held direction
/// is the direction of, the one it was held to first — the ones the fit may lie on. `None`:
/// nothing holds it.
fn hold_direction<'a>(free: DVec3, refs: &'a [ConstrainRef], mode: &ConstrainMode, solve_across: impl Fn(DVec3) -> DVec3) -> Option<HeldDirection<'a>> {
    match mode {
        ConstrainMode::Free => None,
        ConstrainMode::Force { relation, reference } => {
            let r = refs.iter().find(|x| &x.key == reference)?;
            let d = r.dir;
            if *relation == Relation::Across {
                return Some(HeldDirection { dir: facing(solve_across(d), free), relations: vec![found(Relation::Across, r, (90.0 - line_angle_deg(free, d)).abs())], same: Vec::new() });
            }
            Some(HeldDirection { dir: facing(d, free), relations: vec![found(*relation, r, line_angle_deg(free, d))], same: vec![r] })
        }
        ConstrainMode::Auto => {
            // The nearest reference the fit runs along, within the tolerance. Of two as near
            // as each other the one that stands first: the part's own planes before the
            // measured elements before the surfaces fitted earlier, which were held to those in
            // their turn — "parallel to the XY plane" says more than "parallel to Surface 7",
            // which is parallel to the XY plane.
            let mut best: Option<usize> = None;
            let mut best_tilt = f64::INFINITY;
            for (i, r) in refs.iter().enumerate() {
                let tilt = line_angle_deg(free, r.dir);
                if tilt <= TILT_TOLERANCE_DEG && tilt < best_tilt - 1e-9 {
                    best = Some(i);
                    best_tilt = tilt;
                }
            }
            if let Some(b) = best {
                let dir = refs[b].dir;
                let mut same = vec![&refs[b]];
                same.extend(refs.iter().enumerate().filter(|(i, r)| *i != b && line_angle_deg(r.dir, dir) < 1e-6).map(|(_, r)| r));
                return Some(HeldDirection { dir: facing(dir, free), relations: vec![found(Relation::Along, &refs[b], best_tilt)], same });
            }
            // Else the references it stands square to: the nearest, and one more that is not
            // the same direction again — two walls of a box hold a third.
            let mut square: Vec<(&ConstrainRef, f64)> = refs.iter().map(|r| (r, (90.0 - line_angle_deg(free, r.dir)).abs())).filter(|x| x.1 <= TILT_TOLERANCE_DEG).collect();
            square.sort_by(|a, b| a.1.total_cmp(&b.1));
            let &(first, first_tilt) = square.first()?;
            if let Some(&(second, second_tilt)) = square.iter().find(|x| line_angle_deg(x.0.dir, first.dir) > 5.0) {
                let both = normalize(first.dir.cross(second.dir));
                // Two references a degree off square to each other can leave their cross
                // product further from the fit than either tolerance meant.
                if let Some(both) = both.filter(|b| line_angle_deg(*b, free) <= 2.0 * TILT_TOLERANCE_DEG) {
                    return Some(HeldDirection { dir: facing(both, free), relations: vec![found(Relation::Across, first, first_tilt), found(Relation::Across, second, second_tilt)], same: Vec::new() });
                }
            }
            Some(HeldDirection { dir: facing(solve_across(first.dir), free), relations: vec![found(Relation::Across, first, first_tilt)], same: Vec::new() })
        }
    }
}

/// The RMS and the largest of the points' distances to a plane.
fn plane_residuals(points: &[DVec3], origin: DVec3, normal: DVec3) -> (f64, f64) {
    let mut sum = 0.0;
    let mut max: f64 = 0.0;
    for p in points {
        let d = (p.x - origin.x) * normal.x + (p.y - origin.y) * normal.y + (p.z - origin.z) * normal.z;
        sum += d * d;
        if d.abs() > max {
            max = d.abs();
        }
    }
    ((sum / points.len() as f64).sqrt(), max)
}

/// A plane through the points held to the references as the mode says. The free fit is the
/// total-least-squares plane; with the normal held, the best plane is the one through the
/// centroid (the mean offset), and a plane within `COINCIDENCE_MM` of a parallel reference plane
/// is that plane. `None`: too few points, or all in a line.
pub fn constrain_plane(points: &[DVec3], refs: &[ConstrainRef], mode: &ConstrainMode) -> Option<ConstrainedPlane> {
    if points.len() < 3 {
        return None;
    }
    let m = moments_of(points);
    let (values, vectors) = eigen_of(&m);
    // Ascending: the first vector is the normal. A line of points has two directions of no
    // spread and no plane.
    if !(values[1] > 1e-12 * values[2].max(1e-300)) {
        return None;
    }
    let free_normal = vectors[0];
    let (rms_free, max_free) = plane_residuals(points, m.mean, free_normal);
    let Some(held) = hold_direction(free_normal, refs, mode, |r| least_spread_across(&m, r)) else {
        return Some(ConstrainedPlane { origin: m.mean, normal: free_normal, rms: rms_free, rms_free, max: max_free, relations: Vec::new() });
    };
    let mut origin = m.mean;
    let mut relations = held.relations;
    // On the first parallel reference plane it lies within reach of — which need not be the
    // one that gave the direction: the floor is parallel to the XY plane, and coincident with
    // the step measured beside it.
    let forced = matches!(mode, ConstrainMode::Force { relation: Relation::On, .. });
    for r in &held.same {
        if r.kind != RefKind::Plane {
            continue;
        }
        let shift = (m.mean - r.origin).dot(held.dir);
        if !forced && !(matches!(mode, ConstrainMode::Auto) && shift.abs() <= COINCIDENCE_MM) {
            continue;
        }
        origin = m.mean - held.dir * shift;
        relations[0] = FoundRelation { shift_mm: Some(shift.abs()), ..found(Relation::On, r, line_angle_deg(free_normal, r.dir)) };
        break;
    }
    let (rms, max) = plane_residuals(points, origin, held.dir);
    Some(ConstrainedPlane { origin, normal: held.dir, rms, rms_free, max, relations })
}

/// Distance of a point from the cylinder surface (positive outside), the unit radial direction
/// there and how far along the axis it lies. NaN on the axis, where the radial direction is
/// undefined.
fn radial(c: &Cylinder, p: DVec3) -> (f64, DVec3, f64) {
    let q = p - c.point;
    let t = q.x * c.axis.x + q.y * c.axis.y + q.z * c.axis.z;
    let w = DVec3::new(q.x - t * c.axis.x, q.y - t * c.axis.y, q.z - t * c.axis.z);
    let rho = (w.x * w.x + w.y * w.y + w.z * w.z).sqrt();
    if !(rho > 1e-12) {
        return (f64::NAN, DVec3::ZERO, t);
    }
    (rho - c.r, w / rho, t)
}

/// The RMS of the points' distances from a cylinder, those on its axis left out; infinite with
/// none.
fn cylinder_rms(points: &[DVec3], c: &Cylinder) -> f64 {
    let (mut sum, mut m) = (0.0, 0usize);
    for p in points {
        let e = radial(c, *p).0;
        if !e.is_finite() {
            continue;
        }
        sum += e * e;
        m += 1;
    }
    if m == 0 {
        f64::INFINITY
    } else {
        (sum / m as f64).sqrt()
    }
}

/// The RMS and the largest of the points' distances from a cylinder.
fn cylinder_residuals(points: &[DVec3], c: &Cylinder) -> (f64, f64) {
    let mut sum = 0.0;
    let mut max: f64 = 0.0;
    for p in points {
        let q = *p - c.point;
        let t = q.dot(c.axis);
        let d = (q - c.axis * t).length() - c.r;
        sum += d * d;
        if d.abs() > max {
            max = d.abs();
        }
    }
    ((sum / points.len() as f64).sqrt(), max)
}

/// Best circle in the plane perpendicular to a fixed axis — that is the cylinder's position and
/// radius for that direction. Algebraic (Coope) fit followed by the 2D form of the
/// orthogonal-distance fixed-point iteration.
pub fn fit_cylinder_on_axis(points: &[DVec3], axis: DVec3) -> Option<Cylinder> {
    if points.len() < 5 {
        return None;
    }
    let d = normalize_or_z(axis);
    let (u, v) = ortho_basis(d);
    let n = points.len() as f64;
    let centre = points.iter().fold(DVec3::ZERO, |s, p| s + *p) / n;
    // Project onto the (u, v) plane through the centroid.
    let flat: Vec<DVec2> = points.iter().map(|p| {
        let q = *p - centre;
        DVec2::new(q.x * u.x + q.y * u.y + q.z * u.z, q.x * v.x + q.y * v.y + q.z * v.z)
    }).collect();
    // The circle itself is the shared 2D kernel's job.
    let (c2, r) = fit_circle_2d(&flat)?;
    Some(Cylinder { point: DVec3::new(centre.x + c2.x * u.x + c2.y * v.x, centre.y + c2.x * u.y + c2.y * v.y, centre.z + c2.x * u.z + c2.y * v.z), axis: d, r })
}

/// Geometric (orthogonal-distance) refinement of all five cylinder degrees of freedom by damped
/// Gauss–Newton. The axis direction is perturbed inside the plane it is normal to (two angles),
/// which keeps it a unit vector without a constraint equation, and the axis point slides in the
/// same plane — the two degrees of freedom that would otherwise just walk along the axis are
/// simply not in the parametrisation.
pub fn refine_cylinder_geometric(points: &[DVec3], init: Cylinder, max_iter: usize) -> Cylinder {
    if points.len() < 6 {
        return init;
    }
    let mut c = init;
    let mut cost = cylinder_rms(points, &c);
    if !cost.is_finite() {
        return init;
    }
    let mut lambda = 1e-6;
    for _ in 0..max_iter {
        let (u, v) = ortho_basis(c.axis);
        let mut jtj = [0.0; 25];
        let mut jtf = [0.0; 5];
        let mut row = [0.0; 5];
        let mut m = 0;
        for p in points {
            let (e, w, t) = radial(&c, *p);
            if !e.is_finite() {
                continue;
            }
            let su = w.x * u.x + w.y * u.y + w.z * u.z;
            let sv = w.x * v.x + w.y * v.y + w.z * v.z;
            row[0] = -su;
            row[1] = -sv;
            row[2] = -t * su;
            row[3] = -t * sv;
            row[4] = -1.0;
            for a in 0..5 {
                jtf[a] += row[a] * e;
                for b in a..5 {
                    jtj[a * 5 + b] += row[a] * row[b];
                }
            }
            m += 1;
        }
        if m < 6 {
            break;
        }
        for a in 0..5 {
            for b in 0..a {
                jtj[a * 5 + b] = jtj[b * 5 + a];
            }
        }
        let mut applied = false;
        let mut converged = false;
        for _attempt in 0..6 {
            let mut mat = jtj;
            for a in 0..5 {
                mat[a * 5 + a] *= 1.0 + lambda;
            }
            let rhs: Vec<f64> = jtf.iter().map(|x| -x).collect();
            let Some(step) = solve_linear(5, &mut mat, &rhs) else {
                lambda *= 10.0;
                continue;
            };
            let axis = normalize_or_z(DVec3::new(c.axis.x + step[2] * u.x + step[3] * v.x, c.axis.y + step[2] * u.y + step[3] * v.y, c.axis.z + step[2] * u.z + step[3] * v.z));
            let cand = Cylinder {
                point: DVec3::new(c.point.x + step[0] * u.x + step[1] * v.x, c.point.y + step[0] * u.y + step[1] * v.y, c.point.z + step[0] * u.z + step[1] * v.z),
                axis,
                r: c.r + step[4],
            };
            if !(cand.r > 0.0) || !cand.point.x.is_finite() {
                lambda *= 10.0;
                continue;
            }
            let cand_cost = cylinder_rms(points, &cand);
            if cand_cost.is_finite() && cand_cost <= cost {
                converged = cost - cand_cost < 1e-12 * cost.max(cand.r);
                c = cand;
                cost = cand_cost;
                lambda = (lambda * 0.3).max(1e-9);
                applied = true;
                break;
            }
            lambda *= 10.0;
        }
        if !applied || converged {
            break;
        }
    }
    c
}

/// The free cylinder through the points, with nothing but the points to start from: a patch of
/// a cylinder has its axis near one of its three principal directions — the long way of a long
/// patch, the middle one of a short wide one — so each is tried as the start of the geometric
/// fit and the best kept.
pub fn fit_cylinder_free(points: &[DVec3]) -> Option<Cylinder> {
    if points.len() < 6 {
        return None;
    }
    let m = moments_of(points);
    let (_, vectors) = eigen_of(&m);
    let mut best: Option<Cylinder> = None;
    let mut best_rms = f64::INFINITY;
    for start in vectors {
        let Some(seeded) = fit_cylinder_on_axis(points, start) else { continue };
        let c = refine_cylinder_geometric(points, seeded, 60);
        if !(c.r > 0.0) || !c.r.is_finite() {
            continue;
        }
        let (rms, _) = cylinder_residuals(points, &c);
        if rms < best_rms {
            best = Some(c);
            best_rms = rms;
        }
    }
    best
}

/// A cylinder through the points held to the references: the axis along a reference direction
/// or square to it, fitted again with the axis fixed (the best circle in the plane square to
/// it); an axis within `COINCIDENCE_MM` of a parallel reference axis is that axis, the radius the
/// points' mean distance from it.
pub fn constrain_cylinder(points: &[DVec3], refs: &[ConstrainRef], mode: &ConstrainMode) -> Option<ConstrainedCylinder> {
    let free = fit_cylinder_free(points)?;
    let free_axis = free.axis;
    let (free_rms, free_max) = cylinder_residuals(points, &free);
    let mean = moments_of(points).mean;
    // The point of the axis beside the middle of the points.
    let middle = |c: &Cylinder| {
        let t = (mean.x - c.point.x) * c.axis.x + (mean.y - c.point.y) * c.axis.y + (mean.z - c.point.z) * c.axis.z;
        DVec3::new(c.point.x + t * c.axis.x, c.point.y + t * c.axis.y, c.point.z + t * c.axis.z)
    };
    // Square to a reference with a turn left free: the free axis laid into the plane square to
    // it. The fit on that axis settles the rest.
    let Some(held) = hold_direction(free_axis, refs, mode, |r| normalize(free_axis - r * free_axis.dot(r)).unwrap_or(free_axis)) else {
        return Some(ConstrainedCylinder { origin: middle(&free), axis: free_axis, radius: free.r, rms: free_rms, rms_free: free_rms, max: free_max, relations: Vec::new() });
    };
    let mut c = fit_cylinder_on_axis(points, held.dir)?;
    let mut relations = held.relations;
    let forced = matches!(mode, ConstrainMode::Force { relation: Relation::On, .. });
    for r in &held.same {
        if r.kind != RefKind::Axis {
            continue;
        }
        let off = c.point - r.origin;
        let apart = (off - held.dir * off.dot(held.dir)).length();
        if forced || (matches!(mode, ConstrainMode::Auto) && apart <= COINCIDENCE_MM) {
            let o = r.origin;
            let mut sum = 0.0;
            for p in points {
                let q = *p - o;
                let t = q.dot(held.dir);
                sum += (q - held.dir * t).length();
            }
            c = Cylinder { point: o, axis: held.dir, r: sum / points.len() as f64 };
            relations[0] = FoundRelation { shift_mm: Some(apart), ..found(Relation::On, r, line_angle_deg(free_axis, r.dir)) };
            break;
        }
    }
    let (rms, max) = cylinder_residuals(points, &c);
    Some(ConstrainedCylinder { origin: middle(&c), axis: held.dir, radius: c.r, rms, rms_free: free_rms, max, relations })
}

/// The rectangle of a fitted plane that is drawn and sewn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PlanePatch {
    pub corner: DVec3,
    pub u: DVec3,
    pub v: DVec3,
    pub width: f64,
    pub height: f64,
}

/// The rectangle of a fitted plane that is drawn and sewn: round the points as they fall on the
/// plane, its sides along the way the region runs, grown by `expand_pct` of each extent — a
/// surface is trimmed against its neighbours and has to reach them.
pub fn plane_patch(points: &[DVec3], origin: DVec3, normal: DVec3, expand_pct: f64) -> PlanePatch {
    let (u0, v0) = ortho_basis(normal);
    // The region's long way within the plane: the larger eigenvector of the points' 2 × 2
    // scatter there.
    let (mut aa, mut ab, mut bb, mut ma, mut mb) = (0.0, 0.0, 0.0, 0.0, 0.0);
    let n = points.len() as f64;
    let at = |p: &DVec3, w: DVec3| (p.x - origin.x) * w.x + (p.y - origin.y) * w.y + (p.z - origin.z) * w.z;
    for p in points {
        ma += at(p, u0);
        mb += at(p, v0);
    }
    ma /= n;
    mb /= n;
    for p in points {
        let (a, b) = (at(p, u0) - ma, at(p, v0) - mb);
        aa += a * a;
        ab += a * b;
        bb += b * b;
    }
    let turn = 0.5 * (2.0 * ab).atan2(aa - bb);
    let (st, ct) = turn.sin_cos();
    let u = DVec3::new(u0.x * ct + v0.x * st, u0.y * ct + v0.y * st, u0.z * ct + v0.z * st);
    let v = normal.cross(u);
    let (mut u_lo, mut u_hi, mut v_lo, mut v_hi) = (f64::INFINITY, f64::NEG_INFINITY, f64::INFINITY, f64::NEG_INFINITY);
    for p in points {
        let (a, b) = (at(p, u), at(p, v));
        u_lo = u_lo.min(a);
        u_hi = u_hi.max(a);
        v_lo = v_lo.min(b);
        v_hi = v_hi.max(b);
    }
    let gu = ((u_hi - u_lo) * (expand_pct / 100.0)).max(1e-3);
    let gv = ((v_hi - v_lo) * (expand_pct / 100.0)).max(1e-3);
    let corner = DVec3::new(
        origin.x + (u_lo - gu) * u.x + (v_lo - gv) * v.x,
        origin.y + (u_lo - gu) * u.y + (v_lo - gv) * v.y,
        origin.z + (u_lo - gu) * u.z + (v_lo - gv) * v.z,
    );
    PlanePatch { corner, u, v, width: u_hi - u_lo + 2.0 * gu, height: v_hi - v_lo + 2.0 * gv }
}

/// The stretch of a fitted cylinder that is drawn.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CylinderPatch {
    pub from: f64,
    pub to: f64,
    pub u: DVec3,
    pub v: DVec3,
    pub start: f64,
    pub sweep: f64,
    pub full: bool,
}

/// The stretch of a fitted cylinder that is drawn: along the axis from `from` to `to` of
/// `origin`, round it from the angle `start` through `sweep` (radians, about `axis` from `u`
/// towards `v`) — where the points are, grown by `expand_pct` each way. `full` when what is
/// left open would be a sliver: the whole tube then.
pub fn cylinder_patch(points: &[DVec3], origin: DVec3, axis: DVec3, expand_pct: f64) -> CylinderPatch {
    let (u, v) = ortho_basis(axis);
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    let mut angles: Vec<f64> = Vec::with_capacity(points.len());
    for p in points {
        let q = *p - origin;
        let t = q.dot(axis);
        lo = lo.min(t);
        hi = hi.max(t);
        angles.push(q.dot(v).atan2(q.dot(u)));
    }
    let grow = ((hi - lo) * (expand_pct / 100.0)).max(1e-3);
    // The points cover the turn but for their largest gap; the arc runs from the gap's far side
    // round to its near side.
    angles.sort_by(|a, b| a.total_cmp(b));
    let (first, last) = (angles.first().copied().unwrap_or(f64::NAN), angles.last().copied().unwrap_or(f64::NAN));
    let mut gap = first + 2.0 * PI - last;
    let mut start = first;
    for i in 1..angles.len() {
        if angles[i] - angles[i - 1] > gap {
            gap = angles[i] - angles[i - 1];
            start = angles[i];
        }
    }
    let covered = 2.0 * PI - gap;
    let more = covered * (expand_pct / 100.0);
    let sweep = covered + 2.0 * more;
    let full = sweep > 2.0 * PI - (20.0 * PI) / 180.0;
    CylinderPatch { from: lo - grow, to: hi + grow, u, v, start: start - more, sweep: if full { 2.0 * PI } else { sweep }, full }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fit2d::Mulberry32;

    fn rad(deg: f64) -> f64 {
        (deg * PI) / 180.0
    }

    fn angle_deg(a: DVec3, b: DVec3) -> f64 {
        (a.dot(b).abs().min(1.0).acos() * 180.0) / PI
    }

    fn unit(v: [f64; 3]) -> DVec3 {
        normalize(DVec3::from(v)).unwrap_or(DVec3::Z)
    }

    /// A point as the scan's Float32Array holds it.
    fn f32ed(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x as f32 as f64, y as f32 as f64, z as f32 as f64)
    }

    /// A 40 × 30 mm patch of a plane through `origin` with the given normal, with scatter along
    /// the normal.
    fn plane_points(origin: [f64; 3], normal: [f64; 3], noise: f64, seed: u32) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(seed);
        let n = unit(normal);
        let u = unit(n.cross(if n.x.abs() < 0.9 { DVec3::X } else { DVec3::Y }).to_array());
        let v = n.cross(u);
        let mut out = Vec::new();
        for i in 0..=40 {
            for j in 0..=30 {
                let (a, b, e) = ((i - 20) as f64, (j - 15) as f64, rand.gaussian() * noise);
                out.push(f32ed(origin[0] + a * u.x + b * v.x + e * n.x, origin[1] + a * u.y + b * v.y + e * n.y, origin[2] + a * u.z + b * v.z + e * n.z));
            }
        }
        out
    }

    /// A patch of a cylinder: `span` degrees round, 30 mm long.
    fn cylinder_points(origin: [f64; 3], axis: [f64; 3], radius: f64, span: f64, noise: f64, seed: u32) -> Vec<DVec3> {
        let mut rand = Mulberry32::new(seed);
        let d = unit(axis);
        let u = unit(d.cross(if d.x.abs() < 0.9 { DVec3::X } else { DVec3::Y }).to_array());
        let v = d.cross(u);
        let mut out = Vec::new();
        for i in 0..=60 {
            for j in 0..=30 {
                let (a, t, r) = (rad((i as f64 / 60.0) * span), (j - 15) as f64, radius + rand.gaussian() * noise);
                out.push(f32ed(
                    origin[0] + t * d.x + r * (a.cos() * u.x + a.sin() * v.x),
                    origin[1] + t * d.y + r * (a.cos() * u.y + a.sin() * v.y),
                    origin[2] + t * d.z + r * (a.cos() * u.z + a.sin() * v.z),
                ));
            }
        }
        out
    }

    fn reference(key: &str, kind: RefKind, name: &str, origin: [f64; 3], dir: [f64; 3]) -> ConstrainRef {
        ConstrainRef { key: key.to_string(), kind, name: name.to_string(), origin: DVec3::from(origin), dir: DVec3::from(dir) }
    }

    fn xy() -> ConstrainRef {
        reference("datum:xy", RefKind::Plane, "XY plane", [0.0; 3], [0.0, 0.0, 1.0])
    }
    fn yz() -> ConstrainRef {
        reference("datum:yz", RefKind::Plane, "YZ plane", [0.0; 3], [1.0, 0.0, 0.0])
    }
    fn zx() -> ConstrainRef {
        reference("datum:zx", RefKind::Plane, "ZX plane", [0.0; 3], [0.0, 1.0, 0.0])
    }
    fn datums() -> Vec<ConstrainRef> {
        vec![xy(), yz(), zx()]
    }

    fn close(a: f64, b: f64, digits: i32) -> bool {
        (a - b).abs() < 10f64.powi(-digits) / 2.0
    }

    #[test]
    fn a_top_face_0_3_degrees_off_level_is_made_level_at_the_points_mean_height() {
        let tilted = [rad(0.3).sin(), 0.0, rad(0.3).cos()];
        let fit = constrain_plane(&plane_points([5.0, 5.0, 12.0], tilted, 0.02, 3), &datums(), &ConstrainMode::Auto).expect("a plane");
        assert_eq!(fit.normal, DVec3::new(0.0, 0.0, 1.0));
        assert!(close(fit.origin.z, 12.0, 2));
        assert_eq!(fit.relations.len(), 1);
        assert_eq!((fit.relations[0].relation, fit.relations[0].reference.as_str()), (Relation::Along, "datum:xy"));
        assert!(close(fit.relations[0].tilt_deg, 0.3, 1));
        // Holding it costs what the tilt was over the patch, and no more.
        assert!(fit.rms > fit.rms_free);
        assert!(fit.rms < 0.1);
    }

    #[test]
    fn keeps_the_way_the_free_normal_faced() {
        let fit = constrain_plane(&plane_points([0.0, 0.0, -8.0], [0.002, 0.0, -1.0], 0.02, 3), &datums(), &ConstrainMode::Auto).expect("a plane");
        // The eigenvector's sign is arbitrary; whichever it was, the held normal is the
        // datum's or its opposite, never something between.
        assert_eq!(fit.normal.z.abs(), 1.0);
    }

    #[test]
    fn a_face_2_degrees_off_is_left_as_fitted() {
        // Tilted towards neither axis: leaning along X alone it would still be square to the ZX
        // plane, and rightly held so.
        let tilted = [rad(2.0).sin() * rad(40.0).cos(), rad(2.0).sin() * rad(40.0).sin(), rad(2.0).cos()];
        let fit = constrain_plane(&plane_points([0.0, 0.0, 12.0], tilted, 0.02, 3), &datums(), &ConstrainMode::Auto).expect("a plane");
        assert!(fit.relations.is_empty());
        assert!(angle_deg(fit.normal, DVec3::from(tilted)) < 0.05);
        assert_eq!(fit.rms, fit.rms_free);
    }

    #[test]
    fn within_a_millimetre_of_a_parallel_reference_plane_it_is_that_plane() {
        let floor = reference("element:4", RefKind::Plane, "Plane 4", [0.0, 0.0, 12.4], [0.0, 0.0, 1.0]);
        let fit = constrain_plane(&plane_points([60.0, 0.0, 12.0], [0.003, 0.0, 1.0], 0.02, 3), std::slice::from_ref(&floor), &ConstrainMode::Auto).expect("a plane");
        assert_eq!((fit.relations[0].relation, fit.relations[0].reference.as_str()), (Relation::On, "element:4"));
        assert!(close(fit.relations[0].shift_mm.unwrap_or(f64::NAN), 0.4, 1));
        assert!(close(fit.origin.z, 12.4, 6));
        // Further than that it is parallel only.
        let apart = constrain_plane(&plane_points([60.0, 0.0, 12.0 - COINCIDENCE_MM - 0.5], [0.003, 0.0, 1.0], 0.02, 3), &[floor], &ConstrainMode::Auto).expect("a plane");
        assert_eq!(apart.relations[0].relation, Relation::Along);
    }

    #[test]
    fn of_two_references_as_near_as_each_other_the_first_names_the_relation_and_a_coincident_one_among_them_is_still_found() {
        // A step measured in Measure, level like the XY plane and 12.3 mm up.
        let step = reference("element:9", RefKind::Plane, "Plane 9", [0.0, 0.0, 12.3], [0.0, 0.0, 1.0]);
        let refs = [xy(), step];
        let far = constrain_plane(&plane_points([0.0, 0.0, 40.0], [0.002, 0.0, 1.0], 0.02, 3), &refs, &ConstrainMode::Auto).expect("a plane");
        assert_eq!((far.relations[0].relation, far.relations[0].reference.as_str()), (Relation::Along, "datum:xy"));
        // Beside the step it is on the step, though the XY plane gave the direction.
        let beside = constrain_plane(&plane_points([0.0, 0.0, 12.0], [0.002, 0.0, 1.0], 0.02, 3), &refs, &ConstrainMode::Auto).expect("a plane");
        assert_eq!((beside.relations[0].relation, beside.relations[0].reference.as_str()), (Relation::On, "element:9"));
        assert!(close(beside.origin.z, 12.3, 6));
    }

    #[test]
    fn a_wall_square_to_the_floor_within_the_tolerance_is_made_square_its_turn_about_the_floors_normal_left_to_the_points() {
        // A wall turned 25° about Z and leaning 0.35° out of upright.
        let lean = rad(0.35);
        let upright = [rad(25.0).cos(), rad(25.0).sin(), 0.0];
        let leaning = [upright[0] * lean.cos(), upright[1] * lean.cos(), lean.sin()];
        let fit = constrain_plane(&plane_points([10.0, 10.0, 5.0], leaning, 0.02, 3), &[xy()], &ConstrainMode::Auto).expect("a plane");
        assert_eq!(fit.relations.len(), 1);
        assert_eq!((fit.relations[0].relation, fit.relations[0].reference.as_str()), (Relation::Across, "datum:xy"));
        assert!(fit.normal.z.abs() < 1e-9);
        assert!(angle_deg(fit.normal, DVec3::from(upright)) < 0.05);
    }

    #[test]
    fn square_to_two_references_nothing_is_left_to_fit_it_is_their_cross_product() {
        let almost_y = unit([rad(0.4).sin(), 1.0, rad(0.3).sin()]);
        // Square to XY within 0.3° and to YZ within 0.4°, and those two leave only Y — with no
        // plane among the references to be parallel to.
        let fit = constrain_plane(&plane_points([0.0, 20.0, 0.0], almost_y.to_array(), 0.02, 3), &[xy(), yz()], &ConstrainMode::Auto).expect("a plane");
        let mut keys: Vec<&str> = fit.relations.iter().map(|r| r.reference.as_str()).collect();
        keys.sort_unstable();
        assert_eq!(keys, vec!["datum:xy", "datum:yz"]);
        assert!(angle_deg(fit.normal, DVec3::Y) < 1e-6);
    }

    #[test]
    fn a_forced_relation_holds_however_far_off_the_fit_was_and_free_forbids_them_all() {
        let tilted = [rad(4.0).sin(), 0.0, rad(4.0).cos()];
        let points = plane_points([0.0, 0.0, 12.0], tilted, 0.02, 3);
        let forced = constrain_plane(&points, &datums(), &ConstrainMode::Force { relation: Relation::Along, reference: "datum:xy".to_string() }).expect("a plane");
        assert_eq!(forced.normal, DVec3::new(0.0, 0.0, 1.0));
        assert!(close(forced.relations[0].tilt_deg, 4.0, 1));
        let on = constrain_plane(&points, &datums(), &ConstrainMode::Force { relation: Relation::On, reference: "datum:xy".to_string() }).expect("a plane");
        assert!(close(on.origin.z, 0.0, 9));
        let level = constrain_plane(&plane_points([0.0, 0.0, 12.0], [0.001, 0.0, 1.0], 0.02, 3), &datums(), &ConstrainMode::Free).expect("a plane");
        assert!(level.relations.is_empty());
        assert_ne!(level.normal.z, 1.0);
    }

    #[test]
    fn has_no_plane_through_a_line_of_points() {
        let line: Vec<DVec3> = (0..10).map(|i| DVec3::new(i as f64, 0.0, 0.0)).collect();
        assert_eq!(constrain_plane(&line, &datums(), &ConstrainMode::Auto), None);
    }

    #[test]
    fn finds_a_third_of_a_cylinder_from_its_points_alone() {
        let axis = unit([0.3, -0.2, 1.0]);
        let c = fit_cylinder_free(&cylinder_points([4.0, 5.0, 6.0], axis.to_array(), 8.0, 120.0, 0.02, 5)).expect("a cylinder");
        assert!(close(c.r, 8.0, 1));
        assert!(angle_deg(c.axis, axis) < 0.2);
    }

    #[test]
    fn a_bore_0_3_degrees_off_upright_is_made_upright_and_fitted_again_on_that_axis() {
        let axis = [rad(0.3).sin(), 0.0, rad(0.3).cos()];
        let fit = constrain_cylinder(&cylinder_points([10.0, -4.0, 0.0], axis, 6.0, 200.0, 0.02, 5), &datums(), &ConstrainMode::Auto).expect("a cylinder");
        assert_eq!(angle_deg(fit.axis, DVec3::Z), 0.0);
        assert!(close(fit.radius, 6.0, 1));
        assert!(close(fit.origin.x, 10.0, 1));
        assert!(close(fit.origin.y, -4.0, 1));
        assert_eq!((fit.relations[0].relation, fit.relations[0].reference.as_str(), fit.relations[0].ref_kind), (Relation::Along, "datum:xy", RefKind::Plane));
        assert_eq!(relation_words(FitShape::Cylinder, Relation::Along, RefKind::Plane), "Perpendicular to");
    }

    #[test]
    fn within_a_millimetre_of_a_parallel_reference_axis_it_takes_that_axis() {
        let bore = reference("surface:3", RefKind::Axis, "Surface 3", [10.3, -4.2, 40.0], [0.0, 0.0, 1.0]);
        let fit = constrain_cylinder(&cylinder_points([10.0, -4.0, 0.0], [0.002, 0.0, 1.0], 9.0, 240.0, 0.02, 5), &[bore], &ConstrainMode::Auto).expect("a cylinder");
        assert_eq!((fit.relations[0].relation, fit.relations[0].reference.as_str()), (Relation::On, "surface:3"));
        assert!(close(fit.relations[0].shift_mm.unwrap_or(f64::NAN), 0.3f64.hypot(0.2), 1));
        assert!(close(fit.origin.x, 10.3, 6));
        assert!(close(fit.origin.y, -4.2, 6));
        // The radius is the points' mean distance from the axis they were moved to: over two
        // thirds of a turn that is the true radius and a share of the 0.36 mm the axis moved,
        // not the true radius again.
        assert!((fit.radius - 9.0).abs() < 0.36);
        assert!(fit.rms > fit.rms_free);
    }

    #[test]
    fn an_axis_lying_0_4_degrees_out_of_a_plane_is_laid_into_it() {
        let axis = unit([rad(30.0).cos(), rad(30.0).sin(), rad(0.4).sin()]);
        let fit = constrain_cylinder(&cylinder_points([0.0, 0.0, 10.0], axis.to_array(), 5.0, 180.0, 0.02, 5), &[xy()], &ConstrainMode::Auto).expect("a cylinder");
        assert_eq!((fit.relations[0].relation, fit.relations[0].reference.as_str()), (Relation::Across, "datum:xy"));
        assert!(fit.axis.z.abs() < 1e-9);
        assert_eq!(relation_words(FitShape::Cylinder, Relation::Across, RefKind::Plane), "Parallel to");
    }

    /// Not a ScanRuler test: the drawn stretch of a patch covers the points, grown as asked.
    #[test]
    fn draws_a_plane_patch_round_its_points_and_a_cylinder_patch_over_its_arc() {
        let points = plane_points([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 0.0, 3);
        let patch = plane_patch(&points, DVec3::ZERO, DVec3::Z, 10.0);
        assert!(close(patch.width.max(patch.height), 48.0, 6));
        assert!(close(patch.width.min(patch.height), 36.0, 6));
        let tube = cylinder_points([0.0, 0.0, 0.0], [0.0, 0.0, 1.0], 5.0, 90.0, 0.0, 5);
        let arc = cylinder_patch(&tube, DVec3::ZERO, DVec3::Z, 0.0);
        assert!(!arc.full);
        assert!(close(arc.sweep, PI / 2.0, 4));
        assert!(close(arc.to - arc.from, 30.0 + 2e-3, 6));
    }
}
