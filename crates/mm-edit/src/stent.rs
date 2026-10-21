//! Virtual stenting: place a stent of a given diameter and length between two cross-sections
//! of a vessel and reshape the vessel around it, to compare the flow through treated anatomy.
//!
//! **Placement** ([`place_stent`]). The stent's axis blends two paths between the sections:
//! the vessel's centerline as it is, and the smoothest path joining the sections, which is
//! where a free stent held at both would settle. That path is a line (G0), or a Hermite
//! spline leaving each section along its normal (G1) and also continuing the vessel's bend
//! there (G2), its tangent lengths chosen for the least bending energy for its length
//! (length × ∫κ² ds, least for a circular arc). *Straightness* mixes the two paths at equal
//! fractions of their length, and the axis is eased to bend no tighter than the stent's
//! diameter (where the vessel's course jogs, e.g. through an eccentric narrowing). Past each section the paths go on along the vessel and along
//! the smooth path's end direction (its bend fading out for G2). The stent keeps its length
//! along its own axis and is centred on the two sections, so a stent longer than the
//! stretch reaches equally far past each.
//!
//! **Bare (uncovered) stent** ([`apply_stent`]). A stent can only open the vessel:
//! 1. Where the stent grips the wall (the vessel no wider than the stent, letting go by 1.6×
//!    its radius), each slice of the vessel is carried from the centerline onto the stent's
//!    axis, moved and turned with it, so the vessel straightens with the stent. Part of a
//!    stent floating in a wider lumen moves nothing. The move fades out over the end blend past
//!    the stent's ends, and up side branches. Only the stented vessel's own wall moves: wall
//!    passing near it (the vessel turning back, another vessel) does not.
//! 2. The lumen is joined with the stent's tube (a mesh union). The tube is lofted along the
//!    stent from rings that, in each direction across, lie *stiffness*'s share of the way from
//!    the native wall (lightly smoothed) to the stent's radius: a perfect tube with stiffness
//!    1, the vessel unchanged with 0, and in between the native shape easing into the round
//!    stent. Past each end it eases into the wall over *end blend*. Wall inside the tube gives
//!    way to the tube's own surface, however folded or nearly closed it was; wall outside
//!    it, and side branches, are left alone. Where the union cannot be made (the tube
//!    reaches an open end of the mesh), the wall is pushed out to the tube instead.
//! 3. The reshaped region is remeshed.
//!
//! **Covered stent** (stent graft). The stented stretch is cut out and replaced by the same
//! rings: the lumen there is the stent, narrowed to its diameter where the vessel was
//! wider (an aneurysm or a dilated stretch is excluded, and side branches opening along it are
//! sealed off and removed), and open as far as the stiffness allows where it was narrower.

use crate::bridge::{aligned_loops, commit, quad_band, strip_with, AlignedLoops};
use crate::cut::{local_plane_cut_with, CutFill, KeepSide, LocalCutResult, PlaneCutParams, Side};
use crate::loft::{blend_join, EndContinuity};
use crate::util::smoothstep;
use crate::{EditError, EditResult};
use mm_core::hash::{FastMap, FastSet};
use mm_core::{Aabb, Bvh, DQuat, DVec3, Mesh, Plane, Ray};
use std::cell::Cell;
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use rayon::prelude::*;
use std::f64::consts::TAU;

/// A cross-section of the vessel at one end of the stretch to stent.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StentSection {
    /// Centre of the cross-section (on the vessel's axis).
    pub center: DVec3,
    /// Normal of the cross-section's plane (either sign).
    pub normal: DVec3,
}

/// The vessel's centerline through both sections and on past each, with the vessel's radius
/// at each point.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VesselPath {
    pub points: Vec<DVec3>,
    pub radii: Vec<f64>,
}

impl VesselPath {
    /// Length of the centerline between two sections (each where the centerline passes
    /// nearest its centre).
    pub fn span(&self, sections: [StentSection; 2]) -> Option<f64> {
        let c = Curve::new(&self.points, &self.radii)?;
        Some((c.project(sections[1].center) - c.project(sections[0].center)).abs())
    }
}

/// The stent and how it sits in the vessel.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StentParams {
    /// Nominal (fully expanded) diameter.
    pub diameter: f64,
    /// Length along the stent's axis.
    pub length: f64,
    /// 0: the stent follows the vessel's centerline; 1: the smoothest path between the
    /// sections.
    pub straightness: f64,
    /// How the smoothest path meets the sections: G0 free (a line), G1 along their normals,
    /// G2 also continuing the vessel's bend.
    pub continuity: EndContinuity,
    /// Radial force in 0..1: how far narrower wall is opened towards the stent's diameter.
    pub stiffness: f64,
    /// Distance past each end over which the wall eases back to its own shape.
    pub end_blend: f64,
    /// A covered stent (stent graft): the lumen along it becomes the stent's tube.
    pub covered: bool,
    /// Remesh the reshaped region to even triangles.
    pub remesh: bool,
    /// Edge length for the remesh (default: the region's mean before reshaping).
    pub edge_length: Option<f64>,
}

impl Default for StentParams {
    fn default() -> Self {
        Self {
            diameter: 8.0,
            length: 40.0,
            straightness: 0.5,
            continuity: EndContinuity::G1,
            stiffness: 1.0,
            end_blend: 2.0,
            covered: false,
            remesh: true,
            edge_length: None,
        }
    }
}

fn check(params: &StentParams) -> EditResult<()> {
    let unit = |x: f64| (0.0..=1.0).contains(&x);
    let positive = |x: f64| x.is_finite() && x > 0.0;
    if !positive(params.diameter) || !positive(params.length) || !unit(params.straightness) || !unit(params.stiffness) || !(params.end_blend >= 0.0 && params.end_blend.is_finite()) || params.edge_length.is_some_and(|h| !positive(h)) {
        return Err(EditError::InvalidParameter("the stent needs a positive diameter and length, straightness and stiffness in 0..1 and an end blend of 0 or more".into()));
    }
    Ok(())
}

/// The stent's axis and the vessel's centerline it was placed from, sampled evenly along the
/// axis from `margins[0]` before the stent's start to `margins[1]` past its end.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StentPlacement {
    /// Arc length along the axis of each sample: 0 at the stent's start, `length` at its end.
    pub s: Vec<f64>,
    /// The stent's axis.
    pub axis: Vec<DVec3>,
    /// The matching point of the vessel's centerline (where that slice of vessel came from).
    pub vessel: Vec<DVec3>,
    /// The vessel's radius there (from the centerline).
    pub vessel_radius: Vec<f64>,
    /// The stent's nominal radius.
    pub radius: f64,
    pub length: f64,
    /// Where the two sections lie along the axis (so `sections[0]` is how far the stent
    /// reaches back past the first; negative if it stops short of it).
    pub sections: [f64; 2],
    /// Length of the vessel's centerline between the sections.
    pub centerline_length: f64,
    /// Length of the stent's axis between the sections.
    pub path_length: f64,
    /// How far the samples run on past each end: the end blend. There the axis is the path
    /// along which the vessel returns from the stent's end onto its own centerline, which it
    /// meets at the end of the end blend. Nothing beyond moves.
    pub margins: [f64; 2],
    /// How sharply the vessel bends back past each end: its radius times the return path's
    /// curvature (from 0.4 the wall starts to crease on the inside of the turn, at 1 it folds;
    /// a longer end blend or less straightness eases it).
    pub return_bend: [f64; 2],
}

impl StentPlacement {
    /// How far the stent reaches past each section (negative: it stops short).
    pub fn overhang(&self) -> [f64; 2] {
        [self.sections[0], self.length - self.sections[1]]
    }

    fn locate(&self, s: f64) -> (usize, f64) {
        let k = self.s.len();
        let i = self.s.partition_point(|&x| x <= s).clamp(1, k - 1) - 1;
        (i, ((s - self.s[i]) / (self.s[i + 1] - self.s[i])).clamp(0.0, 1.0))
    }

    /// The point of the axis at `s`.
    pub fn point(&self, s: f64) -> DVec3 {
        let (i, f) = self.locate(s);
        self.axis[i].lerp(self.axis[i + 1], f)
    }

    /// The axis' direction at `s`.
    pub fn tangent(&self, s: f64) -> DVec3 {
        let h = 0.5 * (self.s[1] - self.s[0]);
        (self.point(s + h) - self.point(s - h)).normalize_or_zero()
    }

    /// Rotation-minimising frames at `count + 1` even steps from the stent's start to its
    /// end: point, tangent and two directions across.
    pub fn frames(&self, count: usize) -> Vec<[DVec3; 4]> {
        let count = count.max(1);
        let mut out: Vec<[DVec3; 4]> = Vec::with_capacity(count + 1);
        for j in 0..=count {
            let s = self.length * j as f64 / count as f64;
            let (x, t) = (self.point(s), self.tangent(s));
            let u = match out.last() {
                None => t.any_orthonormal_vector(),
                Some(&[x0, t0, u0, _]) => reflect_frame(x0, t0, u0, x, t),
            };
            out.push([x, t, u, t.cross(u)]);
        }
        out
    }

    /// Rings of the stent's surface (for previews): `count + 1` rings of `around` points.
    pub fn rings(&self, around: usize, count: usize) -> Vec<Vec<DVec3>> {
        self.frames(count)
            .into_iter()
            .map(|[x, _, u, v]| (0..around).map(|i| x + (u * (TAU * i as f64 / around as f64).cos() + v * (TAU * i as f64 / around as f64).sin()) * self.radius).collect())
            .collect()
    }
}

/// Double reflection (Wang et al. 2008): carry the frame direction `u0` at `x0` (tangent
/// `t0`) to `x1` (tangent `t1`) with the least rotation about the curve.
fn reflect_frame(x0: DVec3, t0: DVec3, u0: DVec3, x1: DVec3, t1: DVec3) -> DVec3 {
    let v1 = x1 - x0;
    let c1 = v1.length_squared();
    if c1 < 1e-300 {
        return u0;
    }
    let ul = u0 - v1 * (2.0 / c1 * v1.dot(u0));
    let tl = t0 - v1 * (2.0 / c1 * v1.dot(t0));
    let v2 = t1 - tl;
    let c2 = v2.length_squared();
    let u1 = if c2 < 1e-300 { ul } else { ul - v2 * (2.0 / c2 * v2.dot(ul)) };
    (u1 - t1 * u1.dot(t1)).normalize_or(t1.any_orthonormal_vector())
}

fn smoother(x: f64) -> f64 {
    let x = x.clamp(0.0, 1.0);
    x * x * x * (10.0 - 15.0 * x + 6.0 * x * x)
}

/// A polyline by arc length, carrying a value per point, continued straight past its ends.
#[derive(Clone, Debug)]
struct Curve {
    pts: Vec<DVec3>,
    vals: Vec<f64>,
    cum: Vec<f64>,
}

impl Curve {
    fn new(points: &[DVec3], vals: &[f64]) -> Option<Self> {
        let (mut pts, mut vs): (Vec<DVec3>, Vec<f64>) = (Vec::new(), Vec::new());
        for (i, &p) in points.iter().enumerate() {
            if p.is_finite() && pts.last().is_none_or(|q: &DVec3| q.distance(p) > 1e-12) {
                pts.push(p);
                vs.push(vals.get(i).copied().unwrap_or(0.0));
            }
        }
        if pts.len() < 2 {
            return None;
        }
        let mut cum = vec![0.0];
        for w in pts.windows(2) {
            cum.push(cum[cum.len() - 1] + w[0].distance(w[1]));
        }
        Some(Self { pts, vals: vs, cum })
    }

    fn sampled(f: impl Fn(f64) -> DVec3, n: usize) -> Option<Self> {
        let pts: Vec<DVec3> = (0..=n).map(|i| f(i as f64 / n as f64)).collect();
        Self::new(&pts, &[])
    }

    /// Resampled evenly and smoothed (box filtered three times, `width` across), its ends
    /// kept.
    fn smoothed(&self, width: f64) -> Self {
        let step = (width / 4.0).min(self.len() / 8.0).max(1e-12);
        let n = ((self.len() / step).ceil() as usize).max(2);
        let mut pts: Vec<DVec3> = (0..=n).map(|i| self.at(self.len() * i as f64 / n as f64)).collect();
        let vals: Vec<f64> = (0..=n).map(|i| self.value(self.len() * i as f64 / n as f64)).collect();
        let half = ((0.5 * width / step).round() as usize).max(1);
        for _ in 0..3 {
            let old = pts.clone();
            for (i, p) in pts.iter_mut().enumerate().take(n).skip(1) {
                let h = half.min(i).min(n - i);
                *p = old[i - h..=i + h].iter().sum::<DVec3>() / (2 * h + 1) as f64;
            }
        }
        Self::new(&pts, &vals).unwrap_or_else(|| self.clone())
    }

    fn reversed(&self) -> Self {
        let pts: Vec<DVec3> = self.pts.iter().rev().copied().collect();
        let vals: Vec<f64> = self.vals.iter().rev().copied().collect();
        Self::new(&pts, &vals).expect("a reversed curve stays a curve")
    }

    fn len(&self) -> f64 {
        self.cum[self.cum.len() - 1]
    }

    fn locate(&self, s: f64) -> (usize, f64) {
        let i = self.cum.partition_point(|&c| c <= s).clamp(1, self.pts.len() - 1) - 1;
        (i, ((s - self.cum[i]) / (self.cum[i + 1] - self.cum[i])).clamp(0.0, 1.0))
    }

    fn at(&self, s: f64) -> DVec3 {
        let n = self.pts.len();
        if s < 0.0 {
            return self.pts[0] + (self.pts[1] - self.pts[0]).normalize() * s;
        }
        if s > self.len() {
            return self.pts[n - 1] + (self.pts[n - 1] - self.pts[n - 2]).normalize() * (s - self.len());
        }
        let (i, f) = self.locate(s);
        self.pts[i].lerp(self.pts[i + 1], f)
    }

    fn value(&self, s: f64) -> f64 {
        let (i, f) = self.locate(s);
        self.vals[i] + (self.vals[i + 1] - self.vals[i]) * f
    }

    /// Arc length of the point nearest `p`.
    fn project(&self, p: DVec3) -> f64 {
        let mut best = (f64::INFINITY, 0.0);
        for i in 0..self.pts.len() - 1 {
            let (a, ab) = (self.pts[i], self.pts[i + 1] - self.pts[i]);
            let f = ((p - a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0);
            let d = p.distance_squared(a + ab * f);
            if d < best.0 {
                best = (d, self.cum[i] + f * (self.cum[i + 1] - self.cum[i]));
            }
        }
        best.1
    }
}

/// A quintic Hermite curve (a cubic when its second derivatives are the cubic's).
#[derive(Clone, Copy, Debug)]
struct Spline {
    p: [DVec3; 2],
    t: [DVec3; 2],
    a: [DVec3; 2],
}

impl Spline {
    /// From `p[0]` leaving along `dirs[0]` (tangent length `m[0]`) to `p[1]` arriving along
    /// `dirs[1]`, with curvature vectors `bend` (per arc length) at the ends for G2.
    fn new(p: [DVec3; 2], dirs: [DVec3; 2], m: [f64; 2], bend: Option<[DVec3; 2]>) -> Self {
        let d = p[1] - p[0];
        let t = [dirs[0] * m[0], dirs[1] * m[1]];
        let a = match bend {
            Some(k) => [k[0] * m[0] * m[0], k[1] * m[1] * m[1]],
            None => [6.0 * d - 4.0 * t[0] - 2.0 * t[1], -6.0 * d + 2.0 * t[0] + 4.0 * t[1]],
        };
        Self { p, t, a }
    }

    fn at(&self, t: f64) -> DVec3 {
        let (t2, t3, t4, t5) = (t * t, t.powi(3), t.powi(4), t.powi(5));
        let h0 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5;
        let h1 = t - 6.0 * t3 + 8.0 * t4 - 3.0 * t5;
        let h2 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5;
        let h3 = 0.5 * t3 - t4 + 0.5 * t5;
        let h4 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5;
        let h5 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5;
        self.p[0] * h0 + self.t[0] * h1 + self.a[0] * h2 + self.a[1] * h3 + self.t[1] * h4 + self.p[1] * h5
    }

    /// Bending energy for the length: length × ∫κ² ds (scale-free; a circular arc's is its
    /// turn squared, and no curve with the same total turn does better).
    fn bending(&self) -> f64 {
        const N: usize = 64;
        let q: Vec<DVec3> = (0..=N).map(|i| self.at(i as f64 / N as f64)).collect();
        let seg: Vec<f64> = q.windows(2).map(|w| w[0].distance(w[1])).collect();
        let len: f64 = seg.iter().sum();
        let mut e = 0.0;
        for i in 1..N {
            let turn = (q[i] - q[i - 1]).angle_between(q[i + 1] - q[i]);
            let l = 0.5 * (seg[i - 1] + seg[i]);
            if l > 0.0 && turn.is_finite() {
                e += turn * turn / l;
            }
        }
        len * e
    }
}

/// The spline from `p[0]` to `p[1]` along `dirs` with the least bending for its length: its
/// two tangent lengths searched from those of the circular arc between the directions.
fn smoothest(p: [DVec3; 2], dirs: [DVec3; 2], bend: Option<[DVec3; 2]>) -> Spline {
    let chord = p[0].distance(p[1]);
    let turn = dirs[0].angle_between(dirs[1]);
    let arc = if turn.is_finite() && turn > 1e-6 { chord * (0.5 * turn) / (0.5 * turn).sin() } else { chord };
    let make = |x: [f64; 2]| Spline::new(p, dirs, [chord * x[0].exp(), chord * x[1].exp()], bend);
    let (lo, hi) = (0.05f64.ln(), 4.0f64.ln());
    let mut x = [(arc / chord).ln().clamp(lo, hi); 2];
    let mut best = make(x).bending();
    let mut step = 0.4;
    while step > 1e-3 {
        let mut improved = false;
        for d in [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 1.0], [-1.0, -1.0], [1.0, -1.0], [-1.0, 1.0]] {
            let y = [(x[0] + d[0] * step).clamp(lo, hi), (x[1] + d[1] * step).clamp(lo, hi)];
            let e = make(y).bending();
            if e < best - 1e-12 * best.abs() {
                (best, x, improved) = (e, y, true);
            }
        }
        if !improved {
            step *= 0.5;
        }
    }
    make(x)
}

/// The smooth path continued past a section: from `p` along `t`, bending with `bend` (the
/// curvature vector at the section) fading out over `fade`; `n` points `ds` apart.
fn continue_path(p: DVec3, t: DVec3, bend: DVec3, fade: f64, ds: f64, n: usize) -> Vec<DVec3> {
    const SUB: usize = 4;
    let (mut p, mut t, mut s) = (p, t.normalize(), 0.0);
    let h = ds / SUB as f64;
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        for _ in 0..SUB {
            let w = (1.0 - s / fade).max(0.0);
            let k = (bend - t * bend.dot(t)) * w;
            p += t * h + k * (0.5 * h * h);
            t = (t + k * h).normalize();
            s += h;
        }
        out.push(p);
    }
    out
}

/// Place a stent between two cross-sections along the vessel's centerline (see the module
/// documentation).
pub fn place_stent(path: &VesselPath, sections: [StentSection; 2], params: &StentParams) -> EditResult<StentPlacement> {
    check(params)?;
    let mut centre = Curve::new(&path.points, &path.radii).ok_or_else(|| EditError::InvalidParameter("the vessel's centerline needs two points or more".into()))?;
    // Smoothed over about half the vessel's radius: a traced centerline kinks where rings are
    // partial (by an open end) or merge (at a junction), and the vessel's slices turn with it.
    let mean_radius = centre.vals.iter().sum::<f64>() / centre.vals.len() as f64;
    centre = centre.smoothed((0.5 * mean_radius.max(0.5 * params.diameter)).max(1e-9));
    let (mut sa, mut sb) = (centre.project(sections[0].center), centre.project(sections[1].center));
    if sa > sb {
        centre = centre.reversed();
        (sa, sb) = (centre.len() - sa, centre.len() - sb);
    }
    let gap = sb - sa;
    let r = 0.5 * params.diameter;
    if gap < 1e-6 * (r + centre.len()) {
        return Err(EditError::InvalidParameter("the two sections lie at the same place along the vessel".into()));
    }
    let length = params.length;
    let ds = (0.25 * r).min(length / 80.0).min(gap / 16.0);
    let [a, b] = sections;
    // The centerline, shifted to run through the sections' centres.
    let (oa, ob) = (a.center - centre.at(sa), b.center - centre.at(sb));
    let vessel = |s: f64| centre.at(s) + oa.lerp(ob, smoothstep((s - sa) / gap));
    let reach = |s: f64| (1.5 * centre.value(s).max(r)).max(2.0 * ds);
    let tangent = |s: f64| (vessel(s + 0.5 * reach(s)) - vessel(s - 0.5 * reach(s))).normalize_or_zero();
    let along = |n: DVec3, c: DVec3| {
        let n = n.normalize_or_zero();
        if n.dot(c).abs() < 0.3 { c } else { n * n.dot(c).signum() }
    };
    let (ta, tb) = (along(a.normal, tangent(sa)), along(b.normal, tangent(sb)));
    if ta == DVec3::ZERO || tb == DVec3::ZERO {
        return Err(EditError::Failed("the vessel's centerline has no direction at a section".into()));
    }
    let bend = |s: f64, t: DVec3| {
        let h = reach(s);
        let k = (vessel(s + h) - 2.0 * vessel(s) + vessel(s - h)) / (h * h);
        k - t * k.dot(t)
    };
    let g2 = params.continuity == EndContinuity::G2;
    let bends = if g2 { [bend(sa, ta), bend(sb, tb)] } else { [DVec3::ZERO; 2] };
    let (ideal, dirs) = if params.continuity == EndContinuity::G0 || a.center.distance(b.center) < 1e-9 * gap {
        let d = (b.center - a.center).normalize_or(ta);
        (Curve::new(&[a.center, a.center + d * a.center.distance(b.center).max(1e-9 * gap)], &[]), [d, d])
    } else {
        let sp = smoothest([a.center, b.center], [ta, tb], g2.then_some(bends));
        (Curve::sampled(|u| sp.at(u), 512), [ta, tb])
    };
    let ideal = ideal.ok_or_else(|| EditError::Failed("no path between the sections".into()))?;
    let w = params.straightness;
    // Between the sections: both paths at equal fractions of their length.
    let n = ((gap.max(ideal.len()) / ds).ceil() as usize).max(8);
    let mut span: Vec<(DVec3, DVec3, f64)> = (0..=n)
        .map(|i| {
            let u = i as f64 / n as f64;
            let s = sa + gap * u;
            let c = vessel(s);
            (c.lerp(ideal.at(ideal.len() * u), w), c, s)
        })
        .collect();
    // A stent bends no tighter than its diameter: where the vessel's course jogs (an
    // eccentric narrowing) its axis is eased, between the sections it is held at.
    let mut axis: Vec<DVec3> = span.iter().map(|x| x.0).collect();
    limit_bend(&mut axis, MAX_BEND / r);
    for (x, p) in span.iter_mut().zip(axis) {
        x.0 = p;
    }
    let span_len: f64 = span.windows(2).map(|x| x[0].0.distance(x[1].0)).sum();
    // Past each end the vessel is back on its own course by the end of the end blend: nothing
    // beyond it moves. (A few samples at least, where there is no end blend.)
    let ret = params.end_blend.max(2.0 * ds);
    let past = 1.25 * ((0.5 * (length - span_len)).max(0.0) + ret) + 4.0 * ds;
    let m = (past / ds).ceil() as usize;
    let ext_a = continue_path(a.center, -dirs[0], bends[0], 2.0 * r, ds, m);
    let ext_b = continue_path(b.center, dirs[1], bends[1], 2.0 * r, ds, m);
    let mut raw: Vec<(DVec3, DVec3, f64)> = Vec::with_capacity(2 * m + n + 1);
    for j in (1..=m).rev() {
        let s = sa - j as f64 * ds;
        let c = vessel(s);
        raw.push((c.lerp(ext_a[j - 1], w), c, s));
    }
    raw.extend_from_slice(&span);
    for (j, e) in ext_b.iter().enumerate() {
        let s = sb + (j + 1) as f64 * ds;
        let c = vessel(s);
        raw.push((c.lerp(*e, w), c, s));
    }
    let mut sigma = vec![0.0];
    for x in raw.windows(2) {
        sigma.push(sigma[sigma.len() - 1] + x[0].0.distance(x[1].0));
    }
    let (sig_a, sig_b) = (sigma[m], sigma[m + n]);
    let start = 0.5 * (sig_a + sig_b - length);
    // Both paths at arc length `x` along the stent's axis.
    let at = |x: f64| {
        let j = sigma.partition_point(|&c| c <= x).clamp(1, sigma.len() - 1) - 1;
        let f = ((x - sigma[j]) / (sigma[j + 1] - sigma[j]).max(1e-300)).clamp(0.0, 1.0);
        let ((p0, c0, s0), (p1, c1, s1)) = (raw[j], raw[j + 1]);
        (p0.lerp(p1, f), c0.lerp(c1, f), s0 + (s1 - s0) * f)
    };
    // Past each end the vessel returns from the stent's end onto its own course along the
    // path of least bending that leaves the stent along its direction and meets the
    // centerline tangentially at the end of the end blend.
    let ends = [(start, -1.0), (start + length, 1.0)];
    let return_path = |e: usize, l: f64| {
        let (x, out) = ends[e];
        let ((pe, _, sc), (pn, _, _)) = (at(x), at(x + out * ds));
        let ((_, c1, _), (_, c2, _)) = (at(x + out * l), at(x + out * (l + ds)));
        let dirs = [(pn - pe).normalize_or(DVec3::Z), (c2 - c1).normalize_or(DVec3::Z)];
        let path = Curve::sampled(|t| smoothest([pe, c1], dirs, None).at(t), 128).unwrap_or_else(|| Curve::new(&[pe, pe + dirs[0] * ds], &[]).unwrap());
        let steps = 64;
        let h = path.len() / steps as f64;
        let bend = (1..steps).map(|j| (path.at(j as f64 * h) - path.at((j - 1) as f64 * h)).angle_between(path.at((j + 1) as f64 * h) - path.at(j as f64 * h)) / h).filter(|k| k.is_finite()).fold(0.0, f64::max);
        (path, centre.value(sc).max(r) * bend)
    };
    let returns: Vec<(Curve, f64, f64)> = (0..2)
        .map(|e| {
            let (path, sharp) = return_path(e, ret);
            (path, ret, sharp)
        })
        .collect();
    let beyond = |e: usize, u: f64| {
        let (path, l, _) = &returns[e];
        let (_, c, sc) = at(ends[e].0 + ends[e].1 * u);
        (path.at(path.len() * (u / l).min(1.0)), c, sc)
    };
    let margins = [returns[0].1, returns[1].1];
    let return_bend = [returns[0].2, returns[1].2];
    let (from, to) = (start - margins[0], start + length + margins[1]);
    if from < sigma[0] || to > sigma[sigma.len() - 1] {
        return Err(EditError::Failed("the stent's path ran short".into()));
    }
    let k = ((to - from) / ds).ceil() as usize + 1;
    let mut pl = StentPlacement {
        radius: r,
        length,
        sections: [sig_a - start, sig_b - start],
        centerline_length: span.windows(2).map(|x| x[0].1.distance(x[1].1)).sum(),
        path_length: span_len,
        margins,
        return_bend,
        ..Default::default()
    };
    for i in 0..k {
        let x = from + (to - from) * i as f64 / (k - 1) as f64;
        let (p, c, sc) = if x < start {
            beyond(0, start - x)
        } else if x > start + length {
            beyond(1, x - start - length)
        } else {
            at(x)
        };
        pl.s.push(x - start);
        pl.axis.push(p);
        pl.vessel.push(c);
        pl.vessel_radius.push(centre.value(sc));
    }
    Ok(pl)
}

/// Where moving the wall tears it, the moves are evened out. Each slice of the vessel is
/// carried as a whole, and where the vessel's course turns tighter than the wall's distance
/// from it (it jogs through an eccentric narrowing), wall a few millimetres off the course can
/// be nearer one slice or another from one vertex to the next, and so carried apart: a step.
/// Edges stretched or squeezed by more than half their length mark it (turning a slice as a
/// whole changes no edge). Around them a zone of the moved wall takes moves interpolated
/// smoothly from its rim (a harmonic fill), widened until no edge in it is torn. A step
/// becomes a smooth change a few millimetres wide; the rest is untouched.
pub(crate) fn relax(mesh: &Mesh, moves: &mut [(u32, DVec3)]) {
    const MOST: f64 = 1.0;
    let n = moves.len();
    let index: FastMap<u32, usize> = moves.iter().enumerate().map(|(i, m)| (m.0, i)).collect();
    // Each moved vertex's neighbours: moved ones by index, unmoved ones as fixed points.
    let neighbours: Vec<Vec<(Option<usize>, u32)>> = moves.iter().map(|&(v, _)| mesh.vertex_neighbors(v).map(|u| (index.get(&u).copied(), u)).collect()).collect();
    let mut shift: Vec<DVec3> = moves.iter().map(|&(v, p)| p - mesh.vertex(v)).collect();
    let torn = |shift: &[DVec3]| -> Vec<usize> {
        (0..n)
            .into_par_iter()
            .filter(|&i| {
                let p = mesh.vertex(moves[i].0);
                neighbours[i].iter().any(|&(j, u)| {
                    let q = mesh.vertex(u);
                    let before = p.distance(q);
                    let after = (p + shift[i]).distance(q + j.map_or(DVec3::ZERO, |j| shift[j]));
                    (after - before).abs() > MOST * before
                })
            })
            .collect()
    };
    let mut zone = vec![false; n];
    for _ in 0..12 {
        let bad = torn(&shift);
        if bad.is_empty() {
            break;
        }
        // The zone: the torn vertices and three rings around them (of the moved wall).
        let mut grow: Vec<usize> = bad;
        for _ in 0..3 {
            let next: Vec<usize> = grow.iter().flat_map(|&i| neighbours[i].iter().filter_map(|&(j, _)| j)).collect();
            for &i in &grow {
                zone[i] = true;
            }
            grow = next.into_iter().filter(|&j| !zone[j]).collect();
        }
        for &i in &grow {
            zone[i] = true;
        }
        let members: Vec<usize> = (0..n).filter(|&i| zone[i]).collect();
        for _ in 0..300 {
            let old = shift.clone();
            for &i in &members {
                let around = &neighbours[i];
                shift[i] = around.iter().map(|&(j, _)| j.map_or(DVec3::ZERO, |j| old[j])).sum::<DVec3>() / around.len().max(1) as f64;
            }
        }
    }
    for (m, d) in moves.iter_mut().zip(shift) {
        m.1 = mesh.vertex(m.0) + d;
    }
}

/// Past the end blend (`end_blend`) at either end, by the vessel's slice nearest `p` (as the
/// carry decides what it moves). Nothing there changes.
fn past_end(pl: &StentPlacement, end_blend: f64, p: DVec3) -> bool {
    let k = pl.vessel.len();
    let mut best = (f64::INFINITY, 0.0);
    for i in 0..k - 1 {
        let (a, ab) = (pl.vessel[i], pl.vessel[i + 1] - pl.vessel[i]);
        let f = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
        let d = p.distance_squared(a + ab * f);
        if d < best.0 {
            best = (d, i as f64 + f);
        }
    }
    let i = (best.1.floor() as usize).min(k - 2);
    let s = pl.s[i] + (pl.s[i + 1] - pl.s[i]) * (best.1 - i as f64);
    best.1 <= 0.0 || best.1 >= (k - 1) as f64 || s < -end_blend || s > pl.length + end_blend
}

/// The sharpest a stent bends: its radius times its axis' curvature (a bend radius of the
/// stent's diameter). A tube bent tighter than its radius folds through itself.
const MAX_BEND: f64 = 0.5;

/// Ease `pts` where they bend with a curvature over `max` (the ends stay put): the sharpest
/// stretches are smoothed until none is.
fn limit_bend(pts: &mut [DVec3], max: f64) {
    let n = pts.len();
    if n < 3 {
        return;
    }
    let curvature = |p: &[DVec3], i: usize| {
        let (a, b, c) = (p[i - 1], p[i], p[i + 1]);
        2.0 * (b - a).cross(c - b).length() / (a.distance(b) * b.distance(c) * a.distance(c)).max(1e-300)
    };
    for _ in 0..2000 {
        let sharp: Vec<usize> = (1..n - 1).filter(|&i| curvature(pts, i) > max).collect();
        if sharp.is_empty() {
            return;
        }
        // Each sharp point and its neighbours half way to their neighbours' mean.
        let mut mark = vec![false; n];
        for &i in &sharp {
            mark[i.saturating_sub(2)..=(i + 2).min(n - 1)].fill(true);
        }
        let old = pts.to_vec();
        for i in 1..n - 1 {
            if mark[i] {
                pts[i] = old[i] * 0.5 + (old[i - 1] + old[i + 1]) * 0.25;
            }
        }
    }
}

/// What [`apply_stent`] did.
#[derive(Clone, Debug, Default)]
pub struct StentReport {
    /// Vertices moved, and the largest move.
    pub moved: usize,
    pub max_shift: f64,
    /// Narrowest lumen along the stent, before and after: the wall's distance from the
    /// stent's axis (the slice's 10th percentile).
    pub narrowest: [f64; 2],
    /// The stent's axis runs outside the vessel somewhere (through its wall: the bend is too
    /// tight for that straightness).
    pub outside: bool,
    /// Open ends of the mesh the reshaping moved (e.g. an inlet close past a straightened
    /// stent's end), which a flow model would see moved.
    pub moved_openings: usize,
    /// Covered: open ends (of side branches) removed with the excluded wall.
    pub sealed_openings: usize,
    /// Covered: triangles of wall excluded by the stent and removed.
    pub removed_triangles: usize,
    /// The reshaped region's triangles.
    pub triangles: Vec<u32>,
    /// The input's triangles that were reshaped (ids in the input mesh), e.g. to tell
    /// crossings the stent made from ones that were already there.
    pub source_triangles: Vec<u32>,
    /// A bare stent's wall was pushed out rather than joined to the stent's tube, and why.
    pub pushed: Option<String>,
    /// Covered: how wide the vessel is where each end of the graft joins it, over the stent
    /// (the cut's equivalent radius over the stent's). Well over 1, the graft lands in a far
    /// wider lumen, where a real one would leave a gap; the flare to the wall there can fold.
    pub landing: [f64; 2],
    /// The remesh of the stretched wall was undone: it made the wall run into itself.
    pub remesh_undone: bool,
}

fn tangents(pts: &[DVec3]) -> Vec<DVec3> {
    let k = pts.len();
    (0..k).map(|i| (pts[(i + 1).min(k - 1)] - pts[i.saturating_sub(1)]).normalize_or(DVec3::Z)).collect()
}

fn lerp_at(v: &[DVec3], x: f64) -> DVec3 {
    let i = (x.floor().max(0.0) as usize).min(v.len() - 2);
    v[i].lerp(v[i + 1], (x - i as f64).clamp(0.0, 1.0))
}

fn lerp_f(v: &[f64], x: f64) -> f64 {
    let i = (x.floor().max(0.0) as usize).min(v.len() - 2);
    v[i] + (v[i + 1] - v[i]) * (x - i as f64).clamp(0.0, 1.0)
}

fn box_blur(v: &[f64], half: usize) -> Vec<f64> {
    (0..v.len())
        .map(|i| {
            let (a, b) = (i.saturating_sub(half), (i + half).min(v.len() - 1));
            v[a..=b].iter().sum::<f64>() / (b - a + 1) as f64
        })
        .collect()
}

fn percentile(v: &mut [f64], q: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    Some(v[((v.len() - 1) as f64 * q) as usize])
}

/// Reshape `mesh` around a stent placed by [`place_stent`] (see the module documentation).
/// `owner` marks the vertices of the stented vessel itself (indexed by vertex id), e.g. those
/// whose centerline ring lies on the path; without it, vertices within 2.5 radii of the
/// centerline count. Other vertices near them (side branches) move less with distance.
///
/// A bare stent opens the vessel by joining its tube to the lumen (see [`open_with_tube`]);
/// where that cannot be done (the tube reaches an open end of the mesh, or the union fails),
/// the wall is pushed out instead. The mesh is rebuilt where the tube joins it, so vertex
/// and triangle ids change there.
pub fn apply_stent(mesh: &mut Mesh, pl: &StentPlacement, params: &StentParams, owner: Option<&[bool]>) -> EditResult<StentReport> {
    apply_stent_with(mesh, pl, params, owner, None)
}

/// [`apply_stent`], reporting progress (0..1) to `progress`, which returns `false` to cancel:
/// the work then stops at the next step (between the reshaping, the union, the remesh and the
/// checks, and within the remesh) with the error "cancelled", leaving `mesh` in a valid state
/// but part way (a superseded preview's mesh is dropped anyway).
pub fn apply_stent_with(mesh: &mut Mesh, pl: &StentPlacement, params: &StentParams, owner: Option<&[bool]>, progress: Option<&dyn Fn(f32) -> bool>) -> EditResult<StentReport> {
    check(params)?;
    // Reported never lower than before: the fallback below goes over the same share again.
    let last = Cell::new(0f32);
    let report_up = |x: f32| {
        last.set(last.get().max(x));
        progress.is_none_or(|f| f(last.get()))
    };
    let all = Progress { f: progress.map(|_| &report_up as &dyn Fn(f32) -> bool), from: 0.0, to: 1.0 };
    all.check(0.0)?;
    if params.stiffness <= 0.0 && !params.covered {
        return reshape(mesh, pl, params, owner, true, None, all);
    }
    if params.covered {
        // The tube's rings from the native lumen, measured on the straightened vessel.
        let mut carried = mesh.clone();
        reshape(&mut carried, pl, &StentParams { covered: false, remesh: false, ..*params }, owner, false, None, all.part(0.0, 0.2))?;
        let h = params.edge_length.unwrap_or_else(|| carried.average_edge_length());
        let frames = pl.frames(((pl.length / h).round() as usize).max(3));
        let rings = stent_rings(&carried, &frames, pl.radius, params.stiffness, h);
        drop(carried);
        all.check(0.25)?;
        let s = (0..frames.len()).map(|j| pl.length * j as f64 / (frames.len() - 1) as f64).collect();
        let profile = Profile { s, frames, rings };
        let mut report = reshape(mesh, pl, params, owner, true, Some(&profile), all.part(0.25, 1.0))?;
        // The lumen along a covered stent is its tube (the 10th percentile around each ring).
        report.narrowest[1] = profile.rings.iter().filter_map(|ring| percentile(&mut ring.radii.clone(), 0.1)).fold(f64::INFINITY, f64::min);
        return Ok(report);
    }
    // Straightened only, then opened by the stent's tube.
    let mut carried = mesh.clone();
    let mut report = reshape(&mut carried, pl, params, owner, false, None, all.part(0.0, 0.25))?;
    match open_with_tube(&mut carried, pl, params, all.part(0.25, 1.0)) {
        Ok(None) => {
            *mesh = carried;
            Ok(report)
        }
        Ok(Some(opened)) => {
            // The input's triangles near the tube (by place: ids changed where the
            // straightened wall was remeshed).
            let pad = DVec3::splat(opened.reach);
            let (lo, hi) = opened.centre.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, b), &p| (a.min(p - pad), b.max(p + pad)));
            let near = |p: DVec3| p.cmpge(lo).all() && p.cmple(hi).all() && opened.centre.windows(2).any(|w| point_segment_distance(p, w[0], w[1]) < opened.reach);
            report.source_triangles.extend(live_triangles_where(mesh, |t| mesh.tri_positions(t).iter().any(|&p| near(p))));
            report.source_triangles.sort_unstable();
            report.source_triangles.dedup();
            *mesh = carried;
            // (Opening never narrows the lumen: the two measures differ a little.)
            report.narrowest[1] = opened.narrowest[1].max(report.narrowest[0]);
            report.max_shift = report.max_shift.max(opened.widened);
            report.triangles = opened.triangles;
            Ok(report)
        }
        // A cancelled union is not a failed one: nothing is pushed out instead.
        Err(e) if is_cancelled(&e) || !all.at(0.5) => Err(cancelled()),
        Err(e) => {
            drop(carried);
            let mut report = reshape(mesh, pl, params, owner, true, None, all.part(0.25, 1.0))?;
            report.pushed = Some(e.to_string());
            Ok(report)
        }
    }
}

/// The caller's progress callback over a share (`from`..`to`) of the work, as a check for
/// cancellation between steps.
#[derive(Clone, Copy)]
struct Progress<'a> {
    f: Option<&'a dyn Fn(f32) -> bool>,
    from: f32,
    to: f32,
}

impl Progress<'_> {
    /// Report `x` (0..1) of this share; `false` once the work is cancelled.
    fn at(&self, x: f32) -> bool {
        self.f.is_none_or(|f| f(self.from + (self.to - self.from) * x.clamp(0.0, 1.0)))
    }

    /// [`Progress::at`] as a result: the "cancelled" error once cancelled.
    fn check(&self, x: f32) -> EditResult<()> {
        if self.at(x) {
            Ok(())
        } else {
            Err(cancelled())
        }
    }

    /// The share `a`..`b` of this one.
    fn part(&self, a: f32, b: f32) -> Self {
        let at = |x: f32| self.from + (self.to - self.from) * x;
        Progress { f: self.f, from: at(a), to: at(b) }
    }
}

fn cancelled() -> EditError {
    EditError::Failed("cancelled".into())
}

fn is_cancelled(e: &EditError) -> bool {
    matches!(e, EditError::Failed(s) if s == "cancelled")
}

/// Remesh with the progress reported over `p`'s share; `None` where it was cancelled.
fn remesh_within(mesh: &mut Mesh, params: &mm_remesh::RemeshParams, p: Progress) -> Option<Result<mm_remesh::RemeshReport, mm_remesh::RemeshError>> {
    let sub = |x: f32| p.at(x);
    let r = mm_remesh::remesh(mesh, params, p.f.map(|_| &sub as &dyn Fn(f32) -> bool));
    match r {
        Ok(rep) if rep.cancelled => None,
        r => Some(r),
    }
}

/// The live triangles for which `keep` holds, in id order (found in parallel).
fn live_triangles_where(m: &Mesh, keep: impl Fn(u32) -> bool + Sync) -> Vec<u32> {
    (0..m.max_tid() as u32).into_par_iter().filter(|&t| m.is_triangle(t) && keep(t)).collect()
}

fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// One ring of the stent's lumen, across its frame: the wall's distance from the axis in each
/// direction (see [`ring_angle`]; infinite where there is none) and the ring's own radius
/// there.
#[derive(Clone, Debug)]
struct Ring {
    walls: Vec<f64>,
    radii: Vec<f64>,
}

impl Ring {
    /// The native lumen's radius about the axis: the nearest wall.
    fn native(&self) -> f64 {
        self.walls.iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// The `j`th of `around` directions across a ring, turned by an irrational fraction of a step,
/// so the tube's vertices do not line up exactly with a regular wall's (a union cannot
/// resolve a vertex lying on an edge).
fn ring_angle(j: usize, around: usize) -> f64 {
    TAU * (j as f64 + 0.381_966) / around as f64
}

/// The stent's lumen at each frame (point, tangent and two directions across, as
/// [`StentPlacement::frames`]), keeping the native lumen's shape as far as *stiffness* lets it:
/// in each direction the ring lies *stiffness*'s share of the way from the native wall to the
/// stent's radius (2 mm across natively, a 6 mm stent and 50 % make 4 mm), never wider than
/// the stent. So 100 % makes a round tube, 0 % the native lumen itself, and in between its
/// shape eases into the round stent. The native outline is smoothed a little around each ring
/// (over about ±7°) and along the stent (over a quarter of its radius), so the mesh's
/// roughness does not carry into the tube, but never beyond the wall, so with no stiffness
/// nothing opens.
fn stent_rings(mesh: &Mesh, frames: &[[DVec3; 4]], r: f64, st: f64, h: f64) -> Vec<Ring> {
    let around = ((TAU * r / h).ceil() as usize).clamp(16, 720);
    let near: Vec<u32> = {
        let pad = DVec3::splat(2.0 * r);
        let (lo, hi) = frames.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, b), f| (a.min(f[0] - pad), b.max(f[0] + pad)));
        live_triangles_where(mesh, |t| mesh.tri_positions(t).iter().any(|p| p.cmpge(lo).all() && p.cmple(hi).all()))
    };
    let bvh = Bvh::build_subset(mesh, &near);
    let walls: Vec<Vec<f64>> = frames
        .par_iter()
        .map(|&[x, _, u, v]| (0..around).map(|j| ring_angle(j, around)).map(|a| bvh.raycast(mesh, &Ray::new(x, u * a.cos() + v * a.sin())).map_or(f64::INFINITY, |hit| hit.t)).collect())
        .collect();
    let radii: Vec<Vec<f64>> = if st >= 1.0 {
        vec![vec![r; around]; frames.len()]
    } else {
        // The outline, smoothed (no wall, or wall wider than the stent: the stent's radius).
        let mut outline: Vec<Vec<f64>> = walls.iter().map(|w| w.iter().map(|&x| x.min(r)).collect()).collect();
        let half_angle = (around / 48).max(1);
        let half_ring = if frames.len() > 2 { ((0.25 * r / (frames[1][0] - frames[0][0]).length().max(1e-9)).round() as usize).max(1) } else { 0 };
        for _ in 0..2 {
            let old = outline.clone();
            for (i, row) in outline.iter_mut().enumerate() {
                let rings = i.saturating_sub(half_ring)..=(i + half_ring).min(old.len() - 1);
                let count = (rings.clone().count() * (2 * half_angle + 1)) as f64;
                for (j, x) in row.iter_mut().enumerate() {
                    *x = rings.clone().flat_map(|ii| (0..=2 * half_angle).map(move |dj| (ii, (j + around + dj - half_angle) % around))).map(|(ii, jj)| old[ii][jj]).sum::<f64>() / count;
                }
            }
        }
        outline
            .iter()
            .zip(&walls)
            .map(|(o, w)| o.iter().zip(w).map(|(&smooth, &wall)| {
                let native = smooth.min(wall);
                r - (1.0 - st) * (r - native).max(0.0)
            }).collect())
            .collect()
    };
    walls.into_iter().zip(radii).map(|(walls, radii)| Ring { walls, radii }).collect()
}

/// The stent's lumen as a function of where along the stent and which way across: [`Ring`]s
/// measured at frames, read in any frame (a covered stent's tube has its own).
struct Profile {
    s: Vec<f64>,
    frames: Vec<[DVec3; 4]>,
    rings: Vec<Ring>,
}

impl Profile {
    /// The ring's radius at `s` along the stent in direction `dir` (across the axis).
    fn radius(&self, s: f64, dir: DVec3) -> f64 {
        let at = |j: usize| {
            let [_, _, u, v] = self.frames[j];
            let radii = &self.rings[j].radii;
            let n = radii.len();
            let x = (dir.dot(v).atan2(dir.dot(u)).rem_euclid(TAU) / TAU * n as f64 - 0.381_966).rem_euclid(n as f64);
            let (i, f) = (x.floor() as usize % n, x - x.floor());
            radii[i] * (1.0 - f) + radii[(i + 1) % n] * f
        };
        let j = self.s.partition_point(|&x| x <= s).clamp(1, self.s.len().max(2) - 1) - 1;
        let k = (j + 1).min(self.s.len() - 1);
        let f = ((s - self.s[j]) / (self.s[k] - self.s[j]).max(1e-300)).clamp(0.0, 1.0);
        at(j) * (1.0 - f) + at(k) * f
    }
}

/// The bare stent's tube joined to the lumen: how it went.
struct TubeOpening {
    /// The joined region's triangles (ids in the new mesh).
    triangles: Vec<u32>,
    /// The tube's centreline, and how far from it the wall can have changed.
    centre: Vec<DVec3>,
    reach: f64,
    /// Narrowest lumen along the stent before and after (the 10th percentile of each ring's
    /// wall distances).
    narrowest: [f64; 2],
    /// The most the lumen widened.
    widened: f64,
}

/// Open the lumen to the stent: the new lumen is the old one joined with the stent's tube
/// (a mesh union). Along the stent the tube is round at the stent's radius with *stiffness*
/// 1; with less, part way from the wall towards it, ray by ray from the axis. Past each end it
/// eases over *end blend* to a ring just inside the wall. Where the wall is about as wide as
/// the tube or wider, the tube keeps just inside it, so the two never run along one another
/// (a union cannot tell which surface is outside there) and nothing changes. The tube follows
/// the stent's axis eased to bend no tighter than the stent's diameter.
///
/// The mesh's open ends are capped for the union and opened again after. `None`: the tube
/// lies inside the lumen everywhere (nothing to open). Errors where the union fails or the
/// tube reaches an open end. Cancelled through `p` between its steps.
fn open_with_tube(mesh: &mut Mesh, pl: &StentPlacement, params: &StentParams, p: Progress) -> EditResult<Option<TubeOpening>> {
    // (Checked between the steps over the whole mesh, a few tens of ms each on a big one.)
    p.check(0.0)?;
    let (r, length, st) = (pl.radius, pl.length, params.stiffness);
    // The tube eases into the wall over the end blend and ends there: nothing beyond changes.
    let blend = params.end_blend;
    let k = pl.s.len();
    let (lo, hi) = (pl.s.partition_point(|&x| x < -blend), pl.s.partition_point(|&x| x <= length + blend).min(k));
    if hi < lo + 3 {
        return Ok(None);
    }
    // Edge length: the wall's near the stent.
    let pad = DVec3::splat(2.0 * r);
    let (bmin, bmax) = pl.axis[lo..hi].iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, b), &p| (a.min(p), b.max(p)));
    let (bmin, bmax) = (bmin - pad, bmax + pad);
    let in_box = |m: &Mesh, t: u32| m.tri_positions(t).iter().any(|p| p.cmpge(bmin).all() && p.cmple(bmax).all());
    let boxed: Vec<u32> = live_triangles_where(mesh, |t| in_box(mesh, t));
    if boxed.is_empty() {
        return Ok(None);
    }
    let h = params.edge_length.unwrap_or_else(|| boxed.iter().flat_map(|&t| mesh.tri_edges(t)).map(|e| mesh.edge_length(e)).sum::<f64>() / (3 * boxed.len()) as f64).max(1e-3 * r);
    // The tube's centreline: the axis there, a ring about every edge length, eased.
    let total = pl.s[hi - 1] - pl.s[lo];
    let rings = ((total / h).ceil() as usize).max(2);
    let at = |x: f64| {
        let j = pl.s.partition_point(|&v| v <= x).clamp(1, k - 1) - 1;
        let f = ((x - pl.s[j]) / (pl.s[j + 1] - pl.s[j]).max(1e-300)).clamp(0.0, 1.0);
        pl.axis[j].lerp(pl.axis[j + 1], f)
    };
    let xs: Vec<f64> = (0..=rings).map(|i| pl.s[lo] + total * i as f64 / rings as f64).collect();
    let mut centre: Vec<DVec3> = xs.iter().map(|&x| at(x)).collect();
    limit_bend(&mut centre, MAX_BEND / r);
    // Only wall near the tube, within the stent and its end blend, can change: the union is
    // taken there only (elsewhere it could also change where the mesh already ran into
    // itself), and only that is remeshed. Past the end blend is where the carry stops too:
    // nearest the first or last of the vessel's slices.
    let reach = 1.25 * r + 2.0 * h;
    let off_tube = |p: DVec3| centre.windows(2).map(|w| point_segment_distance(p, w[0], w[1])).fold(f64::INFINITY, f64::min);
    let changes = |p: DVec3| off_tube(p) < reach && !past_end(pl, blend, p);
    // Each vertex's distance from the tube, once (by id; infinite outside the box).
    let off_all = |m: &Mesh, tris: &[u32]| {
        let mut ids: Vec<u32> = tris.iter().flat_map(|&t| m.triangle(t)).collect();
        ids.sort_unstable();
        ids.dedup();
        let d: Vec<f64> = ids.par_iter().map(|&v| off_tube(m.vertex(v))).collect();
        let mut off = vec![f64::INFINITY; m.max_vid()];
        for (&v, d) in ids.iter().zip(d) {
            off[v as usize] = d;
        }
        off
    };
    let near_tube = |m: &Mesh, off: &[f64], t: u32| m.triangle(t).iter().any(|&v| off.get(v as usize).is_some_and(|&d| d < reach));
    let off = off_all(mesh, &boxed);
    p.check(0.01)?;
    // Zero-area and microscopic triangles near the stent fixed first (a union cannot place
    // them, and their coincident corners could not be told apart after it).
    let tiny = 1e-6 * r;
    for _ in 0..4 {
        let mut fixed = false;
        let short: Vec<(u32, u32)> = live_triangles_where(mesh, |t| in_box(mesh, t) && near_tube(mesh, &off, t))
            .into_iter()
            .flat_map(|t| {
                let tv = mesh.triangle(t);
                [(tv[0], tv[1]), (tv[1], tv[2]), (tv[2], tv[0])]
            })
            .filter(|&(a, b)| a < b && mesh.vertex(a).distance(mesh.vertex(b)) < tiny)
            .collect();
        for (a, b) in short {
            if mesh.is_vertex(a) && mesh.is_vertex(b) && mesh.find_edge(a, b).is_some() {
                let mid = (mesh.vertex(a) + mesh.vertex(b)) * 0.5;
                if mesh.collapse_edge(a, b).is_ok() {
                    mesh.set_vertex(a, mid);
                    fixed = true;
                }
            }
        }
        let degenerate: Vec<u32> = live_triangles_where(mesh, |t| in_box(mesh, t) && near_tube(mesh, &off, t) && mm_repair::ops::is_degenerate_triangle(mesh, t, 1e-12));
        for t in degenerate {
            fixed |= mm_repair::ops::fix_degenerate_triangle(mesh, t);
        }
        if !fixed {
            break;
        }
    }
    p.check(0.02)?;
    // (Again after the fixes, which may have added vertices.)
    let boxed: Vec<u32> = live_triangles_where(mesh, |t| in_box(mesh, t));
    let off = off_all(mesh, &boxed);
    let local: Vec<u32> = boxed.iter().copied().filter(|&t| near_tube(mesh, &off, t)).collect();
    if local.is_empty() {
        return Ok(None);
    }
    p.check(0.03)?;
    // Rotation-minimising frames along it, and the stent's rings there.
    let mut frames: Vec<[DVec3; 4]> = Vec::with_capacity(rings + 1);
    for (i, &c) in centre.iter().enumerate() {
        let t = (centre[(i + 1).min(rings)] - centre[i.saturating_sub(1)]).normalize_or(DVec3::Z);
        let u = match frames.last() {
            None => t.any_orthonormal_vector(),
            Some(&[x0, t0, u0, _]) => reflect_frame(x0, t0, u0, c, t),
        };
        frames.push([c, t, u, t.cross(u)]);
    }
    let lumen = stent_rings(mesh, &frames, r, st, h);
    let around = lumen[0].walls.len();
    // Wall this close to a ring counts as at it: half a percent of the radius, or twice the
    // sag of the wall's own facets (a faceted tube of the stent's size is not narrower).
    let margin = (0.005 * r).max(0.25 * h * h / r);
    let mut pos: Vec<DVec3> = Vec::with_capacity((rings + 1) * around + 2);
    let (mut narrowest_before, mut narrowest, mut widened, mut opens) = (f64::INFINITY, f64::INFINITY, 0.0_f64, false);
    for ((&[c, _, u, v], &x), ring) in frames.iter().zip(&xs).zip(&lumen) {
        // The ring's shape along the stent, easing past each end to a round ring just inside
        // the nearest wall.
        let inside = ring.native().min(r) - margin;
        let w = if x < 0.0 { 1.0 - smoother(-x / blend) } else if x > length { 1.0 - smoother((x - length) / blend) } else { 1.0 };
        // In each direction: where the ring opens the wall by more than the margin, the ring;
        // elsewhere just inside the wall, so the two surfaces never run along one another (a
        // union cannot join them) and nothing changes there.
        let here: Vec<f64> = ring
            .walls
            .iter()
            .zip(&ring.radii)
            .map(|(&wall, &target)| {
                let want = inside + (target - inside) * w;
                (if want > wall + margin { want } else { want.min(wall - margin) }).max(0.05 * r)
            })
            .collect();
        for (&wall, &rad) in ring.walls.iter().zip(&here) {
            if rad > wall {
                opens = true;
                widened = widened.max(rad - wall);
            }
        }
        if (0.0..=length).contains(&x) {
            // The 10th percentile of the wall's distance from the axis around the ring, before
            // and after (as the straightened vessel's narrowest is measured).
            let mut walls = ring.walls.clone();
            let mut after: Vec<f64> = ring.walls.iter().zip(&here).map(|(&wall, &rad)| wall.max(rad)).collect();
            if let (Some(b), Some(a)) = (percentile(&mut walls, 0.1), percentile(&mut after, 0.1)) {
                narrowest_before = narrowest_before.min(b);
                narrowest = narrowest.min(a);
            }
        }
        for (j, &rad) in here.iter().enumerate() {
            let a = ring_angle(j, here.len());
            pos.push(c + (u * a.cos() + v * a.sin()) * rad);
        }
    }
    if !opens {
        return Ok(None);
    }
    // The tube, closed by a fan at each end (both inside the lumen).
    let a = around as u32;
    let mut tris: Vec<[u32; 3]> = Vec::with_capacity(2 * rings * around + 2 * around);
    for i in 0..rings as u32 {
        let (p, q) = (i * a, (i + 1) * a);
        for j in 0..a {
            let jn = (j + 1) % a;
            tris.push([p + j, p + jn, q + jn]);
            tris.push([p + j, q + jn, q + j]);
        }
    }
    let (c0, c1) = (pos.len() as u32, pos.len() as u32 + 1);
    pos.push(centre[0]);
    pos.push(centre[rings]);
    let last = rings as u32 * a;
    for j in 0..a {
        let jn = (j + 1) % a;
        tris.push([c0, jn, j]);
        tris.push([c1, last + j, last + jn]);
    }
    let (tube, _) = Mesh::from_triangles(&pos, &tris, None);
    p.check(0.04)?;
    // The vessel closed at its open ends for the union.
    let mut closed = mesh.clone();
    let loops: Vec<Vec<u32>> = mm_core::topology::boundary_loops(&closed).into_iter().map(|l| l.vertices).collect();
    let caps = crate::cap::cap_planar_loops(&mut closed, &loops, None, CutFill::Flat)?;
    if !caps.failed_loops.is_empty() || !closed.is_closed() {
        return Err(EditError::Failed("could not close the vessel's open ends for the union".into()));
    }
    let cap_groups: FastSet<i32> = caps.groups.iter().copied().collect();
    p.check(0.05)?;
    let joined = crate::boolean::mesh_boolean(&closed, &tube, crate::boolean::BooleanOp::Union)?;
    drop(closed);
    p.check(0.45)?;
    let mut out = joined.mesh;
    let tube_group = joined.group_offset;
    // The tube must not reach a cap (an open end), which is opened again now.
    let on_cap = joined.seam_vertices.iter().any(|&v| out.vertex_triangles(v).iter().any(|&t| cap_groups.contains(&out.tri_group(t))));
    if on_cap {
        return Err(EditError::Failed("the stent's tube reaches an open end of the mesh".into()));
    }
    let caps_out: Vec<u32> = live_triangles_where(&out, |t| cap_groups.contains(&out.tri_group(t)));
    out.remove_triangles(caps_out);
    // The tube's triangles take the group of the wall they join (on a tie, the lowest).
    let from_tube = |m: &Mesh, t: u32| m.tri_group(t) >= tube_group && !cap_groups.contains(&m.tri_group(t));
    let mut votes: FastMap<i32, usize> = FastMap::default();
    for &v in &joined.seam_vertices {
        for t in out.vertex_triangles(v) {
            if !from_tube(&out, t) {
                *votes.entry(out.tri_group(t)).or_default() += 1;
            }
        }
    }
    let group = votes.into_iter().max_by_key(|&(g, n)| (n, Reverse(g))).map_or(0, |x| x.0);
    // Back onto the straightened mesh, keeping the ids of everything the union left as it
    // was (it copies untouched triangles exactly): its triangles the union replaced go, and
    // the union's new ones come in. Only near the tube can anything have changed.
    let inside_box = |p: DVec3| p.cmpge(bmin).all() && p.cmple(bmax).all();
    let key = |p: DVec3| [p.x.to_bits(), p.y.to_bits(), p.z.to_bits()];
    let mut old_ids: FastMap<[u64; 3], u32> = FastMap::default();
    for v in local.iter().flat_map(|&t| mesh.triangle(t)) {
        if old_ids.insert(key(mesh.vertex(v)), v).is_some_and(|o| o != v) {
            return Err(EditError::Failed("the wall near the stent has vertices on top of one another".into()));
        }
    }
    p.check(0.47)?;
    let mut kept: FastSet<u32> = FastSet::default();
    let mut fresh: Vec<([u32; 3], i32)> = Vec::new();
    let out_boxed: Vec<u32> = live_triangles_where(&out, |t| out.tri_positions(t).iter().any(|&p| inside_box(p)));
    let out_off = off_all(&out, &out_boxed);
    for &t in &out_boxed {
        let tv = out.triangle(t);
        if !tv.iter().any(|&v| inside_box(out.vertex(v)) && out_off[v as usize] < reach) {
            continue;
        }
        let old: Vec<u32> = tv.iter().filter_map(|&v| old_ids.get(&key(out.vertex(v))).copied()).collect();
        let same = (old.len() == 3).then(|| mesh.vertex_triangles(old[0]).into_iter().find(|&o| {
            let ov = mesh.triangle(o);
            old.iter().all(|v| ov.contains(v))
        })).flatten();
        match same {
            Some(o) => {
                kept.insert(o);
            }
            None => fresh.push((tv, if from_tube(&out, t) { group } else { out.tri_group(t) })),
        }
    }
    let gone: Vec<u32> = local.iter().copied().filter(|t| !kept.contains(t)).collect();
    mesh.remove_triangles(gone);
    // Each corner: the old vertex there if it is still in use, else a new one; all decided
    // before any is added, since added vertices can take the ids of removed ones. New ones are
    // added in the order the corners first come up, so the ids do not depend on a map's order.
    let mut seen: FastSet<u32> = FastSet::default();
    let mut corner: Vec<(u32, Option<u32>)> = Vec::new();
    for (tv, _) in &fresh {
        for &v in tv {
            if seen.insert(v) {
                corner.push((v, old_ids.get(&key(out.vertex(v))).copied().filter(|&o| mesh.is_vertex(o))));
            }
        }
    }
    let ids: FastMap<u32, u32> = corner.into_iter().map(|(v, old)| (v, old.unwrap_or_else(|| mesh.append_vertex(out.vertex(v))))).collect();
    let mut added = Vec::with_capacity(fresh.len());
    for (tv, g) in fresh {
        added.push(mesh.append_triangle(tv.map(|v| ids[&v]), g).map_err(|e| EditError::Failed(format!("joining the stent's tube: {e:?}")))?);
    }
    mesh.remove_isolated_vertices();
    // The joined region: the new triangles and two rings around them.
    let mut region: FastSet<u32> = added.iter().copied().collect();
    let mut may: FastMap<u32, bool> = FastMap::default();
    for _ in 0..2 {
        let mut more: Vec<u32> = region.iter().flat_map(|&t| mesh.triangle(t)).flat_map(|v| mesh.vertex_triangles(v)).filter(|t| !region.contains(t)).collect();
        more.sort_unstable();
        more.dedup();
        let mut verts: Vec<u32> = more.iter().flat_map(|&t| mesh.triangle(t)).filter(|v| !may.contains_key(v)).collect();
        verts.sort_unstable();
        verts.dedup();
        let ok: Vec<bool> = verts.par_iter().map(|&v| changes(mesh.vertex(v))).collect();
        may.extend(verts.into_iter().zip(ok));
        region.extend(more.into_iter().filter(|&t| mesh.triangle(t).iter().all(|v| may[v])));
    }
    let mut region: Vec<u32> = region.into_iter().collect();
    region.sort_unstable();
    // A join that runs into the wall (where it lies close, e.g. at a branch's mouth) is not
    // used: the wall is pushed out instead.
    p.check(0.5)?;
    if runs_into_wall(mesh, &region) {
        return Err(EditError::Failed("the stent's tube, joined to the wall, ran into it".into()));
    }
    if params.remesh {
        p.check(0.55)?;
        let rp = mm_remesh::RemeshParams { target_edge_length: h, iterations: 8, region: Some(region.clone()), ..mm_remesh::RemeshParams::with_target(h) };
        // A remesh that makes the wall run into itself is undone.
        let unremeshed = mesh.clone();
        let remeshed = remesh_within(mesh, &rp, p.part(0.55, 0.9)).ok_or_else(cancelled)?;
        if let Ok(Some(tris)) = remeshed.map(|x| x.region_triangles) {
            p.check(0.92)?;
            if runs_into_wall(mesh, &tris) {
                *mesh = unremeshed;
            } else {
                region = tris;
            }
        }
    }
    mesh.bump_shape_version();
    Ok(Some(TubeOpening { triangles: region, centre, reach, narrowest: [narrowest_before, narrowest], widened }))
}

/// Straighten the vessel with the stent and (`push_wall`) push the wall out to it; covered,
/// replace the stented stretch with the stent's tube, `rings` giving its shape. Cancelled
/// through `p` between its steps.
#[allow(clippy::too_many_arguments)]
fn reshape(mesh: &mut Mesh, pl: &StentPlacement, params: &StentParams, owner: Option<&[bool]>, push_wall: bool, rings: Option<&Profile>, p: Progress) -> EditResult<StentReport> {
    let k = pl.s.len();
    if k < 3 {
        return Err(EditError::InvalidParameter("the stent placement has too few samples".into()));
    }
    p.check(0.0)?;
    let (r, length) = (pl.radius, pl.length);
    let ds = pl.s[1] - pl.s[0];
    let (tp, tc) = (tangents(&pl.axis), tangents(&pl.vessel));
    let beyond = |s: f64| (-s).max(s - length).max(0.0);
    let fade = |x: f64, w: f64| if w <= 0.0 { if x > 0.0 { 0.0 } else { 1.0 } } else { 1.0 - smoother(x / w) };
    let radial: Vec<f64> = pl.s.iter().map(|&s| fade(beyond(s), params.end_blend)).collect();
    let first = pl.s.partition_point(|&s| s < 0.0).min(k - 1);
    let last = pl.s.partition_point(|&s| s <= length).saturating_sub(1).max(first);

    // The vertices of the stented vessel near the samples: where along them they lie (a
    // fractional sample index) and their offset from the centerline.
    let wide = pl.vessel_radius.iter().copied().fold(r, f64::max);
    let pad = DVec3::splat(3.0 * wide);
    let lo = pl.vessel.iter().fold(DVec3::INFINITY, |a, &b| a.min(b)) - pad;
    let hi = pl.vessel.iter().fold(DVec3::NEG_INFINITY, |a, &b| a.max(b)) + pad;
    // The nearest point of the vessel's course (between samples `from` and `to`): where along
    // it (a fractional sample index) and the point.
    let project_in = |p: DVec3, from: f64, to: f64| -> (f64, DVec3) {
        let mut best = (f64::INFINITY, 0.0, pl.vessel[0]);
        for i in (from.floor().max(0.0) as usize)..(to.ceil().max(0.0) as usize).min(k - 1) {
            let (a, ab) = (pl.vessel[i], pl.vessel[i + 1] - pl.vessel[i]);
            let f = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
            let c = a + ab * f;
            let d = p.distance_squared(c);
            if d < best.0 {
                best = (d, i as f64 + f, c);
            }
        }
        (best.1, best.2)
    };
    let project = |p: DVec3| project_in(p, 0.0, (k - 1) as f64);
    // A slice carries wall at most this far from the centerline: wall further off lies near
    // the samples without belonging to them (the vessel passing by again, another vessel).
    let bound = |x: f64| 3.0 * lerp_f(&pl.vessel_radius, x).max(r);
    let near: Vec<u32> = (0..mesh.max_vid() as u32).into_par_iter().filter(|&v| mesh.is_vertex(v) && !(mesh.vertex(v).cmplt(lo).any() || mesh.vertex(v).cmpgt(hi).any())).collect();
    let candidates: Vec<(u32, f64, DVec3)> = near
        .par_iter()
        .filter_map(|&v| {
            let p = mesh.vertex(v);
            let (x, c) = project(p);
            if x <= 0.0 || x >= (k - 1) as f64 || p.distance(c) >= bound(x) {
                return None;
            }
            let mine = match owner {
                Some(o) => o.get(v as usize).copied().unwrap_or(false),
                None => p.distance(c) < 2.5 * lerp_f(&pl.vessel_radius, x).max(r),
            };
            mine.then_some((v, x, p - c))
        })
        .collect();
    // The stented vessel's own wall: the candidates joined, through candidates, to wall close
    // around the stent's stretch of centerline.
    let index: FastMap<u32, usize> = candidates.iter().enumerate().map(|(i, c)| (c.0, i)).collect();
    let mut reached = vec![false; candidates.len()];
    let mut stack: Vec<usize> = (0..candidates.len())
        .filter(|&i| {
            let (_, x, o) = candidates[i];
            (first as f64..=last as f64).contains(&x) && o.length() < 1.5 * lerp_f(&pl.vessel_radius, x).max(r)
        })
        .collect();
    for &i in &stack {
        reached[i] = true;
    }
    while let Some(i) = stack.pop() {
        for w in mesh.vertex_neighbors(candidates[i].0) {
            if let Some(&j) = index.get(&w) {
                if !reached[j] {
                    reached[j] = true;
                    stack.push(j);
                }
            }
        }
    }
    let core: Vec<(u32, f64, DVec3)> = candidates.into_iter().zip(&reached).filter(|(_, &joined)| joined).map(|(c, _)| c).collect();
    let mut is_core = vec![false; mesh.max_vid()];
    for &(v, _, _) in &core {
        {
            is_core[v as usize] = true;
        }
    }
    if core.is_empty() {
        return Err(EditError::Failed("no vessel wall around the stent".into()));
    }
    // Side branches and anything else joined to the vessel near the stent follow it less
    // with distance along the surface.
    let reach = 1.5 * wide;
    let mut dist: FastMap<u32, f64> = core.iter().map(|&(v, _, _)| (v, 0.0)).collect();
    let mut heap: BinaryHeap<Reverse<(u64, u32)>> = core.iter().map(|&(v, _, _)| Reverse((0u64, v))).collect();
    while let Some(Reverse((bits, u))) = heap.pop() {
        let d = f64::from_bits(bits);
        if d > dist[&u] {
            continue;
        }
        let pu = mesh.vertex(u);
        for w in mesh.vertex_neighbors(u) {
            let nd = d + pu.distance(mesh.vertex(w));
            if !is_core[w as usize] && nd < reach && dist.get(&w).is_none_or(|&o| nd < o) {
                dist.insert(w, nd);
                heap.push(Reverse((nd.to_bits(), w)));
            }
        }
    }
    let mut verts: Vec<(u32, f64, DVec3, f64)> = core.iter().map(|&(v, x, o)| (v, x, o, 1.0)).collect();
    // (In vertex order, not the map's.)
    let mut nearby: Vec<(u32, f64)> = dist.into_iter().collect();
    nearby.sort_unstable_by_key(|x| x.0);
    for (v, d) in nearby {
        if !is_core[v as usize] {
            let p = mesh.vertex(v);
            let (x, c) = project(p);
            // (Fading out before the bound, beyond which no slice carries anything.)
            let off = 1.0 - smoother(3.0 * (p.distance(c) / bound(x)) - 2.0);
            if x > 0.0 && x < (k - 1) as f64 && off > 0.0 {
                verts.push((v, x, p - c, (1.0 - smoother(d / reach)) * off));
            }
        }
    }
    // Nothing past the end blend moves: such wall is neither carried nor pushed, nor evened
    // out with its neighbours below.
    verts.retain(|&(_, x, _, _)| lerp_f(&radial, x) > 0.0);


    // Grip: the stent holds the vessel where the wall is no wider than the stent.
    let mut bins: Vec<Vec<f64>> = vec![Vec::new(); k];
    for &(_, x, o) in &core {
        let t = lerp_at(&tc, x).normalize_or_zero();
        bins[(x.round() as usize).min(k - 1)].push((o - t * o.dot(t)).length());
    }
    let wall: Vec<f64> = (0..k).map(|i| percentile(&mut bins[i], 0.3).unwrap_or(pl.vessel_radius[i])).collect();
    let narrow_before = (first..=last).filter_map(|i| percentile(&mut bins[i], 0.1)).fold(f64::INFINITY, f64::min);
    let mut grip: Vec<f64> = wall.iter().map(|&w| 1.0 - smoothstep((w / r - 1.0) / 0.6)).collect();
    for i in 0..first {
        grip[i] = grip[first];
    }
    for i in last + 1..k {
        grip[i] = grip[last];
    }
    let half = ((r / ds).round() as usize).max(1);
    for _ in 0..2 {
        grip = box_blur(&grip, half);
    }
    // (Past the ends the axis itself returns onto the centerline.)
    let carry = grip;
    // Each slice turns to face along the path its centre is carried onto (averaged over half
    // the stent's radius), so the wall is not sheared where the carry fades.
    let carried: Vec<DVec3> = (0..k).map(|i| pl.vessel[i].lerp(pl.axis[i], carry[i])).collect();
    let along = tangents(&carried);
    let half = ((0.5 * r / ds).round() as usize).max(1);
    let rot: Vec<DQuat> = (0..k)
        .map(|i| {
            let window = i.saturating_sub(half)..=(i + half).min(k - 1);
            let t = along[window.clone()].iter().sum::<DVec3>().normalize_or(along[i]);
            let c = tc[window].iter().sum::<DVec3>().normalize_or(tc[i]);
            DQuat::from_rotation_arc(c, t)
        })
        .collect();
    let outside = (first..=last).any(|i| (1.0 - carry[i]) * pl.axis[i].distance(pl.vessel[i]) > 0.8 * wall[i]);

    // Region and edge length before reshaping.
    // Pushed out to the stent's radius, with a soft corner (2 % of the radius) where the
    // push starts, so the wall has no crease at the edge of contact. Wall inside the radius
    // keeps its order away from the axis within a thin band just outside it (half a percent),
    // rather than all landing on the radius: where the wall doubles back in a slice (a shelf
    // where a narrowing meets a wider lumen), its layers would otherwise land on one another.
    let soft = 0.02 * r;
    let band = 0.25 * soft;
    let open = |rho: f64| {
        let d = rho - r;
        let base = if d >= soft {
            rho
        } else if d <= -soft {
            r
        } else {
            r + (d + soft) * (d + soft) / (4.0 * soft)
        };
        // Rising with rho to the radius, then easing out by the end of the soft corner (its
        // slope there stays under the corner's, so the whole stays increasing).
        let order = if d <= 0.0 { band * (rho / r).max(0.0) } else { band * (1.0 - smoother((d / soft).min(1.0))) };
        base + order
    };
    // Each vertex: where the carry takes it, the push from there, and its distance from the
    // stent's axis after (negative off the vessel's own wall).
    let mapped: Vec<(u32, DVec3, DVec3, f64)> = verts
        .par_iter()
        .map(|&(v, x, o, wv)| {
            let (c, p, t) = (lerp_at(&pl.vessel, x), lerp_at(&pl.axis, x), lerp_at(&tp, x).normalize_or_zero());
            let tw = lerp_f(&carry, x) * wv;
            let i = (x.floor().max(0.0) as usize).min(k - 2);
            let q = DQuat::IDENTITY.slerp(rot[i].slerp(rot[i + 1], x - i as f64), wv);
            // Past the stent's end the carry, its turn and the push fade out over the end blend
            // (the push about the vessel's own centre, not an imaginary continuation of the
            // stent), so nothing beyond it moves.
            let fade = lerp_f(&radial, x);
            let carried = c + (p - c) * (tw * fade);
            let xt = carried + DQuat::IDENTITY.slerp(q, fade) * o;
            let rel = xt - carried.lerp(p, fade);
            let perp = rel - t * rel.dot(t);
            let rho = perp.length();
            let push = if push_wall { params.stiffness * fade * wv } else { 0.0 };
            let (xf, rho2) = if rho > 1e-12 * r && push > 0.0 {
                let rho2 = rho + (open(rho) - rho) * push;
                (xt + perp * ((rho2 - rho) / rho), rho2)
            } else {
                (xt, rho)
            };
            (v, xt, xf - xt, if wv >= 1.0 { rho2 } else { -1.0 })
        })
        .collect();
    let mut after: Vec<Vec<f64>> = vec![Vec::new(); k];
    for (&(_, x, _, _), &(_, _, _, rho2)) in verts.iter().zip(&mapped) {
        if rho2 >= 0.0 {
            after[(x.round() as usize).min(k - 1)].push(rho2);
        }
    }
    // The carry, evened out where it would tear the wall, then the push.
    let mut moves: Vec<(u32, DVec3)> = mapped.iter().map(|&(v, p, _, _)| (v, p)).collect();
    relax(mesh, &mut moves);
    for (m, &(_, _, push, _)) in moves.iter_mut().zip(&mapped) {
        m.1 += push;
    }
    let narrow_after = (first..=last).filter_map(|i| percentile(&mut after[i], 0.1)).fold(f64::INFINITY, f64::min);
    p.check(0.3)?;
    // Moves under 1 % of an edge or 0.6 % of the radius (the soft corner's nudge to wall
    // touching the stent) neither need a remesh nor can make new crossings.
    let tiny = (0.01 * mesh.average_edge_length()).max(0.006 * r);
    let mut moved_verts: FastSet<u32> = FastSet::default();
    let mut max_shift: f64 = 0.0;
    for &(v, p) in &moves {
        let d = p.distance(mesh.vertex(v));
        if d > tiny {
            moved_verts.insert(v);
            max_shift = max_shift.max(d);
        }
    }
    let mut region: Vec<u32> = moved_verts.iter().flat_map(|&v| mesh.vertex_triangles(v)).collect();
    region.sort_unstable();
    region.dedup();
    let h = params.edge_length.unwrap_or_else(|| {
        let edges: Vec<f64> = region.iter().flat_map(|&t| mesh.tri_edges(t)).map(|e| mesh.edge_length(e)).collect();
        if edges.is_empty() { mesh.average_edge_length() } else { edges.iter().sum::<f64>() / edges.len() as f64 }
    });
    let before: Vec<[f64; 3]> = region.iter().map(|&t| mesh.tri_edges(t).map(|e| mesh.edge_length(e))).collect();
    let moved_openings = mm_core::topology::boundary_loops(mesh).iter().filter(|l| l.vertices.iter().any(|v| moved_verts.contains(v))).count();
    p.check(0.33)?;
    // (Written all at once, with one new shape version.)
    let pos = mesh.positions_mut();
    for &(v, x) in &moves {
        pos[v as usize] = x;
    }
    mesh.bump_shape_version();
    let mut report = StentReport { moved: moved_verts.len(), max_shift, narrowest: [narrow_before, narrow_after], outside, moved_openings, source_triangles: region.clone(), ..Default::default() };
    // Remeshed: where the wall was stretched or squeezed (not where slices were only carried
    // along), and two rings of triangles around that.
    let mut distorted: FastSet<u32> = region
        .iter()
        .zip(&before)
        .filter(|(t, old)| mesh.tri_edges(**t).iter().zip(old.iter()).any(|(&e, &l)| (mesh.edge_length(e) / l.max(1e-300) - 1.0).abs() > 0.1))
        .map(|(&t, _)| t)
        .collect();
    for _ in 0..2 {
        let ring: Vec<u32> = distorted.iter().flat_map(|&t| mesh.triangle(t)).flat_map(|v| mesh.vertex_triangles(v)).collect();
        distorted.extend(ring);
    }
    // Only wall within the stent and its end blend is remeshed (a region's border stays put).
    let within: FastSet<u32> = moves.iter().map(|m| m.0).collect();
    distorted.retain(|&t| mesh.triangle(t).iter().all(|v| within.contains(v)));
    if params.covered {
        p.check(0.35)?;
        let (tube, sealed, removed, landing) = cover(mesh, pl, h, params.end_blend, rings.ok_or_else(|| EditError::Failed("a covered stent needs its rings".into()))?)?;
        region.retain(|&t| mesh.is_triangle(t));
        distorted.retain(|&t| mesh.is_triangle(t));
        distorted.extend(tube.iter().copied());
        // (Not past the end blend: the cut and its smoothing stop short of it.)
        let mut verts: Vec<u32> = distorted.iter().flat_map(|&t| mesh.triangle(t)).collect();
        verts.sort_unstable();
        verts.dedup();
        let past: FastSet<u32> = verts.par_iter().copied().filter(|&v| past_end(pl, params.end_blend, mesh.vertex(v))).collect();
        distorted.retain(|&t| !mesh.triangle(t).iter().any(|v| past.contains(v)));
        region.extend(tube);
        report.sealed_openings = sealed;
        report.landing = landing;
        report.removed_triangles = removed;
        report.narrowest[1] = report.narrowest[1].min(r);
    }
    region.sort_unstable();
    region.dedup();
    if params.remesh && !distorted.is_empty() {
        p.check(0.5)?;
        let mut part: Vec<u32> = distorted.iter().copied().collect();
        part.sort_unstable();
        let rp = mm_remesh::RemeshParams { target_edge_length: h, iterations: 8, region: Some(part.clone()), ..mm_remesh::RemeshParams::with_target(h) };
        // Kept to go back to: a remesh that makes the wall run into itself (where walls lie
        // close, e.g. a branch's mouth near the stent's end) is undone.
        let unremeshed = mesh.clone();
        let crossed_before = runs_into_wall(mesh, &part);
        p.check(0.55)?;
        let remeshed = remesh_within(mesh, &rp, p.part(0.55, 0.9)).ok_or_else(cancelled)?;
        if let Ok(Some(tris)) = remeshed.map(|x| x.region_triangles) {
            p.check(0.92)?;
            if !crossed_before && runs_into_wall(mesh, &tris) {
                *mesh = unremeshed;
                report.remesh_undone = true;
            } else {
                region.retain(|t| !distorted.contains(t) && mesh.is_triangle(*t));
                region.extend(tris);
            }
        }
    }
    mesh.bump_shape_version();
    report.triangles = region;
    Ok(report)
}

/// Whether any of `tris` crosses another triangle of `m` (one it shares no corner with).
///
/// Only triangles whose boxes come within a hair of the box around `tris` can, so the tree is
/// built over those alone rather than the whole mesh (which cost most of the check on a big
/// mesh). The hair is far wider than the tolerance within which two triangles count as
/// touching, and a pair is tested only where their boxes come within it of one another, so
/// the answer does not depend on how the tree groups the triangles.
fn runs_into_wall(m: &Mesh, tris: &[u32]) -> bool {
    let live: Vec<u32> = tris.iter().copied().filter(|&t| m.is_triangle(t)).collect();
    if live.is_empty() {
        return false;
    }
    let bounds = live.par_iter().map(|&t| Aabb::from_points(m.tri_positions(t))).reduce(|| Aabb::EMPTY, |a, b| a.union(&b));
    let size = bounds.min.abs().max(bounds.max.abs()).max_element().max(1.0);
    let hair = 1e-9 * bounds.diagonal() + 1e-6 * size;
    let zone = bounds.expanded(hair);
    let near = live_triangles_where(m, |t| Aabb::from_points(m.tri_positions(t)).intersects(&zone));
    let bvh = Bvh::build_subset(m, &near);
    live.par_iter().any(|&t| {
        let (tv, tp) = (m.triangle(t), m.tri_positions(t));
        let reach = Aabb::from_points(tp).expanded(hair);
        let mut hit = false;
        bvh.query_aabb(&reach, |o| {
            if !hit && o != t {
                let op = m.tri_positions(o);
                if Aabb::from_points(op).intersects(&reach) && !m.triangle(o).iter().any(|v| tv.contains(v)) && mm_core::intersect::triangles_intersect(tp, op) {
                    hit = true;
                }
            }
        });
        hit
    })
}

/// Covered stent: cut the stented stretch out at the stent's ends and put the stent's tube in
/// its place, shaped as `rings` gives (see [`stent_rings`]), easing into each cut over a short
/// flare. Also returns each cut's width over the stent's ([`StentReport::landing`]). Returns the tube's triangles with the bands blended into
/// it, the number of open ends removed and the number of triangles removed.
fn cover(mesh: &mut Mesh, pl: &StentPlacement, h: f64, end_blend: f64, rings: &Profile) -> EditResult<(Vec<u32>, usize, usize, [f64; 2])> {
    let (r, l) = (pl.radius, pl.length);
    let around = ((TAU * r / h).round() as usize).clamp(12, 4000);
    let rows = ((l / h).round() as usize).max(3);
    let frames = pl.frames(rows);
    let wide = 3.0 * pl.vessel_radius.iter().copied().fold(r, f64::max);
    let outside = || EditError::Failed("each end of a covered stent must lie inside the vessel (this one runs past an open end): move the ends in or shorten the stent".into());
    let cut = |mesh: &mut Mesh, x: DVec3, n: DVec3| -> EditResult<LocalCutResult> {
        let res = local_plane_cut_with(mesh, Plane::from_point_normal(x, n), x, &PlaneCutParams { keep: KeepSide::Both, fill: CutFill::None, ..Default::default() })
            .map_err(|e| match e {
                EditError::CurveDoesNotSeparate => EditError::Failed("the covered stretch is joined to the rest of the vessel another way round, so it cannot be sealed off".into()),
                EditError::NoIntersection => outside(),
                e => e,
            })?;
        if !res.curve_closed || res.curve_distance > wide {
            return Err(outside());
        }
        Ok(res)
    };
    let [x0, t0, ..] = frames[0];
    let [x1, t1, ..] = frames[rows];
    let ca = cut(mesh, x0, -t0)?;
    let cb = cut(mesh, x1, t1)?;
    let pick = |res: &LocalCutResult, side: Side| res.cut.cut_loops.iter().zip(&res.cut.loop_sides).find(|(_, s)| **s == side).map(|(l, _)| l.clone());
    let (Some(outer_a), Some(inner_a), Some(outer_b), Some(inner_b)) = (pick(&ca, Side::Above), pick(&ca, Side::Below), pick(&cb, Side::Above), pick(&cb, Side::Below)) else {
        return Err(EditError::Failed("the vessel was not cut cleanly at the stent's ends".into()));
    };
    let is_loop = |lp: &[u32]| lp.len() >= 3 && (0..lp.len()).all(|i| mesh.find_edge(lp[i], lp[(i + 1) % lp.len()]).is_some_and(|e| mesh.is_boundary_edge(e)));
    if ![&outer_a, &inner_a, &outer_b, &inner_b].iter().all(|lp| is_loop(lp)) {
        return Err(EditError::Failed("the covered stent is too short for the mesh: make it longer or refine the mesh".into()));
    }
    // The excluded stretch: everything joined to the inside of the cuts.
    let seeds: Vec<u32> = inner_a.iter().flat_map(|&v| mesh.vertex_triangles(v)).collect();
    let middle: FastSet<u32> = mm_core::topology::flood_fill_triangles(mesh, &seeds, |_| true).into_iter().collect();
    let touches = |lp: &[u32]| lp.iter().any(|&v| mesh.vertex_triangles(v).iter().any(|t| middle.contains(t)));
    if touches(&outer_a) || touches(&outer_b) || !touches(&inner_b) {
        return Err(EditError::Failed("the covered stretch is joined to the rest of the vessel another way round, so it cannot be sealed off".into()));
    }
    let sealed = mm_core::topology::boundary_loops(mesh)
        .iter()
        .filter(|lp| !lp.vertices.contains(&inner_a[0]) && !lp.vertices.contains(&inner_b[0]) && lp.vertices.iter().all(|&v| mesh.vertex_triangles(v).iter().all(|t| middle.contains(t))))
        .count();
    let removed = middle.len();
    let group = mesh.vertex_triangles(outer_a[0]).first().map_or(0, |&t| mesh.tri_group(t));
    // Removed in id order: the ids they free are taken again by the tube, so the order of a
    // set's walk would show in the result.
    let mut middle: Vec<u32> = middle.into_iter().collect();
    middle.sort_unstable();
    mesh.remove_triangles(middle);
    mesh.remove_isolated_vertices();

    // The stent's tube between the cuts, its rows going round the way the first cut's
    // strip runs.
    let AlignedLoops { p, q, .. } = aligned_loops(mesh, &outer_a, &outer_b, Some([t0, -t1]))?;
    let angle = |x: DVec3, f: &[DVec3; 4]| {
        let d = x - f[0];
        d.dot(f[3]).atan2(d.dot(f[2]))
    };
    let turn: f64 = (0..p.len())
        .map(|i| {
            let d = angle(mesh.vertex(p[(i + 1) % p.len()]), &frames[0]) - angle(mesh.vertex(p[i]), &frames[0]);
            (d + std::f64::consts::PI).rem_euclid(TAU) - std::f64::consts::PI
        })
        .sum();
    let sense = if turn < 0.0 { -1.0 } else { 1.0 };
    let start = angle(mesh.vertex(p[0]), &frames[0]);
    let profile = |lp: &[u32], f: &[DVec3; 4]| -> Vec<(f64, f64)> {
        lp.iter()
            .map(|&v| {
                let d = mesh.vertex(v) - f[0];
                (d.dot(f[2]), d.dot(f[3]))
            })
            .collect()
    };
    let (prof_a, prof_b) = (profile(&p, &frames[0]), profile(&q, &frames[rows]));
    let flare = (0.5 * r).max(2.0 * h).min(l / 3.0);
    // How much wider than the stent each cut is (its equivalent radius over the stent's): a
    // graft landing in a much wider lumen cannot be flared to its wall without folding.
    let wide = [&p, &q].map(|lp| {
        let pts: Vec<DVec3> = lp.iter().map(|&v| mesh.vertex(v)).collect();
        let c = pts.iter().sum::<DVec3>() / pts.len().max(1) as f64;
        let area = 0.5 * (0..pts.len()).map(|i| (pts[i] - c).cross(pts[(i + 1) % pts.len()] - c).length()).sum::<f64>();
        (area / std::f64::consts::PI).sqrt() / r
    });
    let mut tube_rows: Vec<Vec<u32>> = Vec::with_capacity(rows - 1);
    let mut new_vertices = Vec::new();
    for (j, &[x, _, u, v]) in frames.iter().enumerate().take(rows).skip(1) {
        let s = l * j as f64 / rows as f64;
        let (fa, fb) = (smoother(s / flare), smoother((l - s) / flare));
        let row: Vec<u32> = (0..around)
            .map(|i| {
                let a = start + sense * TAU * i as f64 / around as f64;
                let mut rr = rings.radius(s, u * a.cos() + v * a.sin());
                if fa < 1.0 {
                    rr = polar(&prof_a, a).unwrap_or(rr) * (1.0 - fa) + rr * fa;
                }
                if fb < 1.0 {
                    rr = polar(&prof_b, a).unwrap_or(rr) * (1.0 - fb) + rr * fb;
                }
                mesh.append_vertex(x + (u * a.cos() + v * a.sin()) * rr)
            })
            .collect();
        new_vertices.extend_from_slice(&row);
        tube_rows.push(row);
    }
    let close = |v: &[u32]| {
        let mut c = v.to_vec();
        c.push(v[0]);
        c
    };
    let unfolded = |t: [u32; 3]| {
        let pts = t.map(|v| mesh.vertex(v));
        let normal = (pts[1] - pts[0]).cross(pts[2] - pts[0]);
        let surface: DVec3 = t.iter().filter(|&&v| !mesh.vertex_triangles(v).is_empty()).map(|&v| mesh.vertex_normal(v)).sum();
        if normal.dot(surface) < 0.0 { 1e3 * h } else { 0.0 }
    };
    let last_row = &tube_rows[tube_rows.len() - 1];
    let b0 = mesh.vertex(last_row[0]);
    let qs = (0..q.len()).min_by(|&x, &y| mesh.vertex(q[x]).distance_squared(b0).total_cmp(&mesh.vertex(q[y]).distance_squared(b0))).unwrap_or(0);
    let q: Vec<u32> = q[qs..].iter().chain(&q[..qs]).copied().collect();
    let first = strip_with(mesh, &close(&p), &close(&tube_rows[0]), &unfolded);
    let end = strip_with(mesh, &close(last_row), &close(&q), &unfolded);
    let (Some(first), Some(end)) = (first, end) else {
        mesh.remove_isolated_vertices();
        return Err(EditError::Failed("could not join the stent's tube to the vessel".into()));
    };
    let mut tris = first;
    for w in tube_rows.windows(2) {
        tris.extend(quad_band(mesh, &close(&w[0]), &close(&w[1])));
    }
    tris.extend(end);
    let tube = commit(mesh, tris, new_vertices, group)?;
    // The vessel behind each cut is smoothed over a quarter of its diameter, no further than
    // the end blend.
    let across = |lp: &[u32]| {
        let c = lp.iter().map(|&v| mesh.vertex(v)).sum::<DVec3>() / lp.len().max(1) as f64;
        2.0 * lp.iter().map(|&v| mesh.vertex(v).distance(c)).sum::<f64>() / lp.len().max(1) as f64
    };
    let band = (end_blend / across(&p).max(across(&q)).max(1e-9)).min(0.25);
    let mut out = blend_join(mesh, &tube.triangles, [&p, &q], band, &|x| past_end(pl, end_blend, x));
    out.extend(tube.triangles);
    Ok((out, sealed, removed, wide))
}

/// Distance from the origin to the polygon `poly` (2D points, closed) in direction `a`.
fn polar(poly: &[(f64, f64)], a: f64) -> Option<f64> {
    let d = (a.cos(), a.sin());
    let mut best: Option<f64> = None;
    for i in 0..poly.len() {
        let (p, q) = (poly[i], poly[(i + 1) % poly.len()]);
        let e = (q.0 - p.0, q.1 - p.1);
        let det = -d.0 * e.1 + e.0 * d.1;
        if det.abs() < 1e-300 {
            continue;
        }
        let t = (-p.0 * e.1 + e.0 * p.1) / det;
        let s = (d.0 * p.1 - d.1 * p.0) / det;
        if t > 0.0 && (0.0..=1.0).contains(&s) && best.is_none_or(|b| t < b) {
            best = Some(t);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;
    use std::cell::RefCell;

    /// A straight vessel of radius 2 along z (0..60), narrowed to 1 at z = 30.
    fn stenosed() -> (Mesh, VesselPath) {
        let points: Vec<DVec3> = (0..=240).map(|k| DVec3::new(0.0, 0.0, 0.25 * k as f64)).collect();
        let radii: Vec<f64> = points.iter().map(|p| 2.0 - (-((p.z - 30.0) / 3.0).powi(2)).exp()).collect();
        (primitives::tube_along(&points, &radii, 48), VesselPath { points, radii })
    }

    fn sections(path: &VesselPath) -> [StentSection; 2] {
        [80, 160].map(|i| StentSection { center: path.points[i], normal: DVec3::Z })
    }

    /// Everything the result is: ids, positions, triangles and groups, and the report.
    fn same(a: &(Mesh, StentReport), b: &(Mesh, StentReport)) -> bool {
        let (ma, mb) = (&a.0, &b.0);
        ma.max_vid() == mb.max_vid()
            && ma.max_tid() == mb.max_tid()
            && ma.vertex_ids().eq(mb.vertex_ids())
            && ma.vertex_ids().all(|v| ma.vertex(v).to_array().map(f64::to_bits) == mb.vertex(v).to_array().map(f64::to_bits))
            && ma.triangle_ids().eq(mb.triangle_ids())
            && ma.triangle_ids().all(|t| ma.triangle(t) == mb.triangle(t) && ma.tri_group(t) == mb.tri_group(t))
            && format!("{:?}", a.1) == format!("{:?}", b.1)
    }

    fn run(m: &Mesh, path: &VesselPath, params: &StentParams, progress: Option<&dyn Fn(f32) -> bool>) -> EditResult<(Mesh, StentReport)> {
        let pl = place_stent(path, sections(path), params)?;
        let mut out = m.clone();
        let rep = apply_stent_with(&mut out, &pl, params, None, progress)?;
        Ok((out, rep))
    }

    #[test]
    fn progress_changes_nothing_and_runs_give_the_same_mesh() {
        // Bare and covered, stiffness from none to full, remeshed: a callback that never
        // cancels changes nothing, and the result is the same every time (no map's order shows).
        let (m, path) = stenosed();
        for covered in [false, true] {
            for stiffness in [0.0, 0.5, 1.0] {
                let params = StentParams { diameter: 4.0, length: 24.0, straightness: 0.0, stiffness, covered, remesh: true, ..Default::default() };
                let seen = RefCell::new(Vec::new());
                let report = |x: f32| {
                    seen.borrow_mut().push(x);
                    true
                };
                let plain = run(&m, &path, &params, None).unwrap();
                let watched = run(&m, &path, &params, Some(&report)).unwrap();
                let again = run(&m, &path, &params, None).unwrap();
                assert!(same(&plain, &watched), "covered {covered}, stiffness {stiffness}: the progress callback changed the result");
                assert!(same(&plain, &again), "covered {covered}, stiffness {stiffness}: two runs differ");
                let seen = seen.into_inner();
                assert!(!seen.is_empty() && seen.windows(2).all(|w| w[0] <= w[1]) && seen.iter().all(|x| (0.0..=1.0).contains(x)), "{seen:?}");
            }
        }
    }

    #[test]
    fn cancelling_stops_at_the_next_step_without_pushing_the_wall() {
        let (m, path) = stenosed();
        for covered in [false, true] {
            let params = StentParams { diameter: 4.0, length: 24.0, straightness: 0.0, stiffness: 1.0, covered, remesh: true, ..Default::default() };
            // Cancelled at the first report, and after a few (inside the union or the remesh).
            for after in [0, 3, 6] {
                let calls = Cell::new(0);
                let stop = |_: f32| {
                    calls.set(calls.get() + 1);
                    calls.get() <= after
                };
                match run(&m, &path, &params, Some(&stop)) {
                    Err(e) => assert!(is_cancelled(&e), "{e}"),
                    Ok((_, rep)) => panic!("covered {covered}: not cancelled after {after} reports (pushed {:?})", rep.pushed),
                }
                // Not asked again once it said stop.
                assert_eq!(calls.get(), after + 1, "covered {covered}");
            }
        }
    }

    /// The wall check as it was: a tree over the whole mesh, every leaf-mate a candidate.
    fn runs_into_wall_whole(m: &Mesh, tris: &[u32]) -> bool {
        let bvh = Bvh::build(m);
        tris.iter().filter(|&&t| m.is_triangle(t)).any(|&t| {
            let (tv, tp) = (m.triangle(t), m.tri_positions(t));
            let mut hit = false;
            bvh.query_aabb(&Aabb::from_points(tp), |o| {
                if !hit && o != t && !m.triangle(o).iter().any(|v| tv.contains(v)) && mm_core::intersect::triangles_intersect(tp, m.tri_positions(o)) {
                    hit = true;
                }
            });
            hit
        })
    }

    #[test]
    fn the_wall_check_on_nearby_triangles_answers_as_on_the_whole_mesh() {
        // Two spheres overlapping (crossings where they meet) and a tube apart from them; parts
        // of each as the region checked, crossing and clear, big and small.
        let mut m = primitives::icosphere(1.0, 4);
        let mut b = primitives::icosphere(0.8, 4);
        b.transform(|p| p + DVec3::new(1.5, 0.1, 0.05));
        m.append_mesh(&b, 1);
        let path: Vec<DVec3> = (0..=40).map(|i| DVec3::new(5.0, 0.0, 0.1 * i as f64)).collect();
        m.append_mesh(&primitives::tube_along(&path, &vec![0.5; path.len()], 24), 2);
        let all: Vec<u32> = m.triangle_ids().collect();
        for (k, step) in [(0usize, 1usize), (3, 7), (1, 50), (0, 997)] {
            for part in [m.tri_group(0), 1, 2] {
                let tris: Vec<u32> = all.iter().copied().filter(|&t| m.tri_group(t) == part).skip(k).step_by(step).collect();
                assert_eq!(runs_into_wall(&m, &tris), runs_into_wall_whole(&m, &tris), "group {part}, every {step}th from {k}");
            }
            // Near where the spheres meet only.
            let tris: Vec<u32> = all.iter().copied().filter(|&t| (m.tri_centroid(t).x - 0.9).abs() < 0.2).skip(k).step_by(step).collect();
            let crossing = runs_into_wall(&m, &tris);
            assert_eq!(crossing, runs_into_wall_whole(&m, &tris));
            assert!(crossing || step > 50, "the meeting spheres cross");
        }
        // Nothing live to check.
        assert!(!runs_into_wall(&m, &[]));
        assert!(!runs_into_wall(&m, &[m.max_tid() as u32 + 5]));
    }
}
