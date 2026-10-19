//! Virtual stenting: place a stent of a given diameter and length between two cross-sections
//! of a vessel and reshape the vessel around it, to compare the flow through treated anatomy.
//!
//! **Placement** ([`place_stent`]). The stent's axis blends two paths between the sections:
//! the vessel's centerline as it is, and the smoothest path joining the sections, which is
//! where a free stent held at both would settle. That path is a line (G0), or a Hermite
//! spline leaving each section along its normal (G1) and also continuing the vessel's bend
//! there (G2), its tangent lengths chosen for the least bending energy for its length
//! (length × ∫κ² ds, least for a circular arc). *Straightness* mixes the two paths at equal
//! fractions of their length. Past each section the paths go on along the vessel and along
//! the smooth path's end direction (its bend fading out for G2). The stent keeps its length
//! along its own axis and is centred on the two sections, so a stent longer than the
//! stretch reaches equally far past each.
//!
//! **Bare (uncovered) stent** ([`apply_stent`]). A stent can only open the vessel:
//! 1. Where the stent grips the wall (the vessel no wider than the stent, letting go by 1.6×
//!    its radius), each slice of the vessel is carried from the centerline onto the stent's
//!    axis, moved and turned with it, so the vessel straightens with the stent. Part of a
//!    stent floating in a wider lumen moves nothing. The move fades out past the stent's ends
//!    and up side branches.
//! 2. Wall inside the stent's tube is pushed out towards it: all the way with *stiffness* 1
//!    (a perfect tube of the stent's diameter), not at all with 0. Wall outside the tube is
//!    left alone. The push fades out over *end blend* past each end.
//! 3. The reshaped region is remeshed.
//!
//! **Covered stent** (stent graft). After the same steps the stented stretch is cut out and
//! replaced by the stent's tube: the lumen there is the stent, narrowed to its diameter where
//! the vessel was wider (an aneurysm or a dilated stretch is excluded, and side branches
//! opening along it are sealed off and removed), and open as far as the stiffness allows where
//! the vessel was narrower.

use crate::bridge::{aligned_loops, commit, quad_band, strip_with, AlignedLoops};
use crate::cut::{local_plane_cut_with, CutFill, KeepSide, LocalCutResult, PlaneCutParams, Side};
use crate::loft::{blend_join, EndContinuity};
use crate::util::smoothstep;
use crate::{EditError, EditResult};
use mm_core::{Bvh, DQuat, DVec3, Mesh, Plane, Ray};
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap, HashSet};
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
    /// How far the samples run on past each end: the vessel eases back from the stent's path
    /// onto its own over it (longer where the stent leaves the vessel at an angle).
    pub margins: [f64; 2],
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
    let span: Vec<(DVec3, DVec3, f64)> = (0..=n)
        .map(|i| {
            let u = i as f64 / n as f64;
            let s = sa + gap * u;
            let c = vessel(s);
            (c.lerp(ideal.at(ideal.len() * u), w), c, s)
        })
        .collect();
    let span_len: f64 = span.windows(2).map(|x| x[0].0.distance(x[1].0)).sum();
    // Past the sections, far enough for the stent and the longest fade beyond it.
    let longest = params.end_blend.max(12.0 * r);
    let past = 1.25 * ((0.5 * (length - span_len)).max(0.0) + longest) + 4.0 * ds;
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
    // Past each end the vessel eases from the stent's path back onto its own, over a length
    // that keeps the bend this adds gentle for the vessel's radius: a straightened stent's
    // end leaves the vessel at an angle and off its centre, and turning back too sharply
    // would fold the wall on the inside of the turn.
    let fade_length = |x: f64, out: f64| {
        let (p, c, sc) = at(x);
        let (p2, c2, _) = at(x + out * ds);
        let turn = (p2 - p).angle_between(c2 - c);
        let turn = if turn.is_finite() { turn } else { 0.0 };
        let (off, rv) = (p.distance(c), centre.value(sc).max(r));
        let mut l = params.end_blend.max(2.0 * r);
        while l < longest && rv * (1.875 * turn / l + 5.8 * off / (l * l)) > 0.35 {
            l *= 1.1;
        }
        l.min(longest)
    };
    let margins = [fade_length(start, -1.0), fade_length(start + length, 1.0)];
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
        ..Default::default()
    };
    for i in 0..k {
        let x = from + (to - from) * i as f64 / (k - 1) as f64;
        let (p, c, sc) = at(x);
        pl.s.push(x - start);
        pl.axis.push(p);
        pl.vessel.push(c);
        pl.vessel_radius.push(centre.value(sc));
    }
    Ok(pl)
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
    /// Covered: open ends (of side branches) removed with the excluded wall.
    pub sealed_openings: usize,
    /// Covered: triangles of wall excluded by the stent and removed.
    pub removed_triangles: usize,
    /// The reshaped region's triangles.
    pub triangles: Vec<u32>,
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
pub fn apply_stent(mesh: &mut Mesh, pl: &StentPlacement, params: &StentParams, owner: Option<&[bool]>) -> EditResult<StentReport> {
    check(params)?;
    let k = pl.s.len();
    if k < 3 {
        return Err(EditError::InvalidParameter("the stent placement has too few samples".into()));
    }
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
    let project = |p: DVec3| -> (f64, DVec3) {
        let mut best = (f64::INFINITY, 0.0, pl.vessel[0]);
        for i in 0..k - 1 {
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
    let mut core: Vec<(u32, f64, DVec3)> = Vec::new();
    let mut is_core = vec![false; mesh.max_vid()];
    for v in mesh.vertex_ids() {
        let p = mesh.vertex(v);
        if p.cmplt(lo).any() || p.cmpgt(hi).any() {
            continue;
        }
        let (x, c) = project(p);
        if x <= 0.0 || x >= (k - 1) as f64 {
            continue;
        }
        let mine = match owner {
            Some(o) => o.get(v as usize).copied().unwrap_or(false),
            None => p.distance(c) < 2.5 * lerp_f(&pl.vessel_radius, x).max(r),
        };
        if mine {
            core.push((v, x, p - c));
            is_core[v as usize] = true;
        }
    }
    if core.is_empty() {
        return Err(EditError::Failed("no vessel wall around the stent".into()));
    }
    // Side branches and anything else joined to the vessel near the stent follow it less
    // with distance along the surface.
    let reach = 1.5 * wide;
    let mut dist: HashMap<u32, f64> = core.iter().map(|&(v, _, _)| (v, 0.0)).collect();
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
    for (&v, &d) in &dist {
        if !is_core[v as usize] {
            let p = mesh.vertex(v);
            let (x, c) = project(p);
            verts.push((v, x, p - c, 1.0 - smoother(d / reach)));
        }
    }

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
    let ease = |s: f64| if s < 0.0 { fade(-s, pl.margins[0]) } else { fade(s - length, pl.margins[1]) };
    let carry: Vec<f64> = (0..k).map(|i| grip[i] * ease(pl.s[i])).collect();
    // Each slice turns to face along the path its centre is carried onto (averaged over half
    // the stent's radius), so the wall is not sheared where the carry fades.
    let carried: Vec<DVec3> = (0..k).map(|i| pl.vessel[i].lerp(pl.axis[i], carry[i])).collect();
    let along = tangents(&carried);
    let half = ((0.5 * r / ds).round() as usize).max(1);
    let rot: Vec<DQuat> = (0..k)
        .map(|i| {
            let t = along[i.saturating_sub(half)..=(i + half).min(k - 1)].iter().sum::<DVec3>().normalize_or(along[i]);
            DQuat::from_rotation_arc(tc[i], t)
        })
        .collect();
    let outside = (first..=last).any(|i| (1.0 - carry[i]) * pl.axis[i].distance(pl.vessel[i]) > 0.8 * wall[i]);

    // Region and edge length before reshaping.
    let soft = 0.05 * r;
    let open = |rho: f64| {
        let d = rho - r;
        if d >= soft {
            rho
        } else if d <= -soft {
            r
        } else {
            r + (d + soft) * (d + soft) / (4.0 * soft)
        }
    };
    let mut after: Vec<Vec<f64>> = vec![Vec::new(); k];
    let moves: Vec<(u32, DVec3)> = verts
        .iter()
        .map(|&(v, x, o, wv)| {
            let (c, p, t) = (lerp_at(&pl.vessel, x), lerp_at(&pl.axis, x), lerp_at(&tp, x).normalize_or_zero());
            let tw = lerp_f(&carry, x) * wv;
            let i = (x.floor().max(0.0) as usize).min(k - 2);
            let q = DQuat::IDENTITY.slerp(rot[i].slerp(rot[i + 1], x - i as f64), wv);
            let carried = c + (p - c) * tw;
            let xt = carried + q * o;
            // Past the stent's end the push fades out about the vessel's own centre, not an
            // imaginary continuation of the stent.
            let fade = lerp_f(&radial, x);
            let rel = xt - carried.lerp(p, fade);
            let perp = rel - t * rel.dot(t);
            let rho = perp.length();
            let push = params.stiffness * fade * wv;
            let (xf, rho2) = if rho > 1e-12 * r && push > 0.0 {
                let rho2 = rho + (open(rho) - rho) * push;
                (xt + perp * ((rho2 - rho) / rho), rho2)
            } else {
                (xt, rho)
            };
            if wv >= 1.0 {
                after[(x.round() as usize).min(k - 1)].push(rho2);
            }
            (v, xf)
        })
        .collect();
    let narrow_after = (first..=last).filter_map(|i| percentile(&mut after[i], 0.1)).fold(f64::INFINITY, f64::min);
    let tiny = 1e-9 * r;
    let mut moved_verts: HashSet<u32> = HashSet::new();
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
    for &(v, p) in &moves {
        mesh.set_vertex(v, p);
    }
    let mut report = StentReport { moved: moved_verts.len(), max_shift, narrowest: [narrow_before, narrow_after], outside, ..Default::default() };
    if params.covered {
        let (tube, sealed, removed) = cover(mesh, pl, h)?;
        region.retain(|&t| mesh.is_triangle(t));
        region.extend(tube);
        report.sealed_openings = sealed;
        report.removed_triangles = removed;
        report.narrowest[1] = report.narrowest[1].min(r);
    }
    region.sort_unstable();
    region.dedup();
    if params.remesh && !region.is_empty() {
        let rp = mm_remesh::RemeshParams { target_edge_length: h, iterations: 8, region: Some(region.clone()), ..mm_remesh::RemeshParams::with_target(h) };
        if let Ok(Some(tris)) = mm_remesh::remesh(mesh, &rp, None).map(|x| x.region_triangles) {
            region = tris;
        }
    }
    mesh.bump_shape_version();
    report.triangles = region;
    Ok(report)
}

/// Covered stent: cut the stented stretch out at the stent's ends and put the stent's tube in
/// its place — the wall's distance from the axis capped at the stent's radius, easing into
/// each cut over a short flare. Returns the tube's triangles with the bands blended into
/// it, the number of open ends removed and the number of triangles removed.
fn cover(mesh: &mut Mesh, pl: &StentPlacement, h: f64) -> EditResult<(Vec<u32>, usize, usize)> {
    let (r, l) = (pl.radius, pl.length);
    let around = ((TAU * r / h).round() as usize).clamp(12, 4000);
    let rows = ((l / h).round() as usize).max(3);
    let frames = pl.frames(rows);
    // The lumen around the axis: each row's wall distance by angle, capped at the radius.
    let bvh = Bvh::build(mesh);
    let mut rad: Vec<Vec<f64>> = frames
        .iter()
        .map(|&[x, _, u, v]| {
            (0..around)
                .map(|i| {
                    let a = TAU * i as f64 / around as f64;
                    bvh.raycast(mesh, &Ray::new(x, u * a.cos() + v * a.sin())).map_or(r, |hit| hit.t.min(r))
                })
                .collect()
        })
        .collect();
    for _ in 0..2 {
        let old = rad.clone();
        let at = |jj: usize, ii: usize| old[jj.min(rows)][ii % around];
        for (j, row) in rad.iter_mut().enumerate() {
            for (i, x) in row.iter_mut().enumerate() {
                let (jm, jp) = (j.saturating_sub(1), j + 1);
                let (im, ip) = (i + around - 1, i + 1);
                *x = (4.0 * at(j, i) + 2.0 * (at(j, im) + at(j, ip) + at(jm, i) + at(jp, i)) + at(jm, im) + at(jm, ip) + at(jp, im) + at(jp, ip)) / 16.0;
            }
        }
    }
    let wide = 3.0 * pl.vessel_radius.iter().copied().fold(r, f64::max);
    let cut = |mesh: &mut Mesh, x: DVec3, n: DVec3| -> EditResult<LocalCutResult> {
        let res = local_plane_cut_with(mesh, Plane::from_point_normal(x, n), x, &PlaneCutParams { keep: KeepSide::Both, fill: CutFill::None, ..Default::default() })
            .map_err(|e| match e {
                EditError::CurveDoesNotSeparate => EditError::Failed("the covered stretch is joined to the rest of the vessel another way round, so it cannot be sealed off".into()),
                e => e,
            })?;
        if !res.curve_closed || res.curve_distance > wide {
            return Err(EditError::Failed("each end of a covered stent must lie inside the vessel".into()));
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
    let middle: HashSet<u32> = mm_core::topology::flood_fill_triangles(mesh, &seeds, |_| true).into_iter().collect();
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
    mesh.remove_triangles(middle.iter().copied());
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
    let sample = |row: &[f64], a: f64| {
        let x = a.rem_euclid(TAU) / TAU * around as f64;
        let i = x.floor() as usize % around;
        let f = x - x.floor();
        row[i] * (1.0 - f) + row[(i + 1) % around] * f
    };
    let mut tube_rows: Vec<Vec<u32>> = Vec::with_capacity(rows - 1);
    let mut new_vertices = Vec::new();
    for (j, &[x, _, u, v]) in frames.iter().enumerate().take(rows).skip(1) {
        let s = l * j as f64 / rows as f64;
        let (fa, fb) = (smoother(s / flare), smoother((l - s) / flare));
        let row: Vec<u32> = (0..around)
            .map(|i| {
                let a = start + sense * TAU * i as f64 / around as f64;
                let mut rr = sample(&rad[j], a);
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
    let mut out = blend_join(mesh, &tube.triangles, [&p, &q], 0.25);
    out.extend(tube.triangles);
    Ok((out, sealed, removed))
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
