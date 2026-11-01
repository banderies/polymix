// SPDX-License-Identifier: AGPL-3.0-only
//! The radius of a fillet, read off the scan. Ported from ScanRuler
//! (`plugins/reverse/core/filletEstimate.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Every reverse-engineering tool offers "estimate radius from the mesh" on a fillet, since
//! the round on a real part is nobody's guess — it is a tool radius. The reading is taken
//! across the edge at its middle: a section through the scan by the plane perpendicular to
//! the edge, projected into that plane with the edge at the origin and the two faces as lines
//! through it. The scan's points on the flats lie on those lines; the points of the round lie
//! inside both half-spaces — below both faces — and a circle through them, tangent to both
//! faces, is the fillet.

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::fillet_num::{js_num, precision_value};
use crate::slice::SectionCut;
use crate::topo::{EdgeFingerprint, EdgeKind, FaceKind};

/// Why a place read no round (or chamfer), filled in by the reading that gave none: "no
/// scan", "sharp", "S-profile: 45 % on the arc" …
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Miss {
    pub why: Option<String>,
}

/// Write a reason into an optional miss.
pub(crate) fn tell(miss: &mut Option<&mut Miss>, why: impl Into<String>) {
    if let Some(m) = miss.as_deref_mut() {
        m.why = Some(why.into());
    }
}

/// Which shape a corner is read for: a round (a fillet) or a straight cut (a chamfer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum ReadShape {
    #[default]
    Round,
    Chamfer,
}

impl ReadShape {
    /// The other shape: what a reading is checked against.
    pub fn other(self) -> ReadShape {
        match self {
            ReadShape::Round => ReadShape::Chamfer,
            ReadShape::Chamfer => ReadShape::Round,
        }
    }

    /// The shape's word, as the answers write it.
    pub fn word(self) -> &'static str {
        match self {
            ReadShape::Round => "round",
            ReadShape::Chamfer => "chamfer",
        }
    }
}

/// Why a window too few points to read read none: with none near the corner, no scan there;
/// with some, the corner of a coarse scan's polygon, as sharp as it can show.
pub fn few_why(points: &[DVec2], reach: f64) -> &'static str {
    if points.iter().any(|p| p.length() <= 0.1 * reach) {
        "sharp"
    } else {
        "no scan"
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FilletReading {
    pub radius: f64,
    /// How many section points the circle was fitted to.
    pub points: usize,
    /// The RMS of those points about the circle, millimetres.
    pub rms: f64,
    /// The radius's own uncertainty, mm: the points' scatter over how much they say about the
    /// radius — loose on a round with few points or a short arc of them.
    pub sigma: f64,
}

/// The section points that belong to the corner: from the point of all the chains nearest
/// the corner, the chain walked both ways while it stays within `reach`. A section through a
/// thin wall cuts both skins, and the far skin lies inside both faces just as the round does
/// — but it is another part of the chain, joined round the end of the wall, and the walk
/// never gets there.
pub fn corner_window(chains: &[Vec<DVec2>], reach: f64) -> Vec<DVec2> {
    let Some((c, i)) = nearest_corner(chains) else { return Vec::new() };
    let chain = &chains[c];
    let mut out = vec![chain[i]];
    for p in chain[..i].iter().rev() {
        if p.length() > reach {
            break;
        }
        out.push(*p);
    }
    for p in &chain[i + 1..] {
        if p.length() > reach {
            break;
        }
        out.push(*p);
    }
    out
}

/// The point of all the chains nearest the corner: its chain and index.
fn nearest_corner(chains: &[Vec<DVec2>]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize, f64)> = None;
    for (c, chain) in chains.iter().enumerate() {
        for (i, p) in chain.iter().enumerate() {
            let d = p.length();
            if best.is_none_or(|b| d < b.2) {
                best = Some((c, i, d));
            }
        }
    }
    best.map(|b| (b.0, b.1))
}

/// Where the chain about a corner runs off onto another face within the reach: the face it
/// ran along (0 or 1, as the edge has them), how far from the corner along it, mm, and what
/// it does past there, as a place's verdict says it — "step 0.8 mm on face 2" where it steps
/// and runs on beside the same face, "another face 1.7 mm off" where it turns to run beside
/// the other.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEnd {
    pub face: usize,
    pub at: f64,
    pub what: String,
}

/// A corner's window walked along its faces, and the reach it is read to.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceWindow {
    pub points: Vec<DVec2>,
    pub reach: f64,
    /// Set where a face ends within the reach — the reach is then held to `FACE_SHARE` of its
    /// extent.
    pub end: Option<FaceEnd>,
}

/// How much of a face that ends within the reach a round is read over.
const FACE_SHARE: f64 = 0.9;
/// How many points running off both faces in a row end the walk.
const OFF_RUN: usize = 3;

/// Within this angle of a face's direction a run beside it is parallel: cos 25°.
fn cos_parallel() -> f64 {
    (25.0f64).to_radians().cos()
}

/// How far a point is from a face's trace: the half-line from the corner along `d`.
fn off_ray(p: DVec2, d: DVec2) -> f64 {
    let t = p.dot(d).max(0.0);
    (p - d * t).length()
}

/// The section points that belong to the corner when its faces are known: as
/// `corner_window`, the chain walked both ways from the point nearest the corner, but each
/// way only while it keeps to the round and the faces. Once on a face, the walk ends where
/// the chain runs off both faces' traces by more than about three scatters — a key standing
/// 1.7 mm above a housing, walked down its side onto the housing: those points lie nowhere
/// near the profile of any round, and capped as they are they outweigh the round's own.
/// Where a face ends so within the reach, the reach is held to `FACE_SHARE` of how far the
/// face runs: a round larger than the face is not one the scan can show. `n` are the faces'
/// outward normals, `d` the ways they run from the corner, unit length.
pub fn face_window(chains: &[Vec<DVec2>], reach: f64, n: [DVec2; 2], d: [DVec2; 2]) -> FaceWindow {
    let Some((bc, bi)) = nearest_corner(chains) else { return FaceWindow { points: Vec::new(), reach, end: None } };
    // Off a face by more than a few scatters — and more than a tenth of the reach, so a
    // scanned face a tenth off the model's still counts as on it.
    let leave = (3.0 * scatter_on_flats(&corner_window(chains, reach), n[0], n[1], reach)).max(0.1 * reach);
    let chain = &chains[bc];
    let len = chain.len() as isize;
    let walk = |step: isize| -> (Vec<DVec2>, Option<FaceEnd>) {
        let mut points: Vec<DVec2> = Vec::new();
        let mut on: Option<usize> = None;
        let mut at = 0.0;
        let mut off = 0usize;
        let mut j = bi as isize + step;
        while j >= 0 && j < len {
            let p = chain[j as usize];
            if p.length() > reach {
                break;
            }
            let o = [off_ray(p, d[0]), off_ray(p, d[1])];
            let k = if o[0] <= o[1] { 0 } else { 1 };
            if o[k] <= leave {
                on = Some(k);
                at = p.dot(d[k]);
                off = 0;
            } else if let Some(face) = on {
                off += 1;
                if off >= OFF_RUN {
                    let mut past: Vec<DVec2> = Vec::new();
                    let mut q = j - (OFF_RUN as isize - 1) * step;
                    while q >= 0 && q < len && chain[q as usize].length() <= reach {
                        past.push(chain[q as usize]);
                        q += step;
                    }
                    let keep = points.len().saturating_sub(OFF_RUN - 1);
                    points.truncate(keep);
                    let what = past_face(&past, face, at, n, d);
                    return (points, Some(FaceEnd { face, at, what }));
                }
            }
            points.push(p);
            j += step;
        }
        (points, None)
    };
    let (back_points, back_end) = walk(-1);
    let (ahead_points, ahead_end) = walk(1);
    let mut end: Option<FaceEnd> = None;
    for e in [back_end, ahead_end].into_iter().flatten() {
        match &end {
            Some(x) if !(e.at < x.at) => {}
            _ => end = Some(e),
        }
    }
    let held = match &end {
        Some(e) => reach.min(FACE_SHARE * e.at),
        None => reach,
    };
    let points = std::iter::once(chain[bi]).chain(back_points).chain(ahead_points).filter(|p| p.length() <= held).collect();
    FaceWindow { points, reach: held, end }
}

/// What the chain does past where it left face `k`, `at` along it: a run beside the same face
/// is a step, one beside the other face another face, each by how far off that face it runs;
/// else where the face ended.
fn past_face(past: &[DVec2], k: usize, at: f64, n: [DVec2; 2], d: [DVec2; 2]) -> String {
    let mm = |x: f64| js_num(precision_value(x, 2));
    let (Some(first), Some(last)) = (past.first(), past.last()) else {
        return format!("another face {} mm along face {}", mm(at), k + 1);
    };
    let u = *last - *first;
    let l = u.length();
    let beside = |j: usize| l > 1e-6 && u.dot(d[j]).abs() >= cos_parallel() * l;
    let off = |j: usize| {
        let mut o: Vec<f64> = past.iter().map(|p| p.dot(n[j]).abs()).collect();
        o.sort_by(f64::total_cmp);
        o[o.len() >> 1]
    };
    if beside(k) {
        return format!("step {} mm on face {}", mm(off(k)), k + 1);
    }
    let other = 1 - k;
    if beside(other) {
        return format!("another face {} mm off", mm(off(other)));
    }
    format!("another face {} mm along face {}", mm(at), k + 1)
}

/// How far from a face a point still counts as lying on it, for reading the scatter: a thin
/// wall's far skin lies well past this.
const ON_FACE_MM: f64 = 0.3;

/// The scan's scatter about a flat, taken as three times the median distance of the points
/// that lie on the faces — robust to the round's own points, which are the minority near a
/// sharp-ish corner — with a floor for a scan cleaner than the arithmetic. Only points near a
/// face are read: on a thin wall the section runs down the far skin too, and its distance from
/// these faces is the wall, not noise.
fn scatter_on_flats(points: &[DVec2], n1: DVec2, n2: DVec2, reach: f64) -> f64 {
    let mut off: Vec<f64> = Vec::new();
    for p in points {
        if p.length() > reach {
            continue;
        }
        let d = p.dot(n1).abs().min(p.dot(n2).abs());
        if d <= ON_FACE_MM {
            off.push(d);
        }
    }
    if off.len() < 4 {
        return 0.05;
    }
    off.sort_by(f64::total_cmp);
    (3.0 * off[off.len() >> 1]).max(0.02)
}

/// The fillet radius at a corner, from the section points about it.
///
/// `points` are in the section plane with the sharp edge at the origin; `n1` and `n2` are the
/// two faces' outward normals in that plane, unit length. `reach` is how far from the corner
/// points are read — a few times the largest radius expected. Returns `None` when the scan
/// shows no round there: a sharp corner, a chamfer, a gap; `miss` then says why.
///
/// The profile of a filleted corner is a flat, an arc tangent to it, and the other flat
/// tangent to the arc: one unknown, the radius, since the arc's centre lies on the faces'
/// bisector a radius inside each. Every section point near the corner is read against that
/// whole profile — a scan puts only a handful of points on a small round, and reading the
/// round's points alone would lose it in the noise — and the radius is the one the points lie
/// nearest, searched over the reach. A sharp corner is the profile at no radius: a round has
/// to explain the points markedly better than that to be reported.
pub fn estimate_fillet_radius(points: &[DVec2], n1: DVec2, n2: DVec2, reach: f64, mut miss: Option<&mut Miss>) -> Option<FilletReading> {
    let tau = scatter_on_flats(points, n1, n2, reach);
    let pts: Vec<DVec2> = points.iter().copied().filter(|p| p.length() <= reach).collect();
    let n = pts.len();
    if n < 8 {
        tell(&mut miss, few_why(points, reach));
        return None;
    }
    let k = 1.0 + n1.dot(n2);
    if !(k > 1e-6) {
        tell(&mut miss, "the faces nearly one");
        return None;
    }
    // The centre for a radius r is r · b; the tangent points r · (b + n); the flats run from
    // the tangent points away from the corner along e.
    let b = -(n1 + n2) / k;
    let t1 = b + n1;
    let t2 = b + n2;
    let (l1, l2) = (t1.length(), t2.length());
    if !(l1 > 1e-9 && l2 > 1e-9) {
        tell(&mut miss, "the faces nearly one");
        return None;
    }
    let e1 = t1 / l1;
    let e2 = t2 / l2;
    // Distance of a point to the profile of radius r, and whether the arc is what it is
    // nearest.
    let distance = |p: DVec2, r: f64| -> (f64, bool) {
        let c = b * r;
        let p1 = t1 * r;
        let p2 = t2 * r;
        // Beyond both tangent points, on the corner's side: the arc.
        let s1 = (p - p1).dot(e1);
        let s2 = (p - p2).dot(e2);
        if s1 <= 0.0 && s2 <= 0.0 {
            return (((p - c).length() - r).abs(), true);
        }
        let ray = |q: DVec2, e: DVec2| {
            let t = (p - q).dot(e).max(0.0);
            (p - (q + e * t)).length()
        };
        (ray(p1, e1).min(ray(p2, e2)), false)
    };
    // The cost of a radius: the points' squared distances to its profile, each capped a few
    // scatters out so a stray cannot steer the search.
    let cap = 5.0 * tau;
    let cost = |r: f64| -> f64 {
        let mut sum = 0.0;
        for p in &pts {
            let d = distance(*p, r).0.min(cap);
            sum += d * d;
        }
        sum
    };
    let r_min = tau.max(0.02);
    let r_max = reach;
    let (best, _) = log_search(r_min, r_max, &cost);
    let r = golden(best, r_min, r_max, &cost);
    // A sharp corner: the same profile at no radius. The round has to do markedly better
    // than it, and rest on points of its own.
    let sharp = cost(r_min);
    if !(r > 2.0 * tau) || cost(r) > 0.5 * sharp {
        tell(&mut miss, "sharp");
        return None;
    }
    // The points the round rests on: those nearest the arc, within a few scatters of it — and
    // within the corner's own neighbourhood, since on a thin wall the far skin's points are
    // nearest the arc too and lie nowhere near it.
    let mut on_arc = 0usize;
    let mut arc_total = 0usize;
    let mut sum = 0.0;
    let near = 2.0 * r + 3.0 * tau;
    for p in &pts {
        if p.length() > near {
            continue;
        }
        let (d, arc) = distance(*p, r);
        if !arc {
            continue;
        }
        arc_total += 1;
        if d <= 3.0 * tau {
            on_arc += 1;
            sum += d * d;
        }
    }
    if on_arc < 3 {
        tell(&mut miss, "sharp");
        return None;
    }
    let rms = (sum / on_arc as f64).sqrt();
    // A chamfer's points lie on a line: an arc through their middle leaves the ends standing
    // off it; so does an S-profile's.
    if (on_arc as f64) < 0.6 * arc_total as f64 {
        let share = crate::fillet_num::js_round(100.0 * on_arc as f64 / arc_total as f64);
        tell(&mut miss, format!("S-profile: {} % on the arc", js_num(share)));
        return None;
    }
    // The radius's sigma, as a least-squares fit's: a point's distance off the arc, |p − r·b|
    // − r, moves with r by −b·u − 1 (u the unit vector from the centre to it) — nothing at the
    // tangent points, most in the middle of the round, which is where the radius is read.
    let mut jj = 0.0;
    for p in &pts {
        if p.length() > near {
            continue;
        }
        let (d, arc) = distance(*p, r);
        if !arc || d > 3.0 * tau {
            continue;
        }
        let u = *p - b * r;
        let l = u.length();
        if !(l > 1e-12) {
            continue;
        }
        let j = -b.dot(u) / l - 1.0;
        jj += j * j;
    }
    // The scatter is the larger of the fit's and the flats' own: a round that happens to fit
    // its few points closely is no better known than the scan is.
    let scatter = (rms * (on_arc as f64 / (on_arc as f64 - 1.0).max(1.0)).sqrt()).max(tau / 3.0);
    let sigma = if jj > 1e-12 { scatter / jj.sqrt() } else { r };
    Some(FilletReading { radius: r, points: on_arc, rms, sigma })
}

/// How many steps the search over the reach takes, evenly on a log scale.
const SEARCH_STEPS: usize = 160;

/// The best of `SEARCH_STEPS + 1` values from `lo` to `hi` evenly on a log scale: the value
/// and its cost. The first of equal costs is kept.
pub(crate) fn log_search(lo: f64, hi: f64, cost: &dyn Fn(f64) -> f64) -> (f64, f64) {
    let mut best = lo;
    let mut best_cost = f64::INFINITY;
    for i in 0..=SEARCH_STEPS {
        let r = lo * (hi / lo).powf(i as f64 / SEARCH_STEPS as f64);
        let c = cost(r);
        if c < best_cost {
            best_cost = c;
            best = r;
        }
    }
    (best, best_cost)
}

/// A golden-section search for the least cost between the log-scale neighbours of `best`,
/// fifty steps of it: the middle of where it ends.
pub(crate) fn golden(best: f64, lo0: f64, hi0: f64, cost: &dyn Fn(f64) -> f64) -> f64 {
    let ratio = (hi0 / lo0).powf(1.0 / SEARCH_STEPS as f64);
    let mut lo = best / ratio;
    let mut hi = best * ratio;
    for _ in 0..50 {
        let a = lo + (hi - lo) * 0.382;
        let b = lo + (hi - lo) * 0.618;
        if cost(a) < cost(b) {
            hi = b;
        } else {
            lo = a;
        }
    }
    (lo + hi) / 2.0
}

/// Along a face away from the corner: square to its normal, on the side the other face's
/// normal points away from.
fn along_face(n: DVec2, other: DVec2) -> DVec2 {
    let t = DVec2::new(-n.y, n.x);
    if t.dot(other) > 0.0 {
        -t
    } else {
        t
    }
}

/// The chamfer at a corner, from the section points about it — as `estimate_fillet_radius`,
/// with a straight cut in place of the round: the profile is a flat, a line from `s` along it
/// to `s` along the other, and the other flat; one unknown, the setback s, read as the
/// distance along each face (an even chamfer). `None` where the scan shows none.
pub fn estimate_chamfer(points: &[DVec2], n1: DVec2, n2: DVec2, reach: f64, mut miss: Option<&mut Miss>) -> Option<FilletReading> {
    let tau = scatter_on_flats(points, n1, n2, reach);
    let pts: Vec<DVec2> = points.iter().copied().filter(|p| p.length() <= reach).collect();
    if pts.len() < 8 {
        tell(&mut miss, few_why(&pts, reach));
        return None;
    }
    let e1 = along_face(n1, n2);
    let e2 = along_face(n2, n1);
    let seg = |p: DVec2, a: DVec2, b: DVec2| {
        let d = b - a;
        let ll = d.dot(d);
        let t = if ll > 0.0 { ((p - a).dot(d) / ll).clamp(0.0, 1.0) } else { 0.0 };
        (p - (a + d * t)).length()
    };
    let ray = |p: DVec2, a: DVec2, e: DVec2| {
        let t = (p - a).dot(e).max(0.0);
        (p - (a + e * t)).length()
    };
    // A point's distance to the profile of setback s, and whether the cut is what it is
    // nearest.
    let distance = |p: DVec2, s: f64| -> (f64, bool) {
        let a = e1 * s;
        let b = e2 * s;
        let dc = seg(p, a, b);
        let df = ray(p, a, e1).min(ray(p, b, e2));
        if dc <= df {
            (dc, true)
        } else {
            (df, false)
        }
    };
    let cap = 5.0 * tau;
    let cost = |s: f64| -> f64 {
        let mut sum = 0.0;
        for p in &pts {
            let d = distance(*p, s).0.min(cap);
            sum += d * d;
        }
        sum
    };
    let s_min = tau.max(0.02);
    let (best, _) = log_search(s_min, reach, &cost);
    let s = golden(best, s_min, reach, &cost);
    if !(s > 2.0 * tau) || cost(s) > 0.5 * cost(s_min) {
        tell(&mut miss, "sharp");
        return None;
    }
    let mut on = 0usize;
    let mut sum = 0.0;
    let mut jj = 0.0;
    let h = (s * 1e-3).max(1e-4);
    for p in &pts {
        let (d, cut) = distance(*p, s);
        if !cut || d > 3.0 * tau {
            continue;
        }
        on += 1;
        sum += d * d;
        let j = (distance(*p, s + h).0 - distance(*p, s - h).0) / (2.0 * h);
        jj += j * j;
    }
    if on < 3 {
        tell(&mut miss, "sharp");
        return None;
    }
    let rms = (sum / on as f64).sqrt();
    let scatter = (rms * (on as f64 / (on as f64 - 1.0).max(1.0)).sqrt()).max(tau / 3.0);
    Some(FilletReading { radius: s, points: on, rms, sigma: if jj > 1e-12 { scatter / jj.sqrt() } else { s } })
}

/// A chamfer read with its two setbacks apart — a countersink, a 30° bevel — and the cut's
/// own line: `s[i]` along face i from the corner, `dir` the cut's direction in the section,
/// `dir_sigma` its uncertainty in radians.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ChamferSetbacks {
    pub s: [f64; 2],
    pub sigma: [f64; 2],
    pub dir: DVec2,
    pub dir_sigma: f64,
    pub points: usize,
    pub rms: f64,
}

/// The chamfer at a corner with its setbacks free: started from the even reading
/// (`estimate_chamfer`), the points on the cut — nearer it than either face, clear of its ends
/// — fitted with a line, the setbacks where that line meets the faces, and the points taken
/// again against the new profile, a few times over. The sigmas are the line fit's: its
/// offset where it meets each face over the sine of the angle it meets it at.
pub fn estimate_chamfer_setbacks(points: &[DVec2], n1: DVec2, n2: DVec2, reach: f64) -> Option<ChamferSetbacks> {
    let even = estimate_chamfer(points, n1, n2, reach, None)?;
    let tau = scatter_on_flats(points, n1, n2, reach);
    let e = [along_face(n1, n2), along_face(n2, n1)];
    let pts: Vec<DVec2> = points.iter().copied().filter(|p| p.length() <= reach).collect();
    let ray_dist = |p: DVec2, s: f64, k: usize| {
        let t = (p - e[k] * s).dot(e[k]).max(0.0);
        (p - e[k] * (s + t)).length()
    };
    let mut s = [even.radius, even.radius];
    let mut fit: Option<(DVec2, DVec2, Vec<DVec2>)> = None;
    for _ in 0..4 {
        let a = e[0] * s[0];
        let b = e[1] * s[1];
        let dv = b - a;
        let ll = dv.dot(dv);
        if !(ll > 1e-12) {
            return None;
        }
        let mut on: Vec<DVec2> = Vec::new();
        for p in &pts {
            let t = (*p - a).dot(dv) / ll;
            if t < 0.08 || t > 0.92 {
                continue;
            }
            let d = ((p.x - a.x) * dv.y - (p.y - a.y) * dv.x).abs() / ll.sqrt();
            if d <= 3.0 * tau && d <= ray_dist(*p, s[0], 0).min(ray_dist(*p, s[1], 1)) {
                on.push(*p);
            }
        }
        if on.len() < 4 {
            return None;
        }
        // The total-least-squares line: through the centroid, along the scatter's long axis.
        let count = on.len() as f64;
        let g = DVec2::new(on.iter().fold(0.0, |x, p| x + p.x) / count, on.iter().fold(0.0, |x, p| x + p.y) / count);
        let (mut sxx, mut sxy, mut syy) = (0.0, 0.0, 0.0);
        for p in &on {
            sxx += (p.x - g.x) * (p.x - g.x);
            sxy += (p.x - g.x) * (p.y - g.y);
            syy += (p.y - g.y) * (p.y - g.y);
        }
        let th = 0.5 * (2.0 * sxy).atan2(sxx - syy);
        let u = DVec2::new(th.cos(), th.sin());
        // Where the line meets face k: g + λu = μ e_k.
        let meet = |k: usize| {
            let c = e[k].x * u.y - e[k].y * u.x;
            if c.abs() > 1e-6 {
                (g.x * u.y - g.y * u.x) / c
            } else {
                f64::NAN
            }
        };
        let next = [meet(0), meet(1)];
        if !(next[0] > 0.0 && next[1] > 0.0 && next[0] < 1.5 * reach && next[1] < 1.5 * reach) {
            return None;
        }
        s = next;
        fit = Some((g, u, on));
    }
    let (g, u, on) = fit?;
    let count = on.len() as f64;
    let spread: f64 = on.iter().map(|p| (*p - g).dot(u)).fold(0.0, |x, l| x + l * l);
    let sum: f64 = on.iter().fold(0.0, |x, p| {
        let c = (p.x - g.x) * u.y - (p.y - g.y) * u.x;
        x + c * c
    });
    let rms = (sum / count).sqrt();
    let scatter = (rms * (count / (count - 2.0).max(1.0)).sqrt()).max(tau / 3.0);
    if !(spread > 1e-12) {
        return None;
    }
    let sig = |k: usize| {
        let l = (e[k] * s[k] - g).dot(u);
        let off = scatter * (1.0 / count + l * l / spread).sqrt();
        off / (e[k].x * u.y - e[k].y * u.x).abs().max(1e-6)
    };
    Some(ChamferSetbacks { s, sigma: [sig(0), sig(1)], dir: u, dir_sigma: scatter / spread.sqrt(), points: on.len(), rms })
}

/// A straight edge between two flat faces, as a body's topology gives it: where it is, which
/// way it runs, and the faces' outward normals — an edge's fingerprint.
pub type CornerEdge = EdgeFingerprint;

/// Why an edge cannot be read for a round by `read_corner_cut`, or `None` when it can: a
/// straight edge between two flat faces. A curved one goes to `curved_edge`.
pub fn corner_unreadable(edge: &EdgeFingerprint) -> Option<String> {
    if edge.kind != EdgeKind::Line {
        return Some(format!("a {} edge — only a straight edge is read", edge.kind.name()));
    }
    if edge.faces.len() < 2 {
        return Some("an edge with one face — nothing to round between".to_string());
    }
    if edge.faces.iter().any(|f| f.kind != FaceKind::Plane) {
        return Some(format!("an edge between a {} and a {} — only an edge between two flat faces is read", edge.faces[0].kind.name(), edge.faces[1].kind.name()));
    }
    None
}

/// The section plane across an edge at a point along it: its normal (the edge's direction),
/// and the maps into the plane — points with the edge at the origin, X along the first face's
/// outward normal flattened into it, and directions the same way, unit length.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CornerFrame {
    pub normal: DVec3,
    pub at: DVec3,
    pub u: DVec3,
    pub v: DVec3,
}

impl CornerFrame {
    /// A point in the plane, the edge at the origin.
    pub fn to2(&self, p: DVec3) -> DVec2 {
        let d = p - self.at;
        DVec2::new(d.dot(self.u), d.dot(self.v))
    }

    /// A direction in the plane, unit length; +X where it is square to the plane.
    pub fn dir2(&self, p: DVec3) -> DVec2 {
        let q = DVec2::new(p.dot(self.u), p.dot(self.v));
        let l = q.length();
        if l > 1e-9 {
            q / l
        } else {
            DVec2::X
        }
    }
}

/// `None` for a vector too short to have a direction, as ScanRuler's `normalize` has it.
pub(crate) fn unit3(v: DVec3) -> Option<DVec3> {
    let l = v.length();
    if l < 1e-12 || l.is_nan() {
        None
    } else {
        Some(v / l)
    }
}

/// The section plane across an edge at `at` along it. `None` for an edge too short or faces
/// too nearly one.
pub fn corner_frame(edge: &CornerEdge, at: DVec3) -> Option<CornerFrame> {
    let n = unit3(edge.dir)?;
    let n1 = edge.faces.first()?.dir;
    let u = unit3(n1 - n * n1.dot(n))?;
    let v = n.cross(u);
    Some(CornerFrame { normal: n, at, u, v })
}

/// The round at `at` along a straight edge between two flats, read off the scan's cut by the
/// plane square to the edge there (`cut`, the scan's section through `at` with the edge's
/// direction as normal): the panel's estimate and the agent's fillet read both.
pub fn read_corner_cut(edge: &CornerEdge, at: DVec3, cut: &SectionCut, reach: f64, shape: ReadShape, mut miss: Option<&mut Miss>) -> Option<FilletReading> {
    let Some(frame) = corner_frame(edge, at) else {
        tell(&mut miss, "the faces nearly one");
        return None;
    };
    let (Some(f0), Some(f1)) = (edge.faces.first(), edge.faces.get(1)) else {
        tell(&mut miss, "an edge with one face — nothing to round between");
        return None;
    };
    let n1 = frame.dir2(f0.dir);
    let n2 = frame.dir2(f1.dir);
    let w = face_window(&cut_chains(cut, |p| frame.to2(p)), reach, [n1, n2], [face_runs(n1, n2), face_runs(n2, n1)]);
    read_on_window(
        &w,
        |pts, r, m| match shape {
            ReadShape::Chamfer => estimate_chamfer(pts, n1, n2, r, Some(m)),
            ReadShape::Round => estimate_fillet_radius(pts, n1, n2, r, Some(m)),
        },
        miss,
    )
}

/// The way a flat face runs from a corner, in the section: square to its outward normal `n`,
/// away from the side the other face's normal `other` points to — for a convex corner and a
/// concave one alike, whose normals are then turned round.
pub fn face_runs(n: DVec2, other: DVec2) -> DVec2 {
    let k = 1.0 + n.dot(other);
    if k > 1e-6 {
        // From the corner to the tangent point of a round of one: b + n, the centre b on the
        // bisector a radius inside each face.
        let t = -(n + other) / k + n;
        let l = t.length();
        if l > 1e-9 {
            return t / l;
        }
    }
    DVec2::new(-n.y, n.x)
}

/// A reading taken on a corner's face window: with the reach the window holds it to, and
/// where a face ended within the reach and nothing read, the miss says so — "another face
/// 1.7 mm off within the reach — reach ≤ 1.5".
pub fn read_on_window(w: &FaceWindow, estimate: impl FnOnce(&[DVec2], f64, &mut Miss) -> Option<FilletReading>, miss: Option<&mut Miss>) -> Option<FilletReading> {
    let mut own = Miss::default();
    let reading = estimate(&w.points, w.reach, &mut own);
    if reading.is_none() {
        if let Some(m) = miss {
            m.why = match &w.end {
                Some(end) => Some(face_end_why(w.reach, end)),
                None => own.why,
            };
        }
    }
    reading
}

/// A face end as a miss says it: a step as it is ("step 0.8 mm on face 2"), another face with
/// the reach the read was held to ("another face 1.7 mm off within the reach — reach ≤ 1.5").
/// `reach` is the window's.
pub fn face_end_why(reach: f64, end: &FaceEnd) -> String {
    // Rounded down: a tenth, or a hundredth under a millimetre.
    let held = if reach >= 1.0 { (reach * 10.0).floor() / 10.0 } else { (reach * 100.0).floor() / 100.0 };
    if end.what.starts_with("step") {
        end.what.clone()
    } else {
        format!("{} within the reach — reach ≤ {}", end.what, js_num(held))
    }
}

/// A cut's chains mapped into a section plane.
pub fn cut_chains(cut: &SectionCut, to2: impl Fn(DVec3) -> DVec2) -> Vec<Vec<DVec2>> {
    cut.chains.iter().map(|c| c.iter().map(|p| to2(*p)).collect()).collect()
}

/// The chamfer's two setbacks at `at` along a straight edge between two flats, off the cut
/// `read_corner_cut` reads: `s[i]` along the edge's face i.
pub fn read_corner_setbacks(edge: &CornerEdge, at: DVec3, cut: &SectionCut, reach: f64) -> Option<ChamferSetbacks> {
    let frame = corner_frame(edge, at)?;
    let window = corner_window(&cut_chains(cut, |p| frame.to2(p)), reach);
    estimate_chamfer_setbacks(&window, frame.dir2(edge.faces.first()?.dir), frame.dir2(edge.faces.get(1)?.dir), reach)
}

/// Several readings along one edge taken together: the median radius, and a sigma that is the
/// larger of the readings' own (combined) and their spread — a round that changes along the
/// edge is not known to the fit's precision.
pub fn combine_readings(readings: &[FilletReading]) -> Option<FilletReading> {
    if readings.is_empty() {
        return None;
    }
    let mut radii: Vec<f64> = readings.iter().map(|r| r.radius).collect();
    radii.sort_by(f64::total_cmp);
    let mid = radii.len() >> 1;
    let radius = if radii.len() % 2 == 1 { radii[mid] } else { (radii[mid - 1] + radii[mid]) / 2.0 };
    let points: usize = readings.iter().map(|r| r.points).sum();
    let rms = (readings.iter().fold(0.0, |a, r| a + r.rms * r.rms * r.points as f64) / (points.max(1) as f64)).sqrt();
    let fit_sigma = 1.0 / readings.iter().fold(0.0, |a, r| a + 1.0 / r.sigma.max(1e-9).powi(2)).sqrt();
    let count = radii.len() as f64;
    let mean = radii.iter().sum::<f64>() / count;
    let spread = if radii.len() > 1 { (radii.iter().fold(0.0, |a, b| a + (b - mean).powi(2)) / (count - 1.0) / count).sqrt() } else { 0.0 };
    Some(FilletReading { radius, points, rms, sigma: fit_sigma.max(spread) })
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler `plugins/reverse/tests/reFilletEstimate.test.ts`: a rounded
    //! corner sampled with noise gives its radius back, a sharp corner and a chamfer give
    //! nothing, and the corner's window on a section through a thin wall stays on the near
    //! skin.
    use std::f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2};

    use super::*;
    use crate::fillet_testkit::gauss;
    use crate::fit2d::Mulberry32;

    /// The top face, outward +Y.
    const N1: DVec2 = DVec2::new(0.0, 1.0);
    /// The right face, outward +X.
    const N2: DVec2 = DVec2::new(1.0, 0.0);

    /// A corner at the origin between the top face (y = 0, x ≤ 0) and the right face (x = 0,
    /// y ≤ 0), rounded with radius r, sampled every `step` along the profile out to `reach`,
    /// with Gaussian noise of `noise`. The material lies at x ≤ 0, y ≤ 0.
    fn corner(r: f64, reach: f64, step: f64, noise: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(seed);
        let mut out = Vec::new();
        // Along the top face from far left to the round's start at x = −r.
        let mut x = -reach;
        while x < -r {
            out.push(DVec2::new(x, gauss(&mut rand) * noise));
            x += step;
        }
        // The round: centre (−r, −r), from angle 90° down to 0°.
        let arc = PI / 2.0 * r;
        let mut s = 0.0;
        while s <= arc {
            let a = PI / 2.0 - s / r;
            let (nx, ny) = (a.cos(), a.sin());
            let d = gauss(&mut rand) * noise;
            out.push(DVec2::new(-r + (r + d) * nx, -r + (r + d) * ny));
            s += step;
        }
        // Down the right face.
        let mut y = -r;
        while y > -reach {
            out.push(DVec2::new(gauss(&mut rand) * noise, y));
            y -= step;
        }
        out
    }

    #[test]
    fn reads_a_2_mm_round_through_scan_noise() {
        let points = corner(2.0, 8.0, 0.05, 0.02, 5);
        let reading = estimate_fillet_radius(&points, N1, N2, 8.0, None).expect("a round");
        assert!((reading.radius - 2.0).abs() < 0.05, "{reading:?}");
        assert!(reading.points > 20);
        assert!(reading.rms < 0.05);
    }

    #[test]
    fn reads_a_large_round_and_a_small_one() {
        let large = estimate_fillet_radius(&corner(6.0, 20.0, 0.1, 0.03, 5), N1, N2, 20.0, None).expect("a round");
        assert!((large.radius - 6.0).abs() < 0.12, "{large:?}");
        let small = estimate_fillet_radius(&corner(0.5, 4.0, 0.02, 0.01, 5), N1, N2, 4.0, None).expect("a round");
        assert!((small.radius - 0.5).abs() < 0.03, "{small:?}");
    }

    #[test]
    fn finds_no_round_at_a_sharp_corner() {
        assert!(estimate_fillet_radius(&corner(0.0, 8.0, 0.05, 0.02, 5), N1, N2, 8.0, None).is_none());
    }

    #[test]
    fn refuses_a_chamfer_its_points_lie_on_a_line_not_a_circle_tangent_to_both_faces() {
        let mut points = Vec::new();
        let mut x = -8.0;
        while x < -2.0 {
            points.push(DVec2::new(x, 0.0));
            x += 0.05;
        }
        let mut t = 0.0;
        while t <= 1.0 {
            points.push(DVec2::new(-2.0 + 2.0 * t, -2.0 * t));
            t += 0.01;
        }
        let mut y = -2.0;
        while y > -8.0 {
            points.push(DVec2::new(0.0, y));
            y -= 0.05;
        }
        assert!(estimate_fillet_radius(&points, N1, N2, 8.0, None).is_none());
    }

    #[test]
    fn takes_the_corner_s_window_off_the_near_skin_of_a_thin_wall() {
        // One chain: the outer corner rounded at 2 mm, then round the end of the wall and back
        // along the inner skin 1.35 mm inside both faces.
        let outer = corner(2.0, 12.0, 0.1, 0.0, 5);
        let mut inner = Vec::new();
        let mut y = -12.0;
        while y < -1.35 {
            inner.push(DVec2::new(-1.35, y));
            y += 0.1;
        }
        let mut x = -1.35;
        while x > -12.0 {
            inner.push(DVec2::new(x, -1.35));
            x -= 0.1;
        }
        let chain: Vec<DVec2> = outer.into_iter().chain(std::iter::once(DVec2::new(-1.35, -12.0))).chain(inner).collect();
        let window = corner_window(std::slice::from_ref(&chain), 8.0);
        // Everything in the window lies on the outer skin: nothing 1.35 mm inside both faces.
        assert!(window.len() > 50);
        assert!(window.iter().all(|p| !(p.x < -1.3 && p.y < -1.3 && (p.x + 1.35).abs() < 0.01)));
        let reading = estimate_fillet_radius(&window, N1, N2, 8.0, None).expect("a round");
        assert!((reading.radius - 2.0).abs() < 0.05, "{reading:?}");
        // Without the walk, the inner skin pollutes the fit.
        let polluted = estimate_fillet_radius(&chain, N1, N2, 8.0, None);
        assert!(polluted.is_none_or(|p| (p.radius - 2.0).abs() > 0.2), "{polluted:?}");
    }

    /// The same corner cut straight from s along the top to s down the side.
    fn chamfered(s: f64, reach: f64, step: f64, noise: f64, seed: u32) -> Vec<DVec2> {
        let mut rand = Mulberry32::new(seed);
        let mut out = Vec::new();
        let mut x = -reach;
        while x < -s {
            out.push(DVec2::new(x, gauss(&mut rand) * noise));
            x += step;
        }
        let len = s * SQRT_2;
        let mut t = 0.0;
        while t <= len {
            let d = gauss(&mut rand) * noise;
            out.push(DVec2::new(-s + (t / len) * s + d * FRAC_1_SQRT_2, -(t / len) * s + d * FRAC_1_SQRT_2));
            t += step;
        }
        let mut y = -s;
        while y > -reach {
            out.push(DVec2::new(gauss(&mut rand) * noise, y));
            y -= step;
        }
        out
    }

    #[test]
    fn reads_a_45_chamfer_s_setback_with_a_sigma_and_none_on_a_sharp_corner() {
        let c = estimate_chamfer(&chamfered(1.5, 8.0, 0.05, 0.02, 8), N1, N2, 8.0, None).expect("a chamfer");
        assert!((c.radius - 1.5).abs() < 0.03, "{c:?}");
        assert!(c.sigma > 0.0);
        assert!(c.sigma < 0.03);
        assert!(estimate_chamfer(&chamfered(0.0, 8.0, 0.05, 0.02, 8), N1, N2, 8.0, None).is_none());
    }

    #[test]
    fn combines_readings_by_their_median_and_the_larger_of_their_sigma_and_spread() {
        // Not in ScanRuler's tests: `combineReadings` worked through by hand.
        let r = |radius: f64, sigma: f64| FilletReading { radius, points: 10, rms: 0.01, sigma };
        let one = combine_readings(&[r(2.0, 0.01), r(2.2, 0.01), r(2.1, 0.01)]).expect("a reading");
        assert_eq!(one.radius, 2.1);
        assert_eq!(one.points, 30);
        // The spread: √(0.02 / 2 / 3) ≈ 0.0577, over the combined fit's 0.0058.
        assert!((one.sigma - (0.02f64 / 2.0 / 3.0).sqrt()).abs() < 1e-12, "{one:?}");
        assert!(combine_readings(&[]).is_none());
    }
}
