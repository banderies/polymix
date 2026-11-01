// SPDX-License-Identifier: AGPL-3.0-only
//! Whether a smooth loft's skin crosses itself between its sections. Ported from ScanRuler
//! (`plugins/reverse/core/loftFold.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The kernel interpolates every rail through all the sections, so two sections close
//! together that disagree make it overshoot further on: a narrow face of the skin turns
//! inside out (its width goes through zero) and its neighbours cross. The kernel's own check
//! does not test faces crossing: the loft is answered valid, with volume — and the boolean
//! after it comes out empty or wrong. Read here by cutting the side faces across at one v each
//! (all of a smooth loft's faces share their v: the sections lie at the same v on every face),
//! a point every 1/8 of u, and looking whether the cut, laid in its plane, crosses itself. A
//! ruled loft is a face per span, each with a v of its own, and straight between its sections:
//! not read.
//!
//! ScanRuler reads the faces off Open CASCADE; here the kernel's faces come in through
//! `FoldFace` — what a face is, its parameter box, a point at (u, v) and whether (u, v) lies
//! outside its trimmed region — so the check runs on whatever kernel holds the loft.

use std::collections::HashSet;
use std::time::Instant;

use glam::{DVec2, DVec3};
use serde::{Deserialize, Serialize};

use crate::js_fmt::{fixed_num, num};
use crate::loop_piece::{normalize3, piece_points, Profile, PIECE_STEP};

/// How many cuts a fold check takes along the loft.
pub const FOLD_CUTS: usize = 40;
/// Points along a face in a cut: one every 1/8 of its u.
const FOLD_STEPS: usize = 8;
/// Planes within this of parallel are taken as one way, degrees.
const PARALLEL_DEG: f64 = 1.0;

/// A face's parameter box.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct UvBounds {
    pub u_min: f64,
    pub u_max: f64,
    pub v_min: f64,
    pub v_max: f64,
}

/// A face of the kernel's loft, as the fold check reads it.
pub trait FoldFace {
    /// The face's session hash: what `LoftFoldOptions::faces` names it by.
    fn hash(&self) -> u64;
    /// Whether the face is a plane — a cap, which the check leaves out.
    fn is_plane(&self) -> bool;
    fn uv_bounds(&self) -> UvBounds;
    /// The surface's point at (u, v), mm.
    fn point_on_surface(&self, u: f64, v: f64) -> DVec3;
    /// Whether (u, v) lies outside the face's trimmed region.
    fn is_out(&self, u: f64, v: f64) -> bool;
}

/// What the fold check is told.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LoftFoldOptions {
    /// The way across the cuts — the sections' common normal; without it the cuts' own, when
    /// they all agree within PARALLEL_DEG, else each cut's own plane.
    pub normal: Option<DVec3>,
    /// Only these faces, by hash — a loft's own on a body.
    pub faces: Option<HashSet<u64>>,
    /// Only the points inside each face — faces a later step trimmed.
    pub trimmed: bool,
}

/// A cut that crosses itself: its index and its height along the fold's normal (`None`
/// without one).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FoldedCut {
    pub cut: usize,
    pub height: Option<f64>,
}

/// Where a smooth loft folds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct LoftFold {
    /// Where the first cut that crosses itself does so, mm.
    pub at: DVec3,
    /// The way the cuts were laid flat, when it was one for all.
    pub normal: Option<DVec3>,
    /// Each cut that crosses itself.
    pub folded: Vec<FoldedCut>,
    /// How many cuts were read.
    pub of: usize,
    /// What the check took, ms.
    pub ms: f64,
}

/// The plane's normal of a loop of points: Newell's sum, the loop's area vector — `None` for
/// a loop with no area.
fn loop_normal(points: &[DVec3]) -> Option<DVec3> {
    let mut n = DVec3::ZERO;
    for i in 0..points.len() {
        let a = points[i];
        let b = points[(i + 1) % points.len()];
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    normalize3(n)
}

/// A section's outline as points, its middle and its plane's normal.
fn section_of(profile: &Profile) -> (DVec3, Option<DVec3>) {
    let points: Vec<DVec3> = profile.outer.iter().flat_map(|p| piece_points(p, PIECE_STEP)).collect();
    let middle = points.iter().fold(DVec3::ZERO, |s, p| s + *p) * (1.0 / points.len().max(1) as f64);
    (middle, loop_normal(&points))
}

fn parallel(a: DVec3, b: DVec3) -> bool {
    a.dot(b).abs() >= PARALLEL_DEG.to_radians().cos()
}

/// The sections' common normal — `None` when they do not lie in parallel planes (radial
/// stations), and the cuts are laid in planes of their own.
pub fn sections_normal(profiles: &[Profile]) -> Option<DVec3> {
    let normals: Vec<Option<DVec3>> = profiles.iter().map(|p| section_of(p).1).collect();
    let first = (*normals.first()?)?;
    if normals.iter().any(|n| !n.is_some_and(|n| parallel(n, first))) {
        return None;
    }
    Some(first)
}

/// The first point where two segments of a cut laid flat cross, not at a face's end (where
/// neighbours meet), or `None`.
fn crossing(lines: &[Vec<DVec3>], normal: DVec3) -> Option<DVec3> {
    let e1 = normalize3(normal.cross(if normal.x.abs() < 0.9 { DVec3::X } else { DVec3::Y })).unwrap_or(DVec3::X);
    let e2 = normal.cross(e1);
    let flat: Vec<Vec<DVec2>> = lines.iter().map(|line| line.iter().map(|p| DVec2::new(p.dot(e1), p.dot(e2))).collect()).collect();
    let mut lo = DVec2::splat(f64::INFINITY);
    let mut hi = DVec2::splat(f64::NEG_INFINITY);
    for p in flat.iter().flatten() {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    let size = 1e-9f64.max(hi.x - lo.x).max(hi.y - lo.y);
    let eps = 1e-12 * size * size;
    let end_tol = 1e-4 * size;
    let ends: Vec<DVec2> = flat.iter().filter(|l| !l.is_empty()).flat_map(|l| [l[0], l[l.len() - 1]]).collect();
    let orient = |a: DVec2, b: DVec2, c: DVec2| (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
    let mut segments: Vec<(usize, usize)> = Vec::new();
    for (i, line) in flat.iter().enumerate() {
        for j in 0..line.len().saturating_sub(1) {
            segments.push((i, j));
        }
    }
    for s in 0..segments.len() {
        let (al, aat) = segments[s];
        let (p1, p2) = (flat[al][aat], flat[al][aat + 1]);
        for &(bl, bat) in &segments[s + 1..] {
            if al == bl && aat.abs_diff(bat) < 2 {
                continue;
            }
            let (q1, q2) = (flat[bl][bat], flat[bl][bat + 1]);
            let d1 = orient(q1, q2, p1);
            let d2 = orient(q1, q2, p2);
            let d3 = orient(p1, p2, q1);
            let d4 = orient(p1, p2, q2);
            if !((d1 > eps && d2 < -eps) || (d1 < -eps && d2 > eps)) {
                continue;
            }
            if !((d3 > eps && d4 < -eps) || (d3 < -eps && d4 > eps)) {
                continue;
            }
            let f = d1 / (d1 - d2);
            let x = p1 + (p2 - p1) * f;
            if ends.iter().any(|e| (*e - x).length() <= end_tol) {
                continue;
            }
            let a = lines[al][aat];
            let b = lines[al][aat + 1];
            return Some(a + (b - a) * f);
        }
    }
    None
}

/// Where a smooth loft's side faces — those not flat, of `options.faces` when given — cross
/// themselves: FOLD_CUTS cuts across them at one v each, between the ends, each laid flat (see
/// `LoftFoldOptions::normal`) and looked over for two of its pieces crossing. `None` when no
/// cut crosses itself, or there is no curved face to cut.
pub fn loft_fold<F: FoldFace>(all_faces: &[F], options: &LoftFoldOptions) -> Option<LoftFold> {
    let began = Instant::now();
    let faces: Vec<(&F, UvBounds)> = all_faces
        .iter()
        .filter(|f| options.faces.as_ref().is_none_or(|set| set.contains(&f.hash())) && !f.is_plane())
        .map(|f| (f, f.uv_bounds()))
        .collect();
    if faces.is_empty() {
        return None;
    }
    let v_lo = faces.iter().map(|x| x.1.v_min).fold(f64::INFINITY, f64::min);
    let v_hi = faces.iter().map(|x| x.1.v_max).fold(f64::NEG_INFINITY, f64::max);
    if !(v_hi > v_lo) {
        return None;
    }
    let mut cuts: Vec<Vec<Vec<DVec3>>> = Vec::with_capacity(FOLD_CUTS);
    for i in 0..FOLD_CUTS {
        let v = v_lo + ((i as f64 + 0.5) / FOLD_CUTS as f64) * (v_hi - v_lo);
        let mut lines: Vec<Vec<DVec3>> = Vec::new();
        for (f, uv) in &faces {
            if v < uv.v_min || v > uv.v_max {
                continue;
            }
            let mut line: Vec<DVec3> = Vec::new();
            for j in 0..=FOLD_STEPS {
                let u = uv.u_min + (j as f64 / FOLD_STEPS as f64) * (uv.u_max - uv.u_min);
                if options.trimmed && f.is_out(u, v) {
                    if line.len() > 1 {
                        lines.push(std::mem::take(&mut line));
                    }
                    line.clear();
                    continue;
                }
                line.push(f.point_on_surface(u, v));
            }
            if line.len() > 1 {
                lines.push(line);
            }
        }
        cuts.push(lines);
    }
    // Each cut's own plane: its pieces' area vector about its middle — the side faces run
    // round the loft one way, so the pieces of a cut add up as a loop's do.
    let own_normal = |lines: &Vec<Vec<DVec3>>| -> Option<DVec3> {
        let points: Vec<DVec3> = lines.iter().flatten().copied().collect();
        if points.len() < 3 {
            return None;
        }
        let middle = points.iter().fold(DVec3::ZERO, |s, p| s + *p) * (1.0 / points.len() as f64);
        let mut area = DVec3::ZERO;
        for line in lines {
            for j in 0..line.len().saturating_sub(1) {
                area += (line[j] - middle).cross(line[j + 1] - middle);
            }
        }
        normalize3(area)
    };
    let mut normal = options.normal;
    let own: Vec<Option<DVec3>> = cuts.iter().map(own_normal).collect();
    if normal.is_none() {
        let known: Vec<DVec3> = own.iter().flatten().copied().collect();
        if !known.is_empty() && known.iter().all(|n| parallel(*n, known[0])) {
            normal = normalize3(known.iter().fold(DVec3::ZERO, |s, n| s + if n.dot(known[0]) < 0.0 { -*n } else { *n }));
        }
    }
    let mut folded: Vec<FoldedCut> = Vec::new();
    let mut at: Option<DVec3> = None;
    for (i, lines) in cuts.iter().enumerate() {
        let Some(n) = normal.or(own[i]) else { continue };
        if lines.is_empty() {
            continue;
        }
        let Some(x) = crossing(lines, n) else { continue };
        if at.is_none() {
            at = Some(x);
        }
        folded.push(FoldedCut { cut: i, height: normal.map(|nn| x.dot(nn)) });
    }
    let at = at?;
    Some(LoftFold { at, normal, folded, of: FOLD_CUTS, ms: began.elapsed().as_secs_f64() * 1000.0 })
}

/// `Number(x.toFixed(1))`, written as JavaScript writes a number.
fn one(x: f64) -> String {
    num(fixed_num(x, 1))
}

/// The coordinate axis a direction runs along, if it does.
fn axis_of(normal: Option<DVec3>) -> Option<usize> {
    let n = normal?;
    (0..3).find(|&k| n[k].abs() > 1.0 - 1e-6)
}

const XYZ: [&str; 3] = ["x", "y", "z"];

/// "z 9.5 near (18.9, -3.2)" across a coordinate axis, else "near (x, y, z)".
fn where_words(at: DVec3, normal: Option<DVec3>) -> String {
    match axis_of(normal) {
        None => format!("near ({}, {}, {})", one(at.x), one(at.y), one(at.z)),
        Some(axis) => {
            let others: Vec<String> = (0..3).filter(|&k| k != axis).map(|k| one(at[k])).collect();
            format!("{} {} near ({})", XYZ[axis], one(at[axis]), others.join(", "))
        }
    }
}

/// The stretches of the loft that fold, by height — "z 9.3–17.6 and 35.3–36.6" — or by cut,
/// "cuts 2–13 of 40".
fn stretch_words(fold: &LoftFold) -> String {
    let mut runs: Vec<Vec<FoldedCut>> = Vec::new();
    for f in &fold.folded {
        match runs.last_mut() {
            Some(last) if last.last().is_some_and(|l| l.cut + 1 == f.cut) => last.push(*f),
            _ => runs.push(vec![*f]),
        }
    }
    let axis = axis_of(fold.normal);
    let sign = match (axis, fold.normal) {
        (Some(a), Some(n)) if n[a] < 0.0 => -1.0,
        _ => 1.0,
    };
    let words: Vec<String> = runs
        .iter()
        .map(|r| {
            let (a, b) = (r[0], r[r.len() - 1]);
            match (a.height, b.height) {
                (Some(ha), Some(hb)) => {
                    let (x, y) = (ha * sign, hb * sign);
                    let (lo, hi) = if x <= y { (x, y) } else { (y, x) };
                    if r.len() == 1 {
                        one(lo)
                    } else {
                        format!("{}–{}", one(lo), one(hi))
                    }
                }
                _ => {
                    if r.len() == 1 {
                        format!("{}", a.cut + 1)
                    } else {
                        format!("{}–{}", a.cut + 1, b.cut + 1)
                    }
                }
            }
        })
        .collect();
    let list = if words.len() > 1 { format!("{} and {}", words[..words.len() - 1].join(", "), words[words.len() - 1]) } else { words.first().cloned().unwrap_or_default() };
    if fold.folded.first().is_none_or(|f| f.height.is_none()) {
        return format!("cut{} {} of {}", if fold.folded.len() == 1 { "" } else { "s" }, list, fold.of);
    }
    match axis {
        Some(a) => format!("{} {}", XYZ[a], list),
        None => format!("{list} mm along the loft"),
    }
}

/// Where a loft crosses itself, in words: "crosses itself at z 9.5 near (18.9, -3.2) (folded
/// over z 9.3–17.6 and 35.3–36.6)".
pub fn fold_words(fold: &LoftFold) -> String {
    format!("crosses itself at {} (folded over {})", where_words(fold.at, fold.normal), stretch_words(fold))
}

/// A section of a loft, for a refusal: its name and its profile.
#[derive(Debug, Clone, PartialEq)]
pub struct NamedSection {
    pub name: String,
    pub profile: Profile,
}

/// A smooth loft refused for folding, naming its sections: the closest pair, when it lies far
/// closer than the rest (under half their middle spacing) — two sections close together that
/// disagree are what makes the rails overshoot — and the pair the fold lies between, when that
/// is another.
pub fn loft_fold_refusal(fold: &LoftFold, sections: &[NamedSection]) -> String {
    let read: Vec<(DVec3, Option<DVec3>)> = sections.iter().map(|s| section_of(&s.profile)).collect();
    let profiles: Vec<Profile> = sections.iter().map(|s| s.profile.clone()).collect();
    let normal = sections_normal(&profiles);
    let gaps: Vec<f64> = (1..read.len())
        .map(|i| match normal {
            Some(n) => (read[i].0 - read[i - 1].0).dot(n).abs(),
            None => (read[i].0 - read[i - 1].0).length(),
        })
        .collect();
    // The pair the fold lies between: across their planes, or the pair it is nearest.
    let between = match normal {
        Some(n) => {
            let h = fold.at.dot(n);
            (0..gaps.len()).find(|&i| (read[i].0.dot(n) - h) * (read[i + 1].0.dot(n) - h) <= 0.0).unwrap_or(0)
        }
        None => {
            let far = |i: usize| (fold.at - read[i].0).length() + (fold.at - read[i + 1].0).length();
            (0..gaps.len()).fold(0, |best, i| if far(i) < far(best) { i } else { best })
        }
    };
    let mut sorted = gaps.clone();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let median = if sorted.is_empty() {
        f64::NAN
    } else if sorted.len() % 2 == 1 {
        sorted[(sorted.len() - 1) / 2]
    } else {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    };
    let close = (0..gaps.len()).fold(0, |best, i| if gaps[i] < gaps[best] { i } else { best });
    let name = |i: usize| sections.get(i).map_or("", |s| s.name.as_str());
    let pair = |i: usize| format!("{} and {}", name(i), name(i + 1));
    let head = |lies: &str| format!("The smooth loft crosses itself at {}{} (folded over {})", where_words(fold.at, fold.normal), lies, stretch_words(fold));
    if gaps.len() >= 2 && gaps[close] < 0.5 * median {
        let lies = if between != close { format!(", between {}", pair(between)) } else { String::new() };
        return format!("{}: {} are {} mm apart and differ — drop or move one, or make the loft ruled.", head(&lies), pair(close), one(gaps[close]));
    }
    format!(
        "{}: the skin between {} overshoots — the sections differ more than one smooth skin through them all can follow; add a section between them, or make the loft ruled.",
        head(""),
        pair(between)
    )
}

#[cfg(test)]
mod tests {
    //! The pure parts of ScanRuler's `plugins/reverse/tests/reLoftFold.test.ts`: the hump's
    //! sections, their common normal, and the refusal's words. The kernel's loft is Open
    //! CASCADE's there; here the fold check runs on faces given as plain surfaces.
    use super::*;
    use crate::loop_piece::LoopPiece;

    /// The sections' heights and front half-widths.
    const HUMP: [(f64, f64); 5] = [(8.0, 1.4), (21.0, 1.82), (24.0, 2.7), (33.0, 0.55), (38.0, 0.47)];

    /// The hump's outline as a profile: a front edge of half-width w at y 12, slanting sides
    /// to an upright rectangle's, an arc round the back.
    fn hump_profile(z: f64, w: f64) -> Profile {
        let p = |x: f64, y: f64| DVec3::new(x, y, z);
        Profile {
            outer: vec![
                LoopPiece::Line { a: p(-w, 12.0), b: p(w, 12.0) },
                LoopPiece::Line { a: p(w, 12.0), b: p(10.0, 6.0) },
                LoopPiece::Line { a: p(10.0, 6.0), b: p(10.0, 0.0) },
                LoopPiece::Arc { a: p(10.0, 0.0), m: p(0.0, -10.0), b: p(-10.0, 0.0) },
                LoopPiece::Line { a: p(-10.0, 0.0), b: p(-10.0, 6.0) },
                LoopPiece::Line { a: p(-10.0, 6.0), b: p(-w, 12.0) },
            ],
            holes: vec![],
        }
    }

    #[test]
    fn the_humps_sections_share_a_normal_and_tilted_ones_do_not() {
        let profiles: Vec<Profile> = HUMP.iter().map(|&(z, w)| hump_profile(z, w)).collect();
        let n = sections_normal(&profiles).expect("parallel sections");
        assert!((n.z.abs() - 1.0).abs() < 1e-9);
        // One section turned 10° about X: no common normal.
        let mut tilted = profiles.clone();
        let (s, c) = 10f64.to_radians().sin_cos();
        let turn = |p: DVec3| DVec3::new(p.x, c * p.y - s * p.z, s * p.y + c * p.z);
        tilted[2].outer = tilted[2]
            .outer
            .iter()
            .map(|q| match *q {
                LoopPiece::Line { a, b } => LoopPiece::Line { a: turn(a), b: turn(b) },
                LoopPiece::Arc { a, m, b } => LoopPiece::Arc { a: turn(a), m: turn(m), b: turn(b) },
                other => other,
            })
            .collect();
        assert_eq!(sections_normal(&tilted), None);
    }

    #[test]
    fn names_the_close_pair_and_where_it_folds() {
        // A fold as the kernel's hump loft has one: cuts 2–13 and 37–38 of 40 crossing, the
        // first at z 9.1.
        let sections: Vec<NamedSection> = HUMP.iter().map(|&(z, w)| NamedSection { name: format!("Hump z{}", num(z)), profile: hump_profile(z, w) }).collect();
        let span = 38.0 - 8.0;
        let height = |cut: usize| 8.0 + (cut as f64 + 0.5) / FOLD_CUTS as f64 * span;
        let mut folded: Vec<FoldedCut> = (1..=12).map(|cut| FoldedCut { cut, height: Some(height(cut)) }).collect();
        folded.extend((36..=37).map(|cut| FoldedCut { cut, height: Some(height(cut)) }));
        let fold = LoftFold { at: DVec3::new(-0.04, 13.2, height(1)), normal: Some(DVec3::Z), folded, of: FOLD_CUTS, ms: 1.0 };
        let said = loft_fold_refusal(&fold, &sections);
        assert_eq!(
            said,
            "The smooth loft crosses itself at z 9.1 near (0, 13.2), between Hump z8 and Hump z21 (folded over z 9.1–17.4 and 35.4–36.1): Hump z21 and Hump z24 are 3 mm apart and differ — drop or move one, or make the loft ruled."
        );
        assert_eq!(fold_words(&fold), "crosses itself at z 9.1 near (0, 13.2) (folded over z 9.1–17.4 and 35.4–36.1)");
        // Without a common normal the stretches go by cut.
        let by_cut = LoftFold { normal: None, folded: fold.folded.iter().map(|f| FoldedCut { cut: f.cut, height: None }).collect(), ..fold.clone() };
        assert_eq!(fold_words(&by_cut), "crosses itself at near (0, 13.2, 9.1) (folded over cuts 2–13 and 37–38 of 40)");
    }

    /// A side face given as a surface: a polyline through `path(v)` along u, from u 0 to 1.
    struct PathFace<P: Fn(f64) -> Vec<DVec2>> {
        hash: u64,
        path: P,
        out: Option<(f64, f64)>,
    }

    impl<P: Fn(f64) -> Vec<DVec2>> FoldFace for PathFace<P> {
        fn hash(&self) -> u64 {
            self.hash
        }
        fn is_plane(&self) -> bool {
            false
        }
        fn uv_bounds(&self) -> UvBounds {
            UvBounds { u_min: 0.0, u_max: 1.0, v_min: 0.0, v_max: 10.0 }
        }
        fn point_on_surface(&self, u: f64, v: f64) -> DVec3 {
            let path = (self.path)(v);
            let t = u * (path.len() - 1) as f64;
            let k = (t.floor() as usize).min(path.len() - 2);
            let p = path[k] + (path[k + 1] - path[k]) * (t - k as f64);
            DVec3::new(p.x, p.y, v)
        }
        fn is_out(&self, u: f64, _v: f64) -> bool {
            self.out.is_some_and(|(a, b)| u > a && u < b)
        }
    }

    #[test]
    fn finds_where_a_skin_crosses_itself_and_passes_one_that_does_not() {
        // Two side faces round a loop: a straight front from (−1, 0) to (1, 0), and a back
        // arching over it that sinks through the front above v = 5.
        let front = PathFace { hash: 1, path: |_v: f64| vec![DVec2::new(-1.0, 0.0), DVec2::new(1.0, 0.0)], out: None };
        let back = PathFace {
            hash: 2,
            path: |v: f64| {
                let y = if v < 5.0 { 1.0 } else { 1.0 - 2.0 * (v - 5.0) / 5.0 - 0.2 };
                vec![DVec2::new(1.0, 0.0), DVec2::new(0.3, 1.0), DVec2::new(-0.3, y), DVec2::new(-1.0, 0.0)]
            },
            out: None,
        };
        let faces: Vec<Box<dyn FoldFace>> = vec![Box::new(front), Box::new(back)];
        let fold = loft_fold(&faces, &LoftFoldOptions { normal: Some(DVec3::Z), ..Default::default() }).expect("the skin folds");
        assert!(fold.folded.len() > 5);
        assert!(fold.at.z > 5.0 && fold.at.z < 10.0, "{:?}", fold.at);
        assert!(fold.folded.iter().all(|f| f.height.is_some_and(|h| h > 5.0)));
        assert_eq!(fold.of, FOLD_CUTS);
        assert!(fold.ms < 200.0, "{} ms", fold.ms);
        // Without the normal, the cuts' own planes agree and give it.
        let own = loft_fold(&faces, &LoftFoldOptions::default()).expect("the skin folds");
        assert!(own.normal.is_some_and(|n| n.z.abs() > 0.999));
        // A sound skin — the back arching over the front all the way — folds nowhere.
        let sound: Vec<Box<dyn FoldFace>> = vec![
            Box::new(PathFace { hash: 1, path: |_v: f64| vec![DVec2::new(-1.0, 0.0), DVec2::new(1.0, 0.0)], out: None }),
            Box::new(PathFace { hash: 2, path: |_v: f64| vec![DVec2::new(1.0, 0.0), DVec2::new(0.3, 1.0), DVec2::new(-0.3, 1.0), DVec2::new(-1.0, 0.0)], out: None }),
        ];
        assert_eq!(loft_fold(&sound, &LoftFoldOptions { normal: Some(DVec3::Z), ..Default::default() }), None);
        assert_eq!(loft_fold(&sound, &LoftFoldOptions::default()), None);
        // Only the front, by hash: nothing crosses.
        let only = LoftFoldOptions { normal: Some(DVec3::Z), faces: Some([1u64].into_iter().collect()), trimmed: false };
        assert_eq!(loft_fold(&faces, &only), None);
        // The back trimmed where it sinks: nothing left to cross.
        let trimmed_back = PathFace {
            hash: 2,
            path: |v: f64| {
                let y = if v < 5.0 { 1.0 } else { 1.0 - 2.0 * (v - 5.0) / 5.0 - 0.2 };
                vec![DVec2::new(1.0, 0.0), DVec2::new(0.3, 1.0), DVec2::new(-0.3, y), DVec2::new(-1.0, 0.0)]
            },
            out: Some((0.3, 0.8)),
        };
        let trimmed: Vec<Box<dyn FoldFace>> = vec![Box::new(PathFace { hash: 1, path: |_v: f64| vec![DVec2::new(-1.0, 0.0), DVec2::new(1.0, 0.0)], out: None }), Box::new(trimmed_back)];
        assert_eq!(loft_fold(&trimmed, &LoftFoldOptions { normal: Some(DVec3::Z), faces: None, trimmed: true }), None);
    }

    impl FoldFace for Box<dyn FoldFace> {
        fn hash(&self) -> u64 {
            (**self).hash()
        }
        fn is_plane(&self) -> bool {
            (**self).is_plane()
        }
        fn uv_bounds(&self) -> UvBounds {
            (**self).uv_bounds()
        }
        fn point_on_surface(&self, u: f64, v: f64) -> DVec3 {
            (**self).point_on_surface(u, v)
        }
        fn is_out(&self, u: f64, v: f64) -> bool {
            (**self).is_out(u, v)
        }
    }
}
