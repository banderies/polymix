// SPDX-License-Identifier: AGPL-3.0-only
//! How a revolve's profiles stand to its axis: how far they reach along it (what a coil's pitch
//! has to clear), whether they touch or cross it (what a coil cannot have), and where a profile
//! runs across it (what a plain revolve cannot turn). Ported from ScanRuler
//! (`plugins/reverse/core/profiles.ts` profileSpan, profilesMeetAxis, profileAcrossAxis), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! ScanRuler reads the profiles as loops of lifted pieces sampled along their length; here a
//! sketch's regions are lifted the same way, each loop sampled finely enough that an arc's
//! chords stand inside it by a hair.

use glam::DVec3;
use solidcraft_geom::Region2;

use crate::frame::SectionFrame;

/// A profile lifted into the part: its outer loop and its holes, as points in order round each.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct LiftedProfile {
    pub outer: Vec<DVec3>,
    pub holes: Vec<Vec<DVec3>>,
}

/// How finely a loop is sampled to read it against an axis, mm.
const SAMPLE_MM: f64 = 1e-3;

/// A sketch's regions lifted into the part through its frame.
pub fn lift_profiles(frame: &SectionFrame, regions: &[Region2]) -> Vec<LiftedProfile> {
    let lift = |l: &solidcraft_geom::Loop2| -> Vec<DVec3> { l.polyline(SAMPLE_MM).iter().map(|p| frame.to_world(glam::DVec2::new(p.x, p.y))).collect() };
    regions.iter().map(|r| LiftedProfile { outer: lift(&r.outer), holes: r.holes.iter().map(lift).collect() }).collect()
}

/// How far profiles reach along an axis through `origin` along unit `dir`, from the nearest
/// point to the furthest — what a coil's pitch has to clear.
pub fn profile_span(profiles: &[LiftedProfile], origin: DVec3, dir: DVec3) -> (f64, f64) {
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for p in profiles {
        for q in p.outer.iter().chain(p.holes.iter().flatten()) {
            let along = (*q - origin).dot(dir);
            min = min.min(along);
            max = max.max(along);
        }
    }
    (min, max)
}

/// The plane's normal of a loop by Newell's sum, sound for any loop in it.
fn newell(outer: &[DVec3]) -> Option<DVec3> {
    let mut n = DVec3::ZERO;
    for i in 0..outer.len() {
        let a = outer[i];
        let b = outer[(i + 1) % outer.len()];
        n += DVec3::new((a.y - b.y) * (a.z + b.z), (a.z - b.z) * (a.x + b.x), (a.x - b.x) * (a.y + b.y));
    }
    n.try_normalize()
}

/// Whether any of the profiles touches or crosses an axis through `origin` along unit `dir` —
/// what a coil cannot have, every turn of it passing through the axis. An axis in the profile's
/// plane has to keep to one side of the profile, clear of it; one piercing the plane has to
/// pierce it outside the profile, or in a hole of it.
pub fn profiles_meet_axis(profiles: &[LiftedProfile], origin: DVec3, dir: DVec3, tol: f64) -> bool {
    for p in profiles {
        let outer = &p.outer;
        let Some(normal) = newell(outer) else { continue };
        let across = normal.dot(dir);
        let off = |q: DVec3| q - origin;
        if across.abs() < 1e-9 {
            // Along the plane: beside it, clear; in it, on one side of the profile.
            if off(outer[0]).dot(normal).abs() > tol {
                continue;
            }
            let side = normal.cross(dir);
            let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
            for q in outer {
                let d = off(*q).dot(side);
                lo = lo.min(d);
                hi = hi.max(d);
            }
            if lo <= tol && hi >= -tol {
                return true;
            }
            continue;
        }
        // Through the plane: where it pierces it, inside the profile or not.
        let pierce = origin + dir * (off(outer[0]).dot(normal) / across);
        let seed = if normal.x.abs() < 0.9 { DVec3::X } else { DVec3::Y };
        let u = normal.cross(seed).try_normalize().unwrap_or(DVec3::X);
        let v = normal.cross(u);
        let flat = |q: DVec3| {
            let d = q - pierce;
            (d.dot(u), d.dot(v))
        };
        let rim = where_of(outer, &flat, tol);
        if rim == Where::On {
            return true;
        }
        if rim == Where::Out {
            continue;
        }
        let holes: Vec<Where> = p.holes.iter().map(|h| where_of(h, &flat, tol)).collect();
        if holes.contains(&Where::On) || !holes.contains(&Where::In) {
            return true;
        }
    }
    false
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Where {
    On,
    In,
    Out,
}

/// Whether a loop runs through the piercing point (the flat frame's zero), winds round it, or
/// passes it by.
fn where_of(lp: &[DVec3], flat: &dyn Fn(DVec3) -> (f64, f64), tol: f64) -> Where {
    let mut inside = false;
    let n = lp.len();
    if n == 0 {
        return Where::Out;
    }
    let mut j = n - 1;
    for i in 0..n {
        let (xi, yi) = flat(lp[i]);
        let (xj, yj) = flat(lp[j]);
        let ex = xj - xi;
        let ey = yj - yi;
        let l2 = ex * ex + ey * ey;
        let t = if l2 > 0.0 { (-(xi * ex + yi * ey) / l2).clamp(0.0, 1.0) } else { 0.0 };
        if (xi + ex * t).hypot(yi + ey * t) <= tol {
            return Where::On;
        }
        if (yi > 0.0) != (yj > 0.0) && 0.0 < xi - (yi * ex) / ey {
            inside = !inside;
        }
        j = i;
    }
    if inside {
        Where::In
    } else {
        Where::Out
    }
}

/// Where a profile runs across a revolve's axis, or `None` where every profile keeps to one side
/// of it — what a plain revolve cannot turn. Ending on the axis — a pole, as a sphere's arc
/// does — or running along it is fine. An axis in the profile's plane: how far the profile
/// reaches over it on the side it reaches less (mm) and the point that does; an axis piercing
/// the plane inside the profile: that point, and 0.
pub fn profile_across_axis(profiles: &[LiftedProfile], origin: DVec3, dir: DVec3, tol: f64) -> Option<(f64, DVec3)> {
    for p in profiles {
        let outer = &p.outer;
        let Some(normal) = newell(outer) else { continue };
        if outer.is_empty() {
            continue;
        }
        let off = |q: DVec3| q - origin;
        if normal.dot(dir).abs() < 1e-9 {
            if off(outer[0]).dot(normal).abs() > tol {
                continue;
            }
            let side = normal.cross(dir);
            let (mut lo, mut hi) = (0.0, 0.0);
            let (mut lo_at, mut hi_at) = (outer[0], outer[0]);
            for q in outer {
                let d = off(*q).dot(side);
                if d < lo {
                    lo = d;
                    lo_at = *q;
                }
                if d > hi {
                    hi = d;
                    hi_at = *q;
                }
            }
            if lo < -tol && hi > tol {
                return Some(if -lo <= hi { (-lo, lo_at) } else { (hi, hi_at) });
            }
            continue;
        }
        // Piercing the plane: inside the profile, and in none of its holes.
        if profiles_meet_axis(std::slice::from_ref(p), origin, dir, tol) {
            let pierce = origin + dir * (off(outer[0]).dot(normal) / normal.dot(dir));
            return Some((0.0, pierce));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    // ScanRuler's `reCoil.test.ts`: how far profiles reach along an axis, and whether they meet
    // it.
    use super::*;
    use crate::regions::find_regions;
    use crate::sketch::{Shape, Sketch};
    use glam::DVec2;

    /// The XZ plane, seen from −Y: the sheet's X is the part's X, its Y the part's Z.
    fn xz() -> SectionFrame {
        SectionFrame { origin: DVec3::ZERO, normal: DVec3::new(0.0, -1.0, 0.0), u: DVec3::X, v: DVec3::Z }
    }

    fn rect(a: [f64; 2], b: [f64; 2]) -> Vec<LiftedProfile> {
        let mut s = Sketch::default();
        s.add_rectangle(DVec2::new(a[0], a[1]), DVec2::new(b[0], b[1]));
        let regions: Vec<Region2> = find_regions(&s).into_iter().map(|r| r.region).collect();
        lift_profiles(&xz(), &regions)
    }

    #[test]
    fn measures_how_far_profiles_reach_along_an_axis_and_whether_they_meet_it() {
        let clear = rect([8.0, -1.0], [12.0, 2.0]);
        let (min, max) = profile_span(&clear, DVec3::ZERO, DVec3::Z);
        assert!((min + 1.0).abs() < 1e-9 && (max - 2.0).abs() < 1e-9, "{min} {max}");
        // The Z axis runs in the sketch's plane, clear of it, on it, across it.
        assert!(!profiles_meet_axis(&clear, DVec3::ZERO, DVec3::Z, 1e-6));
        assert!(profiles_meet_axis(&rect([0.0, -1.0], [4.0, 1.0]), DVec3::ZERO, DVec3::Z, 1e-6));
        assert!(profiles_meet_axis(&rect([-1.0, -1.0], [3.0, 1.0]), DVec3::ZERO, DVec3::Z, 1e-6));
        // An axis through the plane: outside the profile, inside it, inside a hole.
        assert!(!profiles_meet_axis(&clear, DVec3::ZERO, DVec3::Y, 1e-6));
        assert!(profiles_meet_axis(&clear, DVec3::new(10.0, 0.0, 0.0), DVec3::Y, 1e-6));
        let mut ring = Sketch::default();
        ring.add_shape(Shape::Circle { c: DVec2::new(10.0, 0.0), r: 3.0 });
        ring.add_shape(Shape::Circle { c: DVec2::new(10.0, 0.0), r: 1.0 });
        let washer: Vec<Region2> = find_regions(&ring).into_iter().filter(|r| !r.holes.is_empty()).map(|r| r.region).collect();
        let annulus = lift_profiles(&xz(), &washer);
        assert!(!profiles_meet_axis(&annulus, DVec3::new(10.0, 0.0, 0.0), DVec3::Y, 1e-6));
        assert!(profiles_meet_axis(&annulus, DVec3::new(12.0, 0.0, 0.0), DVec3::Y, 1e-6));
        assert!(profiles_meet_axis(&annulus, DVec3::new(11.0, 0.0, 0.0), DVec3::Y, 1e-6));
    }

    #[test]
    fn names_where_a_profile_runs_across_its_axis() {
        // Clear of the axis, ending on it, and reaching 1 mm over it.
        assert_eq!(profile_across_axis(&rect([8.0, -1.0], [12.0, 1.0]), DVec3::ZERO, DVec3::Z, 1e-6), None);
        assert_eq!(profile_across_axis(&rect([0.0, -1.0], [4.0, 1.0]), DVec3::ZERO, DVec3::Z, 1e-6), None);
        let across = profile_across_axis(&rect([-1.0, -1.0], [3.0, 1.0]), DVec3::ZERO, DVec3::Z, 1e-6);
        assert!(across.is_some_and(|(mm, at)| (mm - 1.0).abs() < 1e-9 && (at.x + 1.0).abs() < 1e-9), "{across:?}");
    }
}
