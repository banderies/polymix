// SPDX-License-Identifier: AGPL-3.0-only
//! A sweep's route: a centreline in 3D given by its corner points and a bend radius, made a
//! chain of straight lines and tangent arcs. A bent tube's bends lie in planes of their own,
//! which no one sketch's path can hold. Pure numbers: the kernel takes the pieces, and the
//! profile is checked against the start where the sweep is built. Ported from ScanRuler
//! (`plugins/reverse/core/sweepRoute.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::js_fmt::{num, round_to};
use crate::loop_piece::{normalize3, LoopPiece};

/// A centreline in the part's frame, mm. Each inner point is a corner the route rounds with
/// a tangent arc: where the straights either side would meet, not a point on the arc.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SweepRoute {
    /// In order, either end first: the sweep starts from the end nearest the profile.
    pub points: Vec<DVec3>,
    /// The radius of every bend `radii` does not name, on the centreline.
    pub bend_radius: f64,
    /// One per inner point in order (points.len() − 2): that bend's own radius, or `None`
    /// for `bend_radius`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radii: Option<Vec<Option<f64>>>,
}

/// Why a route makes no pieces, in words for the user.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
#[error("{0}")]
pub struct RouteError(pub String);

/// A bend of the route as built: the index of its corner among the points given, and its
/// radius.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RouteBend {
    pub index: usize,
    pub radius: f64,
}

/// What a sweep follows: its pieces from the start, the point it starts at, the way it
/// leaves, and its bends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RoutePieces {
    pub pieces: Vec<LoopPiece>,
    pub start: DVec3,
    pub tangent: DVec3,
    pub bends: Vec<RouteBend>,
}

/// Points nearer than this are one point, mm.
const SAME_POINT_MM: f64 = 1e-3;
/// A corner turned by less than this is no bend, degrees: its arc would be a hair at any
/// radius a tube has.
const STRAIGHT_DEG: f64 = 0.1;
/// A corner turned by more than this folds back on itself, degrees.
const FOLD_DEG: f64 = 179.9;
/// What is left of a straight between two arcs under this is none: the arcs meet, mm.
const NO_STRAIGHT_MM: f64 = 1e-3;

/// `Math.round(x * 1000) / 1000 || 0`.
fn r3(x: f64) -> f64 {
    round_to(x, 3)
}

fn point_words(i: usize, p: DVec3) -> String {
    format!("points[{i}] ({}, {}, {})", num(r3(p.x)), num(r3(p.y)), num(r3(p.z)))
}

fn turn_of(a: DVec3, b: DVec3) -> f64 {
    a.dot(b).clamp(-1.0, 1.0).acos()
}

#[derive(Debug, Clone, Copy)]
struct Corner {
    index: usize,
    turn: f64,
    radius: f64,
    slope: f64,
    reach: f64,
}

/// A route as the pieces a sweep follows: straights and tangent arcs, run from the end
/// nearest `from`, with the point it starts at and the way it leaves; or why not. Repeated
/// points and corners with no turn are passed over. A bend that does not fit its straights
/// is refused, with the most it takes.
pub fn route_pieces(route: &SweepRoute, from: DVec3) -> Result<RoutePieces, RouteError> {
    let points = &route.points;
    let refuse = |s: String| Err(RouteError(s));
    if points.len() < 2 {
        return refuse("The route needs two points at least.".into());
    }
    if let Some(radii) = &route.radii {
        if radii.len() != points.len() - 2 {
            return refuse(format!(
                "The route has {} inner points and {} radii — one for each inner point, null for the bend radius.",
                points.len() - 2,
                radii.len()
            ));
        }
    }
    // Repeats go first, then corners with no turn, each read from the points kept: the index
    // a refusal says stays the one given.
    let mut kept: Vec<usize> = vec![0];
    for i in 1..points.len() {
        let last = kept[kept.len() - 1];
        if (points[i] - points[last]).length() >= SAME_POINT_MM {
            kept.push(i);
        }
    }
    if kept.len() < 2 {
        return refuse("The route’s points are all one point.".into());
    }
    if (points[0] - points[kept[kept.len() - 1]]).length() < SAME_POINT_MM {
        return refuse("The route ends where it starts — a route has two ends; a closed path is a sketch’s, drawn as arcs.".into());
    }
    let mut j = 1;
    while j + 1 < kept.len() {
        let into = normalize3(points[kept[j]] - points[kept[j - 1]]).unwrap_or(DVec3::ZERO);
        let out = normalize3(points[kept[j + 1]] - points[kept[j]]).unwrap_or(DVec3::ZERO);
        let turn = turn_of(into, out).to_degrees();
        if turn > FOLD_DEG {
            return refuse(format!("The route folds back on itself at {}.", point_words(kept[j], points[kept[j]])));
        }
        if turn < STRAIGHT_DEG {
            kept.remove(j);
        } else {
            j += 1;
        }
    }
    let at: Vec<DVec3> = kept.iter().map(|&i| points[i]).collect();
    let dirs: Vec<DVec3> = (1..at.len()).map(|j| normalize3(at[j] - at[j - 1]).unwrap_or(DVec3::ZERO)).collect();
    let runs: Vec<f64> = (1..at.len()).map(|j| (at[j] - at[j - 1]).length()).collect();
    // Per corner, how far its arc reaches back along each straight.
    let corners: Vec<Corner> = kept
        .iter()
        .enumerate()
        .map(|(j, &index)| {
            if j == 0 || j == kept.len() - 1 {
                return Corner { index, turn: 0.0, radius: 0.0, slope: 0.0, reach: 0.0 };
            }
            let turn = turn_of(dirs[j - 1], dirs[j]);
            let radius = route.radii.as_ref().and_then(|r| r.get(index - 1).copied().flatten()).unwrap_or(route.bend_radius);
            Corner { index, turn, radius, slope: (turn / 2.0).tan(), reach: radius * (turn / 2.0).tan() }
        })
        .collect();
    for j in 1..kept.len() - 1 {
        let c = corners[j];
        if !(c.radius > 0.0) {
            return refuse(format!(
                "The bend at {} needs a radius above 0 — a route’s corners are rounded, a sharp one has no section to carry round it.",
                point_words(c.index, at[j])
            ));
        }
        let before = runs[j - 1] - corners[j - 1].reach;
        let after = runs[j] - corners[j + 1].reach;
        if c.reach <= before.min(after) + NO_STRAIGHT_MM {
            continue;
        }
        let room = 0f64.max(before.min(after));
        let straight = if after < before {
            format!("straight to {}", point_words(kept[j + 1], at[j + 1]))
        } else {
            format!("straight from {}", point_words(kept[j - 1], at[j - 1]))
        };
        return refuse(format!(
            "The bend at {} does not fit: a radius of {} mm turning {}° takes {} mm of each straight, and the {} leaves it {} mm. It takes a radius of {} mm at most, its neighbours as they are.",
            point_words(c.index, at[j]),
            num(r3(c.radius)),
            num(r3(c.turn.to_degrees())),
            num(r3(c.reach)),
            straight,
            num(r3(room)),
            num((room / c.slope * 1000.0).floor() / 1000.0)
        ));
    }
    let mut pieces: Vec<LoopPiece> = Vec::new();
    let mut was = at[0];
    for j in 1..at.len() {
        let c = corners[j];
        let end = at[j] + dirs[j - 1] * -c.reach;
        if (end - was).length() > NO_STRAIGHT_MM {
            pieces.push(LoopPiece::Line { a: was, b: end });
            was = end;
        }
        if j == at.len() - 1 {
            break;
        }
        // The arc's middle: from the corner towards the centre along the bisector, as far as
        // the corner stands off the arc.
        let inward = normalize3(dirs[j] - dirs[j - 1]).unwrap_or(DVec3::ZERO);
        let exit = at[j] + dirs[j] * c.reach;
        pieces.push(LoopPiece::Arc { a: was, m: at[j] + inward * (c.radius * (1.0 / (c.turn / 2.0).cos() - 1.0)), b: exit });
        was = exit;
    }
    // An arc starts along the straight it rounds, had it any length left: so the way the
    // route leaves is the first straight's, at either end.
    let from_end = (at[at.len() - 1] - from).length() < (at[0] - from).length();
    let bends: Vec<RouteBend> = corners[1..corners.len() - 1].iter().map(|c| RouteBend { index: c.index, radius: c.radius }).collect();
    if from_end {
        let back: Vec<LoopPiece> = pieces.iter().rev().map(|p| p.reversed()).collect();
        return Ok(RoutePieces { pieces: back, start: at[at.len() - 1], tangent: dirs[dirs.len() - 1] * -1.0, bends });
    }
    Ok(RoutePieces { pieces, start: at[0], tangent: dirs[0], bends })
}

#[cfg(test)]
mod tests {
    //! Ported from ScanRuler's `plugins/reverse/tests/reSweepRoute.test.ts`, the route as
    //! pieces; the sweeps built through the kernel are the app's to test.
    use super::*;

    const AT0: DVec3 = DVec3::ZERO;

    fn v(x: f64, y: f64, z: f64) -> DVec3 {
        DVec3::new(x, y, z)
    }

    fn route(points: &[DVec3], bend_radius: f64) -> SweepRoute {
        SweepRoute { points: points.to_vec(), bend_radius, radii: None }
    }

    fn pieces(r: &SweepRoute, from: DVec3) -> RoutePieces {
        match route_pieces(r, from) {
            Ok(p) => p,
            Err(e) => panic!("{e}"),
        }
    }

    fn refused(r: &SweepRoute) -> String {
        match route_pieces(r, AT0) {
            Ok(_) => panic!("the route was not refused"),
            Err(e) => e.0,
        }
    }

    fn kinds(p: &[LoopPiece]) -> String {
        p.iter().map(|x| x.kind_name()).collect::<Vec<_>>().join(" ")
    }

    fn ends(p: &LoopPiece) -> (DVec3, DVec3) {
        match *p {
            LoopPiece::Line { a, b } | LoopPiece::Arc { a, b, .. } => (a, b),
            LoopPiece::Circle { center, .. } => (center, center),
        }
    }

    /// A centreline in three bend planes, each corner a square turn.
    fn three_planes() -> Vec<DVec3> {
        vec![v(0.0, 0.0, 0.0), v(0.0, 0.0, 50.0), v(50.0, 0.0, 50.0), v(50.0, 50.0, 50.0), v(50.0, 50.0, 100.0)]
    }

    #[test]
    fn makes_straights_and_tangent_arcs_each_joined_to_the_next() {
        let made = pieces(&route(&three_planes(), 10.0), AT0);
        assert_eq!(kinds(&made.pieces), "line arc line arc line arc line");
        assert_eq!(made.start, v(0.0, 0.0, 0.0));
        assert_eq!(made.tangent, v(0.0, 0.0, 1.0));
        for i in 1..made.pieces.len() {
            assert_eq!((ends(&made.pieces[i - 1]).1 - ends(&made.pieces[i]).0).length(), 0.0);
        }
        // The first bend: 10 back from the corner each way, its middle on the bisector 10 from
        // the centre at (10, 0, 40).
        let LoopPiece::Arc { a, m, b } = made.pieces[1] else { panic!("not an arc") };
        assert_eq!(a, v(0.0, 0.0, 40.0));
        assert!((b.x - 10.0).abs() < 1e-9 && b.y == 0.0 && (b.z - 50.0).abs() < 1e-9);
        assert!(((m - v(10.0, 0.0, 40.0)).length() - 10.0).abs() < 1e-9);
        assert!(((a - v(10.0, 0.0, 40.0)).length() - 10.0).abs() < 1e-9);
    }

    #[test]
    fn runs_from_the_end_nearest_the_profile() {
        let made = pieces(&route(&three_planes(), 10.0), v(50.0, 50.0, 99.0));
        assert_eq!(made.start, v(50.0, 50.0, 100.0));
        assert_eq!(made.tangent, v(-0.0, -0.0, -1.0));
        assert!(made.tangent.x.is_sign_negative() && made.tangent.y.is_sign_negative());
        assert_eq!(ends(&made.pieces[0]).0, v(50.0, 50.0, 100.0));
        assert_eq!(ends(&made.pieces[made.pieces.len() - 1]).1, v(0.0, 0.0, 0.0));
    }

    #[test]
    fn refuses_a_bend_its_straights_cannot_take_naming_the_point_and_the_most_it_takes() {
        let points = [v(0.0, 0.0, 0.0), v(0.0, 0.0, 20.0), v(30.0, 0.0, 20.0)];
        let said = refused(&route(&points, 25.0));
        assert!(said.contains("points[1] (0, 0, 20)"), "{said}");
        assert!(said.contains("straight from points[0]"), "{said}");
        assert!(said.contains("takes a radius of 20 mm at most"), "{said}");
        // The most it says fits, the arc using up the straight wholly.
        assert_eq!(kinds(&pieces(&route(&points, 20.0), AT0).pieces), "arc line");
        // Two corners sharing a straight: the second is told what the first leaves of it.
        let shared = [v(0.0, 0.0, 0.0), v(0.0, 0.0, 100.0), v(30.0, 0.0, 100.0), v(30.0, 0.0, 200.0)];
        let said = refused(&route(&shared, 20.0));
        assert!(said.contains("points[1]") && said.contains("takes a radius of 10 mm at most"), "{said}");
        assert_eq!(kinds(&pieces(&route(&shared, 15.0), AT0).pieces), "line arc arc line");
        // Each bend its own radius: 25 and 5 share the 30 and no more.
        let own = |radii: Vec<Option<f64>>| SweepRoute { points: shared.to_vec(), bend_radius: 1.0, radii: Some(radii) };
        assert_eq!(kinds(&pieces(&own(vec![Some(25.0), Some(5.0)]), AT0).pieces), "line arc arc line");
        assert!(refused(&own(vec![Some(25.0), Some(6.0)])).contains("takes a radius of 24 mm at most"));
        assert!(refused(&own(vec![Some(25.0)])).contains("2 inner points and 1 radii"));
    }

    #[test]
    fn passes_over_repeated_points_and_corners_with_no_turn_and_says_of_the_rest_what_is_wrong() {
        let plain = pieces(&route(&[v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), v(0.0, 0.0, 20.0), v(0.0, 0.0, 20.0), v(0.0, 0.0, 40.0), v(30.0, 0.0, 40.0)], 5.0), AT0);
        assert_eq!(kinds(&plain.pieces), "line arc line");
        assert_eq!(ends(&plain.pieces[0]).1, v(0.0, 0.0, 35.0));
        assert_eq!(kinds(&pieces(&route(&[v(0.0, 0.0, 0.0), v(0.0, 0.0, 20.0), v(0.0, 0.0, 40.0)], 5.0), AT0).pieces), "line");
        assert!(refused(&route(&[v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0)], 5.0)).contains("all one point"));
        // The point said is as given, whatever was passed over before it.
        let folds = refused(&route(&[v(0.0, 0.0, 0.0), v(0.0, 0.0, 0.0), v(0.0, 0.0, 20.0), v(0.0, 0.0, 5.0)], 5.0));
        assert!(folds.contains("folds back") && folds.contains("points[2]"), "{folds}");
        assert!(refused(&route(&[v(0.0, 0.0, 0.0), v(0.0, 0.0, 20.0), v(30.0, 0.0, 20.0), v(0.0, 0.0, 0.0)], 5.0)).contains("ends where it starts"));
        assert!(refused(&route(&[v(0.0, 0.0, 0.0), v(0.0, 0.0, 20.0), v(30.0, 0.0, 20.0)], 0.0)).contains("needs a radius above 0"));
    }
}
