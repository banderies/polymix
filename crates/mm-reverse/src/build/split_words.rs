// SPDX-License-Identifier: AGPL-3.0-only
//! What a step that leaves a body in pieces says, and what a cut by a plane that takes the
//! whole body says: which piece kept the body's id and which are new, with where each lies —
//! and, for a cut that leaves nothing, which side the body lies on and the pieces of it
//! elsewhere that the cut could have meant. Ported from ScanRuler
//! (`plugins/reverse/core/splitWords.ts`, and `build.ts` pieceSaid, saySplit, takesAll), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! A feature after a split that names the body works on the piece that kept the id alone; one
//! that names the wrong piece finds it out here rather than as "no closed solid".

use glam::DVec3;

use super::{dv3, resolve_plane, volume, Builder};
use crate::js_number::{js_round, js_string, to_precision_number};
use crate::model::{BodyId, Feature, FeatureKind, Part};

/// Two pieces within this share of the larger's volume are the same size: the id goes to the
/// one nearest where the body stood, not to whichever the kernel listed first.
pub const SPLIT_TIE: f64 = 0.01;

/// A piece of a body as the words name it.
#[derive(Debug, Clone, PartialEq)]
pub struct PieceSaid {
    pub id: BodyId,
    pub volume: f64,
    pub min: DVec3,
    pub max: DVec3,
}

/// How a box lies to a plane, along the side cut away.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Away,
    Kept,
    Across,
}

const AXES: [&str; 3] = ["x", "y", "z"];

/// To two decimals, as the words write it (a negative zero as zero).
fn r2(v: f64) -> String {
    js_string(js_round(v * 100.0) / 100.0 + 0.0)
}

fn at2(p: DVec3) -> String {
    format!("({}, {}, {})", r2(p.x), r2(p.y), r2(p.z))
}

/// A body id as ScanRuler writes it: the feature and a fraction.
pub fn id_words(id: BodyId) -> String {
    js_string(id.as_number())
}

/// The world axis a direction runs along.
fn axis_of(dir: DVec3) -> Option<usize> {
    (0..3).find(|i| dir[*i].abs() > 1.0 - 1e-9)
}

/// A box in words: "x 57.26…59.26, y -8.5…8.5, z 4…9.8".
pub fn box_words(min: DVec3, max: DVec3) -> String {
    (0..3).map(|i| format!("{} {}…{}", AXES[i], r2(min[i]), r2(max[i]))).collect::<Vec<_>>().join(", ")
}

/// Where a plane is: "at z 9.8", or through a point square to a direction.
pub fn plane_words(origin: DVec3, normal: DVec3) -> String {
    match axis_of(normal) {
        Some(i) => format!("at {} {}", AXES[i], r2(origin[i])),
        None => format!("through {} square to {}", at2(origin), at2(normal)),
    }
}

/// The side of a plane a direction points to: "x > -50".
pub fn side_words(origin: DVec3, dir: DVec3) -> String {
    match axis_of(dir) {
        Some(i) => format!("{} {} {}", AXES[i], if dir[i] > 0.0 { ">" } else { "<" }, r2(origin[i])),
        None => format!("the side {} points to, from the plane through {}", at2(dir), at2(origin)),
    }
}

/// How a box lies to a plane, along `away` (the side cut away).
pub fn side_of_box(min: DVec3, max: DVec3, origin: DVec3, away: DVec3) -> Side {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for c in 0..8 {
        let p = DVec3::new(if c & 1 != 0 { max.x } else { min.x }, if c & 2 != 0 { max.y } else { min.y }, if c & 4 != 0 { max.z } else { min.z });
        let d = (p - origin).dot(away);
        lo = lo.min(d);
        hi = hi.max(d);
    }
    if lo >= -1e-6 {
        Side::Away
    } else if hi <= 1e-6 {
        Side::Kept
    } else {
        Side::Across
    }
}

fn piece_words(p: &PieceSaid) -> String {
    format!("{}; {} mm³", box_words(p.min, p.max), js_string(to_precision_number(p.volume, 5)))
}

/// A step that left a body in pieces, in a sentence: `what` ("The cut at z 9.8"), the body's
/// name, the piece that kept the id, and the others — each new (`fresh`) or a piece the body
/// had before, given its id again. Where a new piece is the same size as the one that kept the
/// id, says the id went by where the body stood.
pub fn split_words(what: &str, name: &str, kept: &PieceSaid, others: &[(PieceSaid, bool)]) -> String {
    let n = others.len() + 1;
    let tie = others.iter().any(|(p, fresh)| *fresh && (p.volume - kept.volume).abs() <= SPLIT_TIE * p.volume.max(kept.volume));
    let rest: Vec<String> = others.iter().map(|(p, fresh)| format!("{} {} ({})", id_words(p.id), if *fresh { "is new" } else { "keeps its own" }, piece_words(p))).collect();
    format!(
        "{what} left {name} as {} bodies: {} keeps the id ({}); {}{}. A feature naming {} from here works on that piece alone (a Combine naming it takes them all).",
        if n == 2 { "two".to_string() } else { n.to_string() },
        id_words(kept.id),
        piece_words(kept),
        rest.join("; "),
        if tie { " — the two the same size within 1 %, so the id stays with the piece nearest where the body stood" } else { "" },
        id_words(kept.id),
    )
}

/// A cut by a plane that leaves nothing of its body, in a sentence: the body and where it lies,
/// the side cut away, and — where the body has pieces split off it (`family`, each with how it
/// lies to the plane, and `since`, the steps that split them) — each piece, and the ones the
/// cut would take something of: "cut 85, or keep +1".
pub fn takes_all_words(name: &str, body: &PieceSaid, origin: DVec3, away: DVec3, keep: i8, family: &[(PieceSaid, Side)], since: &str) -> String {
    let flip = if keep > 0 { "-1" } else { "+1" };
    let head = format!("The cut takes all of {name} (body {}, {}): it lies wholly on the side cut away ({})", id_words(body.id), box_words(body.min, body.max), side_words(origin, away));
    if family.is_empty() {
        return format!("{head} — keep {flip} keeps all of it, or move the plane.");
    }
    let i = axis_of(away);
    let place = |p: &PieceSaid| match i {
        Some(i) => format!("{} {}…{}", AXES[i], r2(p.min[i]), r2(p.max[i])),
        None => box_words(p.min, p.max),
    };
    let side_said = |s: Side| match s {
        Side::Away => "wholly on the side cut away",
        Side::Kept => "on the side kept",
        Side::Across => "across the plane",
    };
    let mut all: Vec<(&PieceSaid, Side)> = vec![(body, Side::Away)];
    all.extend(family.iter().map(|(p, s)| (p, *s)));
    let meant: Vec<String> = family.iter().filter(|(_, s)| *s != Side::Away).map(|(p, _)| id_words(p.id)).collect();
    let n = all.len();
    let listed: Vec<String> = all.iter().map(|(p, s)| format!("{} at {} ({})", id_words(p.id), place(p), side_said(*s))).collect();
    format!(
        "{head}. Since {since} it is {} bodies: {} — {}keep {flip}.",
        if n == 2 { "two".to_string() } else { n.to_string() },
        listed.join(", "),
        if meant.is_empty() { String::new() } else { format!("cut {}, or ", meant.join(" or ")) },
    )
}

/// How a step says a split it makes: a cut by a plane where the plane is, a Combine, any other
/// cut (ScanRuler's `saySplit` callers).
pub(crate) fn split_what(part: &Part, f: &Feature) -> String {
    match &f.kind {
        FeatureKind::CutPlane(x) => match resolve_plane(&x.plane, part) {
            Some((o, n)) => format!("The cut {}", plane_words(o, n)),
            None => "The cut".to_string(),
        },
        FeatureKind::Boolean(_) => "The combine".to_string(),
        _ => "The cut".to_string(),
    }
}

impl Builder {
    /// A body as the words name it: the name of the STEP part it came as, else of the feature
    /// that made it.
    pub(crate) fn body_words(&self, id: BodyId) -> String {
        if let Some(l) = self.body(id).and_then(|b| b.label.clone()) {
            return l;
        }
        self.track.name_of(id.maker()).map_or_else(|| id.to_string(), str::to_string)
    }

    /// A solid body standing, as the words of a split name it.
    pub(crate) fn piece_said(&self, id: BodyId) -> Option<PieceSaid> {
        let b = self.body(id)?;
        if b.surface {
            return None;
        }
        let bb = b.shape.outline_bounds();
        if bb.is_empty() {
            return None;
        }
        Some(PieceSaid { id, volume: volume(&b.shape).unwrap_or(0.0), min: dv3(bb.min), max: dv3(bb.max) })
    }

    /// A step that left a body in pieces (`ids`, settle's: the one that kept the id first)
    /// says so on its row: which kept the id and which are new, with where each lies.
    pub(crate) fn say_split(&mut self, ids: &[BodyId]) {
        if ids.len() < 2 {
            return;
        }
        let what = self.track.split_what.clone().unwrap_or_else(|| "The cut".to_string());
        let name = self.body_words(ids[0]);
        let Some(kept) = self.piece_said(ids[0]) else { return };
        let others: Vec<(PieceSaid, bool)> = ids[1..].iter().filter_map(|x| self.piece_said(*x).map(|p| (p, true))).collect();
        self.warn(split_words(&what, &name, &kept, &others));
    }

    /// A cut by a plane that leaves nothing of its body, in words: the body, the side cut away,
    /// and the pieces split off the body with how each lies to the plane — the one meant is
    /// likely among them (ScanRuler's `takesAll`). `normal` is the plane's, `keep` the side
    /// kept along it. For the Cut feature's build (`cut_plane.rs`) to fail by, where the cut
    /// leaves nothing.
    #[allow(dead_code)]
    pub(crate) fn takes_all(&self, body: BodyId, origin: DVec3, normal: DVec3, keep: i8) -> String {
        let away = normal * -(keep as f64).signum();
        let id = self.resolve(body);
        let family: Vec<(PieceSaid, Side)> = self.family_of(body).into_iter().filter(|x| *x != id).filter_map(|x| self.piece_said(x)).map(|p| {
            let s = side_of_box(p.min, p.max, origin, away);
            (p, s)
        }).collect();
        let mut makers: Vec<u32> = Vec::new();
        for (p, _) in &family {
            if !makers.contains(&p.id.maker()) {
                makers.push(p.id.maker());
            }
        }
        let since = makers.iter().map(|m| format!("{} (feature {m})", self.track.name_of(*m).unwrap_or("the feature"))).collect::<Vec<_>>().join(" and ");
        let whole = self.piece_said(id).unwrap_or(PieceSaid { id, volume: 0.0, min: DVec3::ZERO, max: DVec3::ZERO });
        takes_all_words(&self.body_words(id), &whole, origin, away, keep, &family, &since)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(id: BodyId, volume: f64, min: [f64; 3], max: [f64; 3]) -> PieceSaid {
        PieceSaid { id, volume, min: DVec3::from(min), max: DVec3::from(max) }
    }

    #[test]
    fn says_which_piece_kept_the_id_and_where_each_lies() {
        let kept = piece(BodyId::new(2), 45.0, [7.0, 0.0, 0.0], [10.0, 5.0, 3.0]);
        let other = piece(BodyId { feature: 3, island: 0 }, 30.0, [0.0, 0.0, 0.0], [2.0, 5.0, 3.0]);
        let s = split_words("The cut at y 5", "Extrude 2", &kept, &[(other, true)]);
        assert_eq!(s, "The cut at y 5 left Extrude 2 as two bodies: 2 keeps the id (x 7…10, y 0…5, z 0…3; 45 mm³); 3 is new (x 0…2, y 0…5, z 0…3; 30 mm³). A feature naming 2 from here works on that piece alone (a Combine naming it takes them all).");
        assert!(!s.contains("same size"));
        // Three pieces, the two legs alike.
        let a = piece(BodyId::new(2), 30.0, [0.0, 0.0, 0.0], [2.0, 5.0, 3.0]);
        let b = piece(BodyId::new(5), 30.0, [8.0, 0.0, 0.0], [10.0, 5.0, 3.0]);
        let c = piece(BodyId { feature: 5, island: 1 }, 24.0, [0.0, 7.0, 0.0], [4.0, 9.0, 3.0]);
        let s = split_words("The combine", "Extrude 2", &a, &[(b, true), (c, true)]);
        assert!(s.contains("The combine left Extrude 2 as 3 bodies: 2 keeps the id (x 0…2, y 0…5, z 0…3; 30 mm³); 5 is new (x 8…10, y 0…5, z 0…3; 30 mm³); 5.0009765625 is new (x 0…4, y 7…9, z 0…3; 24 mm³)"), "{s}");
        assert!(s.contains("the two the same size within 1 %, so the id stays with the piece nearest where the body stood"));
    }

    #[test]
    fn says_a_cut_that_takes_the_whole_body_and_the_pieces_it_could_have_meant() {
        let body = piece(BodyId::new(2), 30.0, [0.0, 0.0, 0.0], [2.0, 5.0, 3.0]);
        let s = takes_all_words("Extrude 2", &body, DVec3::new(5.0, 0.0, 0.0), DVec3::new(-1.0, 0.0, 0.0), 1, &[], "");
        assert_eq!(s, "The cut takes all of Extrude 2 (body 2, x 0…2, y 0…5, z 0…3): it lies wholly on the side cut away (x < 5) — keep -1 keeps all of it, or move the plane.");
        let other = piece(BodyId { feature: 3, island: 1 }, 30.0, [8.0, 0.0, 0.0], [10.0, 5.0, 3.0]);
        let s = takes_all_words("Extrude 2", &body, DVec3::new(5.0, 0.0, 0.0), DVec3::new(-1.0, 0.0, 0.0), 1, &[(other, Side::Kept)], "Cut 3 (feature 3)");
        assert_eq!(s, "The cut takes all of Extrude 2 (body 2, x 0…2, y 0…5, z 0…3): it lies wholly on the side cut away (x < 5). Since Cut 3 (feature 3) it is two bodies: 2 at x 0…2 (wholly on the side cut away), 3.0009765625 at x 8…10 (on the side kept) — cut 3.0009765625, or keep -1.");
    }

    #[test]
    fn says_planes_and_sides_by_their_axis() {
        assert_eq!(plane_words(DVec3::new(0.0, 0.0, 9.8), DVec3::Z), "at z 9.8");
        assert_eq!(plane_words(DVec3::new(1.0, 2.0, 3.0), DVec3::new(0.6, 0.8, 0.0)), "through (1, 2, 3) square to (0.6, 0.8, 0)");
        assert_eq!(side_words(DVec3::new(-50.0, 0.0, 0.0), DVec3::X), "x > -50");
        assert_eq!(side_of_box(DVec3::ZERO, DVec3::ONE, DVec3::new(2.0, 0.0, 0.0), DVec3::NEG_X), Side::Away);
        assert_eq!(side_of_box(DVec3::ZERO, DVec3::ONE, DVec3::new(0.5, 0.0, 0.0), DVec3::NEG_X), Side::Across);
    }
}
