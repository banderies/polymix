// SPDX-License-Identifier: AGPL-3.0-only
//! The Cut by plane feature's build: a body cut at a plane, the side the plane's normal points
//! to kept — or the other. The plane may be any plane reference, its offset a coordinate plane's
//! coordinate or a distance along its normal, driven by an expression like any other number. A
//! cut that leaves nothing says which side the body lies on, and which of the pieces split off
//! it the cut may have meant. Ported from ScanRuler (`plugins/reverse/core/resolve.ts` the
//! cutPlane step, `build.ts` applyToBody and takesAll, `splitWords.ts` takesAllWords), ©
//! Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use glam::DVec3;
use solidcraft_kernel as kernel;

use super::{describe_plane_ref, dv3, kerr, origins_after, plane_through, resolve_plane, volume, Builder};
use crate::model::{BodyId, CutPlaneFeature, Feature, Part, PlaneRef};

/// What a feature on a sheet that only a solid takes says (ScanRuler's build, for every kind but
/// a Pattern, a Split and a Combine).
pub(crate) const SHEET_NOT_SOLID: &str = "A sheet is not worked on as a solid — split a body with it, or thicken it first. A sheet takes a Pattern, a Split (by sheets or solids) and a Combine (trimmed by a solid, or sewn to a sheet).";

/// The label a Cut's or a Mirror's plane carries while its box has none chosen yet — ScanRuler's
/// draft holds no plane then, and its Create waits for one; a feature keeps a plane, so this
/// one stands in, and a build that meets it says that no plane is chosen.
pub const NO_PLANE: &str = "no plane chosen";

/// The plane a new Cut or Mirror opens with: none chosen yet.
pub fn no_plane() -> PlaneRef {
    let frame = crate::frame::SectionFrame::sketch_on(DVec3::ZERO, DVec3::Z, None).unwrap_or_else(|| crate::frame::SectionFrame { origin: DVec3::ZERO, normal: DVec3::Z, u: DVec3::X, v: DVec3::Y });
    PlaneRef::Fitted { frame, label: Some(NO_PLANE.into()) }
}

/// Whether a plane is one, not the stand-in for none chosen.
pub fn plane_chosen(r: &PlaneRef) -> bool {
    !matches!(r, PlaneRef::Fitted { label: Some(l), .. } if l == NO_PLANE)
}

pub(crate) fn build(b: &mut Builder, part: &Part, _f: &Feature, x: &CutPlaneFeature) -> Result<(), String> {
    if !plane_chosen(&x.plane) {
        return Err("No plane chosen — choose the plane to cut at.".into());
    }
    let (origin, normal) = resolve_plane(&x.plane, part).ok_or_else(|| format!("The plane this cut is made at is gone ({}).", describe_plane_ref(&x.plane, part)))?;
    let (id, state) = b.body_for(x.body)?;
    if state.surface {
        return Err(SHEET_NOT_SOLID.into());
    }
    let plane = plane_through(origin, normal)?;
    let (piece, tool) = kernel::half_space(&state.shape, &plane, x.keep >= 0).map_err(kerr)?;
    let Some(piece) = piece.filter(|p| volume(p).unwrap_or(0.0) > 1e-9) else {
        return Err(takes_all(b, part, x, id, &state.shape, origin, normal));
    };
    let own = b.own_origins(&tool);
    let origins = origins_after(&piece, &[(&state.shape, &state.origins), (&tool, &own)], b.step);
    let ids = b.settle(id, piece, origins, true)?;
    b.touched(ids);
    Ok(())
}

/// A body or a piece of one, as the words name it: its id and its box.
struct PieceSaid {
    id: BodyId,
    min: DVec3,
    max: DVec3,
}

impl PieceSaid {
    fn of(id: BodyId, body: &kernel::Body) -> PieceSaid {
        let bb = body.outline_bounds();
        PieceSaid { id, min: dv3(bb.min), max: dv3(bb.max) }
    }
}

/// How a box lies to a plane, along `away` (the side cut away).
#[derive(Clone, Copy, PartialEq, Eq)]
enum Side {
    Away,
    Kept,
    Across,
}

/// A cut by a plane that leaves nothing of its body, in words (ScanRuler's `takesAll`): the
/// body, the side cut away, and the pieces split off the body with how each lies to the plane —
/// the one meant is likely among them.
fn takes_all(b: &Builder, part: &Part, x: &CutPlaneFeature, id: BodyId, shape: &kernel::Body, origin: DVec3, normal: DVec3) -> String {
    let away = normal * -(x.keep as f64);
    let family: Vec<(PieceSaid, Side)> = b
        .family_of(x.body)
        .into_iter()
        .filter(|p| *p != id)
        .filter_map(|p| b.body(p).filter(|s| !s.surface).map(|s| PieceSaid::of(p, &s.shape)))
        .map(|p| {
            let side = side_of_box(&p, origin, away);
            (p, side)
        })
        .collect();
    let mut makers: Vec<u32> = Vec::new();
    for (p, _) in &family {
        if !makers.contains(&p.id.maker()) {
            makers.push(p.id.maker());
        }
    }
    let since = makers.iter().map(|m| format!("{} (feature {m})", part.feature(*m).map_or("the feature".to_string(), |f| f.name.clone()))).collect::<Vec<_>>().join(" and ");
    takes_all_words(&b.name_of(part, id), &PieceSaid::of(id, shape), origin, away, x.keep, &family, &since)
}

const AXES: [&str; 3] = ["x", "y", "z"];

/// A number to the hundredth, as the words write it (`r2`).
fn r2(v: f64) -> String {
    crate::js_number::js_string(crate::js_number::js_round(v * 100.0) / 100.0 + 0.0)
}

fn at2(p: DVec3) -> String {
    format!("({}, {}, {})", r2(p.x), r2(p.y), r2(p.z))
}

/// An id as ScanRuler writes it: the feature and a fraction.
fn id_words(id: BodyId) -> String {
    crate::js_number::js_string(id.as_number())
}

/// The world axis a direction runs along, or `None`.
fn axis_of(dir: DVec3) -> Option<usize> {
    (0..3).find(|i| dir[*i].abs() > 1.0 - 1e-9)
}

/// A box in words: "x 57.26…59.26, y -8.5…8.5, z 4…9.8".
fn box_words(p: &PieceSaid) -> String {
    (0..3).map(|i| format!("{} {}…{}", AXES[i], r2(p.min[i]), r2(p.max[i]))).collect::<Vec<_>>().join(", ")
}

/// The side of a plane a direction points to: "x > -50".
fn side_words(origin: DVec3, dir: DVec3) -> String {
    match axis_of(dir) {
        Some(i) => format!("{} {} {}", AXES[i], if dir[i] > 0.0 { ">" } else { "<" }, r2(origin[i])),
        None => format!("the side {} points to, from the plane through {}", at2(dir), at2(origin)),
    }
}

/// How a box lies to a plane, along `away`: wholly on the side cut away, wholly on the side
/// kept, or across.
fn side_of_box(p: &PieceSaid, origin: DVec3, away: DVec3) -> Side {
    let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
    for c in 0..8 {
        let q = DVec3::new(if c & 1 != 0 { p.max.x } else { p.min.x }, if c & 2 != 0 { p.max.y } else { p.min.y }, if c & 4 != 0 { p.max.z } else { p.min.z });
        let d = (q - origin).dot(away);
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

/// A cut by a plane that leaves nothing of its body, in a sentence (ScanRuler's
/// `takesAllWords`): the body and where it lies, the side cut away, and — where the body has
/// pieces split off it — each piece, and the ones the cut would take something of: "cut 85, or
/// keep +1".
fn takes_all_words(name: &str, body: &PieceSaid, origin: DVec3, away: DVec3, keep: i8, family: &[(PieceSaid, Side)], since: &str) -> String {
    let flip = if keep > 0 { "-1" } else { "+1" };
    let head = format!("The cut takes all of {name} (body {}, {}): it lies wholly on the side cut away ({})", id_words(body.id), box_words(body), side_words(origin, away));
    if family.is_empty() {
        return format!("{head} — keep {flip} keeps all of it, or move the plane.");
    }
    let i = axis_of(away);
    let where_ = |p: &PieceSaid| match i {
        Some(i) => format!("{} {}…{}", AXES[i], r2(p.min[i]), r2(p.max[i])),
        None => box_words(p),
    };
    let said = |s: Side| match s {
        Side::Away => "wholly on the side cut away",
        Side::Kept => "on the side kept",
        Side::Across => "across the plane",
    };
    let mut all: Vec<(&PieceSaid, Side)> = vec![(body, Side::Away)];
    all.extend(family.iter().map(|(p, s)| (p, *s)));
    let meant: Vec<String> = family.iter().filter(|(_, s)| *s != Side::Away).map(|(p, _)| id_words(p.id)).collect();
    let n = all.len();
    let count = if n == 2 { "two".to_string() } else { n.to_string() };
    let listed = all.iter().map(|(p, s)| format!("{} at {} ({})", id_words(p.id), where_(p), said(*s))).collect::<Vec<_>>().join(", ");
    let cut = if meant.is_empty() { String::new() } else { format!("cut {}, or ", meant.join(" or ")) };
    format!("{head}. Since {since} it is {count} bodies: {listed} — {cut}keep {flip}.")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn piece(id: u32, min: [f64; 3], max: [f64; 3]) -> PieceSaid {
        PieceSaid { id: BodyId::new(id), min: DVec3::from(min), max: DVec3::from(max) }
    }

    #[test]
    fn says_where_a_body_cut_away_whole_lies_and_which_piece_was_meant() {
        let body = piece(2, [0.0, 0.0, 0.0], [10.0, 10.0, 5.0]);
        let away = DVec3::new(0.0, 0.0, -1.0);
        let alone = takes_all_words("Extrude 2", &body, DVec3::new(0.0, 0.0, 9.8), away, 1, &[], "");
        assert_eq!(alone, "The cut takes all of Extrude 2 (body 2, x 0…10, y 0…10, z 0…5): it lies wholly on the side cut away (z < 9.8) — keep -1 keeps all of it, or move the plane.");
        let other = piece(85, [0.0, 0.0, 8.0], [10.0, 10.0, 12.0]);
        let with = takes_all_words("Extrude 2", &body, DVec3::new(0.0, 0.0, 9.8), away, 1, &[(other, Side::Across)], "Cut 5 (feature 5)");
        assert_eq!(with, "The cut takes all of Extrude 2 (body 2, x 0…10, y 0…10, z 0…5): it lies wholly on the side cut away (z < 9.8). Since Cut 5 (feature 5) it is two bodies: 2 at z 0…5 (wholly on the side cut away), 85 at z 8…12 (across the plane) — cut 85, or keep -1.");
    }

    #[test]
    fn tells_how_a_box_lies_to_a_plane() {
        let p = piece(1, [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        assert!(side_of_box(&p, DVec3::new(0.0, 0.0, 2.0), DVec3::Z) == Side::Kept);
        assert!(side_of_box(&p, DVec3::new(0.0, 0.0, -1.0), DVec3::Z) == Side::Away);
        assert!(side_of_box(&p, DVec3::new(0.0, 0.0, 0.5), DVec3::Z) == Side::Across);
    }
}
