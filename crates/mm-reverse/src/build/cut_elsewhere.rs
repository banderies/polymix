// SPDX-License-Identifier: AGPL-3.0-only
//! A cut that also takes material where it was not meant: an extent run on past the part, which
//! goes through the walls it was drawn for and then nicks a far corner of the body. Ported from
//! ScanRuler (`plugins/reverse/core/cutElsewhere.ts`, and `build.ts` cutMakers,
//! noteCutsElsewhere), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! Read off the topology alone: the faces the cut made, joined into groups by the edges they
//! share — each place the cut went through — and a group under `CUT_SHARE` of the largest, or a
//! cylinder's group on the axis of a larger one, named on the row. The build asks it after
//! every cut.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use glam::DVec3;

use super::{BodyState, Builder};
use crate::js_number::{js_round, js_string};
use crate::model::{BodyId, BodyOp, BooleanOp, Feature, FeatureId, FeatureKind};
use crate::topo::{face_distance, match_tolerance, EdgeInfo, FaceInfo, FaceKind};

/// A place a cut took material under this share of the largest place's area is likely not
/// meant.
const CUT_SHARE: f64 = 0.02;
/// A cylinder's place on the axis of another place's cylinder of the same radius is likely not
/// meant under this share of the other's area — two walls a bore goes through both are meant.
const COAXIAL_SHARE: f64 = 0.25;
/// How far off parallel two cylinders' axes may lean, radians, and how far apart they may lie,
/// mm, and still be one axis.
const COAXIAL_ANGLE: f64 = std::f64::consts::PI / 180.0;
const COAXIAL_MM: f64 = 0.05;
/// How many places are named, the smallest first.
const NAMED: usize = 3;

/// A place the cut likely did not mean: where, its area, and how far it lies from the rest of
/// the cut — along the axis of the bore it is on, where it is on one.
#[derive(Debug, Clone, PartialEq)]
pub struct CutElsewhere {
    pub at: DVec3,
    pub area: f64,
    pub off: f64,
    pub bore: Option<f64>,
}

struct Place {
    faces: Vec<usize>,
    area: f64,
    at: DVec3,
}

/// How far a point lies from the line through `at` along `dir`.
fn off_line(p: DVec3, at: DVec3, dir: DVec3) -> f64 {
    let l = if dir.length() > 0.0 { dir.length() } else { 1.0 };
    let d = p - at;
    let along = d.dot(dir) / l;
    (d.dot(d) - along * along).max(0.0).sqrt()
}

/// The places among the faces the cut made (`made`) that it likely did not mean, the smallest
/// first.
pub fn cuts_elsewhere(faces: &[FaceInfo], edges: &[EdgeInfo], made: impl Fn(&FaceInfo) -> bool) -> Vec<CutElsewhere> {
    let own: Vec<usize> = (0..faces.len()).filter(|i| made(&faces[*i])).collect();
    if own.len() < 2 {
        return Vec::new();
    }
    let index: HashMap<u32, usize> = own.iter().enumerate().map(|(j, i)| (faces[*i].hash, j)).collect();
    let mut parent: Vec<usize> = (0..own.len()).collect();
    fn find(p: &mut [usize], j: usize) -> usize {
        let mut r = j;
        while p[r] != r {
            r = p[r];
        }
        let mut k = j;
        while p[k] != r {
            let next = p[k];
            p[k] = r;
            k = next;
        }
        r
    }
    for e in edges {
        let js: Vec<usize> = e.faces.iter().filter_map(|h| index.get(h).copied()).collect();
        for a in 1..js.len() {
            let (x, y) = (find(&mut parent, js[a]), find(&mut parent, js[0]));
            parent[x] = y;
        }
    }
    let mut by_root: Vec<(usize, Vec<usize>)> = Vec::new();
    for (j, i) in own.iter().enumerate() {
        let r = find(&mut parent, j);
        match by_root.iter_mut().find(|(x, _)| *x == r) {
            Some((_, m)) => m.push(*i),
            None => by_root.push((r, vec![*i])),
        }
    }
    let places: Vec<Place> = by_root
        .into_iter()
        .map(|(_, members)| {
            let mut area = 0.0;
            let mut at = DVec3::ZERO;
            for i in &members {
                let fp = &faces[*i].fingerprint;
                area += fp.area;
                at += fp.centroid * fp.area;
            }
            let at = if area > 0.0 { at / area } else { faces[members[0]].fingerprint.centroid };
            Place { faces: members, area, at }
        })
        .collect();
    if places.len() < 2 {
        return Vec::new();
    }
    let mut largest = 0;
    for (i, p) in places.iter().enumerate() {
        if p.area > places[largest].area {
            largest = i;
        }
    }
    let cylinders = |p: &Place| -> Vec<&FaceInfo> { p.faces.iter().map(|i| &faces[*i]).filter(|f| f.fingerprint.kind == FaceKind::Cylinder && f.fingerprint.radius.is_some() && f.fingerprint.center.is_some()).collect() };
    let mut out = Vec::new();
    for (pi, p) in places.iter().enumerate() {
        if pi == largest {
            continue;
        }
        // On the axis of a larger place's bore, of its radius: how far on past the bore's end
        // — a full turn's length is its area over its girth, about its centroid on the axis.
        let mut bore: Option<(f64, f64)> = None;
        for c in cylinders(p) {
            for (qi, q) in places.iter().enumerate() {
                if qi == pi || !(p.area < COAXIAL_SHARE * q.area) {
                    continue;
                }
                for d in cylinders(q) {
                    let (cf, df) = (&c.fingerprint, &d.fingerprint);
                    let (Some(cr), Some(r), Some(cc), Some(dc)) = (cf.radius, df.radius, cf.center, df.center) else { continue };
                    if (cr - r).abs() > 1e-3 * r.max(1.0) {
                        continue;
                    }
                    let denom = cf.dir.length() * df.dir.length();
                    let cos = cf.dir.dot(df.dir).abs() / if denom > 0.0 { denom } else { 1.0 };
                    if cos.min(1.0).acos() > COAXIAL_ANGLE || off_line(cc, dc, df.dir) > COAXIAL_MM {
                        continue;
                    }
                    let l = if df.dir.length() > 0.0 { df.dir.length() } else { 1.0 };
                    let t = (p.at - df.centroid).dot(df.dir).abs() / l;
                    // The kernel keeps a whole bore as faces on one surface: the bore's length is
                    // the area of all of them over its girth.
                    let whole: f64 = cylinders(q).iter().filter(|o| crate::topo::on_one_surface(&o.fingerprint, df)).map(|o| o.fingerprint.area).sum::<f64>().max(df.area);
                    let half = if d.full_turn == Some(true) { whole / (2.0 * std::f64::consts::PI * r) / 2.0 } else { 0.0 };
                    bore = Some((r, (t - half).max(0.0)));
                }
            }
        }
        if let Some((radius, off)) = bore {
            out.push(CutElsewhere { at: p.at, area: p.area, off, bore: Some(radius) });
            continue;
        }
        if !(p.area < CUT_SHARE * places[largest].area) {
            continue;
        }
        let off = places[largest].faces.iter().map(|i| (p.at - faces[*i].fingerprint.centroid).length()).fold(f64::INFINITY, f64::min);
        out.push(CutElsewhere { at: p.at, area: p.area, off, bore: None });
    }
    out.sort_by(|a, b| a.area.total_cmp(&b.area));
    out
}

/// `x.toPrecision(p)` as text, for a figure under one.
fn to_precision_text(x: f64, p: usize) -> String {
    if x == 0.0 || !x.is_finite() {
        return js_string(x);
    }
    let e = x.abs().log10().floor() as i32;
    if e < -6 || e >= p as i32 {
        return js_string(crate::js_number::to_precision_number(x, p));
    }
    crate::js_number::to_fixed(x, (p as i32 - 1 - e).max(0) as usize)
}

/// The places a cut likely did not mean as its row says them, or `None` for none: "Holes also
/// cut at (20.35, −46.76, −68.83): 0.15 mm², 11.4 mm along the R2.25 bore's axis — meant?
/// Shorten the extent."
pub fn cut_elsewhere_words(name: &str, places: &[CutElsewhere], todo: &str) -> Option<String> {
    if places.is_empty() {
        return None;
    }
    let r2 = |x: f64| js_string(js_round(x * 100.0) / 100.0 + 0.0);
    let area = |a: f64| if a < 0.01 { to_precision_text(a, 2) } else { r2(a) };
    let said: Vec<String> = places
        .iter()
        .take(NAMED)
        .map(|p| {
            let off = js_string(js_round(p.off * 10.0) / 10.0);
            let place = match p.bore {
                Some(r) => format!("{off} mm along the R{} bore's axis", r2(r)),
                None => format!("{off} mm from the rest of the cut"),
            };
            format!("({}, {}, {}): {} mm², {place}", r2(p.at.x), r2(p.at.y), r2(p.at.z), area(p.area))
        })
        .collect();
    let more = if places.len() > NAMED { format!(", and {} more", places.len() - NAMED) } else { String::new() };
    Some(format!("{name} also cut at {}{more} — meant? {todo}.", said.join("; ")))
}

/// Which makers' faces a cutting step leaves: its own, or a Combine's tool's not already the
/// body's. `None` for a step that does not cut (ScanRuler's `cutMakers`), read before it is
/// played, while the tool still stands.
pub(crate) enum Mine {
    Own(FeatureId),
    Tool { tool: HashSet<i64>, into: HashSet<i64> },
}

impl Mine {
    fn of(&self, origin: i64) -> bool {
        match self {
            Mine::Own(id) => origin == *id as i64,
            Mine::Tool { tool, into } => tool.contains(&origin) && !into.contains(&origin),
        }
    }
}

pub(crate) fn cut_makers(b: &Builder, f: &Feature) -> Option<Mine> {
    let op = match &f.kind {
        FeatureKind::Hole(_) => return Some(Mine::Own(f.id)),
        FeatureKind::Boolean(c) => {
            if c.op != BooleanOp::Cut {
                return None;
            }
            let into_id = b.resolve(c.body);
            let into: HashSet<i64> = b.body(into_id).map_or(HashSet::new(), |s| s.origins.iter().copied().collect());
            let tool: HashSet<i64> = b.family_of(c.tool).into_iter().filter(|x| *x != into_id).filter_map(|x| b.body(x)).flat_map(|s| s.origins.iter().copied()).collect();
            return Some(Mine::Tool { tool, into });
        }
        FeatureKind::Extrude(x) => x.op,
        FeatureKind::Revolve(x) => x.op,
        FeatureKind::Extract(x) => x.op,
        FeatureKind::Loft(x) => x.op,
        FeatureKind::Sweep(x) => x.op,
        FeatureKind::Thicken(x) => x.op,
        _ => return None,
    };
    (op == BodyOp::Cut).then_some(Mine::Own(f.id))
}

/// A cut that also took material where it was likely not meant — said on its row, with what to
/// change. Only the bodies this cut changed, and only the faces it made there: a tool kept and
/// used for another cut before left its faces on the body that cut made.
pub(crate) fn note_cuts_elsewhere(b: &mut Builder, f: &Feature, mine: &Mine, uncut: &[(BodyId, Arc<BodyState>)]) {
    let todo = match &f.kind {
        FeatureKind::Extrude(_) => "Shorten the extent",
        FeatureKind::Revolve(_) => "Narrow the angle, or the profile",
        FeatureKind::Hole(_) => "Shorten its depth",
        FeatureKind::Boolean(_) => "Trim the tool",
        _ => "Trim the cutting body",
    };
    let mut said = Vec::new();
    for (id, s) in b.bodies.raw.iter() {
        let was = uncut.iter().find(|(x, _)| x == id).map(|(_, w)| w);
        if was.is_some_and(|w| Arc::ptr_eq(w, s)) || s.surface || !s.origins.iter().any(|o| mine.of(*o)) {
            continue;
        }
        let t = s.topology();
        let tol = match_tolerance(s.diagonal());
        let before: Vec<FaceInfo> = was.map_or(Vec::new(), |w| w.topology().faces.iter().filter(|g| mine.of(g.fingerprint.origin)).cloned().collect());
        let made = |x: &FaceInfo| mine.of(x.fingerprint.origin) && !before.iter().any(|g| face_distance(&g.fingerprint, &x.fingerprint, &tol).is_some());
        if let Some(words) = cut_elsewhere_words(&f.name, &cuts_elsewhere(&t.faces, &t.edges, made), todo) {
            said.push(words);
        }
    }
    for w in said {
        b.warn(w);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn says_where_a_cut_also_went_and_how_far() {
        let places = vec![CutElsewhere { at: DVec3::new(20.354, -46.756, -68.831), area: 0.15, off: 11.43, bore: Some(2.25) }];
        assert_eq!(cut_elsewhere_words("Holes", &places, "Shorten the extent").as_deref(), Some("Holes also cut at (20.35, -46.76, -68.83): 0.15 mm², 11.4 mm along the R2.25 bore's axis — meant? Shorten the extent."));
        let tiny = vec![CutElsewhere { at: DVec3::ZERO, area: 0.005, off: 3.0, bore: None }];
        assert_eq!(cut_elsewhere_words("Cut", &tiny, "Trim the tool").as_deref(), Some("Cut also cut at (0, 0, 0): 0.0050 mm², 3 mm from the rest of the cut — meant? Trim the tool."));
        assert_eq!(cut_elsewhere_words("Cut", &[], "x"), None);
    }
}
