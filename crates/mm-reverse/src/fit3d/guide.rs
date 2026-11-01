// SPDX-License-Identifier: AGPL-3.0-only
//! What a fit from a click may be told about the surface it is after, so the search does not
//! settle on the wrong one nearby: which side of the surface the material is on (the inside
//! of a cup, not its underside), the size it should come out at, the direction of its axis,
//! and a box it stays in. A click on a rim sits between two surfaces; without these the
//! larger one wins.
//!
//! Ported from ScanRuler (`src/core/fit/guide.ts`), © Stefan Hermann (CNC Kitchen),
//! AGPL-3.0-only.

use std::collections::{HashMap, HashSet};

use glam::DVec3;
use serde::{Deserialize, Serialize};

use super::cone::RansacConeResult;
use super::cylinder::RansacCylinderResult;
use super::js_text::{js_number, to_fixed};
use super::linalg3::{hypot3, js_min};
use super::ransac::RansacResult;
use crate::mesh_graph::MeshGraph;

/// Which side of the surface the material is on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Side {
    /// A concave surface — a bore, a cup's inside, a countersink; the scan's normals point
    /// toward the axis or centre.
    Inside,
    /// A convex one — a shaft, a ball.
    Outside,
}

/// The radius a surface should have, mm; a fit outside is refused.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct RadiusRange {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub min: Option<f64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max: Option<f64>,
}

/// A box the surface stays inside.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct GuideBox {
    pub min: DVec3,
    pub max: DVec3,
}

/// What a fit from a click is told about the surface it is after.
#[derive(Debug, Clone, Copy, PartialEq, Default, Serialize, Deserialize)]
pub struct FitGuide {
    /// Inside: a concave surface; outside: a convex one. Spheres, cylinders and cones.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side: Option<Side>,
    /// The radius the surface should have, mm; a fit outside is refused.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub radius: Option<RadiusRange>,
    /// A guess at the axis direction (cylinder, cone): only vertices whose normals stand
    /// roughly square to it are searched.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub axis: Option<DVec3>,
    /// The surface stays inside this box.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub within: Option<GuideBox>,
}

/// How far a normal may lean off square to a given axis and still be on the round wall —
/// wider than the scan's noise, narrower than a chamfer.
const AXIS_SQUARE_DEG: f64 = 30.0;

fn axis_sin() -> f64 {
    (AXIS_SQUARE_DEG * std::f64::consts::PI / 180.0).sin()
}

/// A final fit's axis may lie this far off the guess (20°).
fn axis_final_cos() -> f64 {
    (20.0 * std::f64::consts::PI / 180.0).cos()
}

/// An estimate's radius is checked with this much slack: the robust estimate on a patch is
/// rough, the final fit is held to the range.
const RADIUS_SLACK: f64 = 0.1;

/// The guide when it says anything at all.
pub fn has_guide(g: Option<&FitGuide>) -> Option<&FitGuide> {
    g.filter(|g| g.side.is_some() || g.radius.is_some() || g.axis.is_some() || g.within.is_some())
}

/// The guide's axis as a unit vector, if it gives one of any length.
pub fn unit_axis(g: &FitGuide) -> Option<DVec3> {
    let a = g.axis?;
    let l = hypot3(a.x, a.y, a.z);
    if l > 0.0 {
        Some(DVec3::new(a.x / l, a.y / l, a.z / l))
    } else {
        None
    }
}

/// Whether a point lies in the guide's box — always, when it gives none.
pub fn in_box(g: &FitGuide, p: DVec3) -> bool {
    match &g.within {
        None => true,
        Some(b) => p.x >= b.min.x && p.x <= b.max.x && p.y >= b.min.y && p.y <= b.max.y && p.z >= b.min.z && p.z <= b.max.z,
    }
}

/// Whether a radius lies in the range — with the estimate's slack when `rough`.
pub fn radius_in_range(g: &FitGuide, r: f64, rough: bool) -> bool {
    let lo = g.radius.and_then(|r| r.min);
    let hi = g.radius.and_then(|r| r.max);
    if let Some(lo) = lo {
        if r < lo * if rough { 1.0 - RADIUS_SLACK } else { 1.0 } {
            return false;
        }
    }
    if let Some(hi) = hi {
        if r > hi * if rough { 1.0 + RADIUS_SLACK } else { 1.0 } {
            return false;
        }
    }
    true
}

/// An axis close enough to the guess (either way along it).
pub fn axis_agrees(g: &FitGuide, a: DVec3) -> bool {
    match unit_axis(g) {
        None => true,
        Some(u) => (u.x * a.x + u.y * a.y + u.z * a.z).abs() >= axis_final_cos(),
    }
}

/// The patch round the seed with what the guide rules out taken away: the vertices outside
/// the box, and — with an axis — those whose normals do not stand square to it (the floor of
/// a hole, the face round its mouth). With a side or a size, only what bends the way asked
/// is kept (see `bent_as_asked`).
pub fn guide_patch(graph: &MeshGraph, patch: &[u32], g: &FitGuide) -> Vec<u32> {
    let u = unit_axis(g);
    let bend = g.side.is_some() || g.radius.is_some();
    if u.is_none() && g.within.is_none() && !bend {
        return patch.to_vec();
    }
    let sin = axis_sin();
    let mut out = Vec::new();
    for &v in patch {
        let p = graph.positions[v as usize];
        if !in_box(g, p) {
            continue;
        }
        if let Some(u) = u {
            let n = graph.normals[v as usize];
            if (n.x * u.x + n.y * u.y + n.z * u.z).abs() > sin {
                continue;
            }
        }
        out.push(v);
    }
    if bend {
        bent_as_asked(graph, &out, g)
    } else {
        out
    }
}

/// The vertices of a patch that bend the way the guide asks: a flat (a rim, a face round a
/// hole's mouth) goes, and with a side the surface curving the other way goes too. The bend
/// at a vertex is how its neighbours' normals spread, (n_u − n_v)·(u − v)/|u − v|² — 1/R on a
/// sphere, about 1/2R on a cylinder, positive where the surface is convex — averaged once
/// more over the ring so the scan's normal noise mostly cancels. Kept: bends past a floor (a
/// quarter of the largest radius asked for's, or the part's size's when none is), of the
/// sign the side asks for.
fn bent_as_asked(graph: &MeshGraph, patch: &[u32], g: &FitGuide) -> Vec<u32> {
    let (p, n) = (&graph.positions, &graph.normals);
    let mut bend_at: HashMap<u32, f64> = HashMap::with_capacity(patch.len());
    for &v in patch {
        let (pv, nv) = (p[v as usize], n[v as usize]);
        let mut sum = 0.0;
        let mut count = 0usize;
        for &w in graph.neighbours(v) {
            let (pw, nw) = (p[w as usize], n[w as usize]);
            let (dx, dy, dz) = (pw.x - pv.x, pw.y - pv.y, pw.z - pv.z);
            let l2 = dx * dx + dy * dy + dz * dz;
            if !(l2 > 0.0) {
                continue;
            }
            sum += ((nw.x - nv.x) * dx + (nw.y - nv.y) * dy + (nw.z - nv.z) * dz) / l2;
            count += 1;
        }
        bend_at.insert(v, if count > 0 { sum / count as f64 } else { 0.0 });
    }
    let floor = match g.radius.and_then(|r| r.max) {
        Some(max) => 1.0 / (4.0 * max),
        None => 0.25 / graph.bbox_diag,
    };
    let sign = match g.side {
        Some(Side::Outside) => 1.0,
        Some(Side::Inside) => -1.0,
        None => 0.0,
    };
    keep_where(patch, |v| {
        let mut sum = bend_at.get(&v).copied().unwrap_or(0.0);
        let mut count = 1.0;
        for &w in graph.neighbours(v) {
            if let Some(&b) = bend_at.get(&w) {
                sum += b;
                count += 1.0;
            }
        }
        let bend = sum / count;
        if sign != 0.0 {
            sign * bend > floor
        } else {
            bend.abs() > floor
        }
    })
}

/// The share of `idx` whose normals face the way `side` asks, against the outward direction
/// `outward` gives at each point (the unit radial there).
pub fn side_share(graph: &MeshGraph, idx: &[u32], side: Side, outward: impl Fn(DVec3) -> Option<DVec3>) -> f64 {
    let sign = if side == Side::Outside { 1.0 } else { -1.0 };
    let mut good = 0usize;
    let mut seen = 0usize;
    let step = (idx.len() / 400).max(1);
    for &v in idx.iter().step_by(step) {
        let Some(o) = outward(graph.positions[v as usize]) else { continue };
        seen += 1;
        let n = graph.normals[v as usize];
        if sign * (o.x * n.x + o.y * n.y + o.z * n.z) > 0.0 {
            good += 1;
        }
    }
    if seen > 0 {
        good as f64 / seen as f64
    } else {
        0.0
    }
}

/// The vertices of `idx` that pass `keep`, as a new list.
pub fn keep_where(idx: &[u32], mut keep: impl FnMut(u32) -> bool) -> Vec<u32> {
    idx.iter().copied().filter(|&v| keep(v)).collect()
}

/// The candidates an estimate on a patch tries in turn when a guide is given: each one that
/// the guide turns down takes its inliers out of the patch, and the next estimate runs on
/// what is left — a cup's outside found first leaves its inside for the second try.
pub const GUIDED_TRIES: usize = 4;

/// A size the guided search saw and turned down, and why — for a round whose axis lay off
/// the guide's, that axis, a point on it, how far off it lay and how many points it held, so
/// it can be asked for after all.
#[derive(Debug, Clone, PartialEq)]
pub struct GuideSeen {
    pub radius: f64,
    pub why: String,
    pub axis: Option<DVec3>,
    pub center: Option<DVec3>,
    pub off_deg: Option<f64>,
    pub inliers: Option<usize>,
}

impl GuideSeen {
    /// A size turned down for a reason given in words.
    pub fn because(radius: f64, why: impl Into<String>) -> Self {
        Self { radius, why: why.into(), axis: None, center: None, off_deg: None, inliers: None }
    }
}

/// The entries of `items` unique by `key`, in the order each key was first met, each the
/// last entry with that key — what JavaScript's `new Map(items.map(...)).values()` gives.
pub(crate) fn unique_by_last<T: Clone>(items: &[T], key: impl Fn(&T) -> String) -> Vec<T> {
    let mut order: Vec<String> = Vec::new();
    let mut last: HashMap<String, T> = HashMap::new();
    for item in items {
        let k = key(item);
        if !last.contains_key(&k) {
            order.push(k.clone());
        }
        last.insert(k, item.clone());
    }
    order.into_iter().filter_map(|k| last.remove(&k)).collect()
}

/// Why a guided fit found nothing: the sizes it saw and turned down, said plainly with what
/// to try — those in the radius range first, so a right size turned down for its noise is
/// not crowded out by the junk a larger patch found.
pub fn guide_refusal(kind: &str, g: &FitGuide, seen: &[GuideSeen]) -> String {
    let range = match &g.radius {
        Some(r) => format!(
            "{}–{}",
            r.min.map(|m| format!("R{}", js_number(m))).unwrap_or_default(),
            r.max.map(|m| format!("R{}", js_number(m))).unwrap_or_default()
        ),
        None => String::new(),
    };
    let mut found = unique_by_last(seen, |s| format!("{} {}", to_fixed(s.radius, 2), s.why));
    // A stable sort: those in the range first.
    found.sort_by_key(|s| !radius_in_range(g, s.radius, false));
    found.truncate(3);
    let fixed = |v: DVec3, d: usize| {
        let parts: Vec<String> = v.to_array().iter().map(|&x| js_number(to_fixed(x, d).parse::<f64>().unwrap_or(f64::NAN))).collect();
        format!("[{}]", parts.join(", "))
    };
    let seen_as = |s: &GuideSeen| match (s.axis, s.center) {
        (Some(axis), Some(center)) => format!(
            "R{} (axis off by {}°: axis {} through {}, {} points)",
            to_fixed(s.radius, 2),
            s.off_deg.map_or_else(|| "undefined".to_owned(), |d| to_fixed(d, 1)),
            fixed(axis, 3),
            fixed(center, 2),
            s.inliers.map_or_else(|| "undefined".to_owned(), |n| n.to_string())
        ),
        _ => format!("R{} ({})", to_fixed(s.radius, 2), s.why),
    };
    let said = if found.is_empty() { String::new() } else { format!(" Found {}.", found.iter().map(seen_as).collect::<Vec<_>>().join(", ")) };
    let off_axis = if found.iter().any(|s| s.axis.is_some()) { " If that is the surface, give that axis or drop axis." } else { "" };
    let range_said = if range.is_empty() { String::new() } else { format!(" (radius {range})") };
    let within = if g.within.is_some() { " and within" } else { "" };
    format!("No {kind} there within the guide{range_said}.{said}{off_axis} Seed on the surface itself, widen radius, or check side{within}.")
}

/// A round turned down for its axis: see `GuideSeen`.
pub fn axis_off(g: &FitGuide, radius: f64, axis: DVec3, center: DVec3, inliers: usize) -> GuideSeen {
    let u = unit_axis(g);
    let l = hypot3(axis.x, axis.y, axis.z);
    let l = if l == 0.0 || l.is_nan() { 1.0 } else { l };
    let a = DVec3::new(axis.x / l, axis.y / l, axis.z / l);
    let cos = match u {
        Some(u) => js_min(1.0, (u.x * a.x + u.y * a.y + u.z * a.z).abs()),
        None => 1.0,
    };
    GuideSeen { radius, why: "axis off".into(), axis: Some(a), center: Some(center), off_deg: Some((cos.acos() * 180.0) / std::f64::consts::PI), inliers: Some(inliers) }
}

/// A robust estimate that names the points it agrees with.
pub trait Inliers {
    fn inliers(&self) -> &[u32];
}

impl Inliers for RansacResult {
    fn inliers(&self) -> &[u32] {
        &self.inliers
    }
}

impl Inliers for RansacCylinderResult {
    fn inliers(&self) -> &[u32] {
        &self.inliers
    }
}

impl Inliers for RansacConeResult {
    fn inliers(&self) -> &[u32] {
        &self.inliers
    }
}

/// The robust estimate on a patch, steered by the guide: the patch first loses what the
/// guide rules out (see `guide_patch`), then each estimate the guide turns down — `judge`
/// says why — takes its own surface out of the patch (its inliers, and whatever `on` says
/// lies on it) and the next runs on what is left. Every refusal is added to `seen`.
pub fn guided_estimate<C: Inliers>(
    graph: &MeshGraph,
    patch: &[u32],
    guide: &FitGuide,
    mut estimate: impl FnMut(&[u32]) -> Option<C>,
    mut judge: impl FnMut(&C) -> Option<GuideSeen>,
    on: impl Fn(&C, u32) -> bool,
    seen: &mut Vec<GuideSeen>,
) -> Option<C> {
    // A size without a side: the convex and the concave parts of the patch are tried apart,
    // the larger first — a robust estimate on the two shells of a wall at once settles
    // between them.
    let patches = if guide.radius.is_some() && guide.side.is_none() {
        let mut both = vec![
            guide_patch(graph, patch, &FitGuide { side: Some(Side::Outside), ..*guide }),
            guide_patch(graph, patch, &FitGuide { side: Some(Side::Inside), ..*guide }),
        ];
        both.sort_by_key(|p| std::cmp::Reverse(p.len()));
        both
    } else {
        vec![guide_patch(graph, patch, guide)]
    };
    for mut p in patches {
        let mut t = 0;
        while t < GUIDED_TRIES && p.len() >= 30 {
            let Some(c) = estimate(&p) else { break };
            let Some(refusal) = judge(&c) else { return Some(c) };
            seen.push(refusal);
            let gone: HashSet<u32> = c.inliers().iter().copied().collect();
            p = keep_where(&p, |v| !gone.contains(&v) && !on(&c, v));
            t += 1;
        }
    }
    None
}

/// The vertices a guided region grows from: the estimate's inliers that face the side asked
/// for.
pub fn guided_seeds(graph: &MeshGraph, inliers: &[u32], guide: &FitGuide, outward: impl Fn(DVec3) -> Option<DVec3>) -> Vec<u32> {
    let Some(side) = guide.side else { return inliers.to_vec() };
    let sign = if side == Side::Outside { 1.0 } else { -1.0 };
    keep_where(inliers, |v| match outward(graph.positions[v as usize]) {
        Some(o) => {
            let n = graph.normals[v as usize];
            sign * (o.x * n.x + o.y * n.y + o.z * n.z) > 0.0
        }
        None => false,
    })
}

/// Unit radial from a centre point.
pub fn radial_from(c: DVec3) -> impl Fn(DVec3) -> Option<DVec3> {
    move |p| {
        let (dx, dy, dz) = (p.x - c.x, p.y - c.y, p.z - c.z);
        let d = hypot3(dx, dy, dz);
        if d > 1e-12 {
            Some(DVec3::new(dx / d, dy / d, dz / d))
        } else {
            None
        }
    }
}

/// Unit radial off an axis line through `point` along the unit `axis`.
pub fn radial_off(point: DVec3, axis: DVec3) -> impl Fn(DVec3) -> Option<DVec3> {
    move |p| {
        let (qx, qy, qz) = (p.x - point.x, p.y - point.y, p.z - point.z);
        let t = qx * axis.x + qy * axis.y + qz * axis.z;
        let (wx, wy, wz) = (qx - t * axis.x, qy - t * axis.y, qz - t * axis.z);
        let d = hypot3(wx, wy, wz);
        if d > 1e-12 {
            Some(DVec3::new(wx / d, wy / d, wz / d))
        } else {
            None
        }
    }
}

/// The judge every round kind shares: size against the range (rough, for an estimate), side
/// against the inliers' normals.
pub fn judge_round(graph: &MeshGraph, guide: &FitGuide, radius: f64, inliers: &[u32], outward: impl Fn(DVec3) -> Option<DVec3>, rough: bool) -> Option<GuideSeen> {
    if !radius_in_range(guide, radius, rough) {
        return Some(GuideSeen::because(radius, "size"));
    }
    if let Some(side) = guide.side {
        if side_share(graph, inliers, side, outward) < 0.6 {
            return Some(GuideSeen::because(radius, if side == Side::Inside { "an outside" } else { "an inside" }));
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lists_what_it_turned_down_in_the_range_first() {
        let g = FitGuide { radius: Some(RadiusRange { min: Some(20.0), max: Some(30.0) }), ..FitGuide::default() };
        let seen = vec![
            GuideSeen::because(6.5, "size"),
            GuideSeen::because(25.0, "an outside"),
            GuideSeen::because(6.5, "size"),
            GuideSeen::because(10.0, "size"),
            GuideSeen::because(12.0, "size"),
        ];
        assert_eq!(
            guide_refusal("sphere", &g, &seen),
            "No sphere there within the guide (radius R20–R30). Found R25.00 (an outside), R6.50 (size), R10.00 (size). Seed on the surface itself, widen radius, or check side."
        );
        let open = FitGuide { radius: Some(RadiusRange { min: None, max: Some(3.5) }), within: Some(GuideBox { min: DVec3::ZERO, max: DVec3::ONE }), ..FitGuide::default() };
        assert_eq!(guide_refusal("cone", &open, &[]), "No cone there within the guide (radius –R3.5). Seed on the surface itself, widen radius, or check side and within.");
    }

    #[test]
    fn names_an_axis_turned_down() {
        let g = FitGuide { axis: Some(DVec3::new(0.0, 0.5, 1.0)), ..FitGuide::default() };
        let s = axis_off(&g, 3.004, DVec3::new(0.0, -1e-5, 2.0), DVec3::new(0.001, 0.0, -0.0004), 812);
        let said = guide_refusal("cylinder", &g, &[s]);
        assert!(said.contains("R3.00 (axis off by 26.6°: axis [0, 0, 1] through [0, 0, 0], 812 points). If that is the surface, give that axis or drop axis."), "{said}");
    }

    #[test]
    fn judges_the_size_with_slack_for_an_estimate() {
        let g = FitGuide { radius: Some(RadiusRange { min: Some(10.0), max: None }), ..FitGuide::default() };
        assert!(!radius_in_range(&g, 9.5, false));
        assert!(radius_in_range(&g, 9.5, true));
        assert!(!radius_in_range(&g, 8.9, true));
        assert!(has_guide(Some(&FitGuide::default())).is_none());
        assert!(has_guide(Some(&g)).is_some());
        assert!(in_box(&FitGuide { within: Some(GuideBox { min: DVec3::ZERO, max: DVec3::ONE }), ..g }, DVec3::splat(0.5)));
    }
}
