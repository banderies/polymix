// SPDX-License-Identifier: AGPL-3.0-only
//! Rounds and chamfers outside the build: a fillet's radius and a chamfer's distance fitted to
//! the scan ("fit" in place of the number), and a round or a chamfer on a feature that made
//! several solids, one feature a solid. Ported from ScanRuler (`plugins/reverse/app/reFit.ts`
//! fitEdges, combined, lawOf and fitEdgeLaws; `plugins/reverse/app/reFeatureCommands.ts`
//! addOnSolids), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only, without the command plumbing:
//! the caller resolves the edges and the scan's section, and keeps the part the solids' rounds
//! come back in.
//!
//! Each edge is read across as `fillet_read` reads it; the readings make one number, the
//! median, with its sigma — refused, every edge's reading listed, where one is unread or
//! stands more than two sigmas off the rest. One edge read along its length gives a variable
//! fillet's law: one radius where its places agree, a line from end to end where they lie on
//! one, refused where they follow neither.

use crate::build::{build, Built};
use crate::fillet_estimate::ReadShape;
use crate::fillet_read::{read_edge_round, round_reach, EdgeStation, LinearLaw, ReadEdge, ReadOptions, Section};
use crate::model::{BodyId, FeatureId, FeatureKind, Part};
use crate::nominal::{nominal_of, nominal_step, NominalKind};
use crate::pick_query::{stored_where, where_indices, FeatureRef, StoredPick, What, Where, WhereAsk};
use crate::round::{js_round, round4};

/// One edge's reading, as a fit lists them: its value and sigma, or why there is none.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    pub value: Option<f64>,
    pub sigma: Option<f64>,
    pub why: Option<String>,
}

/// A number fitted to the scan: the value, its sigma, the nominal and how many sigmas off it,
/// the points it rests on — and each edge's own reading where the value is a median, the reach
/// it was read over where the default one was widened to read it, a variable fillet's law
/// along one edge, what the fit made of it and what makes it doubtful.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FitReading {
    pub value: f64,
    pub sigma: Option<f64>,
    pub nominal: f64,
    pub off: f64,
    pub sigmas: Option<f64>,
    pub points: usize,
    pub rms: Option<f64>,
    pub readings: Vec<Reading>,
    pub reach: Option<f64>,
    /// A radius fitted along one edge: at its first end and at its other, as a variable fillet
    /// takes it — the value is the places' median.
    pub law: Option<[f64; 2]>,
    pub note: Option<String>,
    pub warning: Option<String>,
}

fn median(xs: &[f64]) -> f64 {
    let mut s = xs.to_vec();
    s.sort_by(f64::total_cmp);
    let m = s.len() >> 1;
    if s.is_empty() {
        return f64::NAN;
    }
    if s.len() % 2 == 1 {
        s[m]
    } else {
        (s[m - 1] + s[m]) / 2.0
    }
}

/// A measured figure's nominal, how far off it lies and in how many sigmas (`intent.ts`
/// rounding).
fn rounding(measured: f64, kind: NominalKind, sigma: Option<f64>) -> (f64, f64, Option<f64>) {
    let step = nominal_step(measured, kind);
    let nominal = nominal_of(measured, kind).unwrap_or_else(|| js_round(measured / step) * step);
    let off = measured - nominal;
    let sigmas = match sigma {
        Some(s) if s > 0.0 => Some(round4(off.abs() / s)),
        _ => None,
    };
    (round4(nominal), round4(off), sigmas)
}

/// A number as JavaScript writes it.
fn num(v: f64) -> String {
    crate::fillet_num::js_num(v)
}

/// Several readings of one number made one: the median, its sigma (the readings' combined, a
/// quarter more for the median), and a refusal when one stands more than two sigmas off the
/// rest — each reading listed, to pick the edges again (ScanRuler's `combined`).
pub fn combined(what: &str, unit: &str, readings: &[Reading], kind: NominalKind) -> Result<FitReading, String> {
    let ok: Vec<(f64, f64)> = readings.iter().filter_map(|r| Some((r.value?, r.sigma?))).collect();
    let listed: Vec<String> = readings
        .iter()
        .enumerate()
        .map(|(i, r)| match r.value {
            None => format!("#{} {}", i + 1, r.why.clone().unwrap_or_else(|| "no reading".into())),
            Some(v) => format!("#{} {} ± {} {unit}", i + 1, num(round4(v)), num(round4(r.sigma.unwrap_or(0.0)))),
        })
        .collect();
    let listed = listed.join("; ");
    if ok.len() < readings.len() || ok.is_empty() {
        return Err(format!("The {what} cannot be fitted: not every one picked reads — {listed}. Pick only those that do, or give a number."));
    }
    let value = median(&ok.iter().map(|r| r.0).collect::<Vec<_>>());
    let pooled = 1.0 / ok.iter().fold(0.0, |a, r| a + 1.0 / r.1.max(1e-9).powi(2)).sqrt();
    let sigma = if ok.len() > 1 { 1.2533 * pooled } else { pooled };
    for r in &ok {
        if (r.0 - value).abs() > 2.0 * r.1.hypot(sigma) {
            return Err(format!("The {what} cannot be one number: the readings spread past two sigmas — {listed}. Split them into features of their own, or give a number."));
        }
    }
    let (nominal, off, sigmas) = rounding(value, kind, Some(sigma));
    Ok(FitReading {
        value: round4(value),
        sigma: Some(round4(sigma)),
        nominal,
        off,
        sigmas,
        points: 0,
        readings: readings.iter().map(|r| Reading { value: r.value.map(round4), sigma: r.sigma.map(round4), why: r.why.clone() }).collect(),
        ..FitReading::default()
    })
}

/// What the places along one edge make of its radius.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LawKind {
    /// One number: each place within two sigmas of their median.
    One,
    /// A line from the first end to the other: each place within two sigmas of the line.
    Line,
    Neither,
}

/// What the places along one edge make of its radius: one number, where each lies within two
/// sigmas of their median (as the edges of a fit must); a line from the first end to the
/// other, where they spread past that about the median but each lies within two sigmas of the
/// line; neither. Sigma is each place's own with the places' pooled (ScanRuler's `lawOf`).
pub fn law_of(stations: &[EdgeStation], value: f64, linear: Option<&LinearLaw>) -> (LawKind, [f64; 2]) {
    let read: Vec<(f64, f64, f64)> = stations.iter().filter_map(|x| Some((x.t, x.radius?, x.sigma?))).collect();
    let pooled = 1.0 / read.iter().fold(0.0, |a, x| a + 1.0 / x.2.max(1e-9).powi(2)).sqrt();
    let sigma = if read.len() > 1 { 1.2533 * pooled } else { pooled };
    let near = |at: &dyn Fn(f64) -> f64| read.iter().all(|x| (x.1 - at(x.0)).abs() <= 2.0 * x.2.hypot(sigma));
    if near(&|_| value) {
        return (LawKind::One, [value, value]);
    }
    let [a, b] = linear.map_or([0.0, 0.0], |l| l.radius);
    if linear.is_some() && a > 0.0 && b > 0.0 && near(&|t| a + (b - a) * t) {
        return (LawKind::Line, [a, b]);
    }
    (LawKind::Neither, [value, value])
}

/// A fillet's radius or a chamfer's distance fitted to the scan across each edge given — the
/// median of their readings (ScanRuler's `fitEdges`). A reach given caps the radius; left out,
/// the part's (`round_reach`) widens by itself where a round reads at its top or not at all.
/// A round that runs from one size to another along an edge is warned of.
pub fn fit_edges(section: &mut Section, edges: &[ReadEdge], shape: ReadShape, reach: Option<f64>, model_size: f64) -> Result<FitReading, String> {
    if edges.is_empty() {
        return Err("A fitted size needs the edges to read it across.".into());
    }
    let mut readings = Vec::new();
    let mut points = 0;
    let mut widened: f64 = 0.0;
    let mut running: Vec<String> = Vec::new();
    for (i, edge) in edges.iter().enumerate() {
        let r = read_edge_round(section, edge, ReadOptions { stations: None, reach: reach.unwrap_or_else(|| round_reach(shape, model_size)), shape, widen: reach.is_none() });
        if let Some(round) = &r.round {
            points += round.residual.points;
        }
        if let Some(w) = r.reach {
            widened = widened.max(w);
        }
        readings.push(match &r.round {
            Some(round) => Reading { value: Some(round.radius), sigma: Some(round.sigma), why: None },
            None => Reading { value: None, sigma: None, why: r.why.clone() },
        });
        // A round that runs from one size to another along its edge: one number misses it, a
        // line along the edge would not.
        if let (ReadShape::Round, Some(round), Some(stations)) = (shape, &r.round, &r.stations) {
            let (kind, law) = law_of(stations, round.radius, r.linear.as_ref());
            if kind == LawKind::Line {
                running.push(format!("#{} runs from R{} to R{} along it", i + 1, num(law[0]), num(law[1])));
            }
        }
    }
    let (what, kind) = if shape == ReadShape::Chamfer { ("chamfer’s distance", NominalKind::Length) } else { ("fillet’s radius", NominalKind::Radius) };
    let mut out = combined(what, "mm", &readings, kind)?;
    out.points = points;
    if widened > 0.0 {
        out.reach = Some(widened);
    }
    if !running.is_empty() {
        out.warning = Some(format!("{} — {{ edge, radius: \"fit\" }} in edges fits {} as a variable fillet.", running.join("; "), if running.len() == 1 { "it" } else { "each" }));
    }
    Ok(out)
}

/// How many places along an edge a radius fitted along it is read at.
pub const LAW_STATIONS: usize = 7;

/// One edge's radius fitted along it at [`LAW_STATIONS`] places: [start, end] — one number both
/// ends where the places agree, a line where they lie on one — refused where they follow
/// neither (ScanRuler's `fitEdgeLaws`, one edge). `index` names the edge in a refusal.
pub fn fit_edge_law(section: &mut Section, edge: &ReadEdge, index: usize, reach: Option<f64>, model_size: f64) -> Result<FitReading, String> {
    let read = read_edge_round(section, edge, ReadOptions { stations: Some(LAW_STATIONS), reach: reach.unwrap_or_else(|| round_reach(ReadShape::Round, model_size)), shape: ReadShape::Round, widen: reach.is_none() });
    let (Some(round), Some(stations)) = (read.round, read.stations.as_ref()) else {
        return Err(format!("The radius along edges[{index}] cannot be fitted: {}.", read.why.clone().unwrap_or_else(|| "no round reads along it".into())));
    };
    let (kind, law) = law_of(stations, round.radius, read.linear.as_ref());
    let listed: Vec<String> = stations.iter().map(|x| format!("t {}: {}", num(x.t), x.radius.map_or("none".to_string(), |r| format!("R{} ± {}", num(round4(r)), num(round4(x.sigma.unwrap_or(0.0))))))).collect();
    if kind == LawKind::Neither {
        let line = read.linear.as_ref().map_or(String::new(), |l| format!(" (a line from R{} to R{} leaves rms {} mm)", num(l.radius[0]), num(l.radius[1]), num(l.rms)));
        return Err(format!("The radius along edges[{index}] follows neither one number nor a line from end to end: {}{line}. Split the edge where it changes, or give radius [start, end].", listed.join("; ")));
    }
    let (nominal, off, sigmas) = rounding(round.radius, NominalKind::Radius, Some(round.sigma));
    let note = match (kind, read.linear.as_ref()) {
        (LawKind::Line, Some(l)) => format!(
            "The radius runs from {} at the edge’s first end to {} at its other: its places spread past two sigmas about one radius (rms {} mm) and lie on a line (rms {} mm) — taken as a variable fillet.",
            num(law[0]),
            num(law[1]),
            num(l.median_rms),
            num(l.rms)
        ),
        _ => format!("One radius along the edge: its places agree within two sigmas — {} at both ends.", num(round.radius)),
    };
    Ok(FitReading {
        value: round.radius,
        sigma: Some(round.sigma),
        nominal,
        off,
        sigmas,
        points: round.residual.points,
        rms: read.linear.as_ref().map(|l| if kind == LawKind::Line { l.rms } else { l.median_rms }),
        readings: stations.iter().map(|x| Reading { value: x.radius.map(round4), sigma: x.sigma.map(round4), why: None }).collect(),
        reach: read.reach,
        law: Some(law),
        note: Some(note),
        warning: None,
    })
}

/// The solids of a feature, as the part stands built: every body it made a piece of (the ids
/// it gave out) and, for a feature on a body, the body it worked on — surfaces left out
/// (ScanRuler's `solidsNamed`).
pub fn solids_of(built: &Built, feature: FeatureId, worked: Option<BodyId>) -> Vec<BodyId> {
    built.bodies.iter().filter(|b| (b.id.feature == feature || Some(b.id) == worked) && !b.surface).map(|b| b.id).collect()
}

/// A name as features are told apart by it: trimmed, any case.
fn name_key(name: &str) -> String {
    name.trim().to_lowercase()
}

/// A name free for a feature: the next "name n" none of `taken` (by [`name_key`]) has
/// (ScanRuler's `freeName`).
fn free_name(name: &str, taken: &[String]) -> String {
    let t = name.trim();
    let base = match t.rfind(char::is_whitespace) {
        Some(i) if !t[i..].trim().is_empty() && t[i..].trim().chars().all(|c| c.is_ascii_digit()) => t[..i].trim_end(),
        _ => t,
    };
    (2..).map(|n| format!("{base} {n}")).find(|c| !taken.contains(&name_key(c))).unwrap_or_else(|| base.to_string())
}

/// What [`rounds_on_solids`] made: the part with the rounds in it, each one's id with the solid
/// it is on and how many edges it takes, and the solids no edge of which answers.
#[derive(Debug, Clone)]
pub struct OnSolids {
    pub part: Part,
    pub made: Vec<(FeatureId, BodyId, usize)>,
    pub left_out: Vec<BodyId>,
    pub note: String,
}

/// A round or a chamfer on a feature of several solids: one feature on each solid the queries
/// hit, named "<name> 1..n" (by default "Fillet on <the feature>"), added together and built
/// once. All or none: a solid where it does not build takes them all out again, by solid in
/// the refusal (ScanRuler's `addOnSolids`). `round` is the Fillet or the Chamfer to place, its
/// body and edges set per solid; the queries are kept with it, so each finds its edges again
/// as the solid changes.
pub fn rounds_on_solids(part: &Part, maker: FeatureId, round: &FeatureKind, queries: &[Where], name: Option<&str>) -> Result<OnSolids, String> {
    let Some(of) = part.feature(maker) else { return Err(format!("There is no feature {maker}.")) };
    let kind = match round {
        FeatureKind::Fillet(_) => "fillet",
        FeatureKind::Chamfer(_) => "chamfer",
        _ => return Err("Only a fillet or a chamfer goes on a feature solid by solid.".into()),
    };
    if queries.is_empty() {
        return Err("Give the edges as queries any of the solids can answer (box, near, kind, madeBy…).".into());
    }
    if queries.iter().any(|w| w.faces.is_some() || w.exclude.is_some()) {
        return Err(format!("{} made several bodies, and faces or exclude by index number the faces and edges of one of them — give queries that any of them can answer (box, near, kind, madeBy…), or name one body.", of.name));
    }
    let built = build(part);
    let solids = solids_of(&built, maker, of.kind.body());
    if solids.is_empty() {
        return Err(format!("{} made no solid to {}.", of.name, if kind == "fillet" { "round" } else { "chamfer" }));
    }
    let feature_id = |r: &FeatureRef| -> Result<i64, String> {
        match r {
            FeatureRef::Id(id) => Ok(*id),
            FeatureRef::Name(n) => part.features.iter().find(|f| &f.name == n).map(|f| f.id as i64).ok_or_else(|| format!("No feature is called {n}.")),
        }
    };
    let ask = WhereAsk { feature_id: &feature_id, sharp: None };
    let mut hit: Vec<(BodyId, Vec<StoredPick<crate::model::EdgeRef>>, Vec<crate::model::EdgeRef>, usize)> = Vec::new();
    for id in &solids {
        let Some(body) = built.body(*id) else { continue };
        let t = &body.topology;
        let mut picks = Vec::new();
        let mut found_all = Vec::new();
        let mut count = 0;
        for q in queries {
            let found = where_indices(q, t, &body.name, What::Edge, &ask)?;
            if found.is_empty() {
                continue;
            }
            count += found.len();
            let prints: Vec<_> = found.iter().filter_map(|i| t.edges.get(*i)).map(|e| e.fingerprint.clone()).collect();
            picks.push(StoredPick { query: stored_where(q, t, What::Edge, &feature_id)?, found: prints.clone() });
            found_all.extend(prints);
        }
        if count > 0 {
            hit.push((*id, picks, found_all, count));
        }
    }
    if hit.is_empty() {
        return Err(format!("No edge of the {} solids of {} answers the queries given.", solids.len(), of.name));
    }
    let base = name.map(str::trim).filter(|n| !n.is_empty()).map(str::to_string).unwrap_or_else(|| format!("{} on {}", if kind == "fillet" { "Fillet" } else { "Chamfer" }, of.name));
    let names: Vec<String> = (1..=hit.len()).map(|k| format!("{base} {k}")).collect();
    // Features go by their names in the commands: none of these may be one another has, in
    // any case (ScanRuler's `requireFreeName`).
    for (k, n) in names.iter().enumerate() {
        if let Some(other) = part.features.iter().find(|f| name_key(&f.name) == name_key(n)) {
            let taken: Vec<String> = part.features.iter().map(|f| name_key(&f.name)).chain(names[..k].iter().map(|x| name_key(x))).collect();
            return Err(format!("Feature {} is already named “{}” — the commands find features by name, so each needs its own: “{}”, say.", other.id, other.name, free_name(n, &taken)));
        }
    }
    let mut out = part.clone();
    let mut made = Vec::new();
    for (k, (solid, picks, prints, count)) in hit.iter().enumerate() {
        let mut one = round.clone();
        match &mut one {
            FeatureKind::Fillet(f) => {
                f.body = *solid;
                f.edges = prints.clone();
                f.edges_where = picks.clone();
            }
            FeatureKind::Chamfer(c) => {
                c.body = *solid;
                c.edges = prints.clone();
                c.edges_where = picks.clone();
            }
            _ => {}
        }
        let id = out.add_named(one, names[k].clone());
        made.push((id, *solid, *count));
    }
    let after = build(&out);
    let failing: Vec<String> = made
        .iter()
        .enumerate()
        .filter_map(|(k, (id, solid, _))| {
            let e = after.errors.get(id)?;
            let first = first_sentence(e);
            let on = built.body(*solid).map_or(solid.to_string(), |b| b.name.clone());
            Some(format!("{} on {on} ({solid}): {first}", names[k]))
        })
        .collect();
    if !failing.is_empty() {
        return Err(format!("{} of the {} {kind}s on {} did not build, and none was added — {}", failing.len(), made.len(), of.name, failing.join(" ")));
    }
    let left_out: Vec<BodyId> = solids.iter().filter(|s| !hit.iter().any(|h| h.0 == **s)).copied().collect();
    let note = format!(
        "One {kind} per solid the query hits, in one undo step{}.",
        if left_out.is_empty() { String::new() } else { format!("; {} of the {} solids left out, no edge of them answering", left_out.len(), solids.len()) }
    );
    Ok(OnSolids { part: out, made, left_out, note })
}

/// A failure as a list of them says it: its first sentence.
fn first_sentence(text: &str) -> String {
    let b = text.as_bytes();
    for (i, c) in text.char_indices() {
        if c == '.' || c == ':' {
            let next = b.get(i + 1).copied();
            if next.is_none() || next == Some(b' ') {
                return text[..=i].to_string();
            }
        }
    }
    text.to_string()
}

#[cfg(test)]
mod tests;
