// SPDX-License-Identifier: AGPL-3.0-only
//! A loft added with its stations, in one go: each section given as a plane gets a sketch of
//! the scan's run there fitted like a station already made (`fit.like`), named "<name> station
//! k"; with `inner`, each station's twin moved in by the wall into "<name> inner k" and a loft
//! of its own through them, "<name> inner" — a body of its own, a thin shell's inside to cut
//! from the outer; and the loft through the stations. Ordinary sketches and lofts, added in
//! order, each refitted or edited on its own afterwards. A plane's offset and the wall may be
//! expressions of the parameters, written onto the sketches. Ported from ScanRuler
//! (`plugins/reverse/app/reFeatureCommands.ts` addLoftStations and loftExprsIn,
//! `app/reSketchInput.ts` prepareFit, fitLike and insetDraft, `app/reFeatureInput.ts`
//! addInsetTwin), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.
//!
//! The run a station is fitted to is read off the cut as ScanRuler's fit reads it: the chains
//! the plane cuts through the scan, kept within a box on the sheet and on one skin of the wall
//! where asked, and of the runs left the one of the most points. Facing and the fit's other
//! options are not taken here.

use glam::DVec2;
use mm_reverse::chain_outline::{runs_kept, side_of, KeepOptions, SheetBox};
use mm_reverse::datums::datum_traces;
use mm_reverse::expression::{evaluate_for, ParameterKind};
use mm_reverse::frame::SectionFrame;
use mm_reverse::model::{plane_at_offset, BodyId, BodyOp, FeatureId, FeatureKind, LoftFeature, Part, PlaneRef, SketchFeature, SketchInset};
use mm_reverse::section_skin::{SideRule, SidedChains, Skin};
use mm_reverse::sketch::{EntityId, ShapeKind, Sketch};
use mm_reverse::solver::{solve_sketch, SolveOptions};
use mm_reverse::stations::{inset_chains, like_station, station_faults, station_loop, InsetChainsAsk, StationLoop, STATION_FOLD_HELP};

/// A section of the loft: a sketch the part has, or a plane to fit a station on.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Section {
    Sketch(FeatureId),
    Plane(PlaneRef),
}

/// How each section given as a plane is fitted: like a station already made, its run read on
/// one skin and within a box on the sheet, where given.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationFit {
    pub(crate) like: FeatureId,
    pub(crate) skin: Option<Skin>,
    pub(crate) within: Option<SheetBox>,
    pub(crate) tolerance: Option<f64>,
}

/// A loft asked for with its stations.
#[derive(Debug, Clone, PartialEq, Default)]
pub(crate) struct LoftStations {
    pub(crate) name: Option<String>,
    pub(crate) sections: Vec<Section>,
    pub(crate) fit: Option<StationFit>,
    pub(crate) inner: Option<f64>,
    pub(crate) ruled: Option<bool>,
    pub(crate) op: Option<BodyOp>,
    pub(crate) target: Option<BodyId>,
    /// `sections.<k>.offset` and `inner`, as expressions of the parameters.
    pub(crate) exprs: Vec<(String, String)>,
}

/// A station made: its sketch, its name and its pieces in order.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationMade {
    pub(crate) id: FeatureId,
    pub(crate) name: String,
    pub(crate) pieces: Vec<(EntityId, ShapeKind)>,
    pub(crate) reversed: bool,
}

/// An inner station made: its sketch, its name, the station it was moved in from, how many
/// pieces moved and the ones kept where they lie.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct TwinMade {
    pub(crate) id: FeatureId,
    pub(crate) name: String,
    pub(crate) of: FeatureId,
    pub(crate) moved: usize,
    pub(crate) kept: Vec<EntityId>,
}

/// What the recipe added: the loft, the stations fitted, and the inner loft with its stations.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct StationsMade {
    pub(crate) loft: FeatureId,
    pub(crate) stations: Vec<StationMade>,
    pub(crate) inner: Option<(FeatureId, Vec<TwinMade>)>,
}

/// A name as names are compared: trimmed, in lower case.
fn name_key(name: &str) -> String {
    name.trim().to_lowercase()
}

/// A name free for a feature: the next "name n" no feature has.
fn free_name(name: &str, taken: &[String]) -> String {
    let trimmed = name.trim();
    let base = match trimmed.rsplit_once(' ') {
        Some((head, tail)) if !tail.is_empty() && tail.chars().all(|c| c.is_ascii_digit()) => head.trim_end(),
        _ => trimmed,
    };
    (2..).map(|n| format!("{base} {n}")).find(|c| !taken.contains(&name_key(c))).unwrap_or_else(|| format!("{base} 2"))
}

/// "Loft 1", or the first such name no feature has.
fn unused_name(part: &Part, base: &str) -> String {
    let taken: Vec<String> = part.features.iter().map(|f| name_key(&f.name)).collect();
    (1..).map(|n| format!("{base} {n}")).find(|c| !taken.contains(&name_key(c))).unwrap_or_else(|| format!("{base} 1"))
}

/// A name no feature has, nor one made earlier in the same go — or why not.
fn require_free_name(part: &Part, name: &str, also: &[String]) -> Result<(), String> {
    let mut taken: Vec<String> = part.features.iter().map(|f| name_key(&f.name)).collect();
    taken.extend(also.iter().map(|n| name_key(n)));
    if let Some(other) = part.features.iter().find(|f| name_key(&f.name) == name_key(name)) {
        return Err(format!("Feature {} is already named “{}” — the commands find features by name, so each needs its own: “{}”, say.", other.id, other.name, free_name(name, &taken)));
    }
    if also.iter().any(|x| name_key(x) == name_key(name)) {
        return Err(format!("Two features in this batch are named “{}” — give each its own (“{}”, say); nothing was added.", name.trim(), free_name(name, &taken)));
    }
    Ok(())
}

/// A loft's expressions evaluated: each section's plane offset, and the wall.
struct LoftExprs {
    offsets: Vec<(usize, String, f64)>,
    inner: Option<(String, f64)>,
}

fn loft_exprs(part: &Part, exprs: &[(String, String)], sections: &[Section]) -> Result<LoftExprs, String> {
    // The part's parameters as the evaluator reads them.
    let table = super::loft_kit::parameter_table(part);
    let mut out = LoftExprs { offsets: Vec::new(), inner: None };
    for (key, expr) in exprs {
        if expr.trim().is_empty() {
            return Err(format!("exprs.{key}: an expression is a string."));
        }
        let value = evaluate_for(expr, ParameterKind::Length, &table, false).map_err(|e| format!("exprs.{key}: “{expr}” — {e}"))?.value;
        if let Some(k) = key.strip_prefix("sections.").and_then(|r| r.strip_suffix(".offset")).and_then(|k| k.parse::<usize>().ok()) {
            if k >= sections.len() {
                return Err(format!("exprs.{key}: there are {} sections — sections.0.offset to sections.{}.offset.", sections.len(), sections.len().saturating_sub(1)));
            }
            if matches!(sections[k], Section::Plane(PlaneRef::Fitted { .. })) {
                return Err(format!("exprs.{key}: a plane fitted where the scan was clicked takes no offset."));
            }
            out.offsets.push((k, expr.clone(), value));
        } else if key == "inner" {
            out.inner = Some((expr.clone(), value));
        } else {
            return Err(format!("exprs.{key}: no such field — a loft has sections.<k>.offset (the k-th section’s plane offset, k from 0) and inner (the inner stations’ wall)."));
        }
    }
    if let Some((expr, value)) = &out.inner {
        if !(*value > 0.0) {
            return Err(format!("exprs.inner: “{expr}” comes to {} mm — the wall is above zero.", mm_reverse::js_fmt::num(*value)));
        }
    }
    Ok(out)
}

/// The scan's run on a station's plane fitted like the template — kept within the box and on
/// the skin the fit asks for, the run of the most points — and solved; or why not.
fn fit_station(like_name: &str, template: &Sketch, lp: &StationLoop, cut: &SidedChains, frame: &SectionFrame, fit: &StationFit) -> Result<(Sketch, bool), String> {
    let rule = SideRule { skin: fit.skin, facing: None };
    let side = fit.skin.map(|_| side_of(cut, &rule));
    let work: Vec<Vec<DVec2>> = cut.chains.iter().flat_map(|c| runs_kept(c, &KeepOptions { sheet_box: fit.within, side: side.as_ref(), test: None })).collect();
    let Some(run) = work.iter().reduce(|a, b| if b.len() > a.len() { b } else { a }) else {
        return Err("The scan is not cut on this plane — there is no chain.".into());
    };
    let made = like_station(template, lp, run, fit.tolerance).map_err(|e| format!("The run could not be fitted like {like_name}: {}", e.0))?;
    let mut sketch = made.sketch;
    let traces = datum_traces(frame);
    let report = solve_sketch(&mut sketch, SolveOptions { datums: &traces, ..Default::default() });
    if !report.ok {
        return Err(format!("The station fitted like {like_name} does not solve: {} of its relations are not met.", report.unmet.len()));
    }
    // The solve holds the template's tangencies, and can fold a piece the fit left whole.
    let ids: Vec<EntityId> = sketch.entities.iter().map(|e| e.id).collect();
    let folds = station_faults(template, lp, &sketch.entities, Some(&ids));
    if !folds.is_empty() {
        return Err(format!("The station fitted like {like_name} folds once solved: {} — {STATION_FOLD_HELP}", folds.join("; ")));
    }
    Ok((sketch, made.reversed))
}

/// The pieces of a sketch's one profile, in order.
fn pieces_of(sketch: &Sketch, name: &str) -> Vec<(EntityId, ShapeKind)> {
    match station_loop(&sketch.entities, name, true) {
        Ok(lp) => lp.pieces.iter().map(|p| (p.id, p.kind)).collect(),
        Err(_) => sketch.entities.iter().filter(|e| !e.construction).map(|e| (e.id, e.kind())).collect(),
    }
}

/// The loft and its stations added to the part, in order, as ScanRuler's `reverse.feature.add
/// { kind: "loft", sections, fit, inner }` adds them; `cut` reads the scan on a plane, each
/// chain on the sheet with its points' outward directions; `frame_of` resolves a plane. All or
/// nothing: a station that cannot be fitted refuses the whole of it.
pub(crate) fn add_loft_stations(part: &mut Part, ask: &LoftStations, cut: &dyn Fn(&SectionFrame) -> SidedChains, frame_of: &dyn Fn(&Part, &PlaneRef) -> Option<SectionFrame>) -> Result<StationsMade, String> {
    let mut items = ask.sections.clone();
    if items.len() < 2 {
        return Err("A loft takes two sketches or more — sketches, or sections to fit its stations on.".into());
    }
    // The stations' numbers by expression: a plane's offset, the wall.
    let exprs = loft_exprs(part, &ask.exprs, &items)?;
    let inner = exprs.inner.as_ref().map(|(_, v)| *v).or(ask.inner);
    for (k, _, value) in &exprs.offsets {
        if let Section::Plane(p) = &items[*k] {
            items[*k] = Section::Plane(plane_at_offset(p, *value));
        }
    }
    let on_planes = items.iter().any(|i| matches!(i, Section::Plane(_)));
    let fit = match (&ask.fit, on_planes) {
        (None, true) => return Err("fit.like names the sketch the stations are fitted like: one station fitted to the scan and closed — its run, and a line across its base drawn on.".into()),
        (Some(_), false) => return Err("fit is for sections given as planes — every section here is a sketch you have.".into()),
        (f, _) => f.clone(),
    };
    let name = ask.name.as_deref().map(str::trim).filter(|n| !n.is_empty()).map_or_else(|| unused_name(part, "Loft"), str::to_string);
    let station_name = |k: usize| format!("{name} station {}", k + 1);
    let twin_name = |k: usize| format!("{name} inner {}", k + 1);
    let inner_name = format!("{name} inner");
    // Every name made checked first: one taken would stop it half-way.
    let mut names: Vec<String> = vec![name.clone()];
    let mut made_names: Vec<String> = items.iter().enumerate().filter(|(_, it)| matches!(it, Section::Plane(_))).map(|(k, _)| station_name(k)).collect();
    if inner.is_some() {
        made_names.extend((0..items.len()).map(twin_name));
        made_names.push(inner_name.clone());
    }
    for n in made_names {
        require_free_name(part, &n, &names)?;
        names.push(n);
    }
    for it in &items {
        if let Section::Sketch(id) = it {
            if part.sketch(*id).is_none() {
                return Err(format!("There is no sketch {id}."));
            }
        }
    }
    let like = match &fit {
        Some(f) => {
            let Some(template) = part.sketch(f.like) else { return Err(format!("fit.like: there is no sketch {}.", f.like)) };
            let like_name = part.feature(f.like).map_or_else(|| format!("Sketch {}", f.like), |x| x.name.clone());
            let lp = station_loop(&template.sketch.entities, &like_name, false).map_err(|e| format!("fit.like: {}", e.0))?;
            Some((like_name, template.sketch.clone(), lp))
        }
        None => None,
    };
    // Each station fitted before any is added: a refusal leaves the part as it was.
    let mut work = part.clone();
    let mut ids = Vec::new();
    let mut stations = Vec::new();
    for (k, it) in items.iter().enumerate() {
        match it {
            Section::Sketch(id) => {
                // A sketch you have takes its offset's expression as an edit.
                if let Some((_, expr, value)) = exprs.offsets.iter().find(|o| o.0 == k) {
                    if let Some(f) = work.feature_mut(*id) {
                        if let FeatureKind::Sketch(s) = &mut f.kind {
                            *s = s.at_offset(*value);
                            s.exprs.insert("offset", expr.clone());
                        }
                    }
                }
                ids.push(*id);
            }
            Section::Plane(plane) => {
                let (Some((like_name, template, lp)), Some(f)) = (&like, &fit) else { return Err("fit.like names the sketch the stations are fitted like.".into()) };
                let frame = frame_of(&work, plane).ok_or("That plane is not there any more.")?;
                let (sketch, reversed) = fit_station(like_name, template, lp, &cut(&frame), &frame, f).map_err(|e| format!("{}: {e}", station_name(k)))?;
                let pieces = pieces_of(&sketch, &station_name(k));
                let mut feature = SketchFeature { plane: plane.clone(), frame, sketch, exprs: Default::default(), image: None, inset: None };
                if let Some((_, expr, _)) = exprs.offsets.iter().find(|o| o.0 == k) {
                    feature.exprs.insert("offset", expr.clone());
                }
                let id = work.add_named(FeatureKind::Sketch(feature), station_name(k));
                stations.push(StationMade { id, name: station_name(k), pieces, reversed });
                ids.push(id);
            }
        }
    }
    let loft = work.add_named(FeatureKind::Loft(LoftFeature { sketches: ids.clone(), ruled: ask.ruled.unwrap_or(true), op: ask.op.unwrap_or(BodyOp::Join), target: ask.target }), name.clone());
    let mut inner_out = None;
    if let Some(by) = inner {
        let mut twins = Vec::new();
        for (k, sid) in ids.iter().enumerate() {
            let Some(source) = work.sketch(*sid).cloned() else { continue };
            let twin = twin_name(k);
            let traces = datum_traces(&source.frame);
            let mut asked = InsetChainsAsk::new(by, &twin);
            asked.datums = &traces;
            let out = inset_chains(&source.sketch, &asked).map_err(|e| format!("{twin}: inset: {}", e.0))?;
            let mut sketch = out.sketch;
            solve_sketch(&mut sketch, SolveOptions { datums: &traces, ..Default::default() });
            let mut exprs_of = mm_reverse::model::Exprs::new();
            if let Some((expr, _)) = &exprs.inner {
                exprs_of.insert("inset", expr.clone());
            }
            // On its station's plane, it follows what moves that plane.
            if let Some(offset) = source.exprs.get("offset") {
                exprs_of.insert("offset", offset.to_string());
            }
            let feature = SketchFeature { plane: source.plane.clone(), frame: source.frame, sketch, exprs: exprs_of, image: None, inset: Some(SketchInset { of: *sid, by, keep: out.kept.clone() }) };
            let id = work.add_named(FeatureKind::Sketch(feature), twin.clone());
            twins.push(TwinMade { id, name: twin, of: *sid, moved: out.moved.len(), kept: out.kept });
        }
        let twin_ids: Vec<FeatureId> = twins.iter().map(|t| t.id).collect();
        let inner_loft = work.add_named(FeatureKind::Loft(LoftFeature { sketches: twin_ids, ruled: ask.ruled.unwrap_or(true), op: BodyOp::New, target: None }), inner_name.clone());
        inner_out = Some((inner_loft, twins));
    }
    *part = work;
    Ok(StationsMade { loft, stations, inner: inner_out })
}

/// What the recipe made, in words: the stations and their pieces, the inner loft.
pub(crate) fn made_words(part: &Part, made: &StationsMade) -> String {
    let name = |id: FeatureId| part.feature(id).map_or_else(|| id.to_string(), |f| f.name.clone());
    let kinds = |p: &[(EntityId, ShapeKind)]| p.iter().map(|(_, k)| format!("{k:?}").to_lowercase()).collect::<Vec<_>>().join(" ");
    let stations: Vec<String> = made.stations.iter().map(|s| format!("{} ({})", s.name, kinds(&s.pieces))).collect();
    let inner = match &made.inner {
        Some((id, twins)) => format!("; {} through {}: a body of its own — cut it from the outer once that is joined to the rest", name(*id), twins.iter().map(|t| t.name.clone()).collect::<Vec<_>>().join(", ")),
        None => String::new(),
    };
    format!("{} through {}{inner}", name(made.loft), if stations.is_empty() { "its sketches".to_string() } else { stations.join(", ") })
}

#[cfg(test)]
mod tests {
    //! ScanRuler's `plugins/reverse/tests/reStations.test.ts` "the stations of a moulded wing,
    //! end to end": an arch on two flared walls, 2 mm thick, ribs inside, growing along its
    //! length — lofted through six stations fitted like the first, its inner loft through their
    //! twins moved in by the wall, in one go.
    use std::f64::consts::PI;

    use glam::{DVec2, DVec3};
    use mm_reverse::build::{build, volume};
    use mm_reverse::frame::WorldAxis;
    use mm_reverse::model::*;
    use mm_reverse::slice::slice_mesh;
    use mm_reverse::sketch::{Anchor, AnchorAt, Entity, Rule, Shape};

    use super::*;

    const DEG: f64 = PI / 180.0;

    struct Wing {
        r: f64,
        rise: f64,
        flare: f64,
        t: f64,
        base: f64,
    }

    fn wing_at(z: f64) -> Wing {
        Wing { r: 24.0 + 0.15 * z, rise: 0.05 * z, flare: (8.0 + 0.1 * z) * DEG, t: 2.0, base: -15.0 }
    }

    fn along(a: DVec2, b: DVec2, n: usize) -> Vec<DVec2> {
        (0..n).map(|i| a + (b - a) * (i as f64 / n as f64)).collect()
    }

    fn wall_foot(r: f64, rise: f64, theta: f64, base: f64, side: f64) -> (DVec2, DVec2) {
        let top = DVec2::new(r * theta.cos(), rise + r * theta.sin());
        let d = if side > 0.0 { DVec2::new(theta.sin(), -theta.cos()) } else { DVec2::new(-theta.sin(), theta.cos()) };
        let k = (base - top.y) / d.y;
        (top, DVec2::new(top.x + d.x * k, base))
    }

    fn wing_profile(z: f64) -> Vec<DVec2> {
        let Wing { r: big, rise, flare, t, base } = wing_at(z);
        let r = big - t;
        let (w, a) = (40usize, 160usize);
        let ribs = [0.3, 0.5, 0.7];
        let mut out = Vec::new();
        let right = wall_foot(big, rise, flare, base, 1.0);
        let left = wall_foot(big, rise, PI - flare, base, -1.0);
        out.extend(along(right.1, right.0, w));
        for i in 0..a {
            let ang = flare + (PI - 2.0 * flare) * i as f64 / a as f64;
            out.push(DVec2::new(big * ang.cos(), rise + big * ang.sin()));
        }
        out.extend(along(left.0, left.1, w));
        let inner_left = wall_foot(r, rise, PI - flare, base, -1.0);
        let inner_right = wall_foot(r, rise, flare, base, 1.0);
        out.extend(along(left.1, inner_left.1, 4));
        out.extend(along(inner_left.1, inner_left.0, w));
        for i in 0..a {
            let s = i as f64 / a as f64;
            let rib = ribs.iter().copied().find(|x| (s - x).abs() < 0.02);
            let ang = PI - flare - (PI - 2.0 * flare) * s;
            let Some(rib) = rib else {
                out.push(DVec2::new(r * ang.cos(), rise + r * ang.sin()));
                continue;
            };
            if (s - rib).abs() >= 0.02 - 1.0 / a as f64 {
                continue;
            }
            let ar = PI - flare - (PI - 2.0 * flare) * rib;
            let c = DVec2::new(ar.cos(), ar.sin());
            let tan = DVec2::new(-ar.sin(), ar.cos());
            let at = |rad: f64, side: f64| DVec2::new(c.x * rad + tan.x * side, rise + c.y * rad + tan.y * side);
            out.extend([at(r, 1.0), at(r - 6.0, 1.0), at(r - 6.0, -1.0), at(r, -1.0)]);
        }
        out.extend(along(inner_right.0, inner_right.1, w));
        out.extend(along(inner_right.1, right.1, 4));
        out
    }

    /// The wing as a scan: its profile at every layer along Z, joined into a tube of triangles
    /// wound out of the material, jittered by `noise`.
    fn wing_mesh(length: f64, step: f64, noise: f64) -> (Vec<DVec3>, Vec<[u32; 3]>) {
        let layers = (length / step).round() as usize;
        let profiles: Vec<Vec<DVec2>> = (0..=layers).map(|l| wing_profile(l as f64 * step)).collect();
        let k = profiles[0].len();
        assert!(profiles.iter().all(|p| p.len() == k), "the layers differ in points");
        let mut s: u32 = 7;
        let mut rand = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f64 / 2f64.powi(32)) * 2.0 - 1.0
        };
        let mut positions = Vec::new();
        for (l, p) in profiles.iter().enumerate() {
            for q in p {
                let (dx, dy) = (rand() * noise, rand() * noise);
                positions.push(DVec3::new(q.x + dx, q.y + dy, l as f64 * step));
            }
        }
        let v = |i: usize, l: usize| (l * k + (i % k)) as u32;
        let mut tris = Vec::new();
        for l in 0..layers {
            for i in 0..k {
                tris.push([v(i, l), v(i + 1, l), v(i + 1, l + 1)]);
                tris.push([v(i, l), v(i + 1, l + 1), v(i, l + 1)]);
            }
        }
        (positions, tris)
    }

    /// The outer skin at z as the stations see it: the right wall up from v = from, the arch,
    /// the left wall down — a noisy run of points.
    fn outer_run(z: f64, from: f64, noise: f64) -> Vec<DVec2> {
        let Wing { r: big, rise, flare, base, .. } = wing_at(z);
        let right = wall_foot(big, rise, flare, base, 1.0);
        let left = wall_foot(big, rise, PI - flare, base, -1.0);
        let k = |y: f64, a: DVec2, b: DVec2| DVec2::new(a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y), y);
        let mut pts = along(k(from, right.1, right.0), right.0, 60);
        for i in 0..200 {
            let ang = flare + (PI - 2.0 * flare) * i as f64 / 200.0;
            pts.push(DVec2::new(big * ang.cos(), rise + big * ang.sin()));
        }
        pts.extend(along(left.0, k(from, left.1, left.0), 60));
        pts.push(k(from, left.1, left.0));
        let mut s: u32 = 11;
        let mut rand = || {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            (s as f64 / 2f64.powi(32)) * 2.0 - 1.0
        };
        pts.into_iter()
            .map(|p| {
                let (dx, dy) = (rand() * noise, rand() * noise);
                DVec2::new(p.x + dx, p.y + dy)
            })
            .collect()
    }

    /// A station drawn as the template is: wall, arch, wall fitted (they carry slice points),
    /// joined tangent, closed by a line across the base.
    fn template_at(z: f64) -> Sketch {
        let Wing { r: big, rise, flare, .. } = wing_at(z);
        let right = wall_foot(big, rise, flare, -15.0, 1.0);
        let left = wall_foot(big, rise, PI - flare, -15.0, -1.0);
        let at = |a: DVec2, b: DVec2, y: f64| DVec2::new(a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y), y);
        let r0 = at(right.1, right.0, -13.0);
        let l0 = at(left.1, left.0, -13.0);
        let run = outer_run(z, -13.0, 0.02);
        let mut s = Sketch::default();
        let fitted = |shape: Shape, points: &[DVec2]| Entity { points: points.to_vec(), scatter: 0.02, ..Entity::new(shape) };
        let wall_r = s.add(fitted(Shape::Line { a: r0, b: right.0 }, &run[..60]));
        let arch = s.add(fitted(Shape::Arc { c: DVec2::new(0.0, rise), r: big, start: flare, sweep: PI - 2.0 * flare }, &run[60..260]));
        let wall_l = s.add(fitted(Shape::Line { a: left.0, b: l0 }, &run[260..]));
        let base = s.add(Entity::new(Shape::Line { a: l0, b: r0 }));
        s.constrain(Rule::Coincident { p: Anchor::new(wall_r, AnchorAt::B), q: Anchor::new(arch, AnchorAt::A) });
        s.constrain(Rule::Coincident { p: Anchor::new(arch, AnchorAt::B), q: Anchor::new(wall_l, AnchorAt::A) });
        s.constrain(Rule::Coincident { p: Anchor::new(wall_l, AnchorAt::B), q: Anchor::new(base, AnchorAt::A) });
        s.constrain(Rule::Coincident { p: Anchor::new(base, AnchorAt::B), q: Anchor::new(wall_r, AnchorAt::A) });
        s.constrain(Rule::Tangent { a: wall_r, b: arch });
        s.constrain(Rule::Tangent { a: arch, b: wall_l });
        s.constrain(Rule::Horizontal { entity: base });
        s.constrain(Rule::Radius { entity: arch, value: big });
        s
    }

    #[test]
    fn fits_five_more_stations_like_the_first_moves_all_six_in_by_the_wall_and_lofts_both_sets() {
        let (positions, triangles) = wing_mesh(60.0, 4.0, 0.02);
        let cut = |frame: &SectionFrame| {
            let c = slice_mesh(&positions, &triangles, frame.origin, frame.normal, 1.0);
            SidedChains {
                chains: c.chains.iter().map(|ch| ch.iter().map(|p| frame.to_sheet(*p)).collect()).collect(),
                normals: c.normals.iter().map(|ns| ns.iter().map(|n| DVec2::new(n.dot(frame.u), n.dot(frame.v))).collect()).collect(),
            }
        };
        let frame_of = |_: &Part, p: &PlaneRef| p.own_frame();
        let mut part = Part::default();
        let first = part.add_named(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 5.0, template_at(5.0))), "Wing 1".into());
        let zs = [15.0, 25.0, 35.0, 45.0, 55.0];
        let mut sections = vec![Section::Sketch(first)];
        sections.extend(zs.iter().map(|z| Section::Plane(PlaneRef::world(WorldAxis::Z, *z))));
        let fit = StationFit { like: first, skin: Some(Skin::Outer), within: Some(SheetBox { min: DVec2::new(-100.0, -13.0), max: DVec2::new(100.0, 100.0) }), tolerance: None };
        let ask = LoftStations { name: Some("Wing".into()), sections, fit: Some(fit), inner: Some(2.0), ruled: Some(false), ..Default::default() };
        let made = add_loft_stations(&mut part, &ask, &cut, &frame_of).unwrap_or_else(|e| panic!("{e}"));
        assert_eq!(made.stations.iter().map(|s| s.name.clone()).collect::<Vec<_>>(), (0..5).map(|k| format!("Wing station {}", k + 2)).collect::<Vec<_>>());
        // Each station the same pieces in the same order as the first, the base line among them.
        for st in &made.stations {
            assert_eq!(st.pieces.iter().map(|p| p.1).collect::<Vec<_>>(), vec![ShapeKind::Line, ShapeKind::Arc, ShapeKind::Line, ShapeKind::Line], "{}", st.name);
        }
        let (inner_loft, twins) = made.inner.clone().unwrap_or_else(|| panic!("no inner"));
        assert_eq!(twins.len(), 6);
        for t in &twins {
            assert_eq!((t.moved, t.kept.clone()), (3, vec![4]), "{}", t.name);
        }
        // The arcs of a station and its twin, R and R − 2.
        let arcs = |id: FeatureId| -> Vec<f64> { part.sketch(id).map_or(Vec::new(), |s| s.sketch.entities.iter().filter_map(|e| if let Shape::Arc { r, .. } = e.shape { Some(r) } else { None }).collect()) };
        let outer_ids: Vec<FeatureId> = std::iter::once(first).chain(made.stations.iter().map(|s| s.id)).collect();
        for (k, (o, t)) in outer_ids.iter().zip(&twins).enumerate() {
            let (ro, ri) = (arcs(*o), arcs(t.id));
            assert_eq!(ro.len(), ri.len());
            for (a, b) in ro.iter().zip(&ri) {
                assert!((a - 2.0 - b).abs() < 1e-3, "station {k}: {a} and {b}");
            }
            let z = if k == 0 { 5.0 } else { zs[k - 1] };
            assert!((ro.iter().copied().fold(0.0, f64::max) - wing_at(z).r).abs() < 0.5, "station {k}: {ro:?}");
        }
        // Two closed solids, the inner inside the outer, the wall between them about 2 mm by
        // the skin's length by the 50 mm lofted; the inner cut from the outer, one solid.
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let body = |id: FeatureId| built.body(BodyId::new(id)).unwrap_or_else(|| panic!("no body of {id}"));
        let (wing, inner) = (body(made.loft), body(inner_loft));
        let (vw, vi) = (volume(&wing.body).unwrap_or(0.0), volume(&inner.body).unwrap_or(0.0));
        assert!(vi < vw, "{vi} {vw}");
        let (bw, bi) = (wing.body.outline_bounds(), inner.body.outline_bounds());
        assert!(bi.min.x > bw.min.x + 1.0 && bi.max.x < bw.max.x - 1.0, "{bw:?} {bi:?}");
        assert!((bi.max.y - (bw.max.y - 2.0)).abs() < 0.5, "{bw:?} {bi:?}");
        assert!((bi.min.y - bw.min.y).abs() < 0.05, "{bw:?} {bi:?}");
        let skin = |z: f64| {
            let w = wing_at(z);
            w.r * (PI - 2.0 * w.flare) + 2.0 * 30.0
        };
        let wall = vw - vi;
        assert!(wall > 2.0 * skin(5.0) * 50.0 * 0.8 && wall < 2.0 * skin(55.0) * 50.0 * 1.1, "{wall}");
        part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(made.loft), tool: BodyId::new(inner_loft), keep: false }));
        let built = build(&part);
        assert!(built.errors.is_empty(), "{:?}", built.errors);
        let shell = built.body(BodyId::new(made.loft)).and_then(|b| volume(&b.body)).unwrap_or(0.0);
        assert!((shell - wall).abs() < 10.0, "{shell} {wall}");
    }

    #[test]
    fn refuses_the_recipe_without_a_station_to_copy_and_names_taken() {
        let cut = |_: &SectionFrame| SidedChains::default();
        let frame_of = |_: &Part, p: &PlaneRef| p.own_frame();
        let mut part = Part::default();
        let ask = LoftStations { sections: vec![Section::Plane(PlaneRef::world(WorldAxis::Z, 15.0)), Section::Plane(PlaneRef::world(WorldAxis::Z, 25.0))], ..Default::default() };
        let said = add_loft_stations(&mut part, &ask, &cut, &frame_of).err().unwrap_or_default();
        assert!(said.starts_with("fit.like names"), "{said}");
        let ask = LoftStations { sections: vec![Section::Plane(PlaneRef::world(WorldAxis::Z, 15.0))], ..Default::default() };
        assert_eq!(add_loft_stations(&mut part, &ask, &cut, &frame_of).err().as_deref(), Some("A loft takes two sketches or more — sketches, or sections to fit its stations on."));
        let a = part.add_named(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 5.0, template_at(5.0))), "Wing 1".into());
        part.add_named(FeatureKind::Sketch(SketchFeature::on_world(WorldAxis::Z, 9.0, template_at(9.0))), "Wing station 2".into());
        let fit = StationFit { like: a, skin: None, within: None, tolerance: None };
        let ask = LoftStations { name: Some("Wing".into()), sections: vec![Section::Sketch(a), Section::Plane(PlaneRef::world(WorldAxis::Z, 25.0))], fit: Some(fit), ..Default::default() };
        let said = add_loft_stations(&mut part, &ask, &cut, &frame_of).err().unwrap_or_default();
        assert_eq!(said, "Feature 2 is already named “Wing station 2” — the commands find features by name, so each needs its own: “Wing station 3”, say.");
        let ask = LoftStations { sections: vec![Section::Sketch(a), Section::Sketch(a)], exprs: vec![("inner".into(), "0".into())], ..Default::default() };
        let said = add_loft_stations(&mut part, &ask, &cut, &frame_of).err().unwrap_or_default();
        assert_eq!(said, "exprs.inner: “0” comes to 0 mm — the wall is above zero.");
        let ask = LoftStations { sections: vec![Section::Sketch(a), Section::Sketch(a)], exprs: vec![("sections.5.offset".into(), "3".into())], ..Default::default() };
        let said = add_loft_stations(&mut part, &ask, &cut, &frame_of).err().unwrap_or_default();
        assert_eq!(said, "exprs.sections.5.offset: there are 2 sections — sections.0.offset to sections.1.offset.");
    }
}
