//! Virtual stent (Vessel → Virtual stent, or from two Measure → Vessel diameter sections):
//! click the vessel at the two ends of the stretch to treat, choose the stent's diameter and
//! length, how straight it holds and how hard it pushes, and whether it is covered; the vessel
//! is reshaped around it (see [`mm_edit::stent`]), previewed live, then kept in place or as a
//! copy.
//!
//! The vessel's centerline is traced from the first section (geodesic rings, as the
//! Centerline tool does) through the second and on past both, and the vertices whose rings lie
//! on it are the stented vessel's own wall; side branches follow them with distance.

use super::*;
use crate::centerline::{self, Centerline, CenterlineParams};
use crate::jobs::Job;
use crate::ui_util;
use mm_edit::EndContinuity;
use rayon::prelude::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// One end of the stretch: the vessel's cross-section through a point of its wall.
#[derive(Clone, Debug)]
pub struct Section {
    pub center: DVec3,
    pub normal: DVec3,
    /// Equivalent diameter (of a circle with the cross-section's area).
    pub diameter: f64,
    /// The cross-section's outline.
    pub ring: Vec<[DVec3; 2]>,
}

impl Section {
    /// The cross-section through `point` on the wall (as Measure → Vessel diameter).
    pub fn at(mesh: &Mesh, bvh: &mm_core::Bvh, point: DVec3) -> Option<Self> {
        let x = mm_edit::estimate_vessel_cross_section(mesh, Some(bvh), point, None).ok()?;
        let curve = mm_edit::nearest_intersection_curve(mesh, x.plane, point).filter(|c| c.closed)?;
        let diameter = x.area.map_or(2.0 * x.radius, |a| 2.0 * (a / std::f64::consts::PI).sqrt());
        Some(Self { center: x.center, normal: x.axis, diameter, ring: curve.segments })
    }

    fn stent(&self) -> mm_edit::StentSection {
        mm_edit::StentSection { center: self.center, normal: self.normal }
    }
}

/// The stent's settings as shown (percentages for the 0..1 ones).
#[derive(Clone, Copy, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
struct StentUi {
    /// The stent's nominal (fully expanded) diameter, mm (0.5..60). Follows the wider end until set.
    diameter: f64,
    /// Length along the stent's own axis, mm (1..400), centred on the two ends. Follows the
    /// centerline between the ends (up to the next 5 mm) until set.
    length: f64,
    /// 0..100 %: 0 follows the vessel's centerline as it is, 100 takes the smoothest path
    /// between the ends, straightening the vessel where the stent grips it.
    straightness: f64,
    /// The smoothest path: G0 a straight line between the ends' centres, G1 leaving each end
    /// square to its cross-section, G2 also continuing the vessel's bend.
    continuity: EndContinuity,
    /// 0..100 %: radial force. Along the stent, in each direction across it, the lumen lies
    /// this share of the way from the vessel's own wall to the stent's radius (2 mm across
    /// natively, a 6 mm stent and 50 % give 4 mm), so the vessel's shape eases into the round
    /// stent: 100 makes a perfect tube of the stent's diameter, 0 leaves the vessel as it is.
    /// A bare stent never pulls wider wall in; a covered one narrows it to the stent.
    stiffness: f64,
    /// How far past each end the wall eases back to its own shape and course, mm (0..40).
    /// Nothing beyond it changes; a straightened stent may need a longer one.
    end_blend: f64,
    /// Stent graft: the lumen along it becomes the stent's tube (wider wall narrowed, side
    /// branches opening along it sealed off and removed).
    covered: bool,
    /// Even triangles over the reshaped region (recommended for CFD).
    remesh: bool,
}

impl Default for StentUi {
    fn default() -> Self {
        Self { diameter: 8.0, length: 40.0, straightness: 50.0, continuity: EndContinuity::G1, stiffness: 100.0, end_blend: 2.0, covered: false, remesh: true }
    }
}

impl StentUi {
    fn params(&self) -> mm_edit::StentParams {
        mm_edit::StentParams {
            diameter: self.diameter,
            length: self.length,
            straightness: self.straightness / 100.0,
            continuity: self.continuity,
            stiffness: self.stiffness / 100.0,
            end_blend: self.end_blend,
            covered: self.covered,
            remesh: self.remesh,
            edge_length: None,
        }
    }
}

/// The vessel's centerline through the two sections and on past both, and which vertices
/// belong to it.
struct VesselTrace {
    path: mm_edit::VesselPath,
    /// The stent's ends: where the path crosses each section's plane (the lumen's centre
    /// there), else the section's centre.
    ends: [DVec3; 2],
    owner: Vec<bool>,
    /// Centerline length between the sections.
    span: f64,
    /// The minimal lumen diameter between the sections: the largest sphere inside the lumen
    /// at its narrowest (where the inscribed-sphere centerline was found).
    min_lumen: Option<f64>,
}

/// What a preview was made from: the sections (by revision) and the settings.
#[derive(Clone, Copy, PartialEq, Debug)]
struct StentKey {
    sections: u64,
    ui: StentUi,
}

struct StentOut {
    source: Arc<Mesh>,
    mesh: Mesh,
    placement: mm_edit::StentPlacement,
    report: mm_edit::StentReport,
    crossings: usize,
}

const SETTLE: Duration = Duration::from_millis(150);

pub struct StentTool {
    object: Option<ObjectId>,
    sections: Vec<Section>,
    /// Bumped whenever the sections change.
    revision: u64,
    ui: StentUi,
    /// Keep the result in the object itself rather than a copy.
    in_place: bool,
    /// The diameter / length were typed in (otherwise they follow the sections).
    sized: [bool; 2],
    trace: Option<(u64, Arc<Mesh>, Arc<VesselTrace>)>,
    trace_job: Option<(u64, Job<VesselTrace>)>,
    job: Option<(StentKey, Job<StentOut>)>,
    preview: Option<(StentKey, StentOut)>,
    changed: Option<(StentKey, Instant)>,
    error: Option<(StentKey, String)>,
    pressed: Option<egui::Pos2>,
}

impl StentTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self {
            object: cx.scene.active,
            sections: Vec::new(),
            revision: 0,
            ui: StentUi::default(),
            in_place: false,
            sized: [false; 2],
            trace: None,
            trace_job: None,
            job: None,
            preview: None,
            changed: None,
            error: None,
            pressed: None,
        }
    }

    /// Start from two cross-sections (e.g. Measure's vessel diameters) of `object`.
    pub fn with_sections(cx: &mut ToolCx, object: ObjectId, sections: [Section; 2]) -> Self {
        let mut t = Self::new(cx);
        t.object = Some(object);
        t.sections = sections.to_vec();
        t.sections_changed();
        t
    }

    fn sections_changed(&mut self) {
        self.revision += 1;
        self.trace = None;
        if let Some((_, j)) = self.trace_job.take() {
            j.cancel();
        }
        if self.sections.len() == 2 && !self.sized[0] {
            // The wider end's diameter, to the half millimetre.
            let d = self.sections.iter().map(|s| s.diameter).fold(0.0, f64::max);
            self.ui.diameter = ((d * 2.0).round() / 2.0).max(0.5);
        }
    }

    fn key(&self) -> Option<StentKey> {
        (self.sections.len() == 2).then_some(StentKey { sections: self.revision, ui: self.ui })
    }

    /// Set an end at the wall point `p`: the first two clicks set ends 1 and 2, later ones
    /// move the nearer end (or end `which`, 0 or 1, when given).
    fn set_end(&mut self, cx: &mut ToolCx, id: ObjectId, p: DVec3, which: Option<usize>) -> Result<usize, String> {
        if self.object != Some(id) {
            self.object = Some(id);
            self.sections.clear();
        }
        if which.is_some_and(|i| i > self.sections.len()) {
            return Err("set end 1 first".into());
        }
        let o = cx.scene.get_mut(id).ok_or("no object")?;
        let bvh = o.bvh();
        let s = Section::at(&o.mesh, &bvh, p).ok_or("No vessel cross-section there: pick the wall of a tube-like vessel.")?;
        let i = match which {
            Some(i) => i,
            None if self.sections.len() < 2 => self.sections.len(),
            None => usize::from(self.sections[0].center.distance(p) > self.sections[1].center.distance(p)),
        };
        if i == self.sections.len() {
            self.sections.push(s);
        } else {
            self.sections[i] = s;
        }
        self.sections_changed();
        Ok(i)
    }

    /// The reshaped vessel for `key` (in the background, or at once for scripts).
    fn build(source: &Mesh, trace: &VesselTrace, sections: [mm_edit::StentSection; 2], key: &StentKey) -> anyhow::Result<(Mesh, mm_edit::StentPlacement, mm_edit::StentReport, usize)> {
        let params = key.ui.params();
        // Centred in the lumen at each section (on the traced path).
        let sections = [0, 1].map(|i| mm_edit::StentSection { center: trace.ends[i], normal: sections[i].normal });
        let placement = mm_edit::place_stent(&trace.path, sections, &params)?;
        let mut mesh = source.clone();
        let report = mm_edit::apply_stent(&mut mesh, &placement, &params, Some(&trace.owner))?;
        // Only crossings the stent made (a mesh may cross itself already, e.g. where
        // vessels were merged without a union).
        let crossings = ops::crossings(&mesh, &report.triangles).saturating_sub(ops::crossings(source, &report.source_triangles));
        Ok((mesh, placement, report, crossings))
    }

    /// Keep the trace and preview up to date: collect finished jobs, and start one for
    /// settings that have settled.
    fn update(&mut self, cx: &mut ToolCx) {
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else { return };
        let source = o.mesh.clone();
        if let Some((rev, j)) = &self.trace_job {
            if let Some(r) = j.poll() {
                let rev = *rev;
                self.trace_job = None;
                match r {
                    Ok(t) if rev == self.revision => {
                        if !self.sized[1] {
                            // The stretch's length, up to the next 5 mm.
                            self.ui.length = ((t.span / 5.0).ceil() * 5.0).max(5.0);
                        }
                        self.trace = Some((rev, source.clone(), Arc::new(t)));
                    }
                    Ok(_) => {}
                    Err(e) => {
                        if let Some(k) = self.key() {
                            self.error = Some((k, format!("{e:#}")));
                        }
                    }
                }
            }
        }
        if let Some((k, j)) = &self.job {
            if let Some(r) = j.poll() {
                let k = *k;
                self.job = None;
                match r {
                    Ok(out) => {
                        self.preview = Some((k, out));
                        self.error = None;
                    }
                    Err(e) => self.error = Some((k, format!("{e:#}"))),
                }
            }
        }
        let Some(key) = self.key() else {
            self.preview = None;
            self.changed = None;
            if let Some((_, j)) = self.job.take() {
                j.cancel();
            }
            return;
        };
        let current = |k: Option<&StentKey>| k == Some(&key);
        if current(self.preview.as_ref().map(|p| &p.0)) || current(self.job.as_ref().map(|j| &j.0)) || current(self.error.as_ref().map(|e| &e.0)) {
            self.changed = None;
            return;
        }
        // The trace first (once per pair of sections and mesh).
        let traced = self.trace.as_ref().is_some_and(|(rev, m, _)| *rev == self.revision && Arc::ptr_eq(m, &source));
        if !traced {
            if self.trace_job.as_ref().is_none_or(|(rev, _)| *rev != self.revision) {
                let sections = [self.sections[0].clone(), self.sections[1].clone()];
                let mesh = source.clone();
                self.trace_job = Some((self.revision, Job::spawn("Centerline", cx.egui.clone(), move |_| trace_vessel(&mesh, &sections))));
            }
            return;
        }
        let now = Instant::now();
        match self.changed {
            Some((k, since)) if k == key && now - since >= SETTLE => self.start(key, cx),
            Some((k, since)) if k == key => cx.egui.request_repaint_after(SETTLE - (now - since)),
            _ => {
                self.changed = Some((key, now));
                cx.egui.request_repaint_after(SETTLE);
            }
        }
    }

    fn start(&mut self, key: StentKey, cx: &mut ToolCx) {
        self.changed = None;
        let Some((_, source, trace)) = self.trace.clone() else { return };
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let sections = [self.sections[0].stent(), self.sections[1].stent()];
        self.job = Some((
            key,
            Job::spawn("Stent", cx.egui.clone(), move |_| {
                let (mesh, placement, report, crossings) = Self::build(&source, &trace, sections, &key)?;
                Ok(StentOut { source, mesh, placement, report, crossings })
            }),
        ));
    }

    /// Keep the reshaped vessel (the preview if it is up to date, else built now), in the
    /// object or as a copy. Returns the status.
    fn apply(&mut self, cx: &mut ToolCx) -> Result<String, String> {
        let key = self.key().ok_or("click the vessel at both ends of the stretch first")?;
        let id = self.object.ok_or("no object")?;
        let o = cx.scene.get(id).ok_or("no object")?;
        let (source, name, frame) = (o.mesh.clone(), o.name.clone(), o.frame);
        let (mesh, report) = match self.preview.take() {
            Some((k, out)) if k == key && Arc::ptr_eq(&out.source, &source) => (out.mesh, out.report),
            _ => {
                let trace = match &self.trace {
                    Some((rev, m, t)) if *rev == self.revision && Arc::ptr_eq(m, &source) => t.clone(),
                    _ => Arc::new(trace_vessel(&source, &[self.sections[0].clone(), self.sections[1].clone()]).map_err(|e| format!("{e:#}"))?),
                };
                let (mesh, _, report, _) = Self::build(&source, &trace, [self.sections[0].stent(), self.sections[1].stent()], &key).map_err(|e| format!("{e:#}"))?;
                (mesh, report)
            }
        };
        if let Some((_, j)) = self.job.take() {
            j.cancel();
        }
        let label = format!("⌀ {} × {} mm{}", trim(self.ui.diameter), trim(self.ui.length), if self.ui.covered { " covered" } else { "" });
        cx.history.push("Stent", cx.scene);
        if self.in_place {
            cx.scene.get_mut(id).unwrap().set_mesh(mesh);
        } else {
            let nid = cx.scene.add(format!("{name} stent {label}"), mesh);
            if let Some(n) = cx.scene.get_mut(nid) {
                n.frame = frame;
            }
            if let Some(o) = cx.scene.get_mut(id) {
                o.visible = false;
            }
            self.object = Some(nid);
        }
        self.sections.clear();
        self.sections_changed();
        self.error = None;
        Ok(format!("Stent {label}: narrowest lumen ⌀ {:.2} → {:.2} mm{}", 2.0 * report.narrowest[0], 2.0 * report.narrowest[1], sealed_text(&report)))
    }
}

/// What the preview warns about.
fn warnings(out: &StentOut) -> Vec<String> {
    let pl = &out.placement;
    let mut w = Vec::new();
    if let Some(e) = (0..2).filter(|&e| pl.return_bend[e] > 0.7).max_by(|&a, &b| pl.return_bend[a].total_cmp(&pl.return_bend[b])) {
        w.push(format!("Past end {} the vessel has to bend back sharply onto its course within the end blend, so its wall may crease: lengthen the end blend or lower the straightness.", e + 1));
    }
    let moved = out.report.moved_openings.saturating_sub(out.report.sealed_openings);
    if moved > 0 {
        w.push(format!("The reshaping reaches {moved} open end{} of the mesh and moves {}: check the inlets and outlets, or lower the straightness.", if moved == 1 { "" } else { "s" }, if moved == 1 { "it" } else { "them" }));
    }
    if out.report.outside {
        w.push("The stent's axis runs outside the vessel (a tight bend): lower the straightness.".into());
    }
    if out.crossings > 0 {
        w.push(format!("The reshaped wall runs into itself ({} new crossings): lower the straightness or the stent's length.", out.crossings));
    }
    // (Only with crossings: a wide landing often joins cleanly.)
    for (e, &wide) in out.report.landing.iter().enumerate() {
        if out.crossings > 0 && wide > 1.5 {
            w.push(format!("End {} of the covered stent lands in a lumen {wide:.1}× its width, where its flare to the wall can fold (a real graft would leave a gap there): shorten the stent or move that end into vessel about the stent's size.", e + 1));
        }
    }
    if let Some(why) = &out.report.pushed {
        w.push(format!("The wall was pushed out to the stent rather than joined to its tube ({why}), so it may be uneven where it was narrowest."));
    }
    w
}

fn trim(x: f64) -> String {
    let s = format!("{x:.2}");
    s.trim_end_matches('0').trim_end_matches('.').to_string()
}

fn sealed_text(r: &mm_edit::StentReport) -> String {
    match r.sealed_openings {
        0 if r.removed_triangles > 0 => "; the stretch is lined with the stent".into(),
        0 => String::new(),
        n => format!("; {n} open end{} sealed off and removed", if n == 1 { "" } else { "s" }),
    }
}

fn point_segment_distance(p: DVec3, a: DVec3, b: DVec3) -> f64 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

/// Trace the vessel's centerline from the first section through the second and on past
/// both (the straightest way at each junction).
fn trace_vessel(mesh: &Mesh, sections: &[Section; 2]) -> anyhow::Result<VesselTrace> {
    let [a, b] = sections;
    let h = mesh.average_edge_length();
    let n = a.normal.normalize_or_zero();
    let sources: Vec<u32> = mesh
        .vertex_ids()
        .filter(|&v| {
            let p = mesh.vertex(v);
            (p - a.center).dot(n).abs() < h && a.ring.iter().any(|s| point_segment_distance(p, s[0], s[1]) < h)
        })
        .collect();
    if sources.is_empty() {
        anyhow::bail!("could not find the first end's cross-section on the surface");
    }
    let cl = centerline::compute(mesh, &sources, 0.5 * a.diameter, &CenterlineParams::default());
    let mut on_branch = vec![false; cl.nodes.len()];
    let mut adjacent: Vec<Vec<usize>> = vec![Vec::new(); cl.nodes.len()];
    for br in &cl.branches {
        for w in br.nodes.windows(2) {
            adjacent[w[0]].push(w[1]);
            adjacent[w[1]].push(w[0]);
        }
        for &i in &br.nodes {
            on_branch[i] = true;
        }
    }
    let near = |p: DVec3| (0..cl.nodes.len()).filter(|&i| on_branch[i]).min_by(|&i, &j| cl.nodes[i].position.distance(p).total_cmp(&cl.nodes[j].position.distance(p)));
    let (Some(na), Some(nb)) = (near(a.center), near(b.center)) else {
        anyhow::bail!("could not trace the vessel's centerline");
    };
    let chain = chain_between(&cl, &adjacent, na, nb).ok_or_else(|| anyhow::anyhow!("the two ends are not on one connected vessel"))?;
    if chain.len() < 2 {
        anyhow::bail!("the two ends are too close together along the vessel");
    }
    let pos = |i: usize| cl.nodes[i].position;
    let length = |c: &[usize]| c.windows(2).map(|w| pos(w[0]).distance(pos(w[1]))).sum::<f64>();
    let reach = (3.0 * length(&chain)).max(20.0 * a.diameter.max(b.diameter));
    let before = walk_on(&cl, &adjacent, chain[0], chain[1], reach);
    let after = walk_on(&cl, &adjacent, chain[chain.len() - 1], chain[chain.len() - 2], reach);
    let nodes: Vec<usize> = before.iter().rev().chain(&chain).chain(&after).copied().collect();
    let on_path: std::collections::HashSet<usize> = nodes.iter().copied().collect();
    let owner = cl.vertex_node.iter().map(|n| on_path.contains(n)).collect();
    let mut path = mm_edit::VesselPath { points: nodes.iter().map(|&i| pos(i)).collect(), radii: nodes.iter().map(|&i| cl.nodes[i].radius).collect() };
    let min_lumen = inscribed_between(mesh, &mut path, sections);
    if min_lumen.is_none() {
        recentre(mesh, &mut path, sections);
    }
    let ends = [a, b].map(|s| crossing(&path.points, s.center, s.normal).filter(|c| c.distance(s.center) < 0.5 * s.diameter).unwrap_or(s.center));
    let span = path.span(ends.map(|center| mm_edit::StentSection { center, normal: DVec3::Z })).unwrap_or(0.0);
    Ok(VesselTrace { path, ends, owner, span, min_lumen })
}

/// Where `path` crosses the plane through `c` square to `n`, nearest `c`.
fn crossing(path: &[DVec3], c: DVec3, n: DVec3) -> Option<DVec3> {
    path.windows(2)
        .filter_map(|w| {
            let (da, db) = ((w[0] - c).dot(n), (w[1] - c).dot(n));
            (da * db <= 0.0 && da != db).then(|| w[0].lerp(w[1], da / (da - db)))
        })
        .min_by(|p, q| p.distance(c).total_cmp(&q.distance(c)))
}

/// Between the sections, replace the traced path by the lumen's centerline through the
/// centres of the largest spheres inside it (see [`centerline::inscribed_path`]), with those
/// spheres' radii. Returns the minimal lumen diameter there; `None` (the path unchanged) where
/// the centerline cannot be found.
fn inscribed_between(mesh: &Mesh, path: &mut mm_edit::VesselPath, sections: &[Section; 2]) -> Option<f64> {
    let near = |c: DVec3| (0..path.points.len()).min_by(|&i, &j| path.points[i].distance(c).total_cmp(&path.points[j].distance(c)));
    let (ia, ib) = (near(sections[0].center)?, near(sections[1].center)?);
    let ((ia, a), (ib, b)) = if ia <= ib { ((ia, sections[0].center), (ib, sections[1].center)) } else { ((ib, sections[1].center), (ia, sections[0].center)) };
    if ib <= ia {
        return None;
    }
    let lumen = centerline::inscribed_path(mesh, &path.points[ia..=ib], &path.radii[ia..=ib], a, b)?;
    let narrowest = 2.0 * lumen.radii.iter().copied().fold(f64::INFINITY, f64::min);
    let mut points: Vec<DVec3> = path.points[..ia].iter().chain(&lumen.points).chain(&path.points[ib + 1..]).copied().collect();
    let radii: Vec<f64> = path.radii[..ia].iter().chain(&lumen.radii).chain(&path.radii[ib + 1..]).copied().collect();
    // Where the traced path gives way to the lumen's centerline (at each end, the deepest point
    // of the lumen near it, which can lie a little aside of the traced path), the two are
    // eased together over about one and a half of the vessel's radii there: a corner in the
    // path would turn the vessel's slices apart as the stent carries them.
    let joins = [(ia, 0.5 * sections[0].diameter.max(sections[1].diameter.min(sections[0].diameter))), (ia + lumen.points.len() - 1, 0.5 * sections[1].diameter)];
    for (j, radius) in joins {
        ease_corner(&mut points, j, 1.5 * radius);
    }
    path.points = points;
    path.radii = radii;
    Some(narrowest)
}

/// Smooth `points` within `reach` (along the path) of point `j`, the points at that distance
/// held, so a corner at `j` becomes a gentle bend.
fn ease_corner(points: &mut [DVec3], j: usize, reach: f64) {
    let n = points.len();
    if n < 3 || j >= n {
        return;
    }
    let (mut lo, mut d) = (j, 0.0);
    while lo > 0 && d < reach {
        d += points[lo].distance(points[lo - 1]);
        lo -= 1;
    }
    let (mut hi, mut d) = (j, 0.0);
    while hi + 1 < n && d < reach {
        d += points[hi].distance(points[hi + 1]);
        hi += 1;
    }
    if hi < lo + 2 {
        return;
    }
    // Resampled evenly in between, then relaxed (the ends of the stretch fixed).
    let stretch: Vec<DVec3> = points[lo..=hi].to_vec();
    let mut cum = vec![0.0];
    for w in stretch.windows(2) {
        cum.push(cum[cum.len() - 1] + w[0].distance(w[1]));
    }
    let total = cum[cum.len() - 1].max(1e-12);
    let at = |s: f64| {
        let i = cum.partition_point(|&c| c <= s).clamp(1, cum.len() - 1) - 1;
        let f = ((s - cum[i]) / (cum[i + 1] - cum[i]).max(1e-12)).clamp(0.0, 1.0);
        stretch[i].lerp(stretch[i + 1], f)
    };
    let m = hi - lo;
    let mut even: Vec<DVec3> = (0..=m).map(|i| at(total * i as f64 / m as f64)).collect();
    for _ in 0..(4 * m).max(20) {
        let old = even.clone();
        for i in 1..m {
            even[i] = old[i] * 0.5 + (old[i - 1] + old[i + 1]) * 0.25;
        }
    }
    points[lo..=hi].copy_from_slice(&even);
}

/// Move the path to the lumen's centre between the sections (the stretch the stent treats).
/// Geodesic rings' centroids can lie millimetres off it, and outside the lumen, through a
/// narrowed stretch that bends or narrows eccentrically (a stenosis); the stent would then
/// push the wall open about a point outside the vessel and fold it onto itself. About the
/// lumen's centre, cross-sections are star-shaped, as the push needs.
///
/// The path is resampled densely enough to follow a narrow channel. Each point's target is the
/// centre of the lumen's cross-section square to the path there (its centroid if well inside,
/// else the centre of the largest circle inside, as for a crescent). Where no outline is found
/// near a point, or the nearest is too far to be this vessel's, the target is interpolated
/// from the neighbours'. Only lumens about as narrow as the ends or narrower move the path
/// (the stent pushes no wider wall, and a wide chamber's centre says nothing about the
/// vessel's course). The moves are smoothed along the path over the lumen's radius and fade
/// in from each section over about the ends' radius; a little past the sections the path
/// keeps its course, which the stent's return onto the vessel follows.
fn recentre(mesh: &Mesh, path: &mut mm_edit::VesselPath, sections: &[Section; 2]) {
    if path.points.len() < 3 {
        return;
    }
    let h = mesh.average_edge_length();
    // Resample: a quarter of the local radius apart (0.2 to 1 mm, never finer than the mesh).
    let (mut pts, mut radii) = (vec![path.points[0]], vec![path.radii[0]]);
    for i in 1..path.points.len() {
        let (a, b) = (path.points[i - 1], path.points[i]);
        let step = (0.25 * path.radii[i - 1].min(path.radii[i])).clamp(0.2_f64.max(0.5 * h), 1.0_f64.max(0.5 * h));
        let k = (a.distance(b) / step).ceil().max(1.0) as usize;
        for j in 1..=k {
            let f = j as f64 / k as f64;
            pts.push(a.lerp(b, f));
            radii.push(path.radii[i - 1] + (path.radii[i] - path.radii[i - 1]) * f);
        }
    }
    let n = pts.len();
    let mut cum = vec![0.0];
    for w in pts.windows(2) {
        cum.push(cum[cum.len() - 1] + w[0].distance(w[1]));
    }
    // The stretch to recentre: between the sections, fading in from each over about the ends'
    // radius (or a quarter of the stretch).
    let at = |c: DVec3| (0..n).min_by(|&i, &j| pts[i].distance(c).total_cmp(&pts[j].distance(c))).map_or(0.0, |i| cum[i]);
    let (sa, sb) = (at(sections[0].center), at(sections[1].center));
    let (sa, sb) = (sa.min(sb), sa.max(sb));
    let wide = 0.5 * sections[0].diameter.max(sections[1].diameter);
    let fade = (0.5 * wide).min(0.25 * (sb - sa)).max(1e-9);
    let smoothstep = |x: f64| {
        let x = x.clamp(0.0, 1.0);
        x * x * (3.0 - 2.0 * x)
    };
    let weight: Vec<f64> = cum.iter().map(|&s| smoothstep((s - sa).min(sb - s) / fade)).collect();
    let active: Vec<usize> = (0..n).filter(|&i| weight[i] > 0.0).collect();
    if active.is_empty() {
        return;
    }
    let pad = DVec3::splat(4.0 * radii.iter().copied().fold(0.0, f64::max) + 2.0 * h);
    let (lo, hi) = active.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(a, b), &i| (a.min(pts[i]), b.max(pts[i])));
    let (lo, hi) = (lo - pad, hi + pad);
    let near: Vec<u32> = mesh.triangle_ids().filter(|&t| mesh.tri_positions(t).iter().any(|p| p.cmpge(lo).all() && p.cmple(hi).all())).collect();
    let bvh = mm_core::Bvh::build_subset(mesh, &near);
    // Points within `mm` of point i along the path, each side.
    let window = |i: usize, mm: f64| -> usize {
        let spacing = (cum[(i + 1).min(n - 1)] - cum[i.saturating_sub(1)]).max(1e-9) / 2.0;
        ((mm / spacing).round() as usize).clamp(1, 40)
    };
    // The lumen's outline in the plane square to the path at point i (the smallest outline
    // around the point, else the nearest close enough), with that plane's normal.
    let lumen = |pts: &[DVec3], i: usize| -> Option<(mm_edit::IntersectionCurve, DVec3)> {
        // The plane follows the path over about a radius, not its jitter.
        let span = window(i, radii[i]);
        let t = (pts[(i + span).min(n - 1)] - pts[i.saturating_sub(span)]).normalize_or_zero();
        if t == DVec3::ZERO {
            return None;
        }
        let p = pts[i];
        let reach = 3.0 * radii[i] + 2.0 * h + 1.0;
        let curves = mm_edit::local_intersection_curves(mesh, &bvh, mm_core::Plane::from_point_normal(p, t), p, reach);
        let closed = curves.into_iter().filter(|c| c.closed && c.area > 0.0);
        let (around, other): (Vec<_>, Vec<_>) = closed.partition(|c| c.encloses(p, t));
        let c = around.into_iter().min_by(|a, b| a.area.total_cmp(&b.area)).or_else(|| other.into_iter().filter(|c| c.distance(p) < 2.0 * radii[i] + 1.0).min_by(|a, b| a.distance(p).total_cmp(&b.distance(p))))?;
        Some((c, t))
    };
    for _ in 0..2 {
        // Targets: each cross-section's centre, with the radius of the largest circle inside.
        let found: Vec<Option<(DVec3, f64)>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if weight[i] <= 0.0 {
                    return None;
                }
                let (c, t) = lumen(&pts, i)?;
                let (centre, r) = c.inscribed_centre(t)?;
                if r > 1.5 * wide {
                    // A chamber: no move here.
                    return Some((pts[i], r));
                }
                let fits = c.encloses(c.centroid, t) && c.distance(c.centroid) >= 0.7 * r;
                let target = if fits { c.centroid } else { centre };
                // Further than that, it is another vessel's lumen.
                (target.distance(pts[i]) <= 3.0 * radii[i].max(r) + h).then_some((target, r))
            })
            .collect();
        // Gaps take their neighbours' moves (interpolated).
        let known: Vec<usize> = active.iter().copied().filter(|&i| found[i].is_some()).collect();
        if known.is_empty() {
            return;
        }
        let mut moves = vec![DVec3::ZERO; n];
        let mut reach = radii.clone();
        for &i in &active {
            let j = known.partition_point(|&k| k < i);
            // Lumens wider than the ends move less, chambers not at all.
            let narrow = found[i].map_or(1.0, |(_, r)| 1.0 - smoothstep((r - wide) / (0.5 * wide)));
            moves[i] = match (j.checked_sub(1).map(|j| known[j]), known.get(j).copied()) {
                (_, Some(k)) if k == i => found[i].unwrap().0 - pts[i],
                (Some(a), Some(b)) => {
                    let f = (i - a) as f64 / (b - a) as f64;
                    (found[a].unwrap().0 - pts[a]).lerp(found[b].unwrap().0 - pts[b], f)
                }
                (Some(a), None) => found[a].unwrap().0 - pts[a],
                (None, Some(b)) => found[b].unwrap().0 - pts[b],
                (None, None) => DVec3::ZERO,
            } * weight[i]
                * narrow;
            if let Some((_, r)) = found[i] {
                reach[i] = r;
            }
        }
        // Smoothed along the path over the lumen's radius (wide lumens smooth more), easing out
        // within two fades past the sections.
        for _ in 0..2 {
            moves = (0..n)
                .map(|i| {
                    let half = window(i, reach[i]);
                    let w = i.saturating_sub(half)..=(i + half).min(n - 1);
                    moves[w.clone()].iter().sum::<DVec3>() / w.count() as f64
                })
                .collect();
        }
        for i in 0..n {
            moves[i] *= 1.0 - smoothstep(((sa - cum[i]).max(cum[i] - sb) - fade) / fade);
        }
        let moved: Vec<DVec3> = (0..n).map(|i| pts[i] + moves[i]).collect();
        // A smoothed point that left a narrow channel takes its own target.
        pts = (0..n)
            .into_par_iter()
            .map(|i| match found[i] {
                Some((target, _)) if weight[i] >= 1.0 => match lumen(&moved, i) {
                    Some((c, t)) if c.encloses(moved[i], t) => moved[i],
                    _ => target,
                },
                _ => moved[i],
            })
            .collect();
    }
    path.points = pts;
    path.radii = radii;
}

/// Nodes from `from` to `to` through the branch tree (breadth first over branch links).
fn chain_between(cl: &Centerline, adjacent: &[Vec<usize>], from: usize, to: usize) -> Option<Vec<usize>> {
    let mut prev = vec![usize::MAX; cl.nodes.len()];
    let mut queue = std::collections::VecDeque::from([from]);
    prev[from] = from;
    while let Some(x) = queue.pop_front() {
        if x == to {
            break;
        }
        for &y in &adjacent[x] {
            if prev[y] == usize::MAX {
                prev[y] = x;
                queue.push_back(y);
            }
        }
    }
    if prev[to] == usize::MAX {
        return None;
    }
    let mut out = vec![to];
    while *out.last().unwrap() != from {
        out.push(prev[*out.last().unwrap()]);
    }
    out.reverse();
    Some(out)
}

/// Nodes on past `end` away from `back`, taking the straightest way at junctions, for about
/// `reach`.
fn walk_on(cl: &Centerline, adjacent: &[Vec<usize>], end: usize, back: usize, reach: f64) -> Vec<usize> {
    let pos = |i: usize| cl.nodes[i].position;
    let (mut prev, mut cur, mut len) = (back, end, 0.0);
    let mut seen: std::collections::HashSet<usize> = [back, end].into_iter().collect();
    let mut out = Vec::new();
    while len < reach {
        let dir = (pos(cur) - pos(prev)).normalize_or_zero();
        let Some(next) = adjacent[cur].iter().copied().filter(|n| !seen.contains(n)).max_by(|&x, &y| (pos(x) - pos(cur)).normalize_or_zero().dot(dir).total_cmp(&(pos(y) - pos(cur)).normalize_or_zero().dot(dir))) else {
            break;
        };
        len += pos(cur).distance(pos(next));
        seen.insert(next);
        out.push(next);
        (prev, cur) = (cur, next);
    }
    out
}

impl Tool for StentTool {
    fn name(&self) -> &'static str {
        "Virtual stent"
    }

    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        // `end:x,y,z` (the nearest wall point), `clear`, `apply`, and the settings:
        // `diameter`, `length`, `straightness` / `stiffness` (%), `ends:g0|g1|g2`, `blend`,
        // `covered:on|off`, `remesh:on|off`, `inplace:on|off`.
        let (key, val) = cmd.split_once(':').unwrap_or((cmd, ""));
        let num = val.parse::<f64>().ok();
        let on = val != "off" && val != "0";
        match key {
            "end" => {
                let v: Vec<f64> = val.split(',').filter_map(|x| x.trim().parse().ok()).collect();
                if let (Some(id), &[x, y, z]) = (self.object.or(cx.scene.active), &v[..]) {
                    let point = cx.scene.get_mut(id).and_then(|o| o.bvh().nearest(&o.mesh, DVec3::new(x, y, z), f64::INFINITY)).map(|h| h.point);
                    if let Some(p) = point {
                        if let Err(e) = self.set_end(cx, id, p, None) {
                            *cx.status = e;
                        }
                    }
                }
            }
            "clear" => {
                self.sections.clear();
                self.sections_changed();
            }
            "apply" => {
                // Scripts wait for the trace, then apply (built at once if no preview).
                match self.apply(cx) {
                    Ok(msg) => *cx.status = msg,
                    Err(e) => *cx.status = format!("Stent: {e}"),
                }
            }
            "diameter" => {
                self.ui.diameter = num.unwrap_or(self.ui.diameter);
                self.sized[0] = true;
            }
            "length" => {
                self.ui.length = num.unwrap_or(self.ui.length);
                self.sized[1] = true;
            }
            "straightness" => self.ui.straightness = num.unwrap_or(self.ui.straightness).clamp(0.0, 100.0),
            "stiffness" => self.ui.stiffness = num.unwrap_or(self.ui.stiffness).clamp(0.0, 100.0),
            "blend" => self.ui.end_blend = num.unwrap_or(self.ui.end_blend).clamp(0.0, 40.0),
            "ends" => {
                self.ui.continuity = match val {
                    "g0" => EndContinuity::G0,
                    "g2" => EndContinuity::G2,
                    _ => EndContinuity::G1,
                }
            }
            "covered" => self.ui.covered = on,
            "remesh" => self.ui.remesh = on,
            "inplace" => self.in_place = on,
            _ => {}
        }
        self.update(cx);
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (k, out) = self.preview.as_ref()?;
        (Some(*k) == self.key()).then_some((self.object?, &out.mesh))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() || self.trace_job.is_some() {
            return false;
        }
        if self.sections.len() == 2 {
            match self.apply(cx) {
                Ok(msg) => *cx.status = msg,
                Err(e) => *cx.status = format!("Stent: {e}"),
            }
        }
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        if let Some((_, j)) = &self.trace_job {
            return Some(("Tracing the centerline".into(), j.progress()));
        }
        self.job.as_ref().map(|(_, j)| ("Stent".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        let Some(key) = self.key() else { return false };
        let done = |k: Option<&StentKey>| k == Some(&key);
        self.job.is_none() && self.trace_job.is_none() && !done(self.preview.as_ref().map(|p| &p.0)) && !done(self.error.as_ref().map(|e| &e.0))
    }

    fn settings(&self) -> Option<serde_json::Value> {
        let mut v = serde_json::to_value(self.ui).ok()?;
        v["in_place"] = self.in_place.into();
        Some(v)
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let mut s = schema_of::<StentUi>();
        s["properties"]["in_place"] = serde_json::json!({"type": "boolean", "description": "Reshape the object itself; otherwise the result is a new object and this one is hidden"});
        Some(s)
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.in_place = v["in_place"].as_bool().unwrap_or(self.in_place);
        let ui: StentUi = serde_json::from_value(v).map_err(|e| format!("Virtual stent: {e}"))?;
        // Typed sizes stop following the ends, as in the panel.
        self.sized[0] |= changes.get("diameter").is_some();
        self.sized[1] |= changes.get("length").is_some();
        self.ui = StentUi { diameter: ui.diameter.clamp(0.5, 60.0), length: ui.length.clamp(1.0, 400.0), straightness: ui.straightness.clamp(0.0, 100.0), stiffness: ui.stiffness.clamp(0.0, 100.0), end_blend: ui.end_blend.clamp(0.0, 40.0), ..ui };
        self.update(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        use serde_json::json;
        let key = self.key();
        let current = self.preview.as_ref().filter(|p| Some(p.0) == key).map(|p| &p.1);
        json!({
            "object": self.object,
            "ends": self.sections.iter().enumerate().map(|(i, s)| json!({
                "end": i + 1,
                "centre_mm": vec3(s.center.to_array(), 3),
                "axis": vec3(s.normal.to_array(), 4),
                "diameter_mm": round(s.diameter, 3),
            })).collect::<Vec<_>>(),
            "next": match self.sections.len() {
                0 | 1 => "set_end (or click the wall) at both ends of the stretch to treat",
                _ if current.is_some() => "adjust the settings, then apply (or accept)",
                _ => "wait for the preview (app_wait)",
            },
            "centerline_between_ends_mm": self.trace.as_ref().filter(|t| t.0 == self.revision).map(|t| round(t.2.span, 2)),
            "minimal_lumen_mm": self.trace.as_ref().filter(|t| t.0 == self.revision).and_then(|t| t.2.min_lumen).map(|d| round(d, 2)),
            "sizes_follow_the_ends": {"diameter": !self.sized[0], "length": !self.sized[1]},
            "computing": self.trace_job.as_ref().map(|_| "tracing the centerline").or(self.job.as_ref().map(|_| "previewing")),
            "preview": current.map(|out| {
                let pl = &out.placement;
                json!({
                    "stent_axis_between_ends_mm": round(pl.path_length, 2),
                    "centerline_between_ends_mm": round(pl.centerline_length, 2),
                    "reaches_past_ends_mm": pl.overhang().map(|x| round(x, 2)),
                    "narrowest_lumen_mm": {"before": round(2.0 * out.report.narrowest[0], 3), "after": round(2.0 * out.report.narrowest[1], 3)},
                    "sealed_openings": out.report.sealed_openings,
                    "removed_triangles": out.report.removed_triangles,
                    "new_crossings": out.crossings,
                    "warnings": warnings(out),
                })
            }),
            "error": self.error.as_ref().filter(|e| Some(e.0) == key).map(|e| e.1.clone()),
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::new(
                "set_end",
                "Put an end of the stretch at the vessel's cross-section through the wall point nearest `at` (as clicking the wall does). Without `end`: the first missing end, else the nearer one",
                serde_json::json!({"at": arg::point("A point in mm on or near the vessel wall"), "end": arg::integer("1 or 2: which end to set")}),
            )
            .required(&["at"]),
            Action::plain("clear", "Remove both ends"),
            Action::plain("apply", "Keep the reshaped vessel (the preview, or built now) in the object or as a copy (see in_place); the ends are cleared for the next stent"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        match action {
            "set_end" => {
                let which = match a.opt_usize("end") {
                    Some(e @ 1..=2) => Some(e - 1),
                    Some(e) => return Err(format!("there is no end {e}: 1 or 2")),
                    None => None,
                };
                let id = self.object.or(cx.scene.active).ok_or("no active object")?;
                let at = DVec3::from_array(a.point("at")?);
                let o = cx.scene.get_mut(id).ok_or("no object")?;
                let p = o.bvh().nearest(&o.mesh, at, f64::INFINITY).ok_or("the object has no surface")?.point;
                let i = self.set_end(cx, id, p, which)?;
                self.update(cx);
                let s = &self.sections[i];
                Ok(format!("End {} at [{:.2}, {:.2}, {:.2}], ⌀ {:.2} mm", i + 1, s.center.x, s.center.y, s.center.z, s.diameter))
            }
            "clear" => {
                self.sections.clear();
                self.sections_changed();
                self.update(cx);
                Ok("Ends cleared".into())
            }
            "apply" => self.apply(cx),
            _ => Err(format!("Virtual stent has no action {action}")),
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Virtual stent");
        if self.object.and_then(|id| cx.scene.get(id)).is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Click the vessel wall at both ends of the stretch to treat (green cross-sections); click again to move the nearer end. The stent is centred on the two ends and the vessel is reshaped around it, previewed as you change the settings.");
        ui.horizontal(|ui| {
            for (i, s) in self.sections.iter().enumerate() {
                ui.label(format!("End {}: ⌀ {:.2}", i + 1, s.diameter));
            }
            if self.sections.is_empty() {
                ui.weak("No ends yet.");
            }
            if !self.sections.is_empty() && ui.small_button("Clear").clicked() {
                self.sections.clear();
                self.sections_changed();
            }
        });
        if let Some((_, _, t)) = self.trace.as_ref().filter(|t| t.0 == self.revision) {
            ui.label(format!("Centerline between the ends: {:.1} mm", t.span));
            if let Some(d) = t.min_lumen {
                ui.label(format!("Minimal lumen ⌀ {d:.2} mm")).on_hover_text("The largest sphere that fits inside the lumen at its narrowest, between the ends");
            }
        }
        ui.separator();
        egui::Grid::new("stent_size").num_columns(2).show(ui, |ui| {
            ui.label("Diameter");
            if ui.add(egui::DragValue::new(&mut self.ui.diameter).speed(0.05).range(0.5..=60.0).max_decimals(2).suffix(" mm")).on_hover_text("The stent's nominal (fully expanded) diameter").changed() {
                self.sized[0] = true;
            }
            ui.end_row();
            ui.label("Length");
            if ui.add(egui::DragValue::new(&mut self.ui.length).speed(0.25).range(1.0..=400.0).max_decimals(1).suffix(" mm")).on_hover_text("Along the stent's own axis; it is centred on the two ends, so any extra length reaches equally far past each").changed() {
                self.sized[1] = true;
            }
            ui.end_row();
        });
        ui.add(egui::Slider::new(&mut self.ui.straightness, 0.0..=100.0).suffix("%").text("Straightness"))
            .on_hover_text("0 % follows the vessel's centerline as it is; 100 % takes the smoothest path between the two ends (where a free stent held at both would settle), straightening the vessel where the stent grips it");
        ui.horizontal(|ui| {
            ui.label("Smoothest path");
            ui.selectable_value(&mut self.ui.continuity, EndContinuity::G0, "G0").on_hover_text("A straight line between the ends' centres");
            ui.selectable_value(&mut self.ui.continuity, EndContinuity::G1, "G1").on_hover_text("Leaves each end square to its cross-section");
            ui.selectable_value(&mut self.ui.continuity, EndContinuity::G2, "G2").on_hover_text("Also continues the vessel's bend at each end");
        });
        ui.add(egui::Slider::new(&mut self.ui.stiffness, 0.0..=100.0).suffix("%").text("Stiffness"))
            .on_hover_text("Radial force: along the stent, each way across, the lumen lies this share of the way from the vessel's own wall to the stent's radius (2 mm vessel, 6 mm stent, 50 % → 4 mm), so the vessel's shape eases into the round stent. 100 % makes a perfect tube of the stent's diameter; 0 % leaves the vessel as it is. A bare stent never pulls wider wall in; a covered one narrows it to the stent");
        ui.add(egui::Slider::new(&mut self.ui.end_blend, 0.0..=40.0).suffix(" mm").text("End blend"))
            .on_hover_text("How far past each end of the stent the wall eases back to its own shape and course. Nothing beyond it changes; a straightened stent may need a longer one");
        ui.checkbox(&mut self.ui.covered, "Covered (stent graft)")
            .on_hover_text("The lumen along the stent becomes the stent's tube: wider wall (an aneurysm, a dilated stretch) is narrowed to the stent's diameter, and side branches opening along it are sealed off and removed");
        ui.checkbox(&mut self.ui.remesh, "Remesh the reshaped wall").on_hover_text("Even triangles over the reshaped region (recommended for CFD)");
        ui.checkbox(&mut self.in_place, "Edit in place").on_hover_text("Reshape this object itself; otherwise the result is a new object (this one is hidden)");
        self.update(cx);
        let key = self.key();
        let current = self.preview.as_ref().filter(|p| Some(p.0) == key).map(|p| &p.1);
        if let Some(out) = current {
            let pl = &out.placement;
            let [a, b] = pl.overhang();
            let past = |x: f64| if x >= 0.0 { format!("{x:.1} mm past") } else { format!("{:.1} mm short of", -x) };
            ui.separator();
            ui.label(format!("Stent axis between the ends: {:.1} mm (centerline {:.1} mm)", pl.path_length, pl.centerline_length));
            ui.label(format!("Reaches {} end 1 and {} end 2", past(a), past(b)));
            ui.label(format!("Narrowest lumen ⌀ {:.2} → {:.2} mm", 2.0 * out.report.narrowest[0], 2.0 * out.report.narrowest[1]));
            let sealed = sealed_text(&out.report);
            if !sealed.is_empty() {
                ui.label(sealed.trim_start_matches("; "));
            }
            for w in warnings(out) {
                ui.colored_label(egui::Color32::from_rgb(240, 170, 80), w);
            }
        }
        let ready = current.is_some();
        ui.horizontal(|ui| {
            if ui.add_enabled(ready, egui::Button::new("Apply")).on_hover_text("Keep the previewed result (Enter)").clicked() {
                match self.apply(cx) {
                    Ok(msg) => *cx.status = msg,
                    Err(e) => self.error = key.map(|k| (k, e)),
                }
            }
            if key.is_some() && !ready && self.error.as_ref().map(|e| e.0) != key {
                ui.spinner();
                ui.weak(if self.trace_job.is_some() { "Tracing the centerline…" } else { "Previewing…" });
            }
        });
        if let Some((_, e)) = self.error.as_ref().filter(|e| Some(e.0) == key) {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.pressed = Some(pos);
        }
        if !input.released {
            return;
        }
        let Some(p0) = self.pressed.take() else { return };
        if p0.distance(pos) > 5.0 {
            return;
        }
        if let Some((id, hit)) = cx.pick(pos, self.object.filter(|_| !self.sections.is_empty())) {
            if let Err(e) = self.set_end(cx, id, hit.point, None) {
                *cx.status = e;
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        let green = Vec4::new(0.3, 1.0, 0.6, 1.0);
        for s in &self.sections {
            for seg in &s.ring {
                ov.line(seg[0], seg[1], green);
            }
        }
        let Some((_, out)) = self.preview.as_ref().filter(|p| Some(p.0) == self.key()) else { return };
        let pl = &out.placement;
        let colour = if self.ui.covered { Vec4::new(0.55, 0.75, 1.0, 1.0) } else { Vec4::new(0.85, 0.88, 0.95, 1.0) };
        let count = ((pl.length / (0.75 * pl.radius).max(0.25)).round() as usize).clamp(4, 200);
        let rings = pl.rings(24, count);
        for (j, ring) in rings.iter().enumerate() {
            let c = if j == 0 || j == count { Vec4::new(1.0, 0.75, 0.3, 1.0) } else { colour };
            for i in 0..ring.len() {
                ov.line(ring[i], ring[(i + 1) % ring.len()], c);
            }
        }
        for i in (0..24).step_by(3) {
            for w in rings.windows(2) {
                ov.line(w[0][i], w[1][i], colour);
            }
        }
        let axis: Vec<DVec3> = (0..pl.s.len()).filter(|&i| (0.0..=pl.length).contains(&pl.s[i])).map(|i| pl.axis[i]).collect();
        for w in axis.windows(2) {
            ov.line(w[0], w[1], Vec4::new(1.0, 0.85, 0.2, 1.0));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        for (i, s) in self.sections.iter().enumerate() {
            if let Some(p) = cx.project(s.center) {
                painter.text(p, egui::Align2::LEFT_BOTTOM, format!("{} ⌀ {:.2}", i + 1, s.diameter), egui::FontId::proportional(14.0), egui::Color32::from_rgb(120, 255, 170));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A vessel of radius 4 bending through 120° on a radius of 25 (with 15 mm straight
    /// leads), narrowed eccentrically to about 1.8 halfway round.
    fn bent_stenosis() -> (Mesh, impl Fn(f64) -> DVec3) {
        let arc = 25.0 * 120f64.to_radians();
        let centre = move |s: f64| -> DVec3 {
            if s < 15.0 {
                DVec3::new(25.0, s - 15.0, 0.0)
            } else if s < 15.0 + arc {
                let a = (s - 15.0) / 25.0;
                DVec3::new(25.0 * a.cos(), 25.0 * a.sin(), 0.0)
            } else {
                let a = 120f64.to_radians();
                DVec3::new(25.0 * a.cos(), 25.0 * a.sin(), 0.0) + DVec3::new(-a.sin(), a.cos(), 0.0) * (s - 15.0 - arc)
            }
        };
        let total = 30.0 + arc;
        let mid = 15.0 + 0.5 * arc;
        let n = (total / 0.35) as usize;
        let path: Vec<DVec3> = (0..=n).map(|i| centre(total * i as f64 / n as f64)).collect();
        let radii: Vec<f64> = (0..=n).map(|i| 4.0 - 2.2 * (-((total * i as f64 / n as f64 - mid) / 4.0).powi(2)).exp()).collect();
        (mm_core::primitives::tube_along(&path, &radii, 72), centre)
    }

    /// Sharpest bend between neighbouring triangles of `tris` (degrees).
    fn sharpest(m: &Mesh, tris: &[u32]) -> f64 {
        let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
        m.edge_ids()
            .filter_map(|e| {
                let [s, t] = m.edge_triangles(e);
                (t != mm_core::INVALID && set.contains(&s) && set.contains(&t)).then(|| m.tri_normal(s).angle_between(m.tri_normal(t)).to_degrees())
            })
            .fold(0.0, f64::max)
    }

    /// Vertices of `m` past the end blend (beyond the stent's first or last slice by more than
    /// half an edge) that `out` moved or rebuilt, of how many there are.
    fn changed_past_the_end_blend(m: &Mesh, out: &Mesh, pl: &mm_edit::StentPlacement) -> (usize, usize) {
        let k = pl.vessel.len();
        let ends = [(pl.vessel[0], (pl.vessel[0] - pl.vessel[1]).normalize()), (pl.vessel[k - 1], (pl.vessel[k - 1] - pl.vessel[k - 2]).normalize())];
        let half = 0.5 * m.average_edge_length();
        let ring = |mesh: &Mesh, v: u32| {
            let mut n: Vec<u32> = mesh.vertex_neighbors(v).collect();
            n.sort_unstable();
            n
        };
        let past: Vec<u32> = m
            .vertex_ids()
            .filter(|&v| {
                let p = m.vertex(v);
                let nearest = (0..k).min_by(|&a, &b| pl.vessel[a].distance(p).total_cmp(&pl.vessel[b].distance(p))).unwrap();
                (nearest == 0 && (p - ends[0].0).dot(ends[0].1) > half) || (nearest == k - 1 && (p - ends[1].0).dot(ends[1].1) > half)
            })
            .collect();
        let changed = past.iter().filter(|&&v| !(out.is_vertex(v) && out.vertex(v) == m.vertex(v) && ring(out, v) == ring(m, v))).count();
        (changed, past.len())
    }

    #[test]
    fn stent_in_a_bent_stenosed_vessel() {
        // Traced from the first end, the stent opens the narrowing; straightened (G0) and
        // covered, the wall stays free of crossings and folds.
        let (m, centre) = bent_stenosis();
        let bvh = mm_core::Bvh::build(&m);
        let wall = |s: f64| {
            let c = centre(s);
            bvh.nearest(&m, c + DVec3::Z * 4.0, f64::INFINITY).unwrap().point
        };
        let a0 = 15.0 + 25.0 * 30f64.to_radians();
        let b0 = 15.0 + 25.0 * 90f64.to_radians();
        let sections = [Section::at(&m, &bvh, wall(a0)).unwrap(), Section::at(&m, &bvh, wall(b0)).unwrap()];
        let trace = trace_vessel(&m, &sections).unwrap();
        assert!((trace.span - (b0 - a0)).abs() < 0.6, "span {} vs {}", trace.span, b0 - a0);
        // Straightened across the bend, the vessel turns back onto its course within the end
        // blend: a longer one for a smooth wall.
        for (ui, label) in [
            (StentUi { length: 30.0, ..Default::default() }, "bare, half straight"),
            (StentUi { length: 30.0, straightness: 100.0, continuity: EndContinuity::G0, end_blend: 8.0, ..Default::default() }, "bare, straight"),
            (StentUi { length: 30.0, straightness: 100.0, continuity: EndContinuity::G0, covered: true, stiffness: 50.0, end_blend: 16.0, ..Default::default() }, "covered, straight"),
        ] {
            let key = StentKey { sections: 1, ui };
            let (out, pl, rep, crossings) = StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &key).unwrap();
            out.check_validity().unwrap();
            assert!(out.is_consistently_oriented());
            let bend = sharpest(&out, &rep.triangles);
            let [a, b] = pl.overhang();
            assert!((a - b).abs() < 1e-6 && a > 0.0, "{label}: centred ({a:.3} / {b:.3})");
            assert_eq!(crossings, 0, "{label}");
            assert!(bend < 30.0, "{label}: {bend:.1}°");
            let want = 4.0 + (ui.stiffness / 100.0) * 4.0;
            assert!(2.0 * rep.narrowest[1] > want - 0.3, "{label}: {:?}", rep.narrowest);
            assert_eq!(mm_core::topology::boundary_loops(&out).len(), 2, "{label}");
            let (changed, past) = changed_past_the_end_blend(&m, &out, &pl);
            assert!(past > 1000 && changed == 0, "{label}: {changed} of {past} vertices past the end blend changed");
        }
        // Too short an end blend for that: still nothing past it changes, and the tool says to
        // lengthen it.
        let ui = StentUi { length: 30.0, straightness: 100.0, continuity: EndContinuity::G0, ..Default::default() };
        let (mesh, placement, report, crossings) = StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &StentKey { sections: 1, ui }).unwrap();
        assert_eq!(changed_past_the_end_blend(&m, &mesh, &placement).0, 0);
        let out = StentOut { source: Arc::new(m.clone()), mesh, placement, report, crossings };
        assert!(warnings(&out).iter().any(|w| w.contains("lengthen the end blend")), "{:?}", warnings(&out));
    }

    #[test]
    fn agents_work_the_tool_through_its_api() {
        let (m, centre) = bent_stenosis();
        let mut scene = crate::scene::Scene::default();
        let id = scene.add("vessel", m);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut t = StentTool::new(&mut cx);
        let names: Vec<&str> = t.actions().iter().map(|a| a.name).collect();
        assert_eq!(names, ["set_end", "clear", "apply"]);
        assert_eq!(t.state(cx.scene)["next"], "set_end (or click the wall) at both ends of the stretch to treat");
        let wall = |s: f64| (centre(s) + DVec3::Z * 4.5).to_array();
        let (a0, b0) = (15.0 + 25.0 * 30f64.to_radians(), 15.0 + 25.0 * 90f64.to_radians());
        let msg = t.act("set_end", &serde_json::json!({"at": wall(a0)}), &mut cx).unwrap();
        assert!(msg.starts_with("End 1 at"), "{msg}");
        t.act("set_end", &serde_json::json!({"at": wall(b0)}), &mut cx).unwrap();
        let st = t.state(cx.scene);
        assert_eq!(st["ends"].as_array().unwrap().len(), 2);
        assert!((st["ends"][0]["diameter_mm"].as_f64().unwrap() - 8.0).abs() < 0.3, "{st}");
        // Moving end 2 explicitly; end 3 does not exist.
        t.act("set_end", &serde_json::json!({"at": wall(b0 + 2.0), "end": 2}), &mut cx).unwrap();
        assert!(t.act("set_end", &serde_json::json!({"at": wall(b0), "end": 3}), &mut cx).is_err());
        // Settings: typed sizes stop following the ends; misspelt ones are refused.
        t.set_settings(&serde_json::json!({"diameter": 7.5, "straightness": 100, "continuity": "G0"}), &mut cx).unwrap();
        assert_eq!(t.settings().unwrap()["diameter"], 7.5);
        assert_eq!(t.state(cx.scene)["sizes_follow_the_ends"]["diameter"], false);
        assert!(t.set_settings(&serde_json::json!({"diamter": 7.0}), &mut cx).is_err());
        assert!(t.actions()[0].check(&serde_json::json!({"at": [1, 2]})).is_err());
        // Applied without waiting for a preview: built at once, as a new object.
        let msg = t.act("apply", &serde_json::Value::Null, &mut cx).unwrap();
        assert!(msg.contains("narrowest lumen"), "{msg}");
        assert_eq!(cx.scene.objects.len(), 2);
        assert!(!cx.scene.get(id).unwrap().visible);
        assert_eq!(cx.history.undo_label(), Some("Stent"));
        assert!(t.state(cx.scene)["ends"].as_array().unwrap().is_empty());
    }

    /// A real mesh (`POLYMIX_STENT_MESH`, local only): stents along the longest branch of its
    /// centerline, traced from the widest opening. Prints counts and timings only.
    #[test]
    #[ignore]
    fn stent_on_a_real_mesh() {
        let Some(path) = std::env::var_os("POLYMIX_STENT_MESH") else { return };
        let m = mm_io::read_mesh(std::path::Path::new(&path)).unwrap().mesh;
        let loops = mm_core::topology::boundary_loops(&m);
        let (sources, r) = match loops.iter().max_by(|a, b| a.perimeter(&m).total_cmp(&b.perimeter(&m))) {
            Some(inlet) => (inlet.vertices.clone(), inlet.perimeter(&m) / std::f64::consts::TAU),
            None => {
                // Closed: from the widest flat cap.
                let ends = super::super::vessel::find_vessel_ends(&m);
                let Some(e) = ends.iter().max_by(|a, b| a.diameter.total_cmp(&b.diameter)) else {
                    eprintln!("{} triangles, closed with no flat caps: skipped", m.triangle_count());
                    return;
                };
                let super::super::vessel::EndKind::Cap(g) = e.kind else { panic!("no cap") };
                let mut v: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_group(t) == g).flat_map(|t| m.triangle(t)).collect();
                v.sort_unstable();
                v.dedup();
                (v, 0.5 * e.diameter)
            }
        };
        let cl = centerline::compute(&m, &sources, r, &CenterlineParams::default());
        let branch = cl.branches.iter().max_by(|a, b| a.length.total_cmp(&b.length)).unwrap();
        let all: Vec<u32> = m.triangle_ids().collect();
        eprintln!("input crossings {}", ops::crossings(&m, &all));
        eprintln!("{} triangles, {} open ends, longest branch {:.1} mm over {} nodes", m.triangle_count(), loops.len(), branch.length, branch.nodes.len());
        let bvh = mm_core::Bvh::build(&m);
        // Ends where the cross-section matches the centerline's ring (a clean stretch of tube).
        let clean: Vec<(f64, Section)> = {
            let mut along = 0.0;
            let mut out = Vec::new();
            for (k, &i) in branch.nodes.iter().enumerate() {
                if k > 0 {
                    along += cl.nodes[i].position.distance(cl.nodes[branch.nodes[k - 1]].position);
                }
                let node = cl.nodes[i];
                let wall = bvh.nearest(&m, node.position, f64::INFINITY).unwrap().point;
                if let Some(sec) = Section::at(&m, &bvh, wall) {
                    if (sec.diameter / (2.0 * node.radius) - 1.0).abs() < 0.25 && sec.center.distance(node.position) < 0.5 * node.radius {
                        out.push((along, sec));
                    }
                }
            }
            out
        };
        eprintln!("{} clean sections along it", clean.len());
        let mut pairs = Vec::new();
        let inner = |i: usize| clean[i].0 > 0.15 * branch.length && clean[i].0 < 0.85 * branch.length;
        for want in [3.0, 5.0] {
            let d = clean.first().map_or(1.0, |c| c.1.diameter);
            if let Some((i, j)) = (0..clean.len()).flat_map(|i| (i + 1..clean.len()).map(move |j| (i, j))).filter(|&(i, j)| inner(i) && inner(j)).min_by(|&(a, b), &(c, e)| ((clean[b].0 - clean[a].0) - want * d).abs().total_cmp(&((clean[e].0 - clean[c].0) - want * d).abs())) {
                pairs.push((clean[i].1.clone(), clean[j].1.clone()));
            }
        }
        for (a, b) in pairs {
            let t0 = std::time::Instant::now();
            let sections = [a, b];
            eprintln!("sections {:.1} mm apart", sections[0].center.distance(sections[1].center));
            let trace = match trace_vessel(&m, &sections) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("trace failed: {e:#}");
                    continue;
                }
            };
            eprintln!("ends: ⌀ {:.2} / {:.2}, span {:.1} mm, traced in {:.0} ms", sections[0].diameter, sections[1].diameter, trace.span, t0.elapsed().as_secs_f64() * 1e3);
            let d = ((sections[0].diameter.max(sections[1].diameter) * 2.0).round() / 2.0).max(0.5);
            let length = (trace.span / 5.0).ceil() * 5.0 + 5.0;
            for (ui, label) in [
                (StentUi { diameter: d, length, straightness: 0.0, ..Default::default() }, "bare, follows"),
                (StentUi { diameter: d, length, straightness: 100.0, ..Default::default() }, "bare, G1 smoothest"),
                (StentUi { diameter: d, length, straightness: 100.0, continuity: EndContinuity::G0, ..Default::default() }, "bare, straight"),
                (StentUi { diameter: d, length, straightness: 50.0, covered: true, ..Default::default() }, "covered"),
            ] {
                let t0 = std::time::Instant::now();
                let key = StentKey { sections: 1, ui };
                match StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &key) {
                    Ok((out, pl, rep, crossings)) => eprintln!(
                        "  {label}: {:.0} ms, valid {}, oriented {}, crossings {crossings}, sharpest {:.1}°, narrowest ⌀ {:.2} -> {:.2}, moved {}, overhang {:.1}/{:.1}, outside {}, open ends {} -> {}, sealed {}",
                        t0.elapsed().as_secs_f64() * 1e3,
                        out.check_validity().is_ok(),
                        out.is_consistently_oriented(),
                        sharpest(&out, &rep.triangles),
                        2.0 * rep.narrowest[0],
                        2.0 * rep.narrowest[1],
                        rep.moved,
                        pl.overhang()[0],
                        pl.overhang()[1],
                        rep.outside,
                        loops.len(),
                        mm_core::topology::boundary_loops(&out).len(),
                        rep.sealed_openings
                    ),
                    Err(e) => eprintln!("  {label}: failed: {e:#}"),
                }
            }
        }
    }

    #[test]
    fn the_path_is_moved_into_an_eccentric_narrowing() {
        // The lumen narrows to radius 0.8 at z = 30 while its centre moves 2.5 to the side; a
        // trace straight along z (as ring centroids can run) passes outside it there.
        let bump = |z: f64| (-((z - 30.0) / 4.0).powi(2)).exp();
        let centre = move |z: f64| DVec3::new(2.5 * bump(z), 0.0, z);
        let radius = move |z: f64| 3.0 - 2.2 * bump(z);
        let zs: Vec<f64> = (0..=240).map(|k| 0.25 * k as f64).collect();
        let m = mm_core::primitives::tube_along(&zs.iter().map(|&z| centre(z)).collect::<Vec<_>>(), &zs.iter().map(|&z| radius(z)).collect::<Vec<_>>(), 64);
        let mut path = mm_edit::VesselPath { points: zs.iter().map(|&z| DVec3::new(0.0, 0.0, z)).collect(), radii: vec![3.0; zs.len()] };
        let end = |z: f64| Section { center: centre(z), normal: DVec3::Z, diameter: 2.0 * radius(z), ring: Vec::new() };
        recentre(&m, &mut path, &[end(16.0), end(44.0)]);
        // Through the narrowing, near the lumen's centre (well inside its radius).
        for p in path.points.iter().filter(|p| (22.0..38.0).contains(&p.z)) {
            let off = p.truncate().distance(centre(p.z).truncate());
            assert!(off < 0.5 * radius(p.z), "{off:.2} off the centre at z = {:.1} (radius {:.2})", p.z, radius(p.z));
        }
        // Smooth (no longer than the lumen's own course), and unchanged past the ends.
        let length: f64 = path.points.windows(2).filter(|w| (16.0..44.0).contains(&w[0].z)).map(|w| w[0].distance(w[1])).sum();
        let course: f64 = zs.windows(2).filter(|w| (16.0..44.0).contains(&w[0])).map(|w| centre(w[0]).distance(centre(w[1]))).sum();
        assert!(length < 1.02 * course, "{length:.2} vs {course:.2}");
        // (The moves ease out within twice the fade, half the ends' radius, past each end.)
        for p in path.points.iter().filter(|p| !(13.0..47.0).contains(&p.z)) {
            assert!(p.truncate().length() < 1e-9, "moved at z = {:.1}", p.z);
        }
    }

    /// A stent between given ends of a project's first object (local only):
    /// `POLYMIX_STENT_PROJECT=<.polymix>`, `POLYMIX_STENT_ENDS="x,y,z;x,y,z"` (the ends'
    /// centres, as `mesh_tool_state` reports them) and optionally `POLYMIX_STENT_AXES` (their
    /// axes; without them each end is the cross-section through the wall nearest its centre),
    /// optionally `POLYMIX_STENT_LENGTH` (mm), `POLYMIX_STENT_STIFFNESS` (%) and
    /// `POLYMIX_STENT_COVERED=1`. Prints crossings and readouts at those settings, without
    /// remeshing, and without straightening or push.
    #[test]
    #[ignore]
    fn stent_on_a_project() {
        let (Some(path), Ok(ends)) = (std::env::var_os("POLYMIX_STENT_PROJECT"), std::env::var("POLYMIX_STENT_ENDS")) else { return };
        let points = |e: &str| -> Vec<DVec3> {
            e.split(';')
                .map(|p| {
                    let v: Vec<f64> = p.split(',').map(|x| x.trim().parse().unwrap()).collect();
                    DVec3::new(v[0], v[1], v[2])
                })
                .collect()
        };
        let m = crate::project::load(std::path::Path::new(&path)).unwrap().objects[0].mesh.clone();
        let bvh = mm_core::Bvh::build(&m);
        let (centres, axes) = (points(&ends), std::env::var("POLYMIX_STENT_AXES").ok().map(|a| points(&a)));
        let sections: Vec<Section> = (0..2)
            .map(|i| match &axes {
                Some(axes) => {
                    let curve = mm_edit::nearest_intersection_curve(&m, mm_core::Plane::from_point_normal(centres[i], axes[i]), centres[i]).unwrap();
                    Section { center: centres[i], normal: axes[i], diameter: curve.equivalent_diameter(), ring: curve.segments }
                }
                None => Section::at(&m, &bvh, bvh.nearest(&m, centres[i], f64::INFINITY).unwrap().point).unwrap(),
            })
            .collect();
        let sections = [sections[0].clone(), sections[1].clone()];
        {
            // The lumen's inscribed-sphere centerline between the ends, on its own.
            let a = &sections[0];
            let h = m.average_edge_length();
            let n = a.normal.normalize();
            let src: Vec<u32> = m.vertex_ids().filter(|&v| (m.vertex(v) - a.center).dot(n).abs() < h && a.ring.iter().any(|sg| point_segment_distance(m.vertex(v), sg[0], sg[1]) < h)).collect();
            let cl = centerline::compute(&m, &src, 0.5 * a.diameter, &CenterlineParams::default());
            let pts: Vec<DVec3> = cl.branches.iter().flat_map(|b| b.nodes.iter().map(|&i| cl.nodes[i].position)).collect();
            let near = |c: DVec3| pts.iter().copied().min_by(|p, q| p.distance(c).total_cmp(&q.distance(c))).unwrap();
            eprintln!("  (traced centerline: {} nodes; nearest to the ends {:.2} / {:.2} mm)", pts.len(), near(sections[0].center).distance(sections[0].center), near(sections[1].center).distance(sections[1].center));
        }
        let t0 = std::time::Instant::now();
        let trace = trace_vessel(&m, &sections).unwrap();
        {
            let pts = &trace.path.points;
            let ia = (0..pts.len()).min_by(|&i, &j| pts[i].distance(trace.ends[0]).total_cmp(&pts[j].distance(trace.ends[0]))).unwrap();
            let ib = (0..pts.len()).min_by(|&i, &j| pts[i].distance(trace.ends[1]).total_cmp(&pts[j].distance(trace.ends[1]))).unwrap();
            let (lo, hi) = (ia.min(ib), ia.max(ib));
            let rmin = trace.path.radii[lo..=hi].iter().copied().fold(f64::INFINITY, f64::min);
            let bvh2 = mm_core::Bvh::build(&m);
            let outside = (lo..=hi).filter(|&i| {
                let p = pts[i];
                let t = (pts[(i + 1).min(pts.len() - 1)] - pts[i.saturating_sub(1)]).normalize();
                let u = t.any_orthonormal_vector();
                (0..12).filter(|k| {
                    let a = *k as f64 * std::f64::consts::TAU / 12.0;
                    let mut hits = Vec::new();
                    bvh2.raycast_all(&m, &mm_core::Ray::new(p, u * a.cos() + t.cross(u) * a.sin()), &mut hits);
                    hits.is_empty()
                }).count() > 3
            }).count();
            eprintln!("  path between the ends: {} points, smallest radius {rmin:.3} mm (minimal lumen ⌀ {:.2}), {outside} points outside the lumen", hi - lo + 1, 2.0 * rmin);
        }
        eprintln!("ends ⌀ {:.2} / {:.2}, span {:.2} mm, traced in {:.0} ms", sections[0].diameter, sections[1].diameter, trace.span, t0.elapsed().as_secs_f64() * 1e3);
        let d = std::env::var("POLYMIX_STENT_DIAMETER").ok().and_then(|l| l.parse().ok()).unwrap_or(((sections[0].diameter.max(sections[1].diameter) * 2.0).floor() / 2.0).max(0.5));
        let length = std::env::var("POLYMIX_STENT_LENGTH").ok().and_then(|l| l.parse().ok()).unwrap_or(((trace.span / 5.0).ceil() * 5.0).max(5.0));
        let stiffness = std::env::var("POLYMIX_STENT_STIFFNESS").ok().and_then(|l| l.parse().ok()).unwrap_or(100.0);
        let covered = std::env::var("POLYMIX_STENT_COVERED").is_ok_and(|v| v == "1");
        let ui = StentUi { diameter: d, length, stiffness, covered, ..Default::default() };
        for (ui, label) in [(ui, "default"), (StentUi { remesh: false, ..ui }, "no remesh"), (StentUi { remesh: false, straightness: 0.0, ..ui }, "no remesh, follows"), (StentUi { remesh: false, stiffness: 0.0, ..ui }, "no remesh, no push")] {
            let t0 = std::time::Instant::now();
            match StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &StentKey { sections: 1, ui }) {
                Ok((out, pl, rep, crossings)) => {
                    // The reshaped wall along the stent: still inside it (not opened), facing the
                    // axis (folded over), and sliver triangles, by mm along the stent.
                    let k = pl.s.len();
                    let near = |p: DVec3| (0..k).min_by(|&i, &j| pl.axis[i].distance_squared(p).total_cmp(&pl.axis[j].distance_squared(p))).unwrap();
                    let mut rows = std::collections::BTreeMap::<i64, [usize; 4]>::new();
                    for &t in &rep.triangles {
                        if !out.is_triangle(t) {
                            continue;
                        }
                        let tp = out.tri_positions(t);
                        let c = (tp[0] + tp[1] + tp[2]) / 3.0;
                        let i = near(c);
                        if pl.s[i] < -3.0 || pl.s[i] > pl.length + 3.0 {
                            continue;
                        }
                        let tan = (pl.axis[(i + 1).min(k - 1)] - pl.axis[i.saturating_sub(1)]).normalize();
                        let radial = (c - pl.axis[i]) - tan * (c - pl.axis[i]).dot(tan);
                        let row = rows.entry(pl.s[i].floor() as i64).or_default();
                        row[0] += 1;
                        if radial.length() < 0.9 * pl.radius && (0.0..pl.length).contains(&pl.s[i]) {
                            row[1] += 1;
                        }
                        if out.tri_normal(t).dot(radial.normalize_or_zero()) < -0.3 {
                            row[2] += 1;
                        }
                        let e = out.tri_edges(t).map(|e| out.edge_length(e));
                        let longest = e.iter().copied().fold(0.0, f64::max);
                        let area = 0.5 * (tp[1] - tp[0]).cross(tp[2] - tp[0]).length();
                        if 2.0 * area / longest.max(1e-12) < 0.1 * longest {
                            row[3] += 1;
                        }
                    }
                    eprintln!("  {label}: region {} triangles ({} valid), source region {}; pushed: {:?}; landing {:.2?}", rep.triangles.len(), rep.triangles.iter().filter(|&&t| out.is_triangle(t)).count(), rep.source_triangles.len(), rep.pushed, rep.landing);
                    {
                        // Where the sharpest creases are, by mm along the stent.
                        let set: std::collections::HashSet<u32> = rep.triangles.iter().copied().collect();
                        let k = pl.s.len();
                        let mut creases: Vec<(f64, f64)> = out.edge_ids().filter_map(|e| {
                            let [a, b] = out.edge_triangles(e);
                            (b != mm_core::INVALID && set.contains(&a) && set.contains(&b)).then(|| {
                                let ang = out.tri_normal(a).angle_between(out.tri_normal(b)).to_degrees();
                                let [p, q] = out.edge_vertices(e).map(|v| out.vertex(v));
                                let c = (p + q) * 0.5;
                                let i = (0..k).min_by(|&x, &y| pl.axis[x].distance_squared(c).total_cmp(&pl.axis[y].distance_squared(c))).unwrap();
                                (ang, pl.s[i])
                            })
                        }).filter(|x| x.0 > 90.0).collect();
                        creases.sort_by(|a, b| b.0.total_cmp(&a.0));
                        eprintln!("  {label}: {} creases over 90°, sharpest at s {:?}", creases.len(), creases.iter().take(6).map(|c| format!("{:.1} ({:.0}°)", c.1, c.0)).collect::<Vec<_>>());
                    }
                    if label == "no remesh, no push" {
                        // Where the move changes most sharply between neighbours (the carry only:
                        // no push, no union), and where that lies along the stent's samples.
                        let disp = |v: u32| out.vertex(v) - m.vertex(v);
                        let mut worst: Vec<(f64, u32)> = m
                            .vertex_ids()
                            .filter(|&v| out.is_vertex(v))
                            .map(|v| {
                                let s = out.vertex_neighbors(v).filter(|&u| m.is_vertex(u)).map(|u| (out.vertex(v).distance(out.vertex(u)) / m.vertex(v).distance(m.vertex(u)).max(1e-9) - 1.0).abs()).fold(0.0, f64::max);
                                (s, v)
                            })
                            .collect();
                        worst.sort_by(|a, b| b.0.total_cmp(&a.0));
                        eprintln!("  carry: samples run s {:.1}..{:.1} (margins {:.1?}, return bend {:.2?})", pl.s[0], pl.s[pl.s.len() - 1], pl.margins, pl.return_bend);
                        for &(st, v) in worst.iter().take(4) {
                            let i = (0..pl.s.len()).min_by(|&x, &y| pl.vessel[x].distance(m.vertex(v)).total_cmp(&pl.vessel[y].distance(m.vertex(v)))).unwrap();
                            eprintln!("    edge stretched by {:.0} % at {:.2?}, moved {:.2?}, nearest sample s {:.1} (axis–vessel there {:.2}), the vessel's own {}", 100.0 * st, m.vertex(v), disp(v), pl.s[i], pl.axis[i].distance(pl.vessel[i]), trace.owner[v as usize]);
                            for u in out.vertex_neighbors(v) {
                                eprintln!("        neighbour {:.2?} moved {:.2?}, own {}", m.vertex(u), disp(u), trace.owner[u as usize]);
                            }
                        }
                        // How sharply the vessel's course (as the carry uses it) turns.
                        let k = pl.s.len();
                        let mut turns: Vec<(f64, f64)> = (1..k - 1).map(|i| {
                            let (a, b, c) = (pl.vessel[i - 1], pl.vessel[i], pl.vessel[i + 1]);
                            let kappa = 2.0 * (b - a).cross(c - b).length() / (a.distance(b) * b.distance(c) * a.distance(c)).max(1e-12);
                            (kappa * pl.vessel_radius[i], pl.s[i])
                        }).collect();
                        turns.sort_by(|a, b| b.0.total_cmp(&a.0));
                        eprintln!("  vessel course: sharpest radius × curvature {:?}", turns.iter().take(6).map(|t| format!("{:.2} at s {:.1}", t.0, t.1)).collect::<Vec<_>>());
                        let raw = &trace.path.points;
                        let mut rt: Vec<(f64, usize)> = (1..raw.len() - 1).map(|i| {
                            let (a, b, c) = (raw[i - 1], raw[i], raw[i + 1]);
                            ((b - a).angle_between(c - b).to_degrees(), i)
                        }).collect();
                        rt.sort_by(|a, b| b.0.total_cmp(&a.0));
                        eprintln!("  traced path ({} points): sharpest turns {:?}", raw.len(), rt.iter().take(6).map(|t| format!("{:.0}° at {:.1?} (step {:.2})", t.0, raw[t.1], raw[t.1].distance(raw[t.1 + 1]))).collect::<Vec<_>>());
                        // Does the course come back near itself (far along it, close in space)?
                        let mut closest = (f64::INFINITY, 0.0, 0.0);
                        for i in 0..k {
                            for j in i + 1..k {
                                if pl.s[j] - pl.s[i] > 4.0 {
                                    let d = pl.vessel[i].distance(pl.vessel[j]);
                                    let gap = d / (pl.s[j] - pl.s[i]);
                                    if gap < closest.0 {
                                        closest = (gap, pl.s[i], pl.s[j]);
                                    }
                                }
                            }
                        }
                        eprintln!("  course self-approach: straight distance over distance along {:.2} between s {:.1} and {:.1} (distance {:.2})", closest.0, closest.1, closest.2, {
                            let i = pl.s.iter().position(|&x| x == closest.1).unwrap();
                            let j = pl.s.iter().position(|&x| x == closest.2).unwrap();
                            pl.vessel[i].distance(pl.vessel[j])
                        });
                        for i in (0..k).step_by(4) {
                            if (-4.0..12.0).contains(&pl.s[i]) {
                                eprintln!("    s {:>5.1}: vessel {:.2?} axis {:.2?} radius {:.2}", pl.s[i], pl.vessel[i], pl.axis[i], pl.vessel_radius[i]);
                            }
                        }
                        let own = m.vertex_ids().filter(|&v| trace.owner[v as usize]).count();
                        let near_axis = m.vertex_ids().filter(|&v| pl.axis.iter().any(|a| a.distance(m.vertex(v)) < 8.0)).collect::<Vec<_>>();
                        eprintln!("  owner: {own} vertices are the vessel's own; within 8 mm of the axis {} of which not own {}", near_axis.len(), near_axis.iter().filter(|&&v| !trace.owner[v as usize]).count());
                    }
                    if let Ok(bx) = std::env::var("POLYMIX_DIFF_BOX") {
                        // How the wall moved in the box (vertices kept by id), and the sharpest
                        // change in that move between neighbours.
                        let c: Vec<f64> = bx.split([',', ';']).map(|x| x.trim().parse().unwrap()).collect();
                        let (lo, hi) = (DVec3::new(c[0], c[1], c[2]), DVec3::new(c[3], c[4], c[5]));
                        let kept = |v: u32| m.is_vertex(v) && out.is_vertex(v) && out.vertex(v).distance(m.vertex(v)) < 20.0;
                        let boxed: Vec<u32> = m.vertex_ids().filter(|&v| kept(v) && m.vertex(v).cmpge(lo).all() && m.vertex(v).cmple(hi).all()).collect();
                        let disp = |v: u32| out.vertex(v) - m.vertex(v);
                        let mx = boxed.iter().map(|&v| disp(v).length()).fold(0.0, f64::max);
                        let mut worst = (0.0, 0u32, 0u32);
                        for &v in &boxed {
                            for u in out.vertex_neighbors(v) {
                                if kept(u) {
                                    let strain = (disp(v) - disp(u)).length() / m.vertex(v).distance(m.vertex(u)).max(1e-9);
                                    if strain > worst.0 {
                                        worst = (strain, v, u);
                                    }
                                }
                            }
                        }
                        let i = (0..pl.s.len()).min_by(|&x, &y| pl.vessel[x].distance(m.vertex(worst.1)).total_cmp(&pl.vessel[y].distance(m.vertex(worst.1)))).unwrap();
                        eprintln!("  {label}: box {} vertices, largest move {mx:.2}, sharpest change in move {:.2} (per mm) at {:.2?} (s {:.1} along the stent, which runs 0..{:.0})", boxed.len(), worst.0, m.vertex(worst.1), pl.s[i], pl.length);
                    }
                    eprintln!("  {label}: by mm along the stent: triangles, inside 0.9 r, facing the axis, slivers (height < 0.1 × longest edge)");
                    for (s, r) in rows {
                        if r[1] + r[2] + r[3] > 0 {
                            eprintln!("    s {s:>4}: {:>6} {:>5} {:>5} {:>5}", r[0], r[1], r[2], r[3]);
                        }
                    }
                    eprintln!(
                    "{label}: {:.0} ms, crossings {crossings}, moved {}, max shift {:.2}, narrowest ⌀ {:.2} -> {:.2}, sharpest {:.1}°, outside {}",
                    t0.elapsed().as_secs_f64() * 1e3,
                    rep.moved,
                    rep.max_shift,
                    2.0 * rep.narrowest[0],
                    2.0 * rep.narrowest[1],
                    sharpest(&out, &rep.triangles),
                    rep.outside
                    );
                }
                Err(e) => eprintln!("{label}: failed: {e:#}"),
            }
        }
    }

    /// Diagnostic (local only): compare `POLYMIX_DIFF_A` (a .polymix project's first object)
    /// with `POLYMIX_DIFF_B` (an exported mesh) vertex by vertex, in the box
    /// `POLYMIX_DIFF_BOX="x0,y0,z0;x1,y1,z1"`.
    #[test]
    #[ignore]
    fn diff_meshes() {
        let (Some(a), Some(b), Ok(bx)) = (std::env::var_os("POLYMIX_DIFF_A"), std::env::var_os("POLYMIX_DIFF_B"), std::env::var("POLYMIX_DIFF_BOX")) else { return };
        let m0 = crate::project::load(std::path::Path::new(&a)).unwrap().objects[0].mesh.clone();
        let mut m1 = mm_io::read_mesh(std::path::Path::new(&b)).unwrap().mesh;
        // Exported with Z and Y flipped.
        if std::env::var_os("POLYMIX_DIFF_FLIP").is_some() {
            m1.transform(|p| DVec3::new(p.x, p.z, -p.y));
        }
        let c: Vec<f64> = bx.split([',', ';']).map(|x| x.trim().parse().unwrap()).collect();
        let (lo, hi) = (DVec3::new(c[0], c[1], c[2]), DVec3::new(c[3], c[4], c[5]));
        eprintln!("bounds A {:.2?}..{:.2?} | B {:.2?}..{:.2?}", m0.bounds().min, m0.bounds().max, m1.bounds().min, m1.bounds().max);
        eprintln!("first vertices A {:.3?} {:.3?} | B {:.3?} {:.3?}", m0.vertex(0), m0.vertex(1), m1.vertex(0), m1.vertex(1));
        let n = m0.vertex_count().min(m1.vertex_count());
        let same = (0..n as u32).filter(|&v| m0.vertex(v).distance(m1.vertex(v)) < 1e-4).count();
        let moved: Vec<u32> = (0..n as u32).filter(|&v| m0.vertex(v).distance(m1.vertex(v)) > 1e-3).collect();
        let new_place = (0..n as u32).filter(|&v| m0.vertex(v).distance(m1.vertex(v)) > 5.0).count();
        let (ml, mh) = moved.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(l, h), &v| (l.min(m0.vertex(v)), h.max(m0.vertex(v))));
        eprintln!("{} / {} vertices; identical by index {same}; moved > 1 µm: {} within {ml:.1?}..{mh:.1?}; over 5 mm (slots reused): {new_place}", m0.vertex_count(), m1.vertex_count(), moved.len());
        // Which of B's vertices are A's, untouched (same position, to float precision).
        let key = |p: DVec3| [p.x as f32, p.y as f32, p.z as f32].map(f32::to_bits);
        let orig: std::collections::HashSet<[u32; 3]> = m0.vertex_ids().map(|v| key(m0.vertex(v))).collect();
        let untouched = |v: u32| orig.contains(&key(m1.vertex(v)));
        let changed: Vec<u32> = m1.vertex_ids().filter(|&v| !untouched(v)).collect();
        let (cl, ch) = changed.iter().fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(l, h), &v| (l.min(m1.vertex(v)), h.max(m1.vertex(v))));
        let inbox = |p: DVec3| p.cmpge(lo).all() && p.cmple(hi).all();
        eprintln!("B: {} vertices changed or new, within {cl:.1?}..{ch:.1?}; in the box {} of {}", changed.len(), changed.iter().filter(|&&v| inbox(m1.vertex(v))).count(), m1.vertex_ids().filter(|&v| inbox(m1.vertex(v))).count());
        // Creases in the box, before and after.
        let creases = |m: &Mesh, label: &str| {
            let mut list: Vec<(f64, DVec3)> = m
                .edge_ids()
                .filter_map(|e| {
                    let [a, b] = m.edge_triangles(e);
                    let [p, q] = m.edge_vertices(e).map(|v| m.vertex(v));
                    let c = (p + q) * 0.5;
                    (b != mm_core::INVALID && inbox(c)).then(|| (m.tri_normal(a).angle_between(m.tri_normal(b)).to_degrees(), c))
                })
                .collect();
            list.sort_by(|a, b| b.0.total_cmp(&a.0));
            let n = |t: f64| list.iter().filter(|x| x.0 > t).count();
            eprintln!("{label}: {} edges in the box; over 30° {}, over 60° {}, over 90° {}; sharpest {:?}", list.len(), n(30.0), n(60.0), n(90.0), list.iter().take(5).map(|x| format!("{:.0}° at {:.2?}", x.0, x.1)).collect::<Vec<_>>());
        };
        creases(&m0, "before");
        creases(&m1, "after");
        // Edge lengths in the box, before and after.
        let lengths = |m: &Mesh, label: &str| {
            let mut l: Vec<f64> = m.edge_ids().filter(|&e| m.edge_vertices(e).iter().all(|&v| inbox(m.vertex(v)))).map(|e| m.edge_length(e)).collect();
            l.sort_by(f64::total_cmp);
            let q = |f: f64| l.get(((l.len().max(1) - 1) as f64 * f) as usize).copied().unwrap_or(0.0);
            eprintln!("{label}: edge lengths in the box median {:.3}, 99 % {:.3}, longest {:.3}; over 3× the median: {}", q(0.5), q(0.99), q(1.0), l.iter().filter(|&&x| x > 3.0 * q(0.5)).count());
        };
        lengths(&m0, "before");
        lengths(&m1, "after");
        // In the box: how far the wall moved, and the sharpest change in that move between
        // neighbouring vertices.
        let inbox = |p: DVec3| p.cmpge(lo).all() && p.cmple(hi).all();
        let boxed: Vec<u32> = (0..n as u32).filter(|&v| inbox(m1.vertex(v))).collect();
        let disp = |v: u32| m1.vertex(v) - m0.vertex(v);
        let mut stats: Vec<f64> = boxed.iter().map(|&v| disp(v).length()).collect();
        stats.sort_by(f64::total_cmp);
        let pct = |q: f64| stats.get(((stats.len().max(1) - 1) as f64 * q) as usize).copied().unwrap_or(0.0);
        eprintln!("box: {} vertices; move min {:.3} median {:.3} max {:.3}", boxed.len(), pct(0.0), pct(0.5), pct(1.0));
        let mut jumps: Vec<(f64, u32, u32)> = boxed
            .iter()
            .flat_map(|&v| m1.vertex_neighbors(v).filter(move |&u| (u as usize) < n).map(move |u| (v, u)))
            .map(|(v, u)| ((disp(v) - disp(u)).length() / m0.vertex(v).distance(m0.vertex(u)).max(1e-9), v, u))
            .collect();
        jumps.sort_by(|a, b| b.0.total_cmp(&a.0));
        for (j, v, u) in jumps.iter().take(8) {
            eprintln!("  strain {j:.2}: v{v} at {:.2?} moved {:.2?} | v{u} moved {:.2?}", m0.vertex(*v), disp(*v), disp(*u));
        }
    }

    /// A stent rebuilt on a mesh from a Polymix project (an `objects/N.bin` taken out of it):
    /// what changed, and how far past the end blend (counts and distances only).
    /// POLYMIX_STENT_BIN=<objects/N.bin>, POLYMIX_STENT_CASE="x,y,z,nx,ny,nz,d;x,y,z,nx,ny,nz,d;
    /// diameter,length,straightness,continuity(0/1/2),stiffness,end_blend[,remesh 0/1[,covered 0/1]]"
    #[test]
    #[ignore]
    fn stent_moves_on_a_real_mesh() {
        let (Some(path), Ok(case)) = (std::env::var_os("POLYMIX_STENT_BIN"), std::env::var("POLYMIX_STENT_CASE")) else { return };
        let m = crate::project::mesh_from_bytes(&std::fs::read(path).unwrap()).unwrap();
        let parts: Vec<Vec<f64>> = case.split(';').map(|p| p.split(',').filter_map(|x| x.trim().parse().ok()).collect()).collect();
        let sec = |v: &[f64]| {
            let (c, n) = (DVec3::new(v[0], v[1], v[2]), DVec3::new(v[3], v[4], v[5]).normalize());
            let ring = mm_edit::nearest_intersection_curve(&m, mm_core::Plane::from_point_normal(c, n), c).unwrap().segments;
            Section { center: c, normal: n, diameter: v[6], ring }
        };
        let sections = [sec(&parts[0]), sec(&parts[1])];
        let s = &parts[2];
        let continuity = match s[3] as i32 { 0 => EndContinuity::G0, 2 => EndContinuity::G2, _ => EndContinuity::G1 };
        let ui = StentUi { diameter: s[0], length: s[1], straightness: s[2], continuity, stiffness: s[4], end_blend: s[5], remesh: s.get(6).is_none_or(|&x| x != 0.0), covered: s.get(7).is_some_and(|&x| x != 0.0) };
        let t0 = Instant::now();
        let trace = trace_vessel(&m, &sections).unwrap();
        eprintln!("traced in {:.1} s: span {:.1} mm, {} path points, {} owner vertices", t0.elapsed().as_secs_f64(), trace.span, trace.path.points.len(), trace.owner.iter().filter(|&&o| o).count());
        let t0 = Instant::now();
        let (out, pl, rep, crossings) = StentTool::build(&m, &trace, [sections[0].stent(), sections[1].stent()], &StentKey { sections: 1, ui }).unwrap();
        eprintln!("built in {:.1} s: samples s {:.1}..{:.1}, margins {:.1?}, return bend {:.2?}, centerline {:.1} / axis {:.1} between the ends, crossings {crossings}, report moved {} max {:.2}",
            t0.elapsed().as_secs_f64(), pl.s[0], pl.s[pl.s.len() - 1], pl.margins, pl.return_bend, pl.centerline_length, pl.path_length, rep.moved, rep.max_shift);
        // Where along the vessel's traced centerline (arc length) and how far off it; the
        // stent's ends there.
        let line = &trace.path.points;
        let mut arc = vec![0.0];
        for w in line.windows(2) {
            arc.push(arc[arc.len() - 1] + w[0].distance(w[1]));
        }
        let along = |p: DVec3| {
            let mut best = (f64::INFINITY, 0.0);
            for i in 0..line.len() - 1 {
                let (a, ab) = (line[i], line[i + 1] - line[i]);
                let f = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
                let d = p.distance(a + ab * f);
                if d < best.0 {
                    best = (d, arc[i] + (arc[i + 1] - arc[i]) * f);
                }
            }
            best
        };
        let at = |x: f64| pl.vessel[pl.s.partition_point(|&s| s < x).min(pl.s.len() - 1)];
        let (sa, sb) = (along(at(0.0)).1, along(at(ui.length)).1);
        let (s0, s1) = (sa.min(sb), sa.max(sb));
        let eb = ui.end_blend;
        let past = |s: f64| (s0 - eb - s).max(s - s1 - eb);
        let past_of = |p: DVec3| past(along(p).1);
        let (changed, past) = changed_past_the_end_blend(&m, &out, &pl);
        eprintln!("past the end blend: {past} vertices, {changed} changed");
        // The rest by distance past the end blend along the traced centerline (moved: the same
        // vertex, ids freed by the union and remesh being reused by new ones).
        let ring = |mesh: &Mesh, v: u32| {
            let mut n: Vec<u32> = mesh.vertex_neighbors(v).collect();
            n.sort_unstable();
            n
        };
        let kept = |v: u32| out.is_vertex(v) && ring(&out, v) == ring(&m, v);
        let mut rows: Vec<(&str, f64, f64)> = Vec::new();
        for v in m.vertex_ids() {
            let d = if kept(v) { out.vertex(v).distance(m.vertex(v)) } else { f64::INFINITY };
            if d > 0.0 {
                rows.push((if d.is_finite() { "moved" } else { "rebuilt" }, past_of(m.vertex(v)), d));
            }
        }
        for class in ["moved", "rebuilt"] {
            let bins = [f64::NEG_INFINITY, 0.0, 2.0, 5.0, 10.0, 20.0, 40.0, f64::INFINITY];
            for w in bins.windows(2) {
                let b: Vec<_> = rows.iter().filter(|r| r.0 == class && r.1 >= w[0] && r.1 < w[1]).collect();
                if !b.is_empty() {
                    let max = b.iter().map(|r| r.2).filter(|d| d.is_finite()).fold(0.0, f64::max);
                    eprintln!("  {class} {:>5.0}..{:<5.0} mm past the end blend: {} (largest move {max:.2} mm)", w[0], w[1], b.len());
                }
            }
        }
    }
}
