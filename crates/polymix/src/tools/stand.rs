//! Stand (Edit → Stand, Vessel → Stand): a base under the object and legs holding it up off
//! it, to set a printed model on a bench. Drag a line across the view (seen from the side)
//! to put the base's top there, or size and place it in the panel; click the object to add a
//! leg (drag a leg's root or foot to move it, right-click either to remove the leg). Accept
//! unites the base and legs with the object (or appends them, or makes them a new object).
//!
//! The legs are built in the background as they change (`mm_edit::leg_mesh`, tens of
//! milliseconds each); the preview shows each leg's latest build, its outline while one is
//! under way. While a leg's end or a number in the panel is being dragged only the outlines
//! follow: the base and the legs are built again when it is let go (building them as it
//! went made the view lag), and a number typed in takes effect on Enter. A leg's root sinks into the object's outer wall by less than half the wall's
//! thickness, so uniting changes only the outer surface (see `mm_edit::stand`).

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use mm_core::{Bvh, DVec2, Plane};
use mm_edit::{BaseFrame, BaseParams, BaseShape, LegParams, LegPlan};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combine {
    /// United with the object.
    Boolean,
    /// Added to the object as separate shells.
    Append,
    NewObject,
}

/// A leg's three diameters (mm).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Sizes {
    diameter: f64,
    root: f64,
    foot: f64,
}

/// Everything a leg's mesh is made from: when any of it changes the leg is planned and built
/// again.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spec {
    root: DVec3,
    foot: DVec3,
    base: BaseFrame,
    thickness: f64,
    params: LegParams,
}

struct Leg {
    /// On the object's surface.
    root: DVec3,
    /// On the base's top, in its (width, depth) coordinates, when placed by hand (it then
    /// moves with the base); otherwise below the root ([`mm_edit::default_foot`]).
    foot: Option<DVec2>,
    sizes: Sizes,
    plan: Option<(Spec, Result<LegPlan, String>)>,
    /// The latest mesh built, and what for (shown until the next one is ready).
    built: Option<(Spec, Arc<Mesh>)>,
    job: Option<(Spec, Job<Mesh>)>,
    failed: Option<(Spec, String)>,
}

impl Leg {
    fn plan(&self) -> Option<&LegPlan> {
        self.plan.as_ref().and_then(|(_, p)| p.as_ref().ok())
    }

    /// Built for `spec`.
    fn ready(&self, spec: &Spec) -> bool {
        self.built.as_ref().is_some_and(|(s, _)| s == spec)
    }
}

enum Drag {
    Root(usize),
    Foot(usize),
    Line(egui::Pos2, egui::Pos2),
}

/// What an accept made: the mesh, the number added to the stand's face groups in it (the base
/// is group 0 of the stand, leg k group k + 1) and the number of legs.
struct Accepted {
    mesh: Mesh,
    offset: i32,
    legs: usize,
}

/// A base's mesh (or why it could not be made) and where and what size it was made for.
type BuiltBase = ((BaseFrame, BaseParams), Result<Arc<Mesh>, String>);

/// Screen distance (points) within which a leg's handle is grabbed.
const GRAB: f32 = 10.0;

pub struct StandTool {
    object: Option<ObjectId>,
    source: Arc<Mesh>,
    bvh: Arc<Bvh>,
    base: BaseFrame,
    base_params: BaseParams,
    /// The base's mesh and what it was built for.
    base_mesh: Option<BuiltBase>,
    /// Fit to object: the margin around the object's footprint.
    margin: f64,
    legs: Vec<Leg>,
    /// The leg last placed or touched (highlighted; Delete removes it).
    current: Option<usize>,
    /// New legs take these; changing them in the panel sets every leg's.
    all: Sizes,
    root_flare: f64,
    foot_flare: f64,
    bend: f64,
    combine: Combine,
    preview: Option<Mesh>,
    /// What the preview shows: the base's and the legs' meshes (by address).
    preview_key: Vec<usize>,
    drag: Option<Drag>,
    /// A number in the panel being dragged (see [`StandTool::held`]).
    panel_held: bool,
    pressed_at: Option<egui::Pos2>,
    job: Option<Job<Accepted>>,
    error: Option<String>,
}

/// `x` rounded to a tenth of a millimetre.
fn tenth(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

impl StandTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, bvh) = match cx.scene.active.and_then(|id| cx.scene.get_mut(id)) {
            Some(o) => (Some(o.id), o.mesh.clone(), o.bvh()),
            None => (None, Arc::new(Mesh::new()), Arc::new(Bvh::build(&Mesh::new()))),
        };
        let diag = source.bounds().diagonal().max(1e-6);
        let up = cx.camera.world_up();
        // The base's width along the world axis most square to up.
        let width = [DVec3::X, DVec3::Y, DVec3::Z].into_iter().min_by(|a, b| a.dot(up).abs().total_cmp(&b.dot(up).abs())).unwrap_or(DVec3::X);
        let d = tenth(diag * 0.02).clamp(0.8, 8.0);
        let margin = tenth(diag * 0.05).clamp(2.0, 20.0);
        let mut t = Self {
            object,
            source,
            bvh,
            base: BaseFrame::new(DVec3::ZERO, up, width),
            base_params: BaseParams { shape: BaseShape::Rectangle, width: 1.0, depth: 1.0, thickness: tenth(diag * 0.03).clamp(1.5, 10.0), corner_radius: margin },
            base_mesh: None,
            margin,
            legs: Vec::new(),
            current: None,
            all: Sizes { diameter: d, root: 2.0 * d, foot: 2.0 * d },
            root_flare: tenth(1.5 * d),
            foot_flare: tenth(1.5 * d),
            bend: 0.5,
            combine: Combine::Boolean,
            preview: None,
            preview_key: Vec::new(),
            drag: None,
            panel_held: false,
            pressed_at: None,
            job: None,
            error: None,
        };
        if t.object.is_some() && t.source.triangle_count() > 0 {
            t.fit_base();
            t.set_gap(tenth(diag * 0.12).clamp(5.0, 40.0));
        }
        t
    }

    fn params(&self, s: Sizes) -> LegParams {
        LegParams { diameter: s.diameter, root_diameter: s.root, foot_diameter: s.foot, root_flare: self.root_flare, foot_flare: self.foot_flare, bend: self.bend }
    }

    /// Where leg `i`'s foot is on the base's top.
    fn foot_of(&self, i: usize) -> DVec3 {
        let leg = &self.legs[i];
        match leg.foot {
            Some(q) => self.base.world(DVec3::new(q.x, q.y, 0.0)),
            None => mm_edit::default_foot(&self.source, &self.bvh, leg.root, &self.base, &self.base_params, leg.sizes.root),
        }
    }

    fn spec(&self, i: usize) -> Spec {
        let leg = &self.legs[i];
        Spec { root: leg.root, foot: self.foot_of(i), base: self.base, thickness: self.base_params.thickness, params: self.params(leg.sizes) }
    }

    /// The object's lowest point's height above the base's top.
    fn gap(&self) -> f64 {
        self.source.vertex_ids().map(|v| (self.source.vertex(v) - self.base.center).dot(self.base.up)).fold(f64::INFINITY, f64::min)
    }

    /// Move the base along its up so the object's lowest point is `gap` above its top.
    fn set_gap(&mut self, gap: f64) {
        let now = self.gap();
        if now.is_finite() {
            self.base.center += self.base.up * (now - gap);
        }
    }

    /// Centre the base under the object (and the legs' feet) and size it to their footprint
    /// plus the margin all round, keeping its height and turn.
    fn fit_base(&mut self) {
        let (u, v) = (self.base.width_axis, self.base.depth_axis());
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        let mut add = |p: DVec3, r: f64| {
            let q = DVec2::new(p.dot(u), p.dot(v));
            lo = lo.min(q - DVec2::splat(r));
            hi = hi.max(q + DVec2::splat(r));
        };
        for vid in self.source.vertex_ids() {
            add(self.source.vertex(vid), 0.0);
        }
        for i in 0..self.legs.len() {
            add(self.foot_of(i), 0.5 * self.legs[i].sizes.foot);
        }
        if !lo.is_finite() || !hi.is_finite() {
            return;
        }
        let c = (lo + hi) * 0.5;
        let here = DVec2::new(self.base.center.dot(u), self.base.center.dot(v));
        self.base.center += u * (c.x - here.x) + v * (c.y - here.y);
        let size = hi - lo + DVec2::splat(2.0 * self.margin);
        self.base_params.width = size.x.max(1e-3);
        self.base_params.depth = size.y.max(1e-3);
        if self.base_params.shape == BaseShape::Ellipse {
            // The ellipse round the rectangle's corners.
            self.base_params.width = (size.x - 2.0 * self.margin) * std::f64::consts::SQRT_2 + 2.0 * self.margin;
            self.base_params.depth = (size.y - 2.0 * self.margin) * std::f64::consts::SQRT_2 + 2.0 * self.margin;
        }
    }

    /// The base's top level (its up the world's), keeping its centre.
    fn level(&mut self, up: DVec3) {
        self.base = BaseFrame::new(self.base.center, up, self.base.width_axis);
    }

    /// Turn the base about its up by `deg` degrees.
    fn turn(&mut self, deg: f64) {
        let q = glam::DQuat::from_axis_angle(self.base.up, deg.to_radians());
        self.base = BaseFrame::new(self.base.center, self.base.up, q * self.base.width_axis);
    }

    /// The base's top under the screen point `pos`, in its (width, depth) coordinates.
    fn on_base(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<DVec2> {
        let ray = cx.ray_at(pos);
        let denom = ray.dir.dot(self.base.up);
        if denom.abs() < 1e-9 {
            return None;
        }
        let t = (self.base.center - ray.origin).dot(self.base.up) / denom;
        let q = self.base.local(ray.at(t));
        (t > 0.0).then_some(DVec2::new(q.x, q.y))
    }

    /// Put the base's top in the plane through the screen stroke `a`–`b` and the view
    /// direction, as wide as the stroke, its up toward the object; centred on the object
    /// front to back (the stroke says nothing about that).
    fn draw_base(&mut self, cx: &ToolCx, a: egui::Pos2, b: egui::Pos2) {
        let centre = self.source.bounds().center();
        let fwd = cx.camera.forward();
        let depth = Plane::from_point_normal(centre, fwd);
        let at = |p: egui::Pos2| {
            let r = cx.ray_at(p);
            r.at(-depth.signed_distance(r.origin) / r.dir.dot(depth.normal))
        };
        let (pa, pb) = (at(a), at(b));
        let along = pb - pa;
        let Some(mut up) = along.cross(fwd).try_normalize() else { return };
        let mid = (pa + pb) * 0.5;
        if up.dot(centre - mid) < 0.0 {
            up = -up;
        }
        let mut base = BaseFrame::new(mid, up, along);
        let v = base.depth_axis();
        let (lo, hi) = self.source.vertex_ids().map(|i| self.source.vertex(i).dot(v)).fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), x| (l.min(x), h.max(x)));
        if lo.is_finite() {
            base.center += v * (0.5 * (lo + hi) - mid.dot(v));
        }
        self.base = base;
        self.base_params.width = along.length().max(1e-3);
    }

    fn add_leg(&mut self, root: DVec3, foot: Option<DVec3>) -> usize {
        let foot = foot.map(|f| {
            let q = self.base.local(f);
            DVec2::new(q.x, q.y)
        });
        self.legs.push(Leg { root, foot, sizes: self.all, plan: None, built: None, job: None, failed: None });
        self.current = Some(self.legs.len() - 1);
        self.legs.len() - 1
    }

    fn remove_leg(&mut self, i: usize) {
        if i < self.legs.len() {
            self.legs.remove(i);
            self.current = (!self.legs.is_empty()).then(|| i.min(self.legs.len() - 1));
            self.drag = None;
        }
    }

    /// Whether an edit is under way, a leg's end dragged in the view or a number dragged in
    /// the panel: the legs' plans (and so the outlines) follow it, while the base's and the
    /// legs' meshes and the preview wait for it to end.
    fn held(&self) -> bool {
        self.panel_held || matches!(self.drag, Some(Drag::Root(_) | Drag::Foot(_)))
    }

    /// The base's mesh, if it was built for the base as it is now.
    fn current_base(&self) -> Option<&Result<Arc<Mesh>, String>> {
        self.base_mesh.as_ref().filter(|(k, _)| *k == (self.base, self.base_params)).map(|(_, m)| m)
    }

    /// Bring everything up to date: the base's mesh, each leg's plan, its build (started in
    /// the background, collected when done) and the preview; while an edit is under way
    /// ([`StandTool::held`]) only the plans.
    fn refresh(&mut self, ctx: &egui::Context) {
        if self.object.is_none() {
            return;
        }
        let held = self.held();
        let key = (self.base, self.base_params);
        if !held && self.base_mesh.as_ref().is_none_or(|(k, _)| *k != key) {
            let edge = (self.base_params.width.min(self.base_params.depth) / 30.0).max(0.2);
            let m = mm_edit::base_mesh(&self.base, &self.base_params, edge).map(Arc::new).map_err(|e| e.to_string());
            self.base_mesh = Some((key, m));
        }
        let specs: Vec<Spec> = (0..self.legs.len()).map(|i| self.spec(i)).collect();
        for (leg, spec) in self.legs.iter_mut().zip(specs) {
            if leg.plan.as_ref().is_none_or(|(s, _)| *s != spec) {
                let plan = mm_edit::plan_leg(&self.source, &self.bvh, spec.root, spec.foot, &spec.base, spec.thickness, &spec.params).map_err(|e| e.to_string());
                leg.plan = Some((spec, plan));
            }
            if let Some((s, job)) = &leg.job {
                if let Some(r) = job.poll() {
                    match r {
                        Ok(m) => leg.built = Some((*s, Arc::new(m))),
                        Err(e) => leg.failed = Some((*s, format!("{e:#}"))),
                    }
                    leg.job = None;
                }
            }
            // One build at a time per leg: a newer spec waits for the one under way.
            let wanted = !leg.ready(&spec) && leg.failed.as_ref().is_none_or(|(s, _)| *s != spec);
            if wanted && leg.job.is_none() && !held {
                if let Some(plan) = leg.plan().cloned() {
                    leg.job = Some((spec, Job::spawn("Stand leg", ctx.clone(), move |p| mm_edit::leg_mesh(&plan, Some(p)).map_err(|e| anyhow::anyhow!("{e}")))));
                }
            }
        }
        if !held {
            self.refresh_preview();
        }
    }

    /// The meshes the preview shows: the base's, then each leg's latest build (none for a
    /// leg that cannot be made as it is now).
    fn shown(&self) -> Vec<&Arc<Mesh>> {
        let base = self.base_mesh.as_ref().and_then(|(_, m)| m.as_ref().ok());
        base.into_iter().chain(self.legs.iter().filter(|l| l.plan().is_some()).filter_map(|l| l.built.as_ref().map(|(_, m)| m))).collect()
    }

    fn refresh_preview(&mut self) {
        let key: Vec<usize> = self.shown().iter().map(|m| Arc::as_ptr(m) as usize).collect();
        if key == self.preview_key && self.preview.is_some() {
            return;
        }
        let mut prev = (*self.source).clone();
        for m in self.shown() {
            prev.append_mesh(m, prev.max_group_id() + 1);
        }
        self.preview = Some(prev);
        self.preview_key = key;
    }

    /// Every leg's current plan, or the first leg's reason it cannot be made.
    fn plans(&self) -> Result<Vec<(Spec, LegPlan)>, String> {
        (0..self.legs.len())
            .map(|i| {
                let spec = self.spec(i);
                match &self.legs[i].plan {
                    Some((s, Ok(p))) if *s == spec => Ok((spec, p.clone())),
                    Some((s, Err(e))) if *s == spec => Err(format!("Leg {}: {e}", i + 1)),
                    _ => mm_edit::plan_leg(&self.source, &self.bvh, spec.root, spec.foot, &spec.base, spec.thickness, &spec.params).map(|p| (spec, p)).map_err(|e| format!("Leg {}: {e}", i + 1)),
                }
            })
            .collect()
    }

    /// Whether the object still has the mesh the stand was made for (otherwise the tool's
    /// work is dropped, with a message, rather than combined with another mesh).
    fn source_is_current(&self, cx: &mut ToolCx) -> bool {
        let current = self.object.and_then(|id| cx.scene.get(id)).is_some_and(|o| Arc::ptr_eq(&o.mesh, &self.source));
        if !current {
            *cx.status = "Stand cancelled: the object changed while the tool was open.".into();
        }
        current
    }

    /// Start combining the stand with the object in the background (applied by
    /// [`StandTool::poll`]). False with an error to show when it cannot be made.
    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() || self.object.is_none() {
            return false;
        }
        let plans = match self.plans() {
            Ok(p) => p,
            Err(e) => {
                self.error = Some(e);
                return false;
            }
        };
        let base = match self.current_base().cloned() {
            Some(Ok(m)) => m,
            Some(Err(e)) => {
                self.error = Some(format!("The base: {e}"));
                return false;
            }
            None => match mm_edit::base_mesh(&self.base, &self.base_params, (self.base_params.width.min(self.base_params.depth) / 30.0).max(0.2)) {
                Ok(m) => Arc::new(m),
                Err(e) => {
                    self.error = Some(format!("The base: {e}"));
                    return false;
                }
            },
        };
        // Legs already built for what they are now are used as they are.
        let legs: Vec<(LegPlan, Option<Arc<Mesh>>)> =
            plans.into_iter().zip(&self.legs).map(|((spec, plan), leg)| (plan, leg.built.as_ref().filter(|(s, _)| *s == spec).map(|(_, m)| m.clone()))).collect();
        let (source, combine) = (self.source.clone(), self.combine);
        self.error = None;
        self.job = Some(Job::spawn("Stand", cx.egui.clone(), move |progress| {
            let n = legs.len();
            let mut meshes = Vec::with_capacity(n);
            for (k, (plan, built)) in legs.into_iter().enumerate() {
                if !progress(0.5 * k as f32 / n.max(1) as f32) {
                    anyhow::bail!("cancelled");
                }
                meshes.push(match built {
                    Some(m) => m,
                    None => Arc::new(mm_edit::leg_mesh(&plan, None).map_err(|e| anyhow::anyhow!("leg {}: {e}", k + 1))?),
                });
            }
            let refs: Vec<&Mesh> = meshes.iter().map(|m| m.as_ref()).collect();
            let joining = |f: f32| progress(0.5 + 0.2 * f);
            let stand = mm_edit::join_stand(&base, &refs, Some(&joining)).map_err(|e| anyhow::anyhow!("joining the legs to the base: {e}"))?;
            if !progress(0.7) {
                anyhow::bail!("cancelled");
            }
            Ok(match combine {
                Combine::Boolean => {
                    let r = mm_edit::mesh_boolean(&source, &stand, mm_edit::BooleanOp::Union).map_err(|e| anyhow::anyhow!("{e}"))?;
                    Accepted { mesh: r.mesh, offset: r.group_offset, legs: n }
                }
                Combine::Append => {
                    let mut m = (*source).clone();
                    let offset = m.max_group_id() + 1;
                    m.append_mesh(&stand, offset);
                    Accepted { mesh: m, offset, legs: n }
                }
                Combine::NewObject => Accepted { mesh: stand, offset: 0, legs: n },
            })
        }));
        true
    }

    /// Apply the accepted stand once it is done, if the object still has the mesh it was made
    /// for. True when the tool is done.
    fn poll(&mut self, cx: &mut ToolCx) -> bool {
        let Some(r) = self.job.as_ref().and_then(|j| j.poll()) else { return false };
        self.job = None;
        match r {
            Ok(a) => {
                if let Some(id) = self.object.filter(|_| self.source_is_current(cx)) {
                    self.apply(cx, id, a);
                }
                true
            }
            Err(e) => {
                let hint = if self.combine == Combine::Boolean { " — use Combine Mode: Append, or make the object solid first" } else { "" };
                self.error = Some(format!("{e:#}{hint}"));
                false
            }
        }
    }

    fn apply(&mut self, cx: &mut ToolCx, id: ObjectId, a: Accepted) {
        cx.history.push("Stand", cx.scene);
        let target = match self.combine {
            Combine::NewObject => {
                let name = cx.scene.get(id).map(|o| format!("{} stand", o.name)).unwrap_or_else(|| "stand".into());
                cx.scene.add(name, a.mesh)
            }
            _ => {
                if let Some(o) = cx.scene.get_mut(id) {
                    o.set_mesh(a.mesh);
                }
                id
            }
        };
        if let Some(o) = cx.scene.get_mut(target) {
            let present: std::collections::HashSet<i32> = o.mesh.triangle_ids().map(|t| o.mesh.tri_group(t)).collect();
            for (k, prefix) in std::iter::once("base").chain(std::iter::repeat_n("leg", a.legs)).enumerate() {
                let g = a.offset + k as i32;
                if present.contains(&g) {
                    let name = super::vessel::next_name(&o.group_names, prefix);
                    o.group_names.insert(g, name);
                }
            }
        }
        let legs = match a.legs {
            1 => "1 leg".to_string(),
            n => format!("{n} legs"),
        };
        *cx.status = match self.combine {
            Combine::Boolean => format!("Stand: a base and {legs} united with the object"),
            Combine::Append => format!("Stand: a base and {legs} added to the object"),
            Combine::NewObject => format!("Stand: a base and {legs}, a new object"),
        };
    }

    /// The leg handle under `pos`: (leg, whether it is the foot).
    fn handle_at(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<(usize, bool)> {
        let mut best: Option<(usize, bool, f32)> = None;
        for i in 0..self.legs.len() {
            for (foot, p) in [(false, self.legs[i].root), (true, self.foot_of(i))] {
                if let Some(d) = cx.project(p).map(|q| q.distance(pos)).filter(|d| *d < GRAB) {
                    if best.is_none_or(|b| d < b.2) {
                        best = Some((i, foot, d));
                    }
                }
            }
        }
        best.map(|(i, foot, _)| (i, foot))
    }

    /// What to point out about each leg: (leg, note, whether it stops the stand being made).
    fn notes(&self) -> Vec<(usize, String, bool)> {
        let mut out = Vec::new();
        for (i, leg) in self.legs.iter().enumerate() {
            match &leg.plan {
                Some((_, Err(e))) => out.push((i, e.clone(), true)),
                Some((spec, Ok(p))) => {
                    if let Some((s, e)) = &leg.failed {
                        if s == spec {
                            out.push((i, format!("could not be built: {e}"), true));
                        }
                    }
                    // Both would cut into the object (and could reach a lumen): not made.
                    if p.clearance < 0.0 {
                        out.push((i, "runs into the object: move its foot or root, or change its bend".into(), true));
                    }
                    if p.buried {
                        out.push((i, "the object rises over its root (a crotch, or another branch close by): move the root or narrow it".into(), true));
                    }
                    if p.overhang > 0.05 {
                        out.push((i, format!("its root is wider than the surface under it ({:.0}% of it overhangs)", 100.0 * p.overhang), false));
                    }
                    let q = spec.base.local(spec.foot);
                    if !self.base_params.contains(DVec2::new(q.x, q.y)) {
                        out.push((i, "its foot is off the base".into(), false));
                    }
                }
                None => {}
            }
        }
        out
    }

    fn size_key(s: &Sizes) -> serde_json::Value {
        serde_json::json!({"diameter_mm": s.diameter, "root_diameter_mm": s.root, "foot_diameter_mm": s.foot})
    }

    /// Apply the diameters given in `a` (any of diameter_mm / root_diameter_mm /
    /// foot_diameter_mm) to `s`.
    fn sizes_from(a: &Args, s: &mut Sizes) -> Result<(), String> {
        for (k, v) in [("diameter_mm", &mut s.diameter), ("root_diameter_mm", &mut s.root), ("foot_diameter_mm", &mut s.foot)] {
            if let Some(x) = a.opt_f64(k) {
                if !(x > 0.0 && x.is_finite()) {
                    return Err(format!("{k} must be positive"));
                }
                *v = x;
            }
        }
        Ok(())
    }

    fn leg_index(&self, a: &Args) -> Result<usize, String> {
        let k = a.usize("leg")?;
        (1..=self.legs.len()).contains(&k).then_some(k - 1).ok_or_else(|| format!("there is no leg {k} (there are {})", self.legs.len()))
    }

    /// The base's settings; true while one of its numbers is being dragged.
    fn base_ui(&mut self, ui: &mut egui::Ui, world_up: DVec3) -> bool {
        let mut held = false;
        ui_util::section(ui, "Base");
        let p = &mut self.base_params;
        ui.horizontal(|ui| {
            ui.label("Shape");
            ui.selectable_value(&mut p.shape, BaseShape::Rectangle, "Rounded rectangle");
            ui.selectable_value(&mut p.shape, BaseShape::Ellipse, "Ellipse");
        });
        fn size(v: &mut f64) -> egui::DragValue<'_> {
            egui::DragValue::new(v).speed(0.1).range(0.1..=1e5).min_decimals(1).max_decimals(2).suffix(" mm").update_while_editing(false)
        }
        egui::Grid::new("stand_base_size").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
            ui.label("Width");
            held |= ui.add(size(&mut p.width)).dragged();
            ui.end_row();
            ui.label("Depth");
            held |= ui.add(size(&mut p.depth)).dragged();
            ui.end_row();
            ui.label("Thickness");
            held |= ui.add(size(&mut p.thickness)).dragged();
            ui.end_row();
            if p.shape == BaseShape::Rectangle {
                ui.label("Corner radius");
                held |= ui.add(egui::DragValue::new(&mut p.corner_radius).speed(0.1).range(0.0..=1e5).max_decimals(2).suffix(" mm").update_while_editing(false)).dragged();
                ui.end_row();
            }
        });
        let mut gap = self.gap();
        ui.horizontal(|ui| {
            ui.label("Below the object").on_hover_text("From the object's lowest point down to the base's top (negative: the base cuts into the object)");
            let r = ui.add(egui::DragValue::new(&mut gap).speed(0.1).max_decimals(2).suffix(" mm").update_while_editing(false));
            held |= r.dragged();
            if r.changed() {
                self.set_gap(gap);
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Fit to object").on_hover_text("Centre the base under the object and its legs' feet, the margin all round").clicked() {
                self.fit_base();
            }
            ui.add(egui::DragValue::new(&mut self.margin).speed(0.1).range(0.0..=1e4).max_decimals(1).prefix("margin ").suffix(" mm").update_while_editing(false));
        });
        ui.horizontal_wrapped(|ui| {
            ui.label("Turn");
            if ui.small_button("⟲").on_hover_text("5° counter-clockwise, seen from above").clicked() {
                self.turn(5.0);
            }
            if ui.small_button("⟳").on_hover_text("5° clockwise").clicked() {
                self.turn(-5.0);
            }
            if ui.button("Level").on_hover_text("Make the base's top level (square to the world's up)").clicked() {
                self.level(world_up);
            }
        });
        held
    }

    /// The legs' settings; true while one of their numbers is being dragged.
    fn legs_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut held = false;
        ui_util::section(ui, "Legs");
        fn drag(v: &mut f64) -> egui::DragValue<'_> {
            egui::DragValue::new(v).speed(0.05).range(0.05..=1e4).min_decimals(1).max_decimals(2).update_while_editing(false)
        }
        let mut remove = None;
        egui::Grid::new("stand_legs").num_columns(5).spacing([8.0, 4.0]).show(ui, |ui| {
            ui.label("");
            ui.label("⌀ mm").on_hover_text("The column's diameter");
            ui.label("root ⌀").on_hover_text("Where the leg meets the object it flares out to this");
            ui.label("foot ⌀").on_hover_text("Where it meets the base it flares out to this");
            ui.end_row();
            ui.label("All").on_hover_text("New legs take these; changing one sets it on every leg");
            let before = self.all;
            held |= ui.add(drag(&mut self.all.diameter)).dragged();
            held |= ui.add(drag(&mut self.all.root)).dragged();
            held |= ui.add(drag(&mut self.all.foot)).dragged();
            ui.end_row();
            for (field, changed) in [(0, self.all.diameter != before.diameter), (1, self.all.root != before.root), (2, self.all.foot != before.foot)] {
                if changed {
                    for leg in &mut self.legs {
                        match field {
                            0 => leg.sizes.diameter = self.all.diameter,
                            1 => leg.sizes.root = self.all.root,
                            _ => leg.sizes.foot = self.all.foot,
                        }
                    }
                }
            }
            for (i, leg) in self.legs.iter_mut().enumerate() {
                let mut label = egui::RichText::new(format!("Leg {}", i + 1));
                if self.current == Some(i) {
                    label = label.strong();
                }
                let info = match leg.plan() {
                    Some(p) => {
                        let wall = p.wall.map_or("a solid part".to_string(), |w| format!("a {w:.2} mm wall"));
                        format!("Root set {:.2} mm into {wall} (never more than {:.0}% of the way to the next surface)", p.embed, 100.0 * mm_edit::stand::PLUG_SHARE)
                    }
                    None => "Cannot be made as it is".into(),
                };
                if ui.selectable_label(self.current == Some(i), label).on_hover_text(info).clicked() {
                    self.current = Some(i);
                }
                held |= ui.add(drag(&mut leg.sizes.diameter)).dragged();
                held |= ui.add(drag(&mut leg.sizes.root)).dragged();
                held |= ui.add(drag(&mut leg.sizes.foot)).dragged();
                if ui.small_button("✕").on_hover_text("Remove this leg").clicked() {
                    remove = Some(i);
                }
                ui.end_row();
            }
        });
        if let Some(i) = remove {
            self.remove_leg(i);
        }
        egui::Grid::new("stand_legs_shape").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
            ui.label("Root flare").on_hover_text("How far up the leg its root's flare runs");
            held |= ui.add(drag(&mut self.root_flare).suffix(" mm")).dragged();
            ui.end_row();
            ui.label("Foot flare").on_hover_text("How far up the leg its foot's flare runs");
            held |= ui.add(drag(&mut self.foot_flare).suffix(" mm")).dragged();
            ui.end_row();
            ui.label("Bend").on_hover_text("How far a leg keeps to its root's direction (square to the object) before it turns down to the base; 0: straight");
            held |= ui.add(egui::Slider::new(&mut self.bend, 0.0..=1.0)).dragged();
            ui.end_row();
        });
        for (i, note, serious) in self.notes() {
            let colour = if serious { egui::Color32::from_rgb(240, 110, 90) } else { egui::Color32::from_rgb(230, 170, 70) };
            ui.colored_label(colour, format!("Leg {}: {note}", i + 1));
        }
        held
    }
}

impl Tool for StandTool {
    fn name(&self) -> &'static str {
        "Stand"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        use polymix_workspace::agent::vec3;
        let p = &self.base_params;
        let shape = match p.shape {
            BaseShape::Rectangle => "rectangle",
            BaseShape::Ellipse => "ellipse",
        };
        let combine = match self.combine {
            Combine::Boolean => "boolean",
            Combine::Append => "append",
            Combine::NewObject => "new_object",
        };
        let mut legs = Self::size_key(&self.all);
        legs["root_flare_mm"] = self.root_flare.into();
        legs["foot_flare_mm"] = self.foot_flare.into();
        legs["bend"] = self.bend.into();
        Some(serde_json::json!({
            "base": {
                "shape": shape, "width_mm": p.width, "depth_mm": p.depth, "thickness_mm": p.thickness, "corner_radius_mm": p.corner_radius,
                "center_mm": vec3(self.base.center.to_array(), 5), "up": vec3(self.base.up.to_array(), 6), "width_axis": vec3(self.base.width_axis.to_array(), 6),
                "margin_mm": self.margin,
            },
            "legs": legs,
            "combine": combine,
        }))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let v3 = |about: &str| serde_json::json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": about});
        let mm = |about: &str| serde_json::json!({"type": "number", "exclusiveMinimum": 0, "description": about});
        Some(serde_json::json!({"type": "object", "properties": {
            "base": {"type": "object", "properties": {
                "shape": {"type": "string", "enum": ["rectangle", "ellipse"], "description": "rectangle: with rounded corners (corner_radius_mm)"},
                "width_mm": mm("Along width_axis"), "depth_mm": mm("Along up × width_axis"), "thickness_mm": mm("Below the top"),
                "corner_radius_mm": {"type": "number", "minimum": 0, "description": "rectangle: radius of the rounded corners"},
                "center_mm": v3("The centre of the base's top face"),
                "up": v3("The way up from the base's top, toward the object"),
                "width_axis": v3("The direction the width runs (turned into the top's plane)"),
                "margin_mm": {"type": "number", "minimum": 0, "description": "fit_base: room left round the object's footprint"},
            }},
            "legs": {"type": "object", "properties": {
                "diameter_mm": mm("Column diameter for new legs; changing it sets every leg's"),
                "root_diameter_mm": mm("Where a leg meets the object it flares out to this; new legs take it, changing it sets every leg's"),
                "foot_diameter_mm": mm("Where a leg meets the base it flares out to this; new legs take it, changing it sets every leg's"),
                "root_flare_mm": mm("How far up each leg its root's flare runs"),
                "foot_flare_mm": mm("How far up each leg its foot's flare runs"),
                "bend": {"type": "number", "minimum": 0, "maximum": 1, "description": "How far legs keep to their root's direction before turning down to the base; 0 straight"},
            }},
            "combine": {"type": "string", "enum": ["boolean", "append", "new_object"], "description": "boolean: united with the object (needs a closed object); append: separate shells of the object; new_object: the base and legs as an object of their own"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.job = None;
        let b = Args(&v["base"]);
        let shape = match b.opt_str("shape") {
            Some("rectangle") => BaseShape::Rectangle,
            Some("ellipse") => BaseShape::Ellipse,
            _ => return Err("base.shape must be rectangle or ellipse".into()),
        };
        let positive = |a: &Args, k: &str| a.f64(k).and_then(|x| if x > 0.0 && x.is_finite() { Ok(x) } else { Err(format!("{k} must be positive")) });
        self.base_params = BaseParams {
            shape,
            width: positive(&b, "width_mm")?,
            depth: positive(&b, "depth_mm")?,
            thickness: positive(&b, "thickness_mm")?,
            corner_radius: b.f64("corner_radius_mm")?.max(0.0),
        };
        let up = DVec3::from_array(b.point("up")?).try_normalize().ok_or("base.up cannot be zero")?;
        self.base = BaseFrame::new(DVec3::from_array(b.point("center_mm")?), up, DVec3::from_array(b.point("width_axis")?));
        self.margin = b.f64("margin_mm")?.max(0.0);
        let l = Args(&v["legs"]);
        let mut all = self.all;
        Self::sizes_from(&l, &mut all)?;
        // As the panel's All row: a shared diameter that changes is set on every leg.
        for leg in &mut self.legs {
            if all.diameter != self.all.diameter {
                leg.sizes.diameter = all.diameter;
            }
            if all.root != self.all.root {
                leg.sizes.root = all.root;
            }
            if all.foot != self.all.foot {
                leg.sizes.foot = all.foot;
            }
        }
        self.all = all;
        self.root_flare = positive(&l, "root_flare_mm")?;
        self.foot_flare = positive(&l, "foot_flare_mm")?;
        self.bend = l.f64("bend")?.clamp(0.0, 1.0);
        self.combine = match v["combine"].as_str() {
            Some("boolean") => Combine::Boolean,
            Some("append") => Combine::Append,
            Some("new_object") => Combine::NewObject,
            _ => return Err("combine must be boolean, append or new_object".into()),
        };
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        let notes = self.notes();
        let legs: Vec<serde_json::Value> = (0..self.legs.len())
            .map(|i| {
                let leg = &self.legs[i];
                let spec = self.spec(i);
                let mut j = Self::size_key(&leg.sizes);
                j["leg"] = (i + 1).into();
                j["root_mm"] = vec3(leg.root.to_array(), 4);
                j["foot_mm"] = vec3(spec.foot.to_array(), 4);
                j["foot_placed_by_hand"] = leg.foot.is_some().into();
                if let Some(p) = leg.plan() {
                    j["root_set_into_object_mm"] = round(p.embed, 3).into();
                    j["wall_under_root_mm"] = p.wall.map(|w| round(w, 3)).into();
                    j["root_overhang_fraction"] = round(p.overhang, 3).into();
                    j["clearance_from_object_mm"] = if p.clearance.is_finite() { round(p.clearance, 3).into() } else { serde_json::Value::Null };
                }
                j["built"] = leg.ready(&spec).into();
                j["triangles"] = leg.built.as_ref().map(|(_, m)| m.triangle_count()).into();
                j["notes"] = notes.iter().filter(|n| n.0 == i).map(|n| n.1.clone()).collect::<Vec<_>>().into();
                j
            })
            .collect();
        let ready = self.current_base().is_some_and(|m| m.is_ok()) && (0..self.legs.len()).all(|i| self.legs[i].ready(&self.spec(i)));
        serde_json::json!({
            "object": self.object,
            "base": {"center_mm": vec3(self.base.center.to_array(), 4), "up": vec3(self.base.up.to_array(), 5), "below_object_mm": round(self.gap(), 3),
                     "triangles": self.current_base().and_then(|m| m.as_ref().ok()).map(|m| m.triangle_count()),
                     "error": self.base_mesh.as_ref().and_then(|(_, m)| m.as_ref().err().cloned())},
            "legs": legs,
            "current_leg": self.current.map(|i| i + 1),
            "preview_ready": ready,
            "combining": self.job.is_some(),
            "error": self.error,
            "next": if self.legs.is_empty() { "add_leg (as clicking the object does), then accept (mesh_tool accept)" } else { "accept (mesh_tool accept) to combine the base and legs with the object" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("add_leg", "Add a leg rooted at the surface point nearest `at` (as clicking the object does); its foot below the root unless `foot` is given", json!({"at": arg::point("A point on or near the object's surface, mm"), "foot": arg::point("Optional: where the foot goes (dropped onto the base's top), mm")})).required(&["at"]),
            Action::new("move_leg", "Move leg `leg`'s root (onto the surface point nearest `root`) and/or its foot (dropped onto the base's top)", json!({"leg": arg::integer("1, 2, …"), "root": arg::point("mm"), "foot": arg::point("mm")})).required(&["leg"]),
            Action::new("size_leg", "Set leg `leg`'s diameters (any of them)", json!({"leg": arg::integer("1, 2, …"), "diameter_mm": arg::number("Column"), "root_diameter_mm": arg::number("Where it meets the object"), "foot_diameter_mm": arg::number("Where it meets the base")})).required(&["leg"]),
            Action::new("remove_leg", "Remove leg `leg`", json!({"leg": arg::integer("1, 2, …")})).required(&["leg"]),
            Action::plain("clear_legs", "Remove every leg"),
            Action::plain("fit_base", "Centre the base under the object and its legs' feet and size it to their footprint plus base.margin_mm all round"),
            Action::new("base_gap", "Move the base along its up so its top is `mm` below the object's lowest point", json!({"mm": arg::number("mm")})).required(&["mm"]),
            Action::new("level_base", "Make the base's top square to `up` (default: the world's up)", json!({"up": arg::point("Optional direction")})),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        self.job = None;
        let on_surface = |t: &Self, p: [f64; 3]| t.bvh.nearest(&t.source, DVec3::from_array(p), f64::INFINITY).map(|h| h.point).ok_or("the object has no surface");
        let r = match action {
            "add_leg" => {
                let root = on_surface(self, a.point("at")?)?;
                let i = self.add_leg(root, a.opt_point("foot").map(DVec3::from_array));
                Ok(format!("Leg {} added", i + 1))
            }
            "move_leg" => {
                let i = self.leg_index(&a)?;
                if let Some(p) = a.opt_point("root") {
                    self.legs[i].root = on_surface(self, p)?;
                }
                if let Some(p) = a.opt_point("foot") {
                    let q = self.base.local(DVec3::from_array(p));
                    self.legs[i].foot = Some(DVec2::new(q.x, q.y));
                }
                self.current = Some(i);
                Ok(format!("Leg {} moved", i + 1))
            }
            "size_leg" => {
                let i = self.leg_index(&a)?;
                Self::sizes_from(&a, &mut self.legs[i].sizes)?;
                self.current = Some(i);
                Ok(format!("Leg {} sized", i + 1))
            }
            "remove_leg" => {
                let i = self.leg_index(&a)?;
                self.remove_leg(i);
                Ok(format!("Leg {} removed", i + 1))
            }
            "clear_legs" => {
                self.legs.clear();
                self.current = None;
                Ok("Legs removed".into())
            }
            "fit_base" => {
                self.fit_base();
                Ok(format!("Base {:.1} × {:.1} mm", self.base_params.width, self.base_params.depth))
            }
            "base_gap" => {
                self.set_gap(a.f64("mm")?);
                Ok(format!("The base's top is {:.2} mm below the object", self.gap()))
            }
            "level_base" => {
                let up = a.opt_point("up").map(DVec3::from_array).unwrap_or_else(|| cx.camera.world_up());
                if up.length_squared() < 1e-24 {
                    return Err("up cannot be zero".into());
                }
                self.level(up);
                Ok("Base levelled".into())
            }
            _ => return Err(format!("Stand has no action {action}")),
        };
        self.refresh(cx.egui);
        r
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Stand");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        if self.poll(cx) {
            return ToolOutcome::Close;
        }
        let combining = self.job.is_some();
        ui.small("Drag a line under the object (seen from the side) to put the base's top there. Click the object to add a leg; drag a leg's ends to move them; right-click an end (or Delete) to remove the leg.");
        let world_up = cx.camera.world_up();
        ui.add_enabled_ui(!combining, |ui| {
            let base_held = self.base_ui(ui, world_up);
            let legs_held = self.legs_ui(ui);
            self.panel_held = base_held || legs_held;
            ui.add_space(4.0);
            egui::ComboBox::from_label("Combine mode")
                .selected_text(match self.combine {
                    Combine::Boolean => "Boolean",
                    Combine::Append => "Append",
                    Combine::NewObject => "New Object",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.combine, Combine::Boolean, "Boolean").on_hover_text("Unite the base and legs with the object (needs a closed object)");
                    ui.selectable_value(&mut self.combine, Combine::Append, "Append").on_hover_text("Add them to the object as separate shells");
                    ui.selectable_value(&mut self.combine, Combine::NewObject, "New Object").on_hover_text("The base and legs as an object of their own");
                });
        });
        self.refresh(ui.ctx());
        let building = self.legs.iter().filter(|l| l.job.is_some()).count();
        if building > 0 && !combining {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label(if building == 1 { "Building a leg…".to_string() } else { format!("Building {building} legs…") });
            });
        }
        if let Some(Err(e)) = self.current_base() {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), format!("The base: {e}"));
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        if combining {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Combining…");
            });
        }
        let blocked = self.notes().iter().any(|n| n.2);
        let (accept, cancel) = ui_util::accept_cancel(ui, !combining && !blocked);
        if accept {
            self.commit(cx);
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if self.job.is_some() {
            return;
        }
        if input.pressed {
            self.pressed_at = Some(pos);
            self.drag = self.handle_at(cx, pos).map(|(i, foot)| {
                self.current = Some(i);
                if foot {
                    Drag::Foot(i)
                } else {
                    Drag::Root(i)
                }
            });
        }
        if input.down {
            match self.drag {
                Some(Drag::Root(i)) => {
                    if let Some((_, hit)) = cx.pick(pos, self.object) {
                        self.legs[i].root = hit.point;
                    }
                }
                Some(Drag::Foot(i)) => {
                    if let Some(q) = self.on_base(cx, pos) {
                        self.legs[i].foot = Some(q);
                    }
                }
                Some(Drag::Line(a, _)) => self.drag = Some(Drag::Line(a, pos)),
                None => {
                    if let Some(a) = self.pressed_at.filter(|a| a.distance(pos) > 8.0) {
                        self.drag = Some(Drag::Line(a, pos));
                    }
                }
            }
        }
        if input.released {
            let clicked = self.pressed_at.take().is_some_and(|p| p.distance(pos) < 5.0);
            match self.drag.take() {
                Some(Drag::Line(a, b)) => self.draw_base(cx, a, b),
                Some(_) => {}
                None if clicked => {
                    if let Some((_, hit)) = cx.pick(pos, self.object) {
                        self.add_leg(hit.point, None);
                    }
                }
                None => {}
            }
        }
        self.refresh(cx.egui);
    }

    fn secondary_click(&mut self, pos: egui::Pos2, cx: &mut ToolCx) {
        if let Some((i, _)) = self.handle_at(cx, pos).filter(|_| self.job.is_none()) {
            self.remove_leg(i);
            self.refresh(cx.egui);
        }
    }

    fn keys(&mut self, input: &egui::InputState, cx: &mut ToolCx) -> bool {
        if (input.key_pressed(egui::Key::Delete) || input.key_pressed(egui::Key::Backspace)) && self.job.is_none() {
            if let Some(i) = self.current {
                self.remove_leg(i);
                self.refresh(cx.egui);
                return true;
            }
        }
        false
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        let outline: Vec<DVec3> = self.base_params.outline(self.base_params.width.min(self.base_params.depth) / 60.0).into_iter().map(|q| self.base.world(DVec3::new(q.x, q.y, 0.0))).collect();
        let base_colour = Vec4::new(1.0, 0.8, 0.3, 0.9);
        for k in 0..outline.len() {
            ov.line(outline[k], outline[(k + 1) % outline.len()], base_colour);
        }
        let notes = self.notes();
        for (i, leg) in self.legs.iter().enumerate() {
            let Some(p) = leg.plan() else { continue };
            let serious = notes.iter().any(|n| n.0 == i && n.2) || p.clearance < 0.0;
            let colour = if serious {
                Vec4::new(1.0, 0.35, 0.3, 1.0)
            } else if notes.iter().any(|n| n.0 == i) {
                Vec4::new(1.0, 0.65, 0.25, 1.0)
            } else if self.current == Some(i) {
                Vec4::new(0.45, 0.85, 1.0, 1.0)
            } else {
                Vec4::new(0.35, 0.65, 0.95, 0.9)
            };
            for ring in [p.footprint(48), p.foot_ring(32)] {
                for k in 0..ring.len() {
                    ov.line(ring[k], ring[(k + 1) % ring.len()], colour);
                }
            }
            for w in p.centreline().windows(2) {
                ov.line(w[0], w[1], colour);
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let outline = egui::Stroke::new(1.5, egui::Color32::from_gray(20));
        for i in 0..self.legs.len() {
            let fill = if self.current == Some(i) { ui_util::ACCENT } else { egui::Color32::from_rgb(90, 160, 230) };
            if let Some(p) = cx.project(self.legs[i].root) {
                painter.circle(p, 6.0, fill, outline);
            }
            if let Some(p) = cx.project(self.foot_of(i)) {
                painter.rect(egui::Rect::from_center_size(p, egui::vec2(10.0, 10.0)), 1.0, fill, outline, egui::StrokeKind::Middle);
            }
        }
        if let Some(Drag::Line(a, b)) = self.drag {
            painter.line_segment([a, b], egui::Stroke::new(2.0, ui_util::ACCENT));
        }
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        Some((self.object?, self.preview.as_ref()?))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_none() {
            self.commit(cx);
        }
        false
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Stand".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        self.legs.iter().any(|l| l.job.is_some())
    }

    fn on_close(&mut self, _cx: &mut ToolCx) {
        for leg in &mut self.legs {
            leg.job = None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// A ball of radius 10 seen from the side (the world's up in view), with the tool open
    /// on it.
    fn with_tool(test: impl FnOnce(&mut StandTool, &mut ToolCx, ObjectId)) {
        let mut scene = Scene::default();
        let id = scene.add("ball", primitives::icosphere(10.0, 4));
        scene.active = Some(id);
        let mut camera = crate::camera::Camera::default();
        let up = camera.world_up();
        let side = if up.dot(DVec3::Y).abs() > 0.5 { -DVec3::Z } else { DVec3::Y };
        camera.look_along(side);
        camera.focus(&scene.bounds().expanded(30.0));
        let (mut history, mut status, ctx) = (History::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut t = StandTool::new(&mut cx);
        test(&mut t, &mut cx, id);
    }

    /// Refresh until every leg is built (or `frames` run out).
    fn settle(t: &mut StandTool, cx: &mut ToolCx, frames: usize) {
        for _ in 0..frames {
            t.refresh(cx.egui);
            if !t.pending() {
                return;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        panic!("the legs were not built");
    }

    /// Draw the panel until it closes the tool (or `frames` run out). True if it closed.
    fn run_panel(t: &mut StandTool, cx: &mut ToolCx, frames: usize) -> bool {
        let ctx = cx.egui.clone();
        for _ in 0..frames {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 1200.0))), ..Default::default() };
            let mut closed = false;
            // The frame's texture changes are not shown anywhere: dropped on purpose.
            ctx.run_ui(raw, |ui| closed = matches!(t.panel(ui, cx), ToolOutcome::Close)).textures_delta.clear();
            if closed {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    }

    fn bottom(t: &StandTool) -> DVec3 {
        -t.base.up * 10.0
    }

    #[test]
    fn opens_with_a_base_under_the_object() {
        with_tool(|t, _cx, _| {
            // The defaults scale with the object (here at their least, a 35 mm diagonal).
            assert!((t.gap() - 5.0).abs() < 1e-6, "gap {}", t.gap());
            assert!(t.base_params.width >= 20.0 + 2.0 * t.margin - 1e-6 && t.base_params.depth >= 20.0 + 2.0 * t.margin - 1e-6);
            let c = t.base.local(DVec3::ZERO);
            assert!(c.x.abs() < 0.1 && c.y.abs() < 0.1, "centred under the ball");
            assert!(t.legs.is_empty() && t.preview.is_none());
        });
    }

    #[test]
    fn legs_build_in_the_background_and_unite_on_accept() {
        with_tool(|t, cx, id| {
            let up = t.base.up;
            let side = t.base.width_axis;
            t.act("add_leg", &serde_json::json!({"at": bottom(t).to_array()}), cx).unwrap();
            t.act("add_leg", &serde_json::json!({"at": (bottom(t) * 0.6 + side * 8.0).to_array()}), cx).unwrap();
            assert!(t.pending(), "building");
            settle(t, cx, 20_000);
            let state = t.state(cx.scene);
            assert_eq!(state["preview_ready"], true, "{state}");
            assert_eq!(state["legs"].as_array().unwrap().len(), 2);
            assert!(state["legs"][0]["notes"].as_array().unwrap().is_empty(), "{state}");
            let preview = t.preview.as_ref().unwrap();
            assert!(preview.triangle_count() > cx.scene.get(id).unwrap().mesh.triangle_count() + 1000, "base and legs in the preview");
            assert!(preview.vertex_ids().any(|v| preview.vertex(v).dot(up) < -15.0), "down to the base");

            let ball = mm_core::measure::volume(&cx.scene.get(id).unwrap().mesh);
            assert!(!t.accept(cx), "the union computes first");
            assert!(t.busy().is_some());
            assert!(run_panel(t, cx, 30_000), "the tool closes once it is done");
            assert_eq!(cx.history.labels().0, vec!["Stand"]);
            let o = cx.scene.get(id).unwrap();
            assert!(o.mesh.is_closed());
            assert_eq!(mm_core::topology::connected_components(&o.mesh).len(), 1, "one piece");
            assert!(mm_core::measure::volume(&o.mesh) > ball + t.base_params.width * t.base_params.depth * t.base_params.thickness * 0.9);
            let mut names: Vec<&str> = o.group_names.values().map(|s| s.as_str()).collect();
            names.sort();
            assert_eq!(names, vec!["base_1", "leg_1", "leg_2"]);
            assert!(cx.status.starts_with("Stand: a base and 2 legs united"), "{}", cx.status);
        });
    }

    #[test]
    fn append_and_new_object_keep_the_ball_as_it_was() {
        for combine in ["append", "new_object"] {
            with_tool(|t, cx, id| {
                let before = cx.scene.get(id).unwrap().mesh.triangle_count();
                t.set_settings(&serde_json::json!({"combine": combine}), cx).unwrap();
                t.act("add_leg", &serde_json::json!({"at": bottom(t).to_array()}), cx).unwrap();
                settle(t, cx, 20_000);
                t.accept(cx);
                assert!(run_panel(t, cx, 30_000));
                let n = cx.scene.objects.len();
                if combine == "append" {
                    assert_eq!(n, 1);
                    let m = &cx.scene.get(id).unwrap().mesh;
                    assert!(m.triangle_count() > before && m.is_closed());
                    assert_eq!(mm_core::topology::connected_components(m).len(), 2, "the ball and the stand, apart");
                } else {
                    assert_eq!(n, 2);
                    assert_eq!(cx.scene.get(id).unwrap().mesh.triangle_count(), before);
                    let stand = cx.scene.objects.iter().find(|o| o.id != id).unwrap();
                    assert_eq!(stand.name, "ball stand");
                    assert!(stand.mesh.is_closed());
                    assert!(stand.group_names.values().any(|n| n == "leg_1"));
                }
            });
        }
    }

    #[test]
    fn clicks_add_legs_drags_move_them_right_clicks_remove_them() {
        with_tool(|t, cx, _| {
            let press = |pos: egui::Pos2| PointerInput { pos: Some(pos), pressed: true, down: true, ..Default::default() };
            let hold = |pos: egui::Pos2| PointerInput { pos: Some(pos), down: true, ..Default::default() };
            let release = |pos: egui::Pos2| PointerInput { pos: Some(pos), released: true, ..Default::default() };
            // Click just above the ball's lowest point, as seen from the side.
            let low = cx.project(bottom(t) + t.base.up * 1.0).unwrap();
            t.pointer(&press(low), cx);
            t.pointer(&release(low), cx);
            assert_eq!(t.legs.len(), 1);
            let root = t.legs[0].root;
            assert!((root.length() - 10.0).abs() < 0.1 && root.dot(t.base.up) < -8.5, "on the ball's underside: {root}");
            // Drag its foot along the base.
            let foot = cx.project(t.foot_of(0)).unwrap();
            let to = foot + egui::vec2(60.0, 0.0);
            t.pointer(&press(foot), cx);
            t.pointer(&hold(to), cx);
            t.pointer(&release(to), cx);
            assert!(t.legs[0].foot.is_some(), "placed by hand");
            assert!(t.foot_of(0).distance(t.base.project(bottom(t))) > 2.0);
            assert!((t.base.local(t.foot_of(0)).z).abs() < 1e-9, "on the base's top");
            // It moves with the base.
            let before = t.base.local(t.foot_of(0));
            t.act("base_gap", &serde_json::json!({"mm": 8.0}), cx).unwrap();
            assert!(t.base.local(t.foot_of(0)).distance(before) < 1e-9);
            // Right-click its root: gone.
            let root = cx.project(t.legs[0].root).unwrap();
            t.secondary_click(root, cx);
            assert!(t.legs.is_empty());
        });
    }

    #[test]
    fn while_a_leg_or_a_number_is_dragged_only_the_outlines_follow() {
        with_tool(|t, cx, _| {
            t.act("add_leg", &serde_json::json!({"at": bottom(t).to_array()}), cx).unwrap();
            settle(t, cx, 20_000);
            let (key, base) = (t.preview_key.clone(), t.base_mesh.as_ref().map(|(k, _)| *k));
            // Drag the leg's root across the ball's underside: its plan follows, nothing builds.
            let at = cx.project(t.legs[0].root).unwrap();
            t.pointer(&PointerInput { pos: Some(at), pressed: true, down: true, ..Default::default() }, cx);
            let to = cx.project(bottom(t) * 0.7 + t.base.width_axis * 4.0).unwrap();
            t.pointer(&PointerInput { pos: Some(to), down: true, ..Default::default() }, cx);
            let root = t.legs[0].root;
            assert!(root.distance(bottom(t)) > 1.0, "moved: {root}");
            assert!(t.legs[0].plan.as_ref().is_some_and(|(s, _)| s.root == root), "the outline follows");
            assert!(!t.pending() && t.preview_key == key, "nothing built while dragging");
            // Let go: the leg is built where it is now.
            t.pointer(&PointerInput { pos: Some(to), released: true, ..Default::default() }, cx);
            assert!(t.pending(), "building");
            settle(t, cx, 20_000);
            assert!(t.legs[0].ready(&t.spec(0)) && t.preview_key != key);
            // A number dragged in the panel: the base waits for it too, and the leg with it.
            let key = t.preview_key.clone();
            t.panel_held = true;
            t.base_params.thickness += 1.0;
            t.refresh(cx.egui);
            assert!(t.current_base().is_none() && t.base_mesh.as_ref().map(|(k, _)| *k) == base, "the base waits");
            assert!(!t.pending() && t.preview_key == key);
            t.panel_held = false;
            t.refresh(cx.egui);
            assert!(t.current_base().is_some_and(|m| m.is_ok()) && t.pending());
            settle(t, cx, 20_000);
            assert_eq!(t.state(cx.scene)["preview_ready"], true);
        });
    }

    #[test]
    fn a_line_drawn_across_the_view_puts_the_base_there() {
        with_tool(|t, cx, _| {
            let up = t.base.up;
            // A level line 15 below the ball's centre, 40 wide.
            let (a, b) = (cx.project(-up * 15.0 - cx.camera.right() * 20.0).unwrap(), cx.project(-up * 15.0 + cx.camera.right() * 20.0).unwrap());
            let press = PointerInput { pos: Some(a), pressed: true, down: true, ..Default::default() };
            t.pointer(&press, cx);
            t.pointer(&PointerInput { pos: Some(b), down: true, ..Default::default() }, cx);
            t.pointer(&PointerInput { pos: Some(b), released: true, ..Default::default() }, cx);
            assert!(t.legs.is_empty(), "a drag is not a click");
            assert!(t.base.up.dot(up) > 0.999, "level, facing the ball: {}", t.base.up);
            assert!((t.gap() - 5.0).abs() < 0.05, "15 below the centre: {}", t.gap());
            assert!((t.base_params.width - 40.0).abs() < 0.05, "{}", t.base_params.width);
            let c = t.base.local(DVec3::ZERO);
            assert!(c.x.abs() < 0.05 && c.y.abs() < 0.05);
        });
    }

    #[test]
    fn a_leg_that_cannot_be_made_stops_the_accept() {
        with_tool(|t, cx, _| {
            t.act("add_leg", &serde_json::json!({"at": bottom(t).to_array()}), cx).unwrap();
            // The base cuts into the ball, above the leg's root.
            t.act("base_gap", &serde_json::json!({"mm": -3.0}), cx).unwrap();
            let notes = t.notes();
            assert!(notes.iter().any(|n| n.0 == 0 && n.2), "{notes:?}");
            assert!(!t.accept(cx));
            assert!(t.job.is_none());
            assert!(t.error.as_deref().is_some_and(|e| e.starts_with("Leg 1:")), "{:?}", t.error);
            t.act("base_gap", &serde_json::json!({"mm": 6.0}), cx).unwrap();
            assert!(t.notes().is_empty());
        });
    }

    #[test]
    fn settings_round_trip_and_set_every_leg() {
        with_tool(|t, cx, _| {
            t.act("add_leg", &serde_json::json!({"at": bottom(t).to_array()}), cx).unwrap();
            t.act("add_leg", &serde_json::json!({"at": (bottom(t) + t.base.width_axis * 3.0).to_array()}), cx).unwrap();
            t.act("size_leg", &serde_json::json!({"leg": 2, "root_diameter_mm": 5.5}), cx).unwrap();
            assert_eq!(t.legs[1].sizes.root, 5.5);
            assert_ne!(t.legs[0].sizes.root, 5.5);
            let s = t.settings().unwrap();
            t.set_settings(&s, cx).unwrap();
            assert_eq!(t.settings().unwrap(), s, "unchanged");
            assert_eq!(t.legs[1].sizes.root, 5.5, "per-leg sizes are kept while the shared ones do not change");
            t.set_settings(&serde_json::json!({"legs": {"diameter_mm": 1.2}, "base": {"shape": "ellipse"}}), cx).unwrap();
            assert!(t.legs.iter().all(|l| l.sizes.diameter == 1.2));
            assert_eq!(t.legs[1].sizes.root, 5.5);
            assert_eq!(t.base_params.shape, BaseShape::Ellipse);
            assert!(t.set_settings(&serde_json::json!({"base": {"widht_mm": 3.0}}), cx).is_err());
            assert!(t.act("remove_leg", &serde_json::json!({"leg": 3}), cx).is_err());
            t.act("remove_leg", &serde_json::json!({"leg": 1}), cx).unwrap();
            assert_eq!(t.legs.len(), 1);
        });
    }

    /// A real mesh (`POLYMIX_STAND_MESH`, local only): an open vessel surface (cut ends),
    /// thickened outward (`POLYMIX_STAND_WALL` mm, default 1.5), then legs under its lowest
    /// points (four, 10 mm apart, skipping roots the tool would not accept) onto a base below;
    /// checks the union is closed and that
    /// every triangle of the lumen's surface is still there. Prints counts and timings.
    #[test]
    #[ignore]
    fn legs_on_a_real_vessel_leave_its_lumen_alone() {
        use std::time::Instant;
        let Some(path) = std::env::var_os("POLYMIX_STAND_MESH") else { return };
        let wall: f64 = std::env::var("POLYMIX_STAND_WALL").ok().and_then(|x| x.parse().ok()).unwrap_or(1.5);
        let surface = mm_io::read_mesh(std::path::Path::new(&path)).unwrap().mesh;
        let t0 = Instant::now();
        let thick = mm_edit::thicken(&surface, &mm_edit::ThickenParams { thickness: wall, ..Default::default() }, None).unwrap();
        eprintln!("{} triangles thickened ({wall} mm) into {} in {:.1?} (rebuilt: {})", surface.triangle_count(), thick.mesh.triangle_count(), t0.elapsed(), thick.rebuilt);
        let shell = thick.mesh;
        let lumen_groups = |g: i32| g != thick.wall_group && !thick.rim_groups.contains(&g);
        let key = |m: &Mesh, t: u32| {
            let mut c: Vec<[u64; 3]> = m.triangle(t).iter().map(|&v| m.vertex(v).to_array().map(f64::to_bits)).collect();
            c.sort();
            c
        };
        let lumen: Vec<Vec<[u64; 3]>> = shell.triangle_ids().filter(|&t| lumen_groups(shell.tri_group(t))).map(|t| key(&shell, t)).collect();
        let bvh = Bvh::build(&shell);
        let b = shell.bounds();
        let size = b.extents();
        let up = DVec3::Z;
        let base = BaseFrame::new(DVec3::new(b.center().x, b.center().y, b.min.z - 15.0), up, DVec3::X);
        let base_params = BaseParams { shape: BaseShape::Rectangle, width: size.x + 20.0, depth: size.y + 20.0, thickness: 4.0, corner_radius: 8.0 };
        let p = LegParams { diameter: 2.5, root_diameter: 5.0, foot_diameter: 6.0, root_flare: 4.0, foot_flare: 3.0, bend: 0.5 };
        let mut legs = Vec::new();
        // Roots from the lowest up, until four legs can be made.
        let mut cands: Vec<DVec3> = shell.triangle_ids().filter(|&t| shell.tri_group(t) == thick.wall_group && shell.tri_normal(t).dot(-up) > 0.8).map(|t| shell.tri_centroid(t)).collect();
        cands.sort_by(|a, c| a.z.total_cmp(&c.z));
        let mut roots: Vec<DVec3> = Vec::new();
        for c in cands {
            if roots.len() < 12 && roots.iter().all(|r| r.distance(c) > 10.0) {
                roots.push(c);
            }
        }
        for (i, &r) in roots.iter().enumerate() {
            if legs.len() == 4 {
                break;
            }
            let foot = mm_edit::default_foot(&shell, &bvh, r, &base, &base_params, p.root_diameter);
            let t0 = Instant::now();
            let plan = mm_edit::plan_leg(&shell, &bvh, r, foot, &base, base_params.thickness, &p).unwrap();
            if plan.buried || plan.clearance < 0.0 {
                eprintln!("leg {} at {:.2}: buried {}, clearance {:.2}: not made (the tool would not accept it)", i + 1, plan.root, plan.buried, plan.clearance);
                continue;
            }
            let t1 = Instant::now();
            let leg = mm_edit::leg_mesh(&plan, None).unwrap();
            eprintln!("leg {} at {:.2} axis {:.3}", i + 1, plan.root, plan.axis);
            eprintln!(
                "leg {}: wall {:?}, embed {:.2}, overhang {:.2}, buried {}, clearance {:.2}; planned in {:.1?}, {} triangles in {:.1?}",
                i + 1, plan.wall.map(|w| (w * 100.0).round() / 100.0), plan.embed, plan.overhang, plan.buried, plan.clearance, t1 - t0, leg.triangle_count(), t1.elapsed()
            );
            legs.push(leg);
        }
        let t0 = Instant::now();
        assert!(legs.len() >= 2, "too few legs could be made");
        let refs: Vec<&Mesh> = legs.iter().collect();
        let stand = mm_edit::join_stand(&mm_edit::base_mesh(&base, &base_params, 3.0).unwrap(), &refs, None).unwrap();
        let all = mm_edit::mesh_boolean(&shell, &stand, mm_edit::BooleanOp::Union).unwrap().mesh;
        eprintln!("joined and united in {:.1?}: {} triangles", t0.elapsed(), all.triangle_count());
        all.check_validity().unwrap();
        assert!(all.is_closed());
        let kept: std::collections::HashSet<Vec<[u64; 3]>> = all.triangle_ids().map(|t| key(&all, t)).collect();
        let lost = lumen.iter().filter(|k| !kept.contains(*k)).count();
        eprintln!("lumen triangles: {}, changed: {lost}", lumen.len());
        assert_eq!(lost, 0);
    }

}
