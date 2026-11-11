//! Mic Pad (Edit → Mic Pad, Vessel → Mic Pad): a flat-topped pad standing off the object's
//! wall, to hold a contact microphone (with ultrasound gel between) over a stenosis of a
//! printed vessel. Click the object to put the pad's centre there: its line (the pad's
//! length) is drawn on the wall, along the vessel; drag the line's ends to turn it about the
//! centre (or set its angle in the panel), drag the centre to move it. The panel sets the
//! base's length and width (the most the pad joins the wall over), the height up to the flat
//! top and the draft (the sides leaning out, so the top is wider than the base). The pad joins
//! the wall only where it faces the pad (it turns away by no more than the contact slope): a
//! base wider than the vessel is cut back to its crown, and the sides lean out from there to
//! the top. Accept unites the pad with the object (or appends it, or makes it a new object),
//! in face group pad_N.
//!
//! The pad is planned and built as it changes (`mm_edit::pad`, milliseconds); while an end,
//! the centre or a number in the panel is dragged only the outlines follow, and the preview
//! is built when it is let go. The pad's underside follows the wall and sinks into it by
//! less than half the wall's thickness, so uniting changes only the outer surface.

use super::stand::Combine;
use super::*;
use crate::jobs::Job;
use crate::ui_util;
use mm_core::Bvh;
use mm_edit::{PadParams, PadPlan};
use std::sync::Arc;

/// Everything the pad is made from: when any of it changes it is planned and built again.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Spec {
    center: DVec3,
    angle: f64,
    params: PadParams,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Drag {
    Center,
    End,
}

/// What an accept made: the mesh and the pad's face group in it.
struct Accepted {
    mesh: Mesh,
    group: i32,
}

/// Screen distance (points) within which a handle is grabbed.
const GRAB: f32 = 10.0;

pub struct MicPadTool {
    object: Option<ObjectId>,
    source: Arc<Mesh>,
    bvh: Arc<Bvh>,
    /// The pad's centre, on the object's surface (`None` until the object is clicked).
    center: Option<DVec3>,
    /// The line's angle (degrees) about the pad's axis from the way the surface runs.
    angle: f64,
    params: PadParams,
    combine: Combine,
    plan: Option<(Spec, Result<PadPlan, String>)>,
    built: Option<(Spec, Result<Arc<Mesh>, String>)>,
    preview: Option<Mesh>,
    /// What the preview shows: the pad's mesh (by address), none for the object alone.
    preview_key: Option<Option<usize>>,
    drag: Option<Drag>,
    /// A number in the panel being dragged (see [`MicPadTool::held`]).
    panel_held: bool,
    pressed_at: Option<egui::Pos2>,
    job: Option<Job<Accepted>>,
    error: Option<String>,
}

/// `x` rounded to a tenth of a millimetre.
fn tenth(x: f64) -> f64 {
    (x * 10.0).round() / 10.0
}

/// An angle (degrees) brought into (−90, 90]: the line is the same turned half a turn.
fn half_turn(deg: f64) -> f64 {
    let a = deg.rem_euclid(180.0);
    if a > 90.0 {
        a - 180.0
    } else {
        a
    }
}

impl MicPadTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, bvh) = match cx.scene.active.and_then(|id| cx.scene.get_mut(id)) {
            Some(o) => (Some(o.id), o.mesh.clone(), o.bvh()),
            None => (None, Arc::new(Mesh::new()), Arc::new(Bvh::build(&Mesh::new()))),
        };
        let diag = source.bounds().diagonal().max(1e-6);
        let w = tenth(diag * 0.03).clamp(2.0, 10.0);
        Self {
            object,
            source,
            bvh,
            center: None,
            angle: 0.0,
            params: PadParams { length: 3.0 * w, width: w, height: w, draft_deg: 30.0, ..PadParams::default() },
            combine: Combine::Boolean,
            plan: None,
            built: None,
            preview: None,
            preview_key: None,
            drag: None,
            panel_held: false,
            pressed_at: None,
            job: None,
            error: None,
        }
    }

    fn spec(&self) -> Option<Spec> {
        Some(Spec { center: self.center?, angle: self.angle, params: self.params })
    }

    /// The pad's plan for it as it is now (none before it is placed, or when it cannot be
    /// planned).
    fn current_plan(&self) -> Option<&PadPlan> {
        let spec = self.spec()?;
        self.plan.as_ref().filter(|(s, _)| *s == spec).and_then(|(_, p)| p.as_ref().ok())
    }

    /// Whether an edit is under way, a handle dragged in the view or a number dragged in the
    /// panel: the plan (and so the outlines) follows it, while the mesh and the preview wait
    /// for it to end.
    fn held(&self) -> bool {
        self.panel_held || self.drag.is_some()
    }

    /// Bring the plan, the pad's mesh and the preview up to date (while an edit is under way
    /// ([`MicPadTool::held`]) only the plan).
    fn refresh(&mut self) {
        if self.object.is_none() {
            return;
        }
        let Some(spec) = self.spec() else {
            self.plan = None;
            self.refresh_preview();
            return;
        };
        if self.plan.as_ref().is_none_or(|(s, _)| *s != spec) {
            let plan = mm_edit::plan_pad(&self.source, &self.bvh, spec.center, spec.angle, &spec.params).map_err(|e| e.to_string());
            self.plan = Some((spec, plan));
        }
        if self.held() {
            return;
        }
        if self.built.as_ref().is_none_or(|(s, _)| *s != spec) {
            if let Some(plan) = self.current_plan().filter(|p| p.ok()) {
                let m = mm_edit::pad_mesh(plan).map(Arc::new).map_err(|e| e.to_string());
                self.built = Some((spec, m));
            }
        }
        self.refresh_preview();
    }

    /// The pad's mesh built for it as it is now, if it can be made.
    fn current_mesh(&self) -> Option<&Arc<Mesh>> {
        let spec = self.spec()?;
        self.current_plan().filter(|p| p.ok())?;
        self.built.as_ref().filter(|(s, _)| *s == spec).and_then(|(_, m)| m.as_ref().ok())
    }

    fn refresh_preview(&mut self) {
        let shown = self.current_mesh().cloned();
        let key = shown.as_ref().map(|m| Arc::as_ptr(m) as usize);
        if self.preview_key == Some(key) {
            return;
        }
        self.preview = shown.map(|m| {
            let mut prev = (*self.source).clone();
            prev.append_mesh(&m, prev.max_group_id() + 1);
            prev
        });
        self.preview_key = Some(key);
    }

    /// What to point out about the pad: (note, whether it stops the pad being made).
    fn notes(&self) -> Vec<(String, bool)> {
        let Some(spec) = self.spec() else { return Vec::new() };
        let mut out = Vec::new();
        match &self.plan {
            Some((s, Err(e))) if *s == spec => out.push((e.clone(), true)),
            Some((s, Ok(p))) if *s == spec => {
                if let Some((b, Err(e))) = &self.built {
                    if *b == spec {
                        out.push((format!("could not be built: {e}"), true));
                    }
                }
                if p.buried {
                    out.push(("another part of the object lies over it or rises into it (a branch close by): move it, turn it, or make it narrower or lower".into(), true));
                }
                if p.cuts_in {
                    out.push(("its flared sides reach into the wall beside it: lower it, lessen the draft, or turn it".into(), true));
                }
                if p.overhang > 0.05 {
                    out.push((format!("it is wider than the surface under it ({:.0}% of the base overhangs): its sides hang down past the wall there", 100.0 * p.overhang), false));
                }
            }
            _ => {}
        }
        out
    }

    /// Whether the object still has the mesh the pad was made for (otherwise the tool's work
    /// is dropped, with a message, rather than combined with another mesh).
    fn source_is_current(&self, cx: &mut ToolCx) -> bool {
        let current = self.object.and_then(|id| cx.scene.get(id)).is_some_and(|o| Arc::ptr_eq(&o.mesh, &self.source));
        if !current {
            *cx.status = "Mic Pad cancelled: the object changed while the tool was open.".into();
        }
        current
    }

    /// Start combining the pad with the object in the background (applied by
    /// [`MicPadTool::poll`]). False with an error to show when it cannot be made.
    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() || self.object.is_none() {
            return false;
        }
        if self.center.is_none() {
            self.error = Some("Click the object where the pad goes first.".into());
            return false;
        }
        let was = self.panel_held;
        self.panel_held = false;
        self.drag = None;
        self.refresh();
        self.panel_held = was;
        if let Some((note, _)) = self.notes().into_iter().find(|n| n.1) {
            self.error = Some(format!("The pad {note}"));
            return false;
        }
        let Some(pad) = self.current_mesh().cloned() else {
            self.error = Some("The pad could not be made here.".into());
            return false;
        };
        let (source, combine) = (self.source.clone(), self.combine);
        self.error = None;
        self.job = Some(Job::spawn("Mic Pad", cx.egui.clone(), move |progress| {
            if !progress(0.1) {
                anyhow::bail!("cancelled");
            }
            Ok(match combine {
                Combine::Boolean => {
                    let r = mm_edit::mesh_boolean(&source, &pad, mm_edit::BooleanOp::Union).map_err(|e| anyhow::anyhow!("{e}"))?;
                    Accepted { mesh: r.mesh, group: r.group_offset }
                }
                Combine::Append => {
                    let mut m = (*source).clone();
                    let group = m.max_group_id() + 1;
                    m.append_mesh(&pad, group);
                    Accepted { mesh: m, group }
                }
                Combine::NewObject => Accepted { mesh: (*pad).clone(), group: 0 },
            })
        }));
        true
    }

    /// Apply the accepted pad once it is done, if the object still has the mesh it was made
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
        cx.history.push("Mic Pad", cx.scene);
        let target = match self.combine {
            Combine::NewObject => {
                let name = cx.scene.get(id).map(|o| format!("{} mic pad", o.name)).unwrap_or_else(|| "mic pad".into());
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
            if o.mesh.triangle_ids().any(|t| o.mesh.tri_group(t) == a.group) {
                let name = super::vessel::next_name(&o.group_names, "pad");
                o.group_names.insert(a.group, name);
            }
        }
        let (tl, tw) = self.params.top();
        let top = format!("its top {tl:.1} × {tw:.1} mm");
        *cx.status = match self.combine {
            Combine::Boolean => format!("Mic Pad: united with the object, {top}"),
            Combine::Append => format!("Mic Pad: added to the object, {top}"),
            Combine::NewObject => format!("Mic Pad: a new object, {top}"),
        };
    }

    /// The handle under `pos`.
    fn handle_at(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<Drag> {
        let plan = self.current_plan().or_else(|| self.plan.as_ref().and_then(|(_, p)| p.as_ref().ok()))?;
        let (a, b) = plan.ends();
        let mut best: Option<(Drag, f32)> = None;
        for (what, p) in [(Drag::Center, plan.center), (Drag::End, a), (Drag::End, b)] {
            if let Some(d) = cx.project(p).map(|q| q.distance(pos)).filter(|d| *d < GRAB) {
                if best.is_none_or(|b| d < b.1) {
                    best = Some((what, d));
                }
            }
        }
        best.map(|(what, _)| what)
    }

    /// The line's angle that points it at the screen point `pos` (seen on the plane square
    /// to the pad's axis through its centre).
    fn angle_at(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<f64> {
        let plan = self.plan.as_ref().and_then(|(_, p)| p.as_ref().ok())?;
        let ray = cx.ray_at(pos);
        let denom = ray.dir.dot(plan.axis);
        let on = if denom.abs() > 1e-6 {
            let t = (plan.center - ray.origin).dot(plan.axis) / denom;
            (t > 0.0).then(|| ray.at(t))?
        } else {
            return None;
        };
        let v = on - plan.center;
        let (x, y) = (v.dot(plan.reference), v.dot(plan.axis.cross(plan.reference)));
        (x.hypot(y) > 1e-9).then(|| half_turn(y.atan2(x).to_degrees()))
    }

    /// The pad's settings; true while one of its numbers is being dragged.
    fn sizes_ui(&mut self, ui: &mut egui::Ui) -> bool {
        let mut held = false;
        fn size(v: &mut f64) -> egui::DragValue<'_> {
            egui::DragValue::new(v).speed(0.05).range(0.1..=1e4).min_decimals(1).max_decimals(2).suffix(" mm").update_while_editing(false)
        }
        let joins = self.current_plan().map(|pl| pl.contact.size());
        let p = &mut self.params;
        ui_util::section(ui, "Base");
        egui::Grid::new("mic_pad_base").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
            ui.label("Length").on_hover_text("Along the line: the most the pad joins the wall over");
            held |= ui.add(size(&mut p.length)).dragged();
            ui.end_row();
            ui.label("Width").on_hover_text("Across the line: the most the pad joins the wall over");
            held |= ui.add(size(&mut p.width)).dragged();
            ui.end_row();
            ui.label("Contact slope").on_hover_text(
                "The pad joins the wall only where the wall turns away from it by no more than this. Past a vessel's crown, down its sides, the base is cut back and the pad's sides lean out from there to the top. On a round vessel 45° joins about 70% of its width, 90° all of it",
            );
            held |= ui.add(egui::DragValue::new(&mut p.contact_deg).speed(0.5).range(1.0..=90.0).max_decimals(1).suffix("°").update_while_editing(false)).dragged();
            ui.end_row();
            if let Some((cl, cw)) = joins {
                ui.label("Joins the wall").on_hover_text("Where the pad meets the wall: the base, cut back where the wall turns away more steeply than the contact slope");
                ui.label(format!("{cl:.1} × {cw:.1} mm"));
                ui.end_row();
            }
        });
        ui_util::section(ui, "Pad");
        egui::Grid::new("mic_pad_pad").num_columns(2).spacing([10.0, 4.0]).show(ui, |ui| {
            ui.label("Height").on_hover_text("From the base up to the flat top");
            held |= ui.add(size(&mut p.height)).dragged();
            ui.end_row();
            ui.label("Draft").on_hover_text("How far each side leans out on the way up, so the top is wider than the base (negative: in)");
            let max = PadParams::MAX_DRAFT;
            held |= ui.add(egui::DragValue::new(&mut p.draft_deg).speed(0.2).range(-max..=max).max_decimals(1).suffix("°").update_while_editing(false)).dragged();
            ui.end_row();
            let (tl, tw) = p.top();
            ui.label("Top face");
            ui.label(format!("{tl:.1} × {tw:.1} mm"));
            ui.end_row();
        });
        held
    }

    /// The line's angle; true while it is being dragged.
    fn angle_ui(&mut self, ui: &mut egui::Ui) -> bool {
        ui_util::section(ui, "Line");
        let mut held = false;
        let along = self.plan.as_ref().and_then(|(_, p)| p.as_ref().ok()).is_none_or(|p| p.follows_surface);
        ui.horizontal_wrapped(|ui| {
            ui.label("Angle").on_hover_text(if along { "About the centre, from the way the vessel runs (0°: along it, 90°: across it)" } else { "About the centre, from a world axis (the surface here runs no way in particular)" });
            let r = ui.add(egui::DragValue::new(&mut self.angle).speed(0.5).range(-90.0..=90.0).max_decimals(1).suffix("°").update_while_editing(false));
            held |= r.dragged();
            if ui.small_button("⟲").on_hover_text("5° counter-clockwise, seen from above the pad").clicked() {
                self.angle = half_turn(self.angle + 5.0);
            }
            if ui.small_button("⟳").on_hover_text("5° clockwise").clicked() {
                self.angle = half_turn(self.angle - 5.0);
            }
            if ui.button(if along { "Along the vessel" } else { "0°" }).clicked() {
                self.angle = 0.0;
            }
        });
        if !along {
            ui.small("No one way the surface runs here: the angle is from a world axis.");
        }
        held
    }
}

impl Tool for MicPadTool {
    fn name(&self) -> &'static str {
        "Mic Pad"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        use polymix_workspace::agent::vec3;
        let p = &self.params;
        let combine = match self.combine {
            Combine::Boolean => "boolean",
            Combine::Append => "append",
            Combine::NewObject => "new_object",
        };
        Some(serde_json::json!({
            "center_mm": self.center.map(|c| vec3(c.to_array(), 5)),
            "angle_deg": self.angle,
            "length_mm": p.length, "width_mm": p.width, "height_mm": p.height, "draft_deg": p.draft_deg, "contact_deg": p.contact_deg,
            "combine": combine,
        }))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let mm = |about: &str| serde_json::json!({"type": "number", "exclusiveMinimum": 0, "description": about});
        let max = PadParams::MAX_DRAFT;
        Some(serde_json::json!({"type": "object", "properties": {
            "center_mm": {"type": ["array", "null"], "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "The pad's centre (moved onto the nearest point of the surface); null: not placed"},
            "angle_deg": {"type": "number", "minimum": -90, "maximum": 90, "description": "The line's angle about the pad's axis from the way the vessel runs at the centre (0 along it, 90 across); from a world axis where the surface runs no way in particular"},
            "length_mm": mm("The base along the line (the most the pad joins the wall over)"),
            "width_mm": mm("The base across the line"),
            "contact_deg": {"type": "number", "exclusiveMinimum": 0, "maximum": 90, "description": "The pad joins the wall only where it turns away from the pad by at most this (degrees); past it (a vessel's sides) the base is cut back and the pad's sides lean out from there to the top. On a round vessel 45 joins about 70% of its width, 90 all of it"},
            "height_mm": mm("From the base up to the flat top"),
            "draft_deg": {"type": "number", "minimum": -max, "maximum": max, "description": "How far each side leans out going up (the top is wider by 2·height·tan(draft) each way); negative leans in"},
            "combine": {"type": "string", "enum": ["boolean", "append", "new_object"], "description": "boolean: united with the object (needs a closed object); append: a separate shell of the object; new_object: the pad as an object of its own"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.job = None;
        let a = Args(&v);
        let positive = |k: &str| a.f64(k).and_then(|x| if x > 0.0 && x.is_finite() { Ok(x) } else { Err(format!("{k} must be positive")) });
        let draft = a.f64("draft_deg")?;
        if draft.is_nan() || draft.abs() > PadParams::MAX_DRAFT {
            return Err(format!("draft_deg must be between −{0} and {0}", PadParams::MAX_DRAFT));
        }
        let contact = a.f64("contact_deg")?;
        if !(contact > 0.0 && contact <= 90.0) {
            return Err("contact_deg must be more than 0 and at most 90".into());
        }
        self.params = PadParams { length: positive("length_mm")?, width: positive("width_mm")?, height: positive("height_mm")?, draft_deg: draft, contact_deg: contact };
        let angle = a.f64("angle_deg")?;
        if !angle.is_finite() {
            return Err("angle_deg must be a number".into());
        }
        self.angle = half_turn(angle);
        self.center = match a.opt_point("center_mm") {
            Some(p) => Some(self.bvh.nearest(&self.source, DVec3::from_array(p), f64::INFINITY).map(|h| h.point).ok_or("the object has no surface")?),
            None => None,
        };
        self.combine = match v["combine"].as_str() {
            Some("boolean") => Combine::Boolean,
            Some("append") => Combine::Append,
            Some("new_object") => Combine::NewObject,
            _ => return Err("combine must be boolean, append or new_object".into()),
        };
        self.refresh();
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        let (tl, tw) = self.params.top();
        let mut j = serde_json::json!({
            "object": self.object,
            "placed": self.center.is_some(),
            "top_mm": [round(tl, 3), round(tw, 3)],
            "notes": self.notes().into_iter().map(|n| n.0).collect::<Vec<_>>(),
            "preview_ready": self.current_mesh().is_some(),
            "combining": self.job.is_some(),
            "error": self.error,
            "next": if self.center.is_none() { "place (as clicking the object does), then accept (mesh_tool accept)" } else { "accept (mesh_tool accept) to unite the pad with the object" },
        });
        if let Some(p) = self.current_plan() {
            j["center_mm"] = vec3(p.center.to_array(), 4);
            j["axis"] = vec3(p.axis.to_array(), 5);
            j["along"] = vec3(p.along.to_array(), 5);
            j["angle_from"] = if p.follows_surface { "the way the vessel runs" } else { "a world axis" }.into();
            j["ends_mm"] = { let (a, b) = p.ends(); serde_json::json!([vec3(a.to_array(), 4), vec3(b.to_array(), 4)]) };
            j["top_center_mm"] = vec3(p.top_center().to_array(), 4);
            j["contact_mm"] = { let (cl, cw) = p.contact.size(); serde_json::json!([round(cl, 3), round(cw, 3)]) };
            j["contact_cut_back"] = p.cut_back().into();
            j["base_above_center_mm"] = round(p.lift, 3).into();
            j["set_into_wall_mm"] = round(p.embed, 3).into();
            j["wall_under_center_mm"] = p.wall.map(|w| round(w, 3)).into();
            j["overhang_fraction"] = round(p.overhang, 3).into();
            j["side_clearance_mm"] = if p.clearance.is_finite() { round(p.clearance, 3).into() } else { serde_json::Value::Null };
            j["triangles"] = self.current_mesh().map(|m| m.triangle_count()).into();
        }
        j
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("place", "Put the pad's centre at the surface point nearest `at` (as clicking the object does)", json!({"at": arg::point("A point on or near the object's surface, mm")})).required(&["at"]),
            Action::new("turn", "Turn the line about the centre by `deg` degrees (counter-clockwise seen from above the pad)", json!({"deg": arg::number("Degrees")})).required(&["deg"]),
            Action::plain("along_vessel", "Turn the line back along the way the vessel runs (angle 0)"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, _cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        self.job = None;
        let r = match action {
            "place" => {
                let p = a.point("at")?;
                self.center = Some(self.bvh.nearest(&self.source, DVec3::from_array(p), f64::INFINITY).map(|h| h.point).ok_or("the object has no surface")?);
                Ok("Pad placed".to_string())
            }
            "turn" => {
                let d = a.f64("deg")?;
                self.angle = half_turn(self.angle + d);
                Ok(format!("The line is at {:.1}°", self.angle))
            }
            "along_vessel" => {
                self.angle = 0.0;
                Ok("The line runs along the vessel".into())
            }
            _ => return Err(format!("Mic Pad has no action {action}")),
        };
        self.refresh();
        r
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Mic Pad");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        if self.poll(cx) {
            return ToolOutcome::Close;
        }
        let combining = self.job.is_some();
        ui.small(if self.center.is_none() {
            "Click the object where the pad goes: its line runs along the vessel there."
        } else {
            "Drag the line's ends to turn it about the centre; drag the centre (or click elsewhere) to move the pad."
        });
        ui.add_enabled_ui(!combining, |ui| {
            let sizes_held = self.sizes_ui(ui);
            let angle_held = self.angle_ui(ui);
            self.panel_held = sizes_held || angle_held;
            ui.add_space(4.0);
            egui::ComboBox::from_label("Combine mode")
                .selected_text(match self.combine {
                    Combine::Boolean => "Boolean",
                    Combine::Append => "Append",
                    Combine::NewObject => "New Object",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.combine, Combine::Boolean, "Boolean").on_hover_text("Unite the pad with the object (needs a closed object)");
                    ui.selectable_value(&mut self.combine, Combine::Append, "Append").on_hover_text("Add it to the object as a separate shell");
                    ui.selectable_value(&mut self.combine, Combine::NewObject, "New Object").on_hover_text("The pad as an object of its own");
                });
        });
        self.refresh();
        if let Some(p) = self.current_plan() {
            let wall = p.wall.map_or("a solid part".to_string(), |w| format!("a {w:.2} mm wall"));
            ui.small(format!("Set {:.2} mm into {wall} at the centre (never more than {:.0}% of the way through).", p.embed, 100.0 * mm_edit::stand::PLUG_SHARE));
            if p.lift > 0.05 {
                ui.small(format!("The wall rises under the pad: its base rests {:.2} mm above the centre.", p.lift));
            }
            if p.cut_back() {
                let (cl, cw) = p.contact.size();
                ui.small(format!("The wall turns away past {cl:.1} × {cw:.1} mm: the pad joins it there, and its sides lean out from there to the top."));
            }
        }
        for (note, serious) in self.notes() {
            let colour = if serious { egui::Color32::from_rgb(240, 110, 90) } else { egui::Color32::from_rgb(230, 170, 70) };
            ui.colored_label(colour, format!("The pad {note}"));
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
        let blocked = self.center.is_none() || self.notes().iter().any(|n| n.1);
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
            self.drag = self.handle_at(cx, pos);
        }
        if input.down {
            match self.drag {
                Some(Drag::Center) => {
                    if let Some((_, hit)) = cx.pick(pos, self.object) {
                        self.center = Some(hit.point);
                    }
                }
                Some(Drag::End) => {
                    if let Some(a) = self.angle_at(cx, pos) {
                        self.angle = a;
                    }
                }
                None => {}
            }
        }
        if input.released {
            let clicked = self.pressed_at.take().is_some_and(|p| p.distance(pos) < 5.0);
            if self.drag.take().is_none() && clicked {
                if let Some((_, hit)) = cx.pick(pos, self.object) {
                    self.center = Some(hit.point);
                }
            }
        }
        self.refresh();
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        let Some(spec) = self.spec() else { return };
        let Some((s, Ok(p))) = &self.plan else { return };
        let stale = *s != spec;
        let serious = self.notes().iter().any(|n| n.1);
        let colour = if serious {
            Vec4::new(1.0, 0.35, 0.3, 1.0)
        } else if !self.notes().is_empty() {
            Vec4::new(1.0, 0.65, 0.25, 1.0)
        } else {
            Vec4::new(0.35, 0.65, 0.95, if stale { 0.5 } else { 0.9 })
        };
        let line_colour = if serious { colour } else { Vec4::new(1.0, 0.85, 0.2, 1.0) };
        let loop_lines = |ov: &mut Overlay, pts: &[DVec3], c: Vec4| {
            for k in 0..pts.len() {
                ov.line(pts[k], pts[(k + 1) % pts.len()], c);
            }
        };
        loop_lines(ov, &p.footprint(), colour);
        let (base, top) = p.corners();
        loop_lines(ov, &base, colour);
        loop_lines(ov, &top, colour);
        for k in 0..4 {
            ov.line(base[k], top[k], colour);
        }
        for w in p.line().windows(2) {
            ov.line(w[0], w[1], line_colour);
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let Some(p) = self.plan.as_ref().and_then(|(_, p)| p.as_ref().ok()) else { return };
        let outline = egui::Stroke::new(1.5, egui::Color32::from_gray(20));
        let fill = |d: Drag| if self.drag == Some(d) { ui_util::ACCENT } else { egui::Color32::from_rgb(90, 160, 230) };
        if let Some(c) = cx.project(p.center) {
            painter.circle(c, 6.0, fill(Drag::Center), outline);
        }
        let (a, b) = p.ends();
        for e in [a, b] {
            if let Some(q) = cx.project(e) {
                painter.rect(egui::Rect::from_center_size(q, egui::vec2(10.0, 10.0)), 1.0, fill(Drag::End), outline, egui::StrokeKind::Middle);
            }
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
        self.job.as_ref().map(|j| ("Mic Pad".to_string(), j.progress()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hollow vessel (an open tube along x of radius 3, thickened outward to 4.5) seen
    /// from above (+z), with the tool open on it.
    fn with_tool(test: impl FnOnce(&mut MicPadTool, &mut ToolCx, ObjectId)) {
        let points: Vec<DVec3> = (0..=120).map(|k| DVec3::new(-15.0 + 0.25 * k as f64, 0.0, 0.0)).collect();
        let tube = mm_core::primitives::tube_along(&points, &vec![3.0; points.len()], 48);
        let shell = mm_edit::thicken(&tube, &mm_edit::ThickenParams { thickness: 1.5, ..Default::default() }, None).unwrap().mesh;
        let mut scene = Scene::default();
        let id = scene.add("vessel", shell);
        scene.active = Some(id);
        let mut camera = crate::camera::Camera::default();
        camera.look_along(-DVec3::Z);
        camera.focus(&scene.bounds().expanded(5.0));
        let (mut history, mut status, ctx) = (History::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut t = MicPadTool::new(&mut cx);
        t.set_settings(&serde_json::json!({"length_mm": 8.0, "width_mm": 4.0, "height_mm": 3.0, "draft_deg": 30.0}), &mut cx).unwrap();
        test(&mut t, &mut cx, id);
    }

    /// Draw the panel until it closes the tool (or `frames` run out). True if it closed.
    fn run_panel(t: &mut MicPadTool, cx: &mut ToolCx, frames: usize) -> bool {
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

    fn press(pos: egui::Pos2) -> PointerInput {
        PointerInput { pos: Some(pos), pressed: true, down: true, ..Default::default() }
    }
    fn hold(pos: egui::Pos2) -> PointerInput {
        PointerInput { pos: Some(pos), down: true, ..Default::default() }
    }
    fn release(pos: egui::Pos2) -> PointerInput {
        PointerInput { pos: Some(pos), released: true, ..Default::default() }
    }

    #[test]
    fn a_click_places_the_pad_along_the_vessel_and_its_ends_turn_it() {
        with_tool(|t, cx, _| {
            assert!(t.preview.is_none() && t.center.is_none());
            let at = cx.project(DVec3::new(2.0, 0.0, 4.5)).unwrap();
            t.pointer(&press(at), cx);
            t.pointer(&release(at), cx);
            let c = t.center.expect("placed");
            assert!(c.distance(DVec3::new(2.0, 0.0, 4.5)) < 0.1, "on the vessel's top: {c}");
            let plan = t.current_plan().unwrap().clone();
            assert!(plan.follows_surface && plan.along.dot(DVec3::X).abs() > 0.99, "along the vessel: {}", plan.along);
            assert!(t.current_mesh().is_some() && t.preview.is_some(), "built and shown");
            // Drag an end round to the side: the line turns about the centre, nothing is built
            // until it is let go.
            let key = t.preview_key;
            let (_, end) = plan.ends();
            let from = cx.project(end).unwrap();
            t.pointer(&press(from), cx);
            assert_eq!(t.drag, Some(Drag::End));
            let to = cx.project(c + DVec3::new(0.0, 4.0, 0.0)).unwrap();
            t.pointer(&hold(to), cx);
            assert!((t.angle.abs() - 90.0).abs() < 2.0, "across the vessel: {}°", t.angle);
            assert!(t.preview_key == key, "nothing built while dragging");
            t.pointer(&release(to), cx);
            let plan = t.current_plan().unwrap();
            assert!(plan.along.dot(DVec3::Y).abs() > 0.99, "{}", plan.along);
            assert!(t.preview_key != key, "built when let go");
            // Drag the centre along the vessel.
            let from = cx.project(c).unwrap();
            t.pointer(&press(from), cx);
            assert_eq!(t.drag, Some(Drag::Center));
            let to = cx.project(DVec3::new(-5.0, 0.0, 4.5)).unwrap();
            t.pointer(&hold(to), cx);
            t.pointer(&release(to), cx);
            assert!(t.center.unwrap().distance(DVec3::new(-5.0, 0.0, 4.5)) < 0.1, "{}", t.center.unwrap());
            // A click elsewhere moves it there.
            let at = cx.project(DVec3::new(6.0, 1.0, 4.4)).unwrap();
            t.pointer(&press(at), cx);
            t.pointer(&release(at), cx);
            assert!(t.center.unwrap().x > 5.0);
        });
    }

    #[test]
    fn accept_unites_the_pad_with_the_vessel() {
        with_tool(|t, cx, id| {
            assert!(!t.accept(cx) && t.error.is_some(), "nothing to accept before the pad is placed");
            t.act("place", &serde_json::json!({"at": [0.0, 0.0, 6.0]}), cx).unwrap();
            let state = t.state(cx.scene);
            assert_eq!(state["preview_ready"], true, "{state}");
            assert!(state["notes"].as_array().unwrap().is_empty(), "{state}");
            let before = mm_core::measure::volume(&cx.scene.get(id).unwrap().mesh);
            assert!(!t.accept(cx), "the union computes first");
            assert!(t.busy().is_some());
            assert!(run_panel(t, cx, 30_000), "the tool closes once it is done");
            assert_eq!(cx.history.labels().0, vec!["Mic Pad"]);
            let o = cx.scene.get(id).unwrap();
            assert!(o.mesh.is_closed());
            assert_eq!(mm_core::topology::connected_components(&o.mesh).len(), 1, "one piece");
            assert!(mm_core::measure::volume(&o.mesh) > before + 8.0 * 4.0 * 3.0);
            assert!(o.group_names.values().any(|n| n == "pad_1"), "{:?}", o.group_names);
            assert!(cx.status.starts_with("Mic Pad: united"), "{}", cx.status);
        });
    }

    #[test]
    fn a_square_pad_joins_only_the_vessels_crown() {
        with_tool(|t, cx, _| {
            t.set_settings(&serde_json::json!({"length_mm": 12.0, "width_mm": 12.0}), cx).unwrap();
            t.act("place", &serde_json::json!({"at": [0.0, 0.0, 6.0]}), cx).unwrap();
            let state = t.state(cx.scene);
            assert_eq!(state["preview_ready"], true, "{state}");
            assert_eq!(state["contact_cut_back"], true, "{state}");
            let joins = state["contact_mm"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect::<Vec<_>>();
            assert!((joins[0] - 12.0).abs() < 1e-6 && joins[1] < 9.0 * 0.75, "{state}");
            assert_eq!(state["top_mm"][0], state["top_mm"][1], "a square top: {state}");
            // All the way round to its sides at 90°.
            t.set_settings(&serde_json::json!({"contact_deg": 90.0}), cx).unwrap();
            let state = t.state(cx.scene);
            assert!(state["contact_mm"][1].as_f64().unwrap() > 8.0, "{state}");
            assert!(t.set_settings(&serde_json::json!({"contact_deg": 0.0}), cx).is_err());
        });
    }

    #[test]
    fn settings_round_trip_and_turns_wrap() {
        with_tool(|t, cx, _| {
            t.act("place", &serde_json::json!({"at": [0.0, 0.0, 6.0]}), cx).unwrap();
            t.act("turn", &serde_json::json!({"deg": 100.0}), cx).unwrap();
            assert!((t.angle + 80.0).abs() < 1e-9, "100° is the same line as −80°: {}", t.angle);
            let s = t.settings().unwrap();
            t.set_settings(&s, cx).unwrap();
            assert_eq!(t.settings().unwrap(), s, "unchanged");
            assert!(t.set_settings(&serde_json::json!({"draft_deg": 80.0}), cx).is_err());
            assert!(t.set_settings(&serde_json::json!({"widht_mm": 3.0}), cx).is_err());
            t.set_settings(&serde_json::json!({"center_mm": null}), cx).unwrap();
            assert!(t.center.is_none() && t.preview.is_none());
            t.act("along_vessel", &serde_json::json!({}), cx).unwrap();
            assert_eq!(t.angle, 0.0);
        });
    }
}
