//! Add Tube (Edit > Add Tube): click two points on the object for the tube's ends (drag an
//! end to move it); the tube is combined with the object by a boolean (outside tubes are
//! added, inside tubes cut a channel), appended, or made a new object.

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use mm_edit::{TubeEnd, TubeParams, TubePath};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TubeType {
    LineOutside,
    LineInside,
    SplineOutside,
    SplineInside,
    AutoRouteMinimal,
    AutoRouteSmooth,
}

impl TubeType {
    const ALL: [TubeType; 6] =
        [TubeType::LineOutside, TubeType::LineInside, TubeType::SplineOutside, TubeType::SplineInside, TubeType::AutoRouteMinimal, TubeType::AutoRouteSmooth];
    fn label(self) -> &'static str {
        match self {
            TubeType::LineOutside => "Line (Outside)",
            TubeType::LineInside => "Line (Inside)",
            TubeType::SplineOutside => "Spline (Outside)",
            TubeType::SplineInside => "Spline (Inside)",
            TubeType::AutoRouteMinimal => "AutoRoute Minimal",
            TubeType::AutoRouteSmooth => "AutoRoute Smooth",
        }
    }
    fn path(self) -> TubePath {
        match self {
            TubeType::LineOutside | TubeType::LineInside => TubePath::Line,
            TubeType::SplineOutside | TubeType::SplineInside => TubePath::Spline,
            TubeType::AutoRouteMinimal => TubePath::AutoRoute { minimal: true },
            TubeType::AutoRouteSmooth => TubePath::AutoRoute { minimal: false },
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combine {
    /// Union for outside tubes, difference for inside ones.
    Boolean,
    Append,
    NewObject,
}

pub struct AddTubeTool {
    object: Option<ObjectId>,
    source: Arc<Mesh>,
    bvh: Arc<mm_core::Bvh>,
    /// The source's bounding-box diagonal (sizes the sliders).
    diag: f64,
    ends: Vec<TubeEnd>,
    kind: TubeType,
    combine: Combine,
    along_normal: bool,
    params: TubeParams,
    tube: Option<Mesh>,
    preview: Option<Mesh>,
    drag: Option<usize>,
    pressed_at: Option<egui::Pos2>,
    dirty: bool,
    error: Option<String>,
    /// The boolean of an accept, computing: the tool applies it and closes once it is done.
    job: Option<Job<Mesh>>,
}

impl AddTubeTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, bvh) = match cx.scene.active.and_then(|id| cx.scene.get_mut(id)) {
            Some(o) => (Some(o.id), o.mesh.clone(), o.bvh()),
            None => (None, Arc::new(Mesh::new()), Arc::new(mm_core::Bvh::build(&Mesh::new()))),
        };
        let diag = source.bounds().diagonal();
        let r = (diag * 0.015).max(1e-6);
        let params = TubeParams { path: TubePath::Spline, inside: true, radius: r, collision_radius: r * 4.0 / 3.0, ..Default::default() };
        Self {
            object,
            source,
            bvh,
            diag,
            ends: Vec::new(),
            // Meshmixer's defaults.
            kind: TubeType::SplineInside,
            combine: Combine::Boolean,
            along_normal: false,
            params,
            tube: None,
            preview: None,
            drag: None,
            pressed_at: None,
            dirty: false,
            error: None,
            job: None,
        }
    }

    /// The tube runs inside the object (and a boolean cuts it out).
    fn inside(&self) -> bool {
        match self.kind {
            TubeType::LineInside | TubeType::SplineInside => true,
            TubeType::AutoRouteMinimal | TubeType::AutoRouteSmooth => self.params.inside,
            _ => false,
        }
    }

    fn rebuild(&mut self) {
        self.dirty = false;
        self.error = None;
        if self.ends.len() < 2 {
            self.tube = None;
            self.preview = None;
            return;
        }
        let mut p = self.params;
        p.path = self.kind.path();
        p.inside = self.inside();
        p.along_normal = self.along_normal && p.path == TubePath::Line;
        let (_, tube) = mm_edit::add_tube(&self.source, &self.bvh, self.ends[0], self.ends[1], &p);
        let mut prev = (*self.source).clone();
        prev.append_mesh(&tube, prev.max_group_id() + 1);
        self.tube = Some(tube);
        self.preview = Some(prev);
    }

    fn end_at(&self, cx: &ToolCx, p: egui::Pos2) -> Option<usize> {
        self.ends.iter().position(|e| cx.project(e.point).is_some_and(|q| q.distance(p) < 10.0))
    }

    fn pick(&self, cx: &mut ToolCx, p: egui::Pos2) -> Option<TubeEnd> {
        let (_, hit) = cx.pick(p, self.object)?;
        Some(TubeEnd { point: hit.point, normal: self.source.tri_normal(hit.tid) })
    }

    /// Whether the object still has the mesh the tube was made for (otherwise the tube is
    /// dropped, with a message, rather than combined with another mesh).
    fn source_is_current(&self, cx: &mut ToolCx) -> bool {
        let current = self.object.and_then(|id| cx.scene.get(id)).is_some_and(|o| Arc::ptr_eq(&o.mesh, &self.source));
        if !current {
            *cx.status = "Add Tube cancelled: the object changed while the tool was open.".into();
        }
        current
    }

    /// Combine the tube with the object. True when the tool is done; false with an error to
    /// show, or while the boolean computes in the background (applied when done, see
    /// [`AddTubeTool::poll`]).
    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_some() {
            return false;
        }
        let (Some(id), Some(tube)) = (self.object, self.tube.as_ref()) else { return true };
        if !self.source_is_current(cx) {
            return true;
        }
        let result = match self.combine {
            Combine::NewObject => {
                cx.history.push("Add Tube", cx.scene);
                let name = cx.scene.get(id).map(|o| format!("{} tube", o.name)).unwrap_or_else(|| "tube".into());
                cx.scene.add(name, tube.clone());
                *cx.status = "Add Tube: new object".into();
                return true;
            }
            // The preview is the source with the tube appended, just what Append makes.
            Combine::Append => self.preview.take().unwrap_or_else(|| {
                let mut m = (*self.source).clone();
                m.append_mesh(tube, m.max_group_id() + 1);
                m
            }),
            Combine::Boolean => {
                let op = if self.inside() { mm_edit::BooleanOp::Difference } else { mm_edit::BooleanOp::Union };
                let (source, tube) = (self.source.clone(), tube.clone());
                self.error = None;
                self.job = Some(Job::spawn("Add Tube", cx.egui.clone(), move |_| mm_edit::mesh_boolean(&source, &tube, op).map(|r| r.mesh).map_err(|e| anyhow::anyhow!("{e}"))));
                return false;
            }
        };
        self.apply(cx, id, result);
        true
    }

    fn apply(&mut self, cx: &mut ToolCx, id: ObjectId, result: Mesh) {
        cx.history.push("Add Tube", cx.scene);
        if let Some(o) = cx.scene.get_mut(id) {
            o.set_mesh(result);
        }
        *cx.status = "Add Tube applied".into();
    }

    /// Apply the accepted boolean once it is done, if the object still has the mesh it was
    /// computed from. True when the tool is done.
    fn poll(&mut self, cx: &mut ToolCx) -> bool {
        let Some(r) = self.job.as_ref().and_then(|j| j.poll()) else { return false };
        self.job = None;
        match r {
            Ok(m) => {
                if let Some(id) = self.object.filter(|_| self.source_is_current(cx)) {
                    self.apply(cx, id, m);
                }
                true
            }
            Err(e) => {
                self.error = Some(format!("{e} — use Combine Mode: Append, or make the object solid first"));
                false
            }
        }
    }
}

impl Tool for AddTubeTool {
    fn name(&self) -> &'static str {
        "Add Tube"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        let key = |k: TubeType| k.label().to_lowercase().replace(['(', ')'], "").replace(' ', "_");
        let combine = match self.combine {
            Combine::Boolean => "boolean",
            Combine::Append => "append",
            Combine::NewObject => "new_object",
        };
        let p = &self.params;
        Some(serde_json::json!({
            "type": key(self.kind), "combine": combine, "along_normal": self.along_normal,
            "radius_mm": p.radius, "end_radius_mm": p.end_radius, "tangent": p.tangent,
            "collision_radius_mm": p.collision_radius, "length_scale": p.length_scale, "solve_steps": p.solve_steps, "route_inside": p.inside,
        }))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let types: Vec<String> = TubeType::ALL.iter().map(|k| k.label().to_lowercase().replace(['(', ')'], "").replace(' ', "_")).collect();
        Some(serde_json::json!({"type": "object", "properties": {
            "type": {"type": "string", "enum": types, "description": "Path: straight line or spline, outside the surface (handle, bypass) or inside (channel); autoroute finds a path around the object"},
            "combine": {"type": "string", "enum": ["boolean", "append", "new_object"], "description": "boolean: outside tubes are added, inside tubes cut a channel (needs a closed object); append: a separate shell of the object; new_object"},
            "along_normal": {"type": "boolean", "description": "Line tubes: run along the first end's normal instead of toward the second end"},
            "radius_mm": {"type": "number"},
            "end_radius_mm": {"type": ["number", "null"], "description": "Radius at the second end; null for the same"},
            "tangent": {"type": "number", "description": "Spline tangent length as a fraction of the distance between the ends (0..1.5)"},
            "collision_radius_mm": {"type": "number", "description": "Autoroute: clearance kept from the surface"},
            "length_scale": {"type": "number", "description": "Autoroute: 0..1, how strongly the route is pulled shorter"},
            "solve_steps": {"type": "integer", "description": "Autoroute: 1..500"},
            "route_inside": {"type": "boolean", "description": "Autoroute: inside the object"},
        }}))
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Add Tube".to_string(), j.progress()))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        // A boolean being computed is for the tube as it was: it is dropped.
        self.job = None;
        let key = |k: TubeType| k.label().to_lowercase().replace(['(', ')'], "").replace(' ', "_");
        self.kind = TubeType::ALL.into_iter().find(|k| v["type"] == key(*k).as_str()).ok_or("unknown tube type")?;
        self.combine = match v["combine"].as_str() {
            Some("boolean") => Combine::Boolean,
            Some("append") => Combine::Append,
            Some("new_object") => Combine::NewObject,
            _ => return Err("combine must be boolean, append or new_object".into()),
        };
        let num = |k: &str| v[k].as_f64().ok_or_else(|| format!("{k} must be a number"));
        self.along_normal = v["along_normal"].as_bool().unwrap_or(false);
        let p = &mut self.params;
        p.radius = num("radius_mm")?.max(1e-6);
        p.end_radius = v["end_radius_mm"].as_f64().filter(|r| *r > 0.0);
        p.tangent = num("tangent")?.clamp(0.0, 1.5);
        p.collision_radius = num("collision_radius_mm")?.max(0.0);
        p.length_scale = num("length_scale")?.clamp(0.0, 1.0);
        p.solve_steps = (v["solve_steps"].as_u64().ok_or("solve_steps must be a whole number")? as usize).clamp(1, 500);
        p.inside = v["route_inside"].as_bool().unwrap_or(false);
        self.rebuild();
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::vec3;
        serde_json::json!({
            "object": self.object,
            "ends": self.ends.iter().enumerate().map(|(i, e)| serde_json::json!({"end": i + 1, "at_mm": vec3(e.point.to_array(), 4), "normal": vec3(e.normal.to_array(), 4)})).collect::<Vec<_>>(),
            "tube_triangles": self.tube.as_ref().map(|t| t.triangle_count()),
            "error": self.error,
            "next": if self.ends.len() < 2 { "set_end twice" } else { "accept (mesh_tool accept) to combine the tube with the object" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("set_end", "Put a tube end at the surface point nearest `at` (as clicking there does): the first missing end, or end `end`", json!({"at": arg::point("A point on or near the surface, mm"), "end": arg::integer("1 or 2")})).required(&["at"]),
            Action::plain("clear", "Remove both ends"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, _cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        self.job = None;
        match action {
            "set_end" => {
                let h = self.bvh.nearest(&self.source, DVec3::from_array(a.point("at")?), f64::INFINITY).ok_or("the object has no surface")?;
                let end = TubeEnd { point: h.point, normal: self.source.tri_normal(h.tid) };
                match a.opt_usize("end") {
                    Some(k @ 1..=2) if k - 1 < self.ends.len() => self.ends[k - 1] = end,
                    Some(k @ 1..=2) if k - 1 == self.ends.len() => self.ends.push(end),
                    Some(k) => return Err(format!("cannot set end {k} now (the tube has {} end(s))", self.ends.len())),
                    None if self.ends.len() < 2 => self.ends.push(end),
                    None => return Err("both ends are set: give `end` to move one".into()),
                }
                self.rebuild();
                Ok(format!("{} end(s) set{}", self.ends.len(), if self.ends.len() == 2 { "; tube previewed" } else { "" }))
            }
            "clear" => {
                self.ends.clear();
                self.rebuild();
                Ok("Ends cleared".into())
            }
            _ => Err(format!("Add Tube has no action {action}")),
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Add Tube");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        if self.poll(cx) {
            return ToolOutcome::Close;
        }
        // The combine mode only matters when the tube is applied: another one only clears the
        // last one's error.
        let before = (self.kind, self.along_normal, self.params);
        let combine = self.combine;
        // While the accepted boolean computes, the tube stays as it is.
        let combining = self.job.is_some();
        ui.add_enabled_ui(!combining, |ui| {
            egui::ComboBox::from_label("Tube type").selected_text(self.kind.label()).show_ui(ui, |ui| {
                for k in TubeType::ALL {
                    ui.selectable_value(&mut self.kind, k, k.label());
                }
            });
            egui::ComboBox::from_label("Combine mode")
                .selected_text(match self.combine {
                    Combine::Boolean => "Boolean",
                    Combine::Append => "Append",
                    Combine::NewObject => "New Object",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.combine, Combine::Boolean, "Boolean").on_hover_text("Outside tubes are added, inside tubes cut a channel (needs a closed object)");
                    ui.selectable_value(&mut self.combine, Combine::Append, "Append").on_hover_text("Add the tube as a separate shell of this object");
                    ui.selectable_value(&mut self.combine, Combine::NewObject, "New Object");
                });
            let line = self.kind.path() == TubePath::Line;
            ui.add_enabled_ui(line, |ui| {
                ui.horizontal(|ui| {
                    ui.label("Direction");
                    ui.selectable_value(&mut self.along_normal, false, "Any Direction");
                    ui.selectable_value(&mut self.along_normal, true, "Face Normal");
                });
            });
            let span = (self.diag * 0.1).max(self.params.radius * 2.0);
            let p = &mut self.params;
            ui.add(egui::Slider::new(&mut p.radius, span * 0.002..=span).logarithmic(true).text("radius"));
            let route = matches!(self.kind, TubeType::AutoRouteMinimal | TubeType::AutoRouteSmooth);
            ui.add_enabled(!line, egui::Slider::new(&mut p.tangent, 0.0..=1.5).text("tangent len"));
            ui.add_enabled(route, egui::Slider::new(&mut p.collision_radius, 0.0..=span).text("collision radius"));
            ui.add_enabled(route, egui::Slider::new(&mut p.length_scale, 0.0..=1.0).text("length scale"));
            ui.add_enabled(route, egui::Slider::new(&mut p.solve_steps, 1..=500).text("solve steps"));
            if route {
                ui.checkbox(&mut p.inside, "Route inside");
            }
            let mut end = p.end_radius.unwrap_or(0.0);
            if ui.add(egui::Slider::new(&mut end, 0.0..=span).text("end radius")).on_hover_text("0 = same as the start").changed() {
                p.end_radius = (end > 0.0).then_some(end);
            }
            ui.small(match self.ends.len() {
                0 => "Click the object to place the start of the tube.",
                1 => "Click the object to place the end of the tube.",
                _ => "Drag an end to move it.",
            });
            if ui.add_enabled(!self.ends.is_empty(), egui::Button::new("Clear ends")).clicked() {
                self.ends.clear();
                self.dirty = true;
            }
        });
        if (self.kind, self.along_normal, self.params) != before {
            self.dirty = true;
        }
        if self.combine != combine {
            self.error = None;
        }
        if self.dirty {
            self.rebuild();
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
        let (accept, cancel) = ui_util::accept_cancel(ui, self.tube.is_some() && !combining);
        if accept && self.commit(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.pressed_at = Some(pos);
            self.drag = self.end_at(cx, pos);
        }
        if input.down {
            if let Some(i) = self.drag {
                // Holding an end still picks the same point again: nothing to rebuild.
                if let Some(e) = self.pick(cx, pos).filter(|e| *e != self.ends[i]) {
                    self.ends[i] = e;
                    self.rebuild();
                }
            }
        }
        if input.released {
            let clicked = self.pressed_at.take().is_some_and(|p| p.distance(pos) < 5.0);
            if self.drag.take().is_none() && clicked && self.ends.len() < 2 {
                if let Some(e) = self.pick(cx, pos) {
                    self.ends.push(e);
                    self.rebuild();
                }
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        for e in &self.ends {
            if let Some(p) = cx.project(e.point) {
                painter.circle(p, 7.0, egui::Color32::from_rgb(220, 60, 50), egui::Stroke::new(1.5, egui::Color32::from_gray(20)));
            }
        }
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        Some((self.object?, self.preview.as_ref()?))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.commit(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn tool(kind: TubeType, combine: Combine) -> AddTubeTool {
        let s = Arc::new(primitives::icosphere(2.0, 3));
        let bvh = Arc::new(mm_core::Bvh::build(&s));
        let n = s.compute_vertex_normals();
        let ex = s.vertex_ids().max_by(|&a, &b| s.vertex(a).x.total_cmp(&s.vertex(b).x)).unwrap();
        let ey = s.vertex_ids().max_by(|&a, &b| s.vertex(a).y.total_cmp(&s.vertex(b).y)).unwrap();
        let mut t = AddTubeTool {
            object: Some(1),
            source: s.clone(),
            bvh,
            diag: s.bounds().diagonal(),
            ends: vec![TubeEnd { point: s.vertex(ex), normal: n[ex as usize] }, TubeEnd { point: s.vertex(ey), normal: n[ey as usize] }],
            kind,
            combine,
            along_normal: false,
            params: TubeParams { radius: 0.2, ..Default::default() },
            tube: None,
            preview: None,
            drag: None,
            pressed_at: None,
            dirty: true,
            error: None,
            job: None,
        };
        t.rebuild();
        t
    }

    /// A sphere of radius 2 in a scene seen from above, with the tool open on it and both
    /// ends set (on +x and on +y).
    fn with_tool(combine: Combine, test: impl FnOnce(&mut AddTubeTool, &mut ToolCx, ObjectId)) {
        let mut scene = Scene::default();
        let id = scene.add("ball", primitives::icosphere(2.0, 3));
        scene.active = Some(id);
        let mut camera = crate::camera::Camera::default();
        camera.look_along(-DVec3::Z);
        camera.focus(&scene.bounds());
        let (mut history, mut status, ctx) = (History::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut t = AddTubeTool::new(&mut cx);
        t.combine = combine;
        t.params.radius = 0.2;
        t.act("set_end", &serde_json::json!({"at": [1.5, 0.0, 1.2]}), &mut cx).unwrap();
        t.act("set_end", &serde_json::json!({"at": [0.0, 1.5, 1.2]}), &mut cx).unwrap();
        test(&mut t, &mut cx, id);
    }

    /// Draw the panel until it closes the tool (or `frames` run out). True if it closed.
    fn run_panel(t: &mut AddTubeTool, cx: &mut ToolCx, frames: usize) -> bool {
        let ctx = cx.egui.clone();
        for _ in 0..frames {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
            let mut closed = false;
            let _ = ctx.run_ui(raw, |ui| closed = matches!(t.panel(ui, cx), ToolOutcome::Close));
            if closed {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    }

    #[test]
    fn a_boolean_accept_is_applied_in_the_background() {
        with_tool(Combine::Boolean, |t, cx, id| {
            let source = cx.scene.get(id).unwrap().mesh.clone();
            let want = mm_edit::mesh_boolean(&source, t.tube.as_ref().unwrap(), mm_edit::BooleanOp::Difference).unwrap().mesh;
            assert!(!t.accept(cx), "the boolean computes first");
            assert!(t.busy().is_some());
            assert!(!t.accept(cx), "accepting again waits too");
            assert!(cx.history.labels().0.is_empty());
            assert!(run_panel(t, cx, 30_000), "the tool closes once the boolean is done");
            assert_eq!(cx.history.labels().0, vec!["Add Tube"]);
            assert_eq!(cx.scene.get(id).unwrap().mesh.to_arrays(), want.to_arrays());
            assert_eq!(*cx.status, "Add Tube applied");
        });
        // The object changed meanwhile: nothing is applied.
        with_tool(Combine::Boolean, |t, cx, id| {
            assert!(!t.accept(cx));
            cx.scene.get_mut(id).unwrap().set_mesh(primitives::cube(1.0, 2));
            let replaced = cx.scene.get(id).unwrap().mesh.clone();
            assert!(run_panel(t, cx, 30_000));
            assert!(cx.history.labels().0.is_empty());
            assert!(Arc::ptr_eq(&cx.scene.get(id).unwrap().mesh, &replaced));
            assert!(cx.status.starts_with("Add Tube cancelled"), "{}", cx.status);
        });
    }

    #[test]
    fn append_applies_the_preview() {
        with_tool(Combine::Append, |t, cx, id| {
            let mut want = (*t.source).clone();
            want.append_mesh(t.tube.as_ref().unwrap(), want.max_group_id() + 1);
            assert!(t.accept(cx), "appending is immediate");
            assert_eq!(cx.history.labels().0, vec!["Add Tube"]);
            assert_eq!(cx.scene.get(id).unwrap().mesh.to_arrays(), want.to_arrays());
        });
    }

    #[test]
    fn holding_an_end_still_does_not_rebuild_the_tube() {
        with_tool(Combine::Append, |t, cx, _| {
            let version = |t: &AddTubeTool| t.preview.as_ref().unwrap().shape_version();
            let p = cx.project(t.ends[0].point).unwrap();
            let at = |pos: egui::Pos2, pressed: bool| PointerInput { pos: Some(pos), pressed, down: true, ..Default::default() };
            t.pointer(&at(p, true), cx);
            assert_eq!(t.drag, Some(0));
            t.pointer(&at(p, false), cx);
            let held = version(t);
            for _ in 0..3 {
                t.pointer(&at(p, false), cx);
                assert_eq!(version(t), held, "the same point: no rebuild");
            }
            let moved = p + egui::vec2(0.0, 25.0);
            t.pointer(&at(moved, false), cx);
            assert_ne!(version(t), held, "moving the end rebuilds the tube");
            t.pointer(&PointerInput { pos: Some(moved), released: true, ..Default::default() }, cx);
            assert_eq!(t.drag, None);
            // Drawing the panel keeps the tube too.
            let v = version(t);
            assert!(!run_panel(t, cx, 2));
            assert_eq!(version(t), v);
        });
    }

    #[test]
    fn outside_tube_adds_volume_inside_tube_removes_it() {
        let sphere = mm_core::measure::volume(&primitives::icosphere(2.0, 3));
        for (kind, more) in [(TubeType::SplineOutside, true), (TubeType::SplineInside, false), (TubeType::LineInside, false)] {
            let t = tool(kind, Combine::Boolean);
            let tube = t.tube.as_ref().unwrap();
            let op = if t.inside() { mm_edit::BooleanOp::Difference } else { mm_edit::BooleanOp::Union };
            let r = mm_edit::mesh_boolean(&t.source, tube, op).unwrap();
            assert!(r.mesh.is_closed());
            let v = mm_core::measure::volume(&r.mesh);
            assert_eq!(v > sphere, more, "{kind:?}: {v} vs {sphere}");
        }
        // Routes follow the "Route inside" option.
        let mut route = tool(TubeType::AutoRouteSmooth, Combine::Append);
        assert!(!route.inside());
        route.params.inside = true;
        assert!(route.inside());
        route.rebuild();
        assert!(route.preview.as_ref().unwrap().triangle_count() > route.source.triangle_count());
    }
}
