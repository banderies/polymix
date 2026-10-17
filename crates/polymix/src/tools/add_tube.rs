//! Add Tube (Edit > Add Tube): click two points on the object for the tube's ends (drag an
//! end to move it); the tube is combined with the object by a boolean (outside tubes are
//! added, inside tubes cut a channel), appended, or made a new object.

use super::*;
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
}

impl AddTubeTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, bvh) = match cx.scene.active.and_then(|id| cx.scene.get_mut(id)) {
            Some(o) => (Some(o.id), o.mesh.clone(), o.bvh()),
            None => (None, Arc::new(Mesh::new()), Arc::new(mm_core::Bvh::build(&Mesh::new()))),
        };
        let r = (source.bounds().diagonal() * 0.015).max(1e-6);
        let params = TubeParams { path: TubePath::Spline, inside: true, radius: r, collision_radius: r * 4.0 / 3.0, ..Default::default() };
        Self {
            object,
            source,
            bvh,
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

    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        let (Some(id), Some(tube)) = (self.object, self.tube.clone()) else { return true };
        if !cx.scene.get(id).is_some_and(|o| Arc::ptr_eq(&o.mesh, &self.source)) {
            *cx.status = "Add Tube cancelled: the object changed while the tool was open.".into();
            return true;
        }
        let result = match self.combine {
            Combine::NewObject => {
                cx.history.push("Add Tube", cx.scene);
                let name = cx.scene.get(id).map(|o| format!("{} tube", o.name)).unwrap_or_else(|| "tube".into());
                cx.scene.add(name, tube);
                *cx.status = "Add Tube: new object".into();
                return true;
            }
            Combine::Append => {
                let mut m = (*self.source).clone();
                m.append_mesh(&tube, m.max_group_id() + 1);
                m
            }
            Combine::Boolean => {
                let op = if self.inside() { mm_edit::BooleanOp::Difference } else { mm_edit::BooleanOp::Union };
                match mm_edit::mesh_boolean(&self.source, &tube, op) {
                    Ok(r) => r.mesh,
                    Err(e) => {
                        self.error = Some(format!("{e} — use Combine Mode: Append, or make the object solid first"));
                        return false;
                    }
                }
            }
        };
        cx.history.push("Add Tube", cx.scene);
        if let Some(o) = cx.scene.get_mut(id) {
            o.set_mesh(result);
        }
        *cx.status = "Add Tube applied".into();
        true
    }
}

impl Tool for AddTubeTool {
    fn name(&self) -> &'static str {
        "Add Tube"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Add Tube");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let before = (self.kind, self.combine, self.along_normal, self.params);
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
        let span = (self.source.bounds().diagonal() * 0.1).max(self.params.radius * 2.0);
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
        if (self.kind, self.combine, self.along_normal, self.params) != before {
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild();
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, self.tube.is_some());
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
                if let Some(e) = self.pick(cx, pos) {
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
        };
        t.rebuild();
        t
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
