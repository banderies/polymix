//! Units / Dimensions (Analysis > Units / Dimensions): read and set the object's size in
//! common units, convert between units, and set the scale from measurements: drag on the
//! model to measure, give each measurement its real length, and the scale that best fits
//! them all is applied (with the remaining average / maximum deviation shown).

use super::*;
use crate::ui_util;
use rayon::prelude::*;
use std::sync::Arc;

/// Display units; model units are taken to be millimetres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Mm,
    Cm,
    M,
    In,
}

impl Unit {
    const ALL: [Unit; 4] = [Unit::Mm, Unit::Cm, Unit::M, Unit::In];
    fn label(self) -> &'static str {
        match self {
            Unit::Mm => "mm",
            Unit::Cm => "cm",
            Unit::M => "m",
            Unit::In => "in",
        }
    }
    /// Millimetres per unit.
    fn mm(self) -> f64 {
        match self {
            Unit::Mm => 1.0,
            Unit::Cm => 10.0,
            Unit::M => 1000.0,
            Unit::In => 25.4,
        }
    }
}

struct Measure {
    a: DVec3,
    b: DVec3,
    /// The real length this measurement should have (model units).
    target: f64,
}

impl Measure {
    fn length(&self) -> f64 {
        self.a.distance(self.b)
    }
}

pub struct UnitsTool {
    object: Option<ObjectId>,
    unit: Unit,
    measures: Vec<Measure>,
    selected: Option<usize>,
    pressed: Option<(egui::Pos2, DVec3)>,
    /// The dimension being dragged or typed, if any.
    gesture: Option<Gesture>,
    /// Where the X, Y and Z fields were drawn last (tests work them).
    #[cfg(test)]
    dim_rects: [egui::Rect; 3],
}

/// A dimension being dragged (or typed, until Enter): its first change pushed the undo step,
/// and every change scales the object as it was before the gesture, not as the last frame
/// left it. A drag is then one undo step, and no frame copies the mesh or snapshots the scene.
struct Gesture {
    object: ObjectId,
    /// The mesh before the gesture (the undo step holds the same one).
    base: Arc<Mesh>,
    center: DVec3,
    extents: DVec3,
    /// The measurements' ends before the gesture.
    measures: Vec<(DVec3, DVec3)>,
    /// The shape version the last change left: any other edit of the mesh ends the gesture.
    shape: u64,
}

/// Least-squares scale `s` minimising Σ (s·Lᵢ − Tᵢ)², with the average and maximum
/// deviation |s·Lᵢ − Tᵢ| left over.
fn fit_scale(pairs: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let num: f64 = pairs.iter().map(|(l, t)| l * t).sum();
    let den: f64 = pairs.iter().map(|(l, _)| l * l).sum();
    if den <= 0.0 || pairs.is_empty() {
        return None;
    }
    let s = num / den;
    let dev: Vec<f64> = pairs.iter().map(|(l, t)| (s * l - t).abs()).collect();
    Some((s, dev.iter().sum::<f64>() / dev.len() as f64, dev.iter().copied().fold(0.0, f64::max)))
}

impl UnitsTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self {
            object: cx.scene.active,
            unit: Unit::Mm,
            measures: Vec::new(),
            selected: None,
            pressed: None,
            gesture: None,
            #[cfg(test)]
            dim_rects: [egui::Rect::NOTHING; 3],
        }
    }

    fn scale_by(&mut self, cx: &mut ToolCx, s: f64, label: &str) {
        // A step of its own: a dimension dragged afterwards starts a new one.
        self.gesture = None;
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else { return };
        if !(s > 0.0 && s.is_finite()) || s == 1.0 {
            return;
        }
        cx.history.push(label, cx.scene);
        let Some(o) = cx.scene.get_mut(id) else { return };
        let c = o.bounds().center();
        o.mesh_mut().transform(|p| c + (p - c) * s);
        for m in &mut self.measures {
            m.a = c + (m.a - c) * s;
            m.b = c + (m.b - c) * s;
        }
        *cx.status = format!("{label}: scaled by {s}");
    }

    /// Give dimension `axis` the length `size` (model units) by scaling the object uniformly
    /// about its centre, as one change of a dimension gesture (see [`Gesture`]).
    fn set_dimension_live(&mut self, cx: &mut ToolCx, axis: usize, size: f64) {
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else { return };
        let n = self.measures.len();
        let going = self.gesture.as_ref().is_some_and(|g| g.object == id && g.measures.len() == n && cx.scene.get(id).is_some_and(|o| o.mesh.shape_version() == g.shape));
        if going {
            let s = self.gesture.as_ref().map_or(0.0, |g| size / g.extents[axis]);
            if !(s > 0.0 && s.is_finite()) {
                return;
            }
            cx.history.touch();
        } else {
            let Some(o) = cx.scene.get(id) else { return };
            let b = o.bounds();
            let s = size / b.extents()[axis];
            if !(s > 0.0 && s.is_finite()) || s == 1.0 {
                return;
            }
            cx.history.push("Set dimensions", cx.scene);
            let measures = self.measures.iter().map(|m| (m.a, m.b)).collect();
            self.gesture = Some(Gesture { object: id, base: o.mesh.clone(), center: b.center(), extents: b.extents(), measures, shape: 0 });
        }
        let (Some(g), Some(o)) = (self.gesture.as_mut(), cx.scene.get_mut(id)) else { return };
        let (c, s) = (g.center, size / g.extents[axis]);
        let base = &*g.base;
        // The same positions Mesh::transform gives the live vertices; the mesh is copied from
        // the undo step once, on the first change.
        let mesh = o.mesh_mut();
        const CHUNK: usize = 1 << 15;
        mesh.positions_mut().par_chunks_mut(CHUNK).zip(base.positions().par_chunks(CHUNK)).enumerate().for_each(|(k, (ps, qs))| {
            for (i, (p, q)) in ps.iter_mut().zip(qs).enumerate() {
                if base.is_vertex((k * CHUNK + i) as u32) {
                    *p = c + (*q - c) * s;
                }
            }
        });
        mesh.bump_shape_version();
        g.shape = mesh.shape_version();
        for (m, (a, b)) in self.measures.iter_mut().zip(&g.measures) {
            m.a = c + (*a - c) * s;
            m.b = c + (*b - c) * s;
        }
        *cx.status = format!("Set dimensions: scaled by {s}");
    }
}

impl UnitsTool {
    /// Apply the scale that best fits the measurements' real lengths.
    fn apply_fit(&mut self, cx: &mut ToolCx) -> Result<f64, String> {
        let pairs: Vec<(f64, f64)> = self.measures.iter().filter(|m| m.target > 0.0).map(|m| (m.length(), m.target)).collect();
        let (s, _, _) = fit_scale(&pairs).ok_or("give at least one measurement its real length")?;
        self.scale_by(cx, s, "Scale from measurements");
        for m in &mut self.measures {
            m.target = m.length();
        }
        Ok(s)
    }
}

impl Tool for UnitsTool {
    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"unit": self.unit.label()}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {"unit": {"type": "string", "enum": Unit::ALL.map(|u| u.label()), "description": "Unit the dimensions and lengths are shown and given in (the model is in mm)"}}}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.unit = Unit::ALL.into_iter().find(|u| v["unit"] == u.label()).ok_or("unit must be mm, cm, m or in")?;
        Ok(())
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::round;
        let k = self.unit.mm();
        let e = self.object.and_then(|id| scene.get(id)).map(|o| o.bounds().extents());
        let pairs: Vec<(f64, f64)> = self.measures.iter().filter(|m| m.target > 0.0).map(|m| (m.length(), m.target)).collect();
        serde_json::json!({
            "object": self.object,
            "unit": self.unit.label(),
            "dimensions": e.map(|e| [e.x, e.y, e.z].map(|x| round(x / k, 5))),
            "measurements": self.measures.iter().enumerate().map(|(i, m)| serde_json::json!({"measurement": i + 1, "length": round(m.length() / k, 5), "real_length": round(m.target / k, 5)})).collect::<Vec<_>>(),
            "fitted_scale": fit_scale(&pairs).map(|(s, avg, max)| serde_json::json!({"scale": round(s, 8), "average_deviation": round(avg / k, 5), "max_deviation": round(max / k, 5)})),
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("set_dimension", "Scale the object uniformly so one dimension has this size (in the unit setting)", json!({"axis": arg::choice(&["x", "y", "z"], "Which dimension"), "size": arg::number("Its new size")})).required(&["axis", "size"]),
            Action::new("scale", "Scale the object uniformly about its centre (e.g. 1000 for m → mm, 25.4 for in → mm)", json!({"factor": arg::number("Scale factor")})).required(&["factor"]),
            Action::new("measure", "Add a measurement between two points (as dragging on the model does), optionally with its real length", json!({"a": arg::point("First point, mm"), "b": arg::point("Second point, mm"), "real_length": arg::number("Its real length (in the unit setting)")})).required(&["a", "b"]),
            Action::new("set_real_length", "Give a measurement its real length (in the unit setting)", json!({"measurement": arg::integer("Its number (from 1)"), "real_length": arg::number("The real length")})).required(&["measurement", "real_length"]),
            Action::new("remove", "Remove a measurement", json!({"measurement": arg::integer("Its number (from 1)")})).required(&["measurement"]),
            Action::plain("apply_fitted_scale", "Scale the object by the scale that best fits the measurements' real lengths"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        let k = self.unit.mm();
        let index = |len: usize| -> Result<usize, String> { a.usize("measurement")?.checked_sub(1).filter(|i| *i < len).ok_or_else(|| "there is no such measurement".to_string()) };
        // Agents' changes are steps of their own.
        self.gesture = None;
        match action {
            "set_dimension" => {
                let o = self.object.and_then(|id| cx.scene.get(id)).ok_or("no active object")?;
                let e = o.bounds().extents();
                let i = ["x", "y", "z"].iter().position(|x| Some(*x) == a.opt_str("axis")).ok_or("axis must be x, y or z")?;
                let size = a.f64("size")? * k;
                if !(size > 0.0 && e[i] > 0.0) {
                    return Err("the size and the current dimension must be positive".into());
                }
                self.scale_by(cx, size / e[i], "Set dimensions");
                Ok(cx.status.clone())
            }
            "scale" => {
                let f = a.f64("factor")?;
                if !(f > 0.0 && f.is_finite()) {
                    return Err("the factor must be positive".into());
                }
                self.scale_by(cx, f, "Scale");
                Ok(cx.status.clone())
            }
            "measure" => {
                let (pa, pb) = (DVec3::from_array(a.point("a")?), DVec3::from_array(a.point("b")?));
                let target = a.opt_f64("real_length").map_or(pa.distance(pb), |t| t * k);
                self.measures.push(Measure { a: pa, b: pb, target });
                self.selected = Some(self.measures.len() - 1);
                Ok(format!("Measurement {}: {:.4} {}", self.measures.len(), pa.distance(pb) / k, self.unit.label()))
            }
            "set_real_length" => {
                let i = index(self.measures.len())?;
                self.measures[i].target = a.f64("real_length")? * k;
                Ok(format!("Measurement {} should be {} {}", i + 1, a.f64("real_length")?, self.unit.label()))
            }
            "remove" => {
                let i = index(self.measures.len())?;
                self.measures.remove(i);
                self.selected = None;
                Ok(format!("Measurement {} removed", i + 1))
            }
            "apply_fitted_scale" => {
                let s = self.apply_fit(cx)?;
                Ok(format!("Scaled by {s:.6}"))
            }
            _ => Err(format!("Units / Dimensions has no action {action}")),
        }
    }

    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Units / Dimensions"
    }

    fn copy_text(&self) -> Option<String> {
        let mut lines: Vec<String> = self.measures.iter().map(|m| format!("measured {:.4} → {:.4}", m.length(), m.target)).collect();
        lines.insert(0, format!("units {}", self.unit.label()));
        Some(lines.join("\n"))
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Units / Dimensions");
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let e = o.bounds().extents();
        ui.horizontal(|ui| {
            ui.label("Dimensions in");
            egui::ComboBox::from_id_salt("unit").selected_text(self.unit.label()).show_ui(ui, |ui| {
                for u in Unit::ALL {
                    ui.selectable_value(&mut self.unit, u, u.label());
                }
            });
        });
        let k = self.unit.mm();
        let mut size = None;
        // A dimension is being dragged, or typed in (a typed value applies on Enter or when
        // the field loses focus, not at every keystroke).
        let mut held = false;
        egui::Grid::new("dims").num_columns(2).show(ui, |ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                let mut v = e[i] / k;
                let r = ui.add(egui::DragValue::new(&mut v).speed(e[i] / k * 0.005).max_decimals(4).suffix(format!(" {}", self.unit.label())).update_while_editing(false));
                held |= r.dragged() || r.has_focus();
                if r.changed() && e[i] > 0.0 && v > 0.0 {
                    size = Some((i, v * k));
                }
                #[cfg(test)]
                {
                    self.dim_rects[i] = r.rect;
                }
                ui.end_row();
            }
        });
        if let Some((axis, size)) = size {
            self.set_dimension_live(cx, axis, size);
        }
        if !held {
            self.gesture = None;
        }
        ui.small("Typing a dimension scales the object uniformly. Mesh files carry no units; medical exports are usually millimetres.");
        ui_util::section(ui, "Convert units");
        ui.horizontal_wrapped(|ui| {
            for (label, s) in [("m → mm", 1000.0), ("cm → mm", 10.0), ("in → mm", 25.4), ("mm → m", 0.001), ("mm → in", 1.0 / 25.4)] {
                if ui.button(label).clicked() {
                    self.scale_by(cx, s, label);
                }
            }
        });
        ui_util::section(ui, "Set from measurements");
        ui.small("Drag on the model to measure, then enter each measurement's real length.");
        let mut remove = None;
        for (i, m) in self.measures.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                if ui.radio(self.selected == Some(i), format!("{:.3}", m.length() / k)).on_hover_text("Measured length").clicked() {
                    self.selected = Some(i);
                }
                ui.label("→");
                let mut t = m.target / k;
                if ui.add(egui::DragValue::new(&mut t).speed(0.01).range(0.0..=1e9).max_decimals(4).suffix(format!(" {}", self.unit.label()))).on_hover_text("Real length").changed() {
                    m.target = t * k;
                }
                if ui.small_button("x").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.measures.remove(i);
            self.selected = None;
            self.gesture = None;
        }
        let pairs: Vec<(f64, f64)> = self.measures.iter().filter(|m| m.target > 0.0).map(|m| (m.length(), m.target)).collect();
        if let Some((s, avg, max)) = fit_scale(&pairs) {
            ui.label(format!("Scale {s:.6}"));
            if pairs.len() > 1 {
                ui.label(format!("Avg deviation {:.4} {u}, max {:.4} {u}", avg / k, max / k, u = self.unit.label()));
            }
            if ui.add_enabled((s - 1.0).abs() > 1e-12, egui::Button::new("Apply scale")).clicked() {
                let _ = self.apply_fit(cx);
            }
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
            self.pressed = cx.pick(pos, self.object).map(|(_, h)| (pos, h.point));
        }
        if input.released {
            if let (Some((p0, a)), Some((_, h))) = (self.pressed.take(), cx.pick(pos, self.object)) {
                if p0.distance(pos) > 5.0 {
                    let m = Measure { a, b: h.point, target: a.distance(h.point) };
                    self.measures.push(m);
                    self.selected = Some(self.measures.len() - 1);
                    self.gesture = None;
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        // Bounding box, edges coloured by axis.
        if let Some(o) = self.object.and_then(|id| cx.scene.get(id)) {
            let b = o.bounds();
            let c = |i: usize| DVec3::new(if i & 1 == 0 { b.min.x } else { b.max.x }, if i & 2 == 0 { b.min.y } else { b.max.y }, if i & 4 == 0 { b.min.z } else { b.max.z });
            let cols = [Vec4::new(0.9, 0.3, 0.3, 1.0), Vec4::new(0.3, 0.85, 0.35, 1.0), Vec4::new(0.35, 0.5, 0.95, 1.0)];
            for i in 0..8 {
                for (axis, bit) in [1usize, 2, 4].iter().enumerate() {
                    if i & bit == 0 {
                        ov.line(c(i), c(i | bit), cols[axis]);
                    }
                }
            }
        }
        for (i, m) in self.measures.iter().enumerate() {
            let col = if self.selected == Some(i) { Vec4::new(1.0, 0.6, 0.2, 1.0) } else { Vec4::new(1.0, 0.85, 0.2, 1.0) };
            ov.line(m.a, m.b, col);
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let k = self.unit.mm();
        let font = egui::FontId::proportional(13.0);
        if let Some(o) = self.object.and_then(|id| cx.scene.get(id)) {
            let b = o.bounds();
            let e = b.extents();
            let mids = [
                (DVec3::new(b.center().x, b.min.y, b.min.z), e.x, egui::Color32::from_rgb(230, 90, 90)),
                (DVec3::new(b.max.x, b.center().y, b.min.z), e.y, egui::Color32::from_rgb(90, 210, 100)),
                (DVec3::new(b.max.x, b.min.y, b.center().z), e.z, egui::Color32::from_rgb(110, 140, 240)),
            ];
            for (p, len, col) in mids {
                if let Some(q) = cx.project(p) {
                    painter.text(q, egui::Align2::CENTER_TOP, format!("{:.3} {}", len / k, self.unit.label()), font.clone(), col);
                }
            }
        }
        for m in &self.measures {
            if let Some(q) = cx.project((m.a + m.b) * 0.5) {
                painter.text(q, egui::Align2::LEFT_BOTTOM, format!("{:.3}", m.length() / k), font.clone(), egui::Color32::from_rgb(255, 220, 80));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_scale() {
        // Two measurements that both say "double it" fit exactly.
        let (s, avg, max) = fit_scale(&[(1.0, 2.0), (3.0, 6.0)]).unwrap();
        assert!((s - 2.0).abs() < 1e-12 && avg < 1e-12 && max < 1e-12);
        // Conflicting ones leave a deviation.
        let (s, avg, max) = fit_scale(&[(1.0, 2.0), (1.0, 3.0)]).unwrap();
        assert!((s - 2.5).abs() < 1e-12 && (avg - 0.5).abs() < 1e-12 && (max - 0.5).abs() < 1e-12);
        assert!(fit_scale(&[]).is_none());
    }

    /// Every live vertex of `m` is `base`'s scaled by `s` about `c`, to rounding.
    fn scaled_from(m: &Mesh, base: &Mesh, c: DVec3, s: f64) -> bool {
        base.vertex_ids().all(|v| (m.vertex(v) - (c + (base.vertex(v) - c) * s)).length() < 1e-9 * (1.0 + s))
    }

    #[test]
    fn a_dimension_drag_or_typed_value_is_one_undo_step() {
        let mut scene = Scene::default();
        let mut m = mm_core::primitives::cube(1.0, 3);
        m.transform(|p| DVec3::new(p.x * 80.0 + 5.0, p.y * 40.0 - 3.0, p.z * 20.0 + 1.0));
        let id = scene.add("box", m);
        scene.active = Some(id);
        let (mut history, mut camera, mut status, ctx) = (History::default(), crate::camera::Camera::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = UnitsTool::new(&mut cx);
        // A measurement scales with the object.
        tool.act("measure", &serde_json::json!({"a": [5.0, -3.0, 1.0], "b": [45.0, -3.0, 1.0]}), &mut cx).unwrap();
        let frame = |tool: &mut UnitsTool, cx: &mut ToolCx, events: Vec<egui::Event>| {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), events, ..Default::default() };
            let _ = ctx.run_ui(raw, |ui| {
                let _ = tool.panel(ui, cx);
            });
        };
        let press = |p: egui::Pos2, pressed: bool| egui::Event::PointerButton { pos: p, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE };
        let base = cx.scene.get(id).unwrap().mesh.clone();
        let b0 = base.bounds();
        assert!((b0.extents().x - 80.0).abs() < 1e-9);
        frame(&mut tool, &mut cx, vec![]);
        frame(&mut tool, &mut cx, vec![]);

        // Drag X to the right over several frames: every frame rescales the box, all of it one
        // undo step.
        let p = tool.dim_rects[0].center();
        frame(&mut tool, &mut cx, vec![egui::Event::PointerMoved(p), press(p, true)]);
        let mut versions = vec![];
        for k in 1..=5 {
            frame(&mut tool, &mut cx, vec![egui::Event::PointerMoved(p + egui::vec2(20.0 * k as f32, 0.0))]);
            versions.push(cx.scene.get(id).unwrap().mesh.shape_version());
        }
        frame(&mut tool, &mut cx, vec![press(p + egui::vec2(100.0, 0.0), false)]);
        frame(&mut tool, &mut cx, vec![]);
        versions.dedup();
        assert!(versions.len() >= 4, "the box follows the drag: {versions:?}");
        assert_eq!(cx.history.labels().0, vec!["Set dimensions"]);
        let e = cx.scene.get(id).unwrap().bounds().extents();
        let s = e.x / 80.0;
        assert!(s > 1.2, "{e:?}");
        assert!(scaled_from(&cx.scene.get(id).unwrap().mesh, &base, b0.center(), s));
        assert!((e.y - 40.0 * s).abs() < 1e-9 && (e.z - 20.0 * s).abs() < 1e-9, "{e:?}");
        assert!((tool.measures[0].length() - 40.0 * s).abs() < 1e-9);

        // Type 120 over the X value: nothing happens until Enter, then one step scales it.
        frame(&mut tool, &mut cx, vec![egui::Event::PointerMoved(p), press(p, true)]);
        frame(&mut tool, &mut cx, vec![press(p, false)]);
        frame(&mut tool, &mut cx, vec![]);
        let before = cx.scene.get(id).unwrap().mesh.shape_version();
        for c in ["1", "2", "0"] {
            frame(&mut tool, &mut cx, vec![egui::Event::Text(c.into())]);
            assert_eq!(cx.scene.get(id).unwrap().mesh.shape_version(), before, "typing does not scale");
        }
        assert_eq!(cx.history.labels().0.len(), 1);
        frame(&mut tool, &mut cx, vec![egui::Event::Key { key: egui::Key::Enter, physical_key: None, pressed: true, repeat: false, modifiers: Default::default() }]);
        frame(&mut tool, &mut cx, vec![]);
        frame(&mut tool, &mut cx, vec![]);
        assert_eq!(cx.history.labels().0, vec!["Set dimensions", "Set dimensions"]);
        let e = cx.scene.get(id).unwrap().bounds().extents();
        assert!((e.x - 120.0).abs() < 1e-9 && (e.y - 60.0).abs() < 1e-9, "{e:?}");
        assert!(scaled_from(&cx.scene.get(id).unwrap().mesh, &base, b0.center(), 1.5));
        assert!((tool.measures[0].length() - 60.0).abs() < 1e-9);

        // Undo goes back to the drag's result, then to the very mesh before it.
        let gen = cx.history.generation();
        cx.history.undo(cx.scene);
        assert!((cx.scene.get(id).unwrap().bounds().extents().x - 80.0 * s).abs() < 1e-9);
        cx.history.undo(cx.scene);
        assert!(Arc::ptr_eq(&cx.scene.get(id).unwrap().mesh, &base));
        assert!(cx.history.generation() > gen);
    }
}
