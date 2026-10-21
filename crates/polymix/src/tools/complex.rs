//! Filter Complex (Edit > Filter Complex): keep the cells of a complex (made by Generate
//! Complex: one face group per cell, the group id being the cell's membership — A = 1,
//! B = 2, A ∩ B = 3, …) by which inputs they are inside, chosen in a checklist of the cells or
//! with an expression (`A & B`, `A - B`, `n == 1`, … see [`mm_edit::CellFilter`]), joined
//! into one object along the faces they share. The preview shows the result in place of the
//! complex; Accept puts it in a new object (the complex is hidden, so other filters can follow)
//! or replaces the complex. Polymix's reading: Meshmixer documents none of its complex tools
//! (see [`mm_edit::complex`]).

use super::*;
use crate::commands::ComplexInput;
use crate::jobs::Job;
use crate::ui_util;
use mm_edit::complex::{cell_label, complex_cells, input_letter, ComplexInfo};
use mm_edit::{CellFilter, FilterGroups, FilterResult};
use std::sync::Arc;

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct FilterComplexSettings {
    /// The cells to keep, by membership (A = 1, B = 2, A ∩ B = 3, C = 4, …: one bit per input).
    pub keep: Vec<u32>,
    /// A test on each cell's membership that sets `keep` when changed: A…F (inside that input),
    /// n (how many inputs the cell is inside), all, &, |, - (and not), ^, !, comparisons and
    /// parentheses, e.g. "A & B", "A - B", "n == 1", "n >= 2".
    pub expression: String,
    /// Face groups of the result: Cells (the cell each face came from), Inputs (the input whose
    /// surface it lies on: A = 1, B = 2, …) or Single.
    pub groups: FilterGroups,
    /// Put the result in a new object and hide the complex (otherwise it replaces the complex).
    pub new_object: bool,
}

/// What a result was computed for.
type Key = (Vec<u32>, FilterGroups);

pub struct FilterComplexTool {
    object: Option<ObjectId>,
    info: Result<ComplexInfo, String>,
    s: FilterComplexSettings,
    /// The expression `keep` came from (None after ticking cells by hand).
    from_expression: Option<String>,
    expr_error: Option<String>,
    job: Option<(Key, Job<FilterResult>)>,
    result: Option<(Key, FilterResult)>,
    error: Option<String>,
}

impl FilterComplexTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let info = object.and_then(|id| cx.scene.get(id)).ok_or_else(|| "No active object.".to_string()).and_then(|o| complex_cells(&o.mesh).map_err(|e| e.to_string()));
        let expression = "n >= 1".to_string();
        let keep = info.as_ref().map(|i| i.cells.iter().map(|c| c.mask).collect()).unwrap_or_default();
        let mut t = Self {
            object,
            info,
            s: FilterComplexSettings { keep, expression: expression.clone(), groups: FilterGroups::Cells, new_object: true },
            from_expression: Some(expression),
            expr_error: None,
            job: None,
            result: None,
            error: None,
        };
        t.start(cx);
        t
    }

    fn key(&self) -> Key {
        let mut keep = self.s.keep.clone();
        keep.sort_unstable();
        keep.dedup();
        (keep, self.s.groups)
    }

    /// Set the kept cells from an expression.
    fn apply_expression(&mut self, text: &str) -> Result<(), String> {
        let f = CellFilter::parse(text)?;
        let Ok(info) = &self.info else { return Err("not a complex".into()) };
        let named = f.inputs_named();
        if named >> info.inputs != 0 {
            return Err(format!("the complex has {} inputs (A to {})", info.inputs, input_letter(info.inputs.saturating_sub(1))));
        }
        self.s.keep = info.cells.iter().filter(|c| f.matches(c.mask, info.inputs)).map(|c| c.mask).collect();
        self.s.expression = text.to_string();
        self.from_expression = Some(text.to_string());
        Ok(())
    }

    fn start(&mut self, cx: &mut ToolCx) {
        if self.info.is_err() {
            return;
        }
        let Some(mesh) = self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone()) else { return };
        let key = self.key();
        if self.result.as_ref().is_some_and(|r| r.0 == key) || self.job.as_ref().is_some_and(|j| j.0 == key) {
            return;
        }
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.error = None;
        if key.0.is_empty() {
            self.result = None;
            return;
        }
        let k = key.clone();
        let mesh: Arc<Mesh> = mesh;
        self.job = Some((key, Job::spawn("Filter Complex", cx.egui.clone(), move |_| Ok(mm_edit::filter_complex(&mesh, &|m| k.0.contains(&m), k.1)?))));
    }

    fn poll(&mut self) {
        let Some((_, j)) = &self.job else { return };
        let Some(r) = j.poll() else { return };
        let (k, _) = self.job.take().unwrap();
        match r {
            Ok(r) => self.result = Some((k, r)),
            Err(e) => self.error = Some(format!("{e:#}")),
        }
    }

    fn ready(&self) -> Option<&FilterResult> {
        self.result.as_ref().filter(|r| r.0 == self.key() && self.job.is_none()).map(|r| &r.1)
    }

    fn summary(r: &FilterResult) -> String {
        let open = if r.open_edges > 0 { format!("; {} open edges left (Inspector or Close Cracks can close them)", r.open_edges) } else { ", closed".to_string() };
        format!("{} cells kept: {} triangles{open}", r.cells, ui_util::format_count(r.mesh.triangle_count()))
    }
}

impl Tool for FilterComplexTool {
    fn name(&self) -> &'static str {
        "Filter Complex"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        self.poll();
        ui_util::section(ui, "Filter Complex");
        let info = match &self.info {
            Ok(i) => i.clone(),
            Err(e) => {
                ui.colored_label(ui.visuals().warn_fg_color, e);
                return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
            }
        };
        let names = self.object.and_then(|id| cx.scene.get(id)).map(|o| o.group_names.clone()).unwrap_or_default();
        let before = self.key();
        ui.label(format!("Keep the cells inside inputs A to {} by", input_letter(info.inputs.saturating_sub(1))));
        let mut text = self.s.expression.clone();
        let mut apply = false;
        ui.horizontal(|ui| {
            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(150.0).hint_text("A & B"));
            r.clone().on_hover_text("A…F: inside that input · n: how many inputs · & and · | or · - and not · ^ one of · ! not · == != < <= > >= · ( )");
            apply |= r.lost_focus() && ui.input(|i| i.key_pressed(egui::Key::Enter));
            apply |= ui.button("Apply").clicked();
        });
        self.s.expression = text.clone();
        ui.horizontal_wrapped(|ui| {
            for (e, tip) in [("n >= 1", "Every cell (the union)"), ("n >= 2", "Where inputs overlap"), ("n == 1", "Inside exactly one input"), ("all", "Inside every input")] {
                if ui.small_button(e).on_hover_text(tip).clicked() {
                    text = e.to_string();
                    apply = true;
                }
            }
        });
        if apply {
            self.expr_error = self.apply_expression(&text).err();
        }
        if let Some(e) = &self.expr_error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        ui.add_space(4.0);
        egui::Grid::new("complex_cells").num_columns(3).striped(true).show(ui, |ui| {
            for c in &info.cells {
                let mut on = self.s.keep.contains(&c.mask);
                if ui.checkbox(&mut on, cell_label(c.mask, info.inputs, None)).changed() {
                    self.s.keep.retain(|&m| m != c.mask);
                    if on {
                        self.s.keep.push(c.mask);
                    }
                    self.from_expression = None;
                }
                ui.weak(names.get(&(c.mask as i32)).cloned().unwrap_or_default());
                ui.weak(format!("{:.4} mm³", c.volume));
                ui.end_row();
            }
        });
        ui.horizontal(|ui| {
            if ui.small_button("All").clicked() {
                self.s.keep = info.cells.iter().map(|c| c.mask).collect();
                self.from_expression = None;
            }
            if ui.small_button("None").clicked() {
                self.s.keep.clear();
                self.from_expression = None;
            }
            if ui.small_button("Invert").clicked() {
                self.s.keep = info.cells.iter().map(|c| c.mask).filter(|m| !self.s.keep.contains(m)).collect();
                self.from_expression = None;
            }
        });
        egui::ComboBox::from_label("Face groups")
            .selected_text(match self.s.groups {
                FilterGroups::Cells => "By cell",
                FilterGroups::Inputs => "By input surface",
                FilterGroups::Single => "One group",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.s.groups, FilterGroups::Cells, "By cell").on_hover_text("The cell each face came from");
                ui.selectable_value(&mut self.s.groups, FilterGroups::Inputs, "By input surface").on_hover_text("The input whose surface the face lies on (A = 1, B = 2, …)");
                ui.selectable_value(&mut self.s.groups, FilterGroups::Single, "One group");
            });
        ui.checkbox(&mut self.s.new_object, "New object").on_hover_text("Keep the complex (hidden) for other filters; otherwise the result replaces it");
        if self.key() != before || (self.result.is_none() && self.job.is_none() && self.error.is_none()) {
            self.start(cx);
        }
        if let Some((_, j)) = &self.job {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Joining cells…");
            });
            cx.egui.request_repaint_after(std::time::Duration::from_millis(50));
            let _ = j;
        } else if let Some(r) = self.ready() {
            let open = r.open_edges > 0;
            let text = Self::summary(r);
            if open {
                ui.colored_label(ui.visuals().warn_fg_color, text);
            } else {
                ui.label(text);
            }
        } else if self.s.keep.is_empty() {
            ui.weak("No cell is kept.");
        }
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, self.ready().is_some());
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        Some((self.object?, &self.ready()?.mesh))
    }

    fn shows_groups(&self) -> bool {
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Filter Complex".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        self.info.is_ok() && self.job.is_none() && self.error.is_none() && !self.s.keep.is_empty() && self.ready().is_none()
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.poll();
        if self.job.is_some() {
            return false;
        }
        let Some(r) = self.ready().cloned() else { return self.info.is_err() || self.s.keep.is_empty() };
        let (Some(id), Ok(info)) = (self.object, &self.info) else { return true };
        let Some(o) = cx.scene.get(id) else { return true };
        let what = self.from_expression.clone().unwrap_or_else(|| {
            let mut keep = self.key().0;
            keep.sort_by_key(|m| std::cmp::Reverse(m.count_ones()));
            keep.iter().map(|&m| cell_label(m, info.inputs, None)).collect::<Vec<_>>().join(", ")
        });
        let name = format!("{} [{what}]", o.name);
        let mut group_names: std::collections::HashMap<i32, String> = Default::default();
        match self.s.groups {
            FilterGroups::Cells => {
                for &m in &self.s.keep {
                    if let Some(n) = o.group_names.get(&(m as i32)) {
                        group_names.insert(m as i32, n.clone());
                    }
                }
            }
            FilterGroups::Inputs => {
                for i in 0..info.inputs {
                    group_names.insert(i as i32 + 1, input_letter(i).to_string());
                }
            }
            FilterGroups::Single => {}
        }
        cx.history.push("Filter Complex", cx.scene);
        let summary = Self::summary(&r);
        if self.s.new_object {
            if let Some(o) = cx.scene.get_mut(id) {
                o.visible = false;
            }
            let nid = cx.scene.add(name, r.mesh);
            if let Some(n) = cx.scene.get_mut(nid) {
                n.group_names = group_names;
            }
            cx.scene.active = Some(nid);
        } else if let Some(o) = cx.scene.get_mut(id) {
            o.set_mesh(r.mesh);
            o.group_names = group_names;
            o.name = name;
        }
        *cx.status = format!("Filter Complex ({what}): {summary}");
        true
    }

    fn settings(&self) -> Option<serde_json::Value> {
        serde_json::to_value(&self.s).ok()
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(schema_of::<FilterComplexSettings>())
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let new: FilterComplexSettings = merged(&self.s, changes).map_err(|e| format!("Filter Complex: {e}"))?;
        let expression = changes.get("expression").is_some();
        self.s.groups = new.groups;
        self.s.new_object = new.new_object;
        if expression {
            self.apply_expression(&new.expression).map_err(|e| format!("Filter Complex: {e}"))?;
        }
        if changes.get("keep").is_some() {
            let known: Vec<u32> = self.info.as_ref().map(|i| i.cells.iter().map(|c| c.mask).collect()).unwrap_or_default();
            if let Some(m) = new.keep.iter().find(|m| !known.contains(m)) {
                return Err(format!("Filter Complex: the complex has no cell {m} (its cells are {known:?})"));
            }
            self.s.keep = new.keep;
            self.from_expression = None;
        }
        self.start(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        let cells = self.info.as_ref().ok().map(|i| {
            i.cells
                .iter()
                .map(|c| serde_json::json!({"mask": c.mask, "cell": cell_label(c.mask, i.inputs, None), "volume_mm3": c.volume, "kept": self.s.keep.contains(&c.mask)}))
                .collect::<Vec<_>>()
        });
        serde_json::json!({
            "object": self.object,
            "complex": self.info.as_ref().err(),
            "inputs": self.info.as_ref().ok().map(|i| i.inputs),
            "cells": cells,
            "computing": self.job.is_some(),
            "result": self.ready().map(|r| serde_json::json!({"cells": r.cells, "triangles": r.mesh.triangle_count(), "open_edges": r.open_edges})),
            "error": self.error,
            "next": "accept (mesh_tool accept) once the result is ready",
        })
    }
}

/// Generate Complex (Edit > Generate Complex) computing in the background: its exact booleans
/// take seconds on large inputs. When they are done the cells become a new object (as
/// [`crate::commands::generate_complex`] makes it) and the tool closes; Esc stops them.
pub struct GenerateComplexTool {
    inputs: Vec<ComplexInput>,
    job: Option<Job<mm_edit::EditResult<Vec<mm_edit::ComplexCell>>>>,
}

impl GenerateComplexTool {
    /// Start on `inputs` (see [`crate::commands::complex_inputs`]).
    pub fn new(inputs: Vec<ComplexInput>, egui: &egui::Context) -> Self {
        let solids = inputs.clone();
        let job = Job::spawn("Generate Complex", egui.clone(), move |progress| Ok(crate::commands::complex_cells(&solids, Some(progress))));
        Self { inputs, job: Some(job) }
    }
}

impl Tool for GenerateComplexTool {
    fn name(&self) -> &'static str {
        "Generate Complex"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Generate Complex");
        let Some(j) = &self.job else { return ToolOutcome::Close };
        if let Some(r) = j.poll() {
            self.job = None;
            match r {
                // Only onto the objects as they were: the cells are made of their meshes.
                Ok(cells) if self.inputs.iter().all(|(id, _, m)| cx.scene.get(*id).is_some_and(|o| Arc::ptr_eq(&o.mesh, m))) => {
                    crate::commands::apply_complex(&mut crate::commands::CmdCx { scene: cx.scene, history: cx.history, status: cx.status }, &self.inputs, cells);
                }
                Ok(_) => *cx.status = "Generate Complex: the objects changed meanwhile; nothing was made.".into(),
                Err(e) => *cx.status = format!("Generate Complex: {e:#}"),
            }
            return ToolOutcome::Close;
        }
        let names: Vec<&str> = self.inputs.iter().map(|i| i.1.as_str()).collect();
        ui.label(format!("Cells of {}…", names.join(", ")));
        ui.horizontal(|ui| {
            polymix_workspace::spinner(ui);
            ui.add(egui::ProgressBar::new(j.progress()).show_percentage());
        });
        if ui.button("Cancel").clicked() && self.cancel(cx) {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Generate Complex".to_string(), j.progress()))
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        // Dropping the job stops its booleans.
        self.job = None;
        *cx.status = "Generate Complex cancelled".into();
        true
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        serde_json::json!({
            "inputs": self.inputs.iter().map(|i| i.0).collect::<Vec<_>>(),
            "computing": self.job.as_ref().map(|j| polymix_workspace::agent::round(j.progress() as f64, 2)),
            "next": "app_wait: the cells become a new object named Complex",
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::commands::{generate_complex, CmdCx};
    use crate::scene::History;
    use mm_core::primitives;
    use std::time::{Duration, Instant};

    #[test]
    fn new_tools_open_with_typed_settings() {
        // As the app's catalog test checks its tools: they open (also on a plain object),
        // and their settings match their schema.
        let mut scene = Scene::default();
        scene.add("cube", primitives::cube(10.0, 2));
        let (mut history, mut camera, mut status) = (History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        for name in ["filtercomplex", "setuv"] {
            let tool = crate::app::make_tool(name, &mut cx).unwrap_or_else(|| panic!("{name} does not open"));
            let (s, schema) = (tool.settings().unwrap(), tool.settings_schema().unwrap());
            let props: Vec<String> = schema["properties"].as_object().unwrap().keys().cloned().collect();
            let keys: Vec<String> = s.as_object().unwrap().keys().cloned().collect();
            assert_eq!(keys.len(), props.len(), "{name}: {keys:?} vs {props:?}");
            assert!(keys.iter().all(|k| props.contains(k)), "{name}: {keys:?} vs {props:?}");
            for a in tool.actions() {
                assert!(a.schema()["properties"].is_object() && !a.about.is_empty(), "{name}: {}", a.name);
            }
        }
        assert!(crate::app::make_tool("nothing", &mut cx).is_none());
    }

    #[test]
    fn filter_complex_tool_keeps_cells_by_expression() {
        let mut scene = Scene::default();
        let a = primitives::cube(2.0, 1);
        let mut b = primitives::cube(2.0, 1);
        b.transform(|p| p + DVec3::new(1.0, 0.3, 0.2));
        scene.add("a", a);
        scene.add("b", b);
        let (mut history, mut camera, mut status) = (History::default(), crate::camera::Camera::default(), String::new());
        generate_complex(&mut CmdCx { scene: &mut scene, history: &mut history, status: &mut status });
        let complex = scene.active.unwrap();
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = FilterComplexTool::new(&mut cx);
        assert_eq!(tool.s.keep.len(), 3, "everything kept at first");
        // The panel draws (and starts the preview).
        for _ in 0..2 {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 900.0))), ..Default::default() };
            let _ = ctx.run_ui(raw, |ui| {
                let _ = tool.panel(ui, &mut cx);
            });
        }
        assert!(tool.job.is_some() || tool.result.is_some());
        assert!(tool.set_settings(&serde_json::json!({"expression": "A & C"}), &mut cx).is_err(), "only two inputs");
        tool.set_settings(&serde_json::json!({"expression": "A - B"}), &mut cx).unwrap();
        assert_eq!(tool.key().0, vec![1]);
        tool.set_settings(&serde_json::json!({"expression": "A"}), &mut cx).unwrap();
        let t0 = Instant::now();
        while tool.ready().is_none() && t0.elapsed() < Duration::from_secs(20) {
            std::thread::sleep(Duration::from_millis(5));
            tool.poll();
        }
        assert!(tool.accept(&mut cx), "{:?}", tool.error);
        let result = cx.scene.active().unwrap();
        assert_eq!(result.name, "Complex [A]");
        assert!(result.mesh.is_closed());
        assert!((mm_core::measure::volume(&result.mesh) - 8.0).abs() < 1e-9);
        assert!(!cx.scene.get(complex).unwrap().visible, "the complex is hidden");
        // A plain object is not a complex.
        cx.scene.active = Some(cx.scene.add("ball", primitives::icosphere(1.0, 1)));
        let tool = FilterComplexTool::new(&mut cx);
        assert!(tool.info.is_err());
    }

    #[test]
    fn generate_complex_in_the_background_makes_the_same_complex() {
        let scene_of = || {
            let mut scene = Scene::default();
            let mut b = primitives::icosphere(1.0, 2);
            b.transform(|p| p + DVec3::new(0.9, 0.1, 0.0));
            let mut c = primitives::icosphere(0.8, 2);
            c.transform(|p| p + DVec3::new(0.4, 0.8, 0.1));
            scene.add("a", primitives::icosphere(1.0, 2));
            scene.add("b", b);
            scene.add("c", c);
            scene
        };
        // At once, as tests and scripts make it.
        let mut direct = scene_of();
        let (mut history, mut status) = (History::default(), String::new());
        generate_complex(&mut CmdCx { scene: &mut direct, history: &mut history, status: &mut status });
        // In the background, as the menu now makes it.
        let mut scene = scene_of();
        let (mut history2, mut camera, mut status2) = (History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history2, camera: &mut camera, status: &mut status2, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let inputs = crate::commands::complex_inputs(cx.scene).unwrap();
        let mut tool = GenerateComplexTool::new(inputs, &ctx);
        assert!(tool.busy().is_some());
        let t0 = Instant::now();
        let mut closed = false;
        while !closed && t0.elapsed() < Duration::from_secs(60) {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 900.0))), ..Default::default() };
            let _ = ctx.run_ui(raw, |ui| closed = matches!(tool.panel(ui, &mut cx), ToolOutcome::Close));
            std::thread::sleep(Duration::from_millis(5));
        }
        assert!(closed && tool.busy().is_none());
        assert_eq!(*cx.status, status);
        let (a, b) = (direct.active().unwrap(), cx.scene.active().unwrap());
        assert_eq!((a.name.as_str(), b.name.as_str()), ("Complex", "Complex"));
        assert_eq!(a.mesh.to_arrays().1, b.mesh.to_arrays().1);
        assert!(a.mesh.to_arrays().0.iter().zip(&b.mesh.to_arrays().0).all(|(p, q)| p.to_array().map(f64::to_bits) == q.to_array().map(f64::to_bits)));
        assert_eq!(a.mesh.to_arrays().2, b.mesh.to_arrays().2);
        assert_eq!(a.group_names, b.group_names);
        assert_eq!(cx.scene.objects.iter().filter(|o| o.visible).count(), 1, "the inputs are hidden");
        assert!(cx.history.can_undo());
        // Esc stops it, leaving the scene as it was.
        let mut scene = scene_of();
        let mut cx = ToolCx { scene: &mut scene, history: &mut history2, camera: &mut camera, status: &mut status2, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = GenerateComplexTool::new(crate::commands::complex_inputs(cx.scene).unwrap(), &ctx);
        assert!(tool.cancel(&mut cx));
        assert!(tool.busy().is_none());
        assert_eq!(cx.scene.objects.len(), 3);
    }
}
