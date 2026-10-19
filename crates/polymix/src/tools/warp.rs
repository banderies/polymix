//! Warp (Select > Deform > Warp): drag handles placed on the selection; the selected
//! region bends to follow while its border and the rest of the mesh stay put.

use super::*;
use crate::ui_util;
use std::sync::Arc;

struct Handle {
    vertex: u32,
    target: DVec3,
    /// Vertices that move with the handle (a small disc around it, inside the selection).
    disc: Vec<u32>,
}

pub struct WarpTool {
    object: Option<ObjectId>,
    source: Arc<Mesh>,
    selection: Vec<u32>,
    handles: Vec<Handle>,
    rigid: bool,
    /// Geodesic radius of the disc each handle carries.
    handle_size: f64,
    preview: Option<Mesh>,
    /// Handle being dragged, with its target and the pointer at the press.
    drag: Option<(usize, DVec3, egui::Pos2)>,
    hover: Option<usize>,
    dirty: bool,
}

const HANDLE_PX: f32 = 7.0;

impl WarpTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, selection) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.mesh.clone(), o.selection.iter().collect()),
            None => (None, Arc::new(Mesh::new()), Vec::new()),
        };
        let handle_size = 3.0 * source.average_edge_length();
        Self { object, source, selection, handles: Vec::new(), rigid: true, handle_size, preview: None, drag: None, hover: None, dirty: false }
    }

    /// Interior selection vertices within `handle_size` of `v` along the surface.
    fn disc(&self, v: u32) -> Vec<u32> {
        let m = &*self.source;
        let selected: std::collections::HashSet<u32> = self.selection.iter().copied().collect();
        let interior = |u: u32| m.vertex_triangles(u).iter().all(|t| selected.contains(t));
        let mut dist: std::collections::HashMap<u32, f64> = std::collections::HashMap::from([(v, 0.0)]);
        let mut stack = vec![v];
        while let Some(x) = stack.pop() {
            let dx = dist[&x];
            for u in m.vertex_neighbors(x) {
                let d = dx + m.vertex(x).distance(m.vertex(u));
                if d <= self.handle_size && interior(u) && d < dist.get(&u).copied().unwrap_or(f64::INFINITY) {
                    dist.insert(u, d);
                    stack.push(u);
                }
            }
        }
        dist.into_keys().collect()
    }

    fn rebuild(&mut self) {
        let src = &self.source;
        let mut handles: Vec<(u32, DVec3)> = Vec::new();
        let mut taken = std::collections::HashSet::new();
        for h in &self.handles {
            let offset = h.target - src.vertex(h.vertex);
            for &v in &h.disc {
                if taken.insert(v) {
                    handles.push((v, src.vertex(v) + offset));
                }
            }
        }
        let mut m = (*self.source).clone();
        for (v, p) in mm_edit::warp(&self.source, &self.selection, &handles, self.rigid) {
            m.set_vertex(v, p);
        }
        self.preview = Some(m);
        self.dirty = false;
    }

    fn current(&self) -> &Mesh {
        self.preview.as_ref().unwrap_or(&self.source)
    }

    fn handle_at(&self, cx: &ToolCx, p: egui::Pos2) -> Option<usize> {
        self.handles
            .iter()
            .enumerate()
            .filter_map(|(i, h)| Some((i, cx.project(h.target)?.distance(p))))
            .filter(|&(_, d)| d < HANDLE_PX + 4.0)
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .map(|x| x.0)
    }

    /// Vertex of the selection's interior under the pointer (on the warped surface).
    fn vertex_at(&self, cx: &ToolCx, p: egui::Pos2) -> Option<u32> {
        let m = self.current();
        let ray = cx.ray_at(p);
        let hit = mm_core::Bvh::build(m).raycast(m, &ray)?;
        let selected: std::collections::HashSet<u32> = self.selection.iter().copied().collect();
        if !selected.contains(&hit.tid) {
            return None;
        }
        m.triangle(hit.tid)
            .into_iter()
            .filter(|&v| m.vertex_triangles(v).iter().all(|t| selected.contains(t)))
            .min_by(|&a, &b| m.vertex(a).distance(hit.point).total_cmp(&m.vertex(b).distance(hit.point)))
    }

    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        let Some(id) = self.object else { return true };
        if self.handles.is_empty() {
            return true;
        }
        if !cx.scene.get(id).is_some_and(|o| Arc::ptr_eq(&o.mesh, &self.source)) {
            *cx.status = "Warp cancelled: the object changed while the tool was open.".into();
            return true;
        }
        if self.dirty || self.preview.is_none() {
            self.rebuild();
        }
        cx.history.push("Warp", cx.scene);
        if let (Some(o), Some(m)) = (cx.scene.get_mut(id), self.preview.take()) {
            let sel = o.selection.clone();
            o.mesh = Arc::new(m);
            o.selection = sel;
            o.touch_selection();
        }
        true
    }
}

impl Tool for WarpTool {
    fn name(&self) -> &'static str {
        "Warp"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"preserve_detail": self.rigid, "handle_size_mm": self.handle_size}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "preserve_detail": {"type": "boolean", "description": "As-rigid-as-possible: surface detail keeps its shape (slower)"},
            "handle_size_mm": {"type": "number", "description": "Radius of the patch each handle carries, along the surface"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.rigid = v["preserve_detail"].as_bool().ok_or("preserve_detail must be true or false")?;
        self.handle_size = v["handle_size_mm"].as_f64().filter(|s| *s >= 0.0).ok_or("handle_size_mm must be a number ≥ 0")?;
        for i in 0..self.handles.len() {
            self.handles[i].disc = self.disc(self.handles[i].vertex);
        }
        self.rebuild();
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::vec3;
        serde_json::json!({
            "object": self.object,
            "selected_triangles": self.selection.len(),
            "handles": self.handles.iter().enumerate().map(|(i, h)| serde_json::json!({"handle": i + 1, "from_mm": vec3(self.source.vertex(h.vertex).to_array(), 4), "to_mm": vec3(h.target.to_array(), 4)})).collect::<Vec<_>>(),
            "next": if self.selection.is_empty() { "select the region to warp first" } else { "add_handle, move_handle, then accept (mesh_tool accept)" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("add_handle", "Put a handle on the selection's interior vertex nearest `at`", json!({"at": arg::point("A point in the selected region, mm")})).required(&["at"]),
            Action::new("move_handle", "Drag a handle to a point (the region bends to follow)", json!({"handle": arg::integer("Its number (from 1)"), "to": arg::point("Where it goes, mm")})).required(&["handle", "to"]),
            Action::new("remove_handle", "Remove a handle", json!({"handle": arg::integer("Its number (from 1)")})).required(&["handle"]),
            Action::plain("clear", "Remove every handle"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, _cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        let index = |n: usize| -> Result<usize, String> { a.usize("handle")?.checked_sub(1).filter(|i| *i < n).ok_or_else(|| "there is no such handle".to_string()) };
        match action {
            "add_handle" => {
                let at = DVec3::from_array(a.point("at")?);
                let m = &*self.source;
                let selected: std::collections::HashSet<u32> = self.selection.iter().copied().collect();
                let v = self
                    .selection
                    .iter()
                    .flat_map(|&t| m.triangle(t))
                    .filter(|&v| m.vertex_triangles(v).iter().all(|t| selected.contains(t)))
                    .min_by(|&x, &y| m.vertex(x).distance(at).total_cmp(&m.vertex(y).distance(at)))
                    .ok_or("the selection has no interior vertices: select a region first")?;
                let disc = self.disc(v);
                self.handles.push(Handle { vertex: v, target: m.vertex(v), disc });
                Ok(format!("Handle {} at [{:.3}, {:.3}, {:.3}]", self.handles.len(), m.vertex(v).x, m.vertex(v).y, m.vertex(v).z))
            }
            "move_handle" => {
                let i = index(self.handles.len())?;
                self.handles[i].target = DVec3::from_array(a.point("to")?);
                self.rebuild();
                Ok(format!("Handle {} moved; previewing", i + 1))
            }
            "remove_handle" => {
                let i = index(self.handles.len())?;
                self.handles.remove(i);
                self.rebuild();
                Ok(format!("Handle {} removed", i + 1))
            }
            "clear" => {
                self.handles.clear();
                self.rebuild();
                Ok("Handles cleared".into())
            }
            _ => Err(format!("Warp has no action {action}")),
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Warp");
        if self.object.is_none() || self.selection.is_empty() {
            ui.label("Select the region to warp first.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Right-click the selection to add a handle, drag a handle to warp, right-click a handle to remove it. The selection's border stays fixed.");
        if ui.checkbox(&mut self.rigid, "Preserve detail").on_hover_text("As-rigid-as-possible: surface detail keeps its shape (slower)").changed() {
            self.dirty = true;
        }
        let max = (self.source.bounds().diagonal() * 0.1).max(self.handle_size);
        if ui.add(egui::Slider::new(&mut self.handle_size, 0.0..=max).text("handle size")).on_hover_text("Radius of the patch each handle moves").changed() {
            for i in 0..self.handles.len() {
                self.handles[i].disc = self.disc(self.handles[i].vertex);
            }
            self.dirty = true;
        }
        ui.label(format!("{} handle(s)", self.handles.len()));
        if ui.add_enabled(!self.handles.is_empty(), egui::Button::new("Clear handles")).clicked() {
            self.handles.clear();
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild();
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, true);
        if accept && self.commit(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn secondary_click(&mut self, pos: egui::Pos2, cx: &mut ToolCx) {
        if let Some(i) = self.handle_at(cx, pos) {
            self.handles.remove(i);
            self.dirty = true;
        } else if let Some(v) = self.vertex_at(cx, pos) {
            if !self.handles.iter().any(|h| h.vertex == v) {
                let target = self.current().vertex(v);
                let disc = self.disc(v);
                self.handles.push(Handle { vertex: v, target, disc });
            }
        }
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if self.drag.is_none() {
            self.hover = self.handle_at(cx, pos);
        }
        if input.pressed {
            if let Some(i) = self.handle_at(cx, pos) {
                self.drag = Some((i, self.handles[i].target, pos));
            }
        }
        if input.down {
            if let Some((i, start, _)) = self.drag {
                // Move in the plane facing the camera through the handle.
                let ray = cx.ray_at(pos);
                let n = cx.camera.forward();
                let denom = ray.dir.dot(n);
                if denom.abs() > 1e-9 {
                    let t = (start - ray.origin).dot(n) / denom;
                    self.handles[i].target = ray.origin + ray.dir * t;
                    self.rebuild();
                }
            }
        }
        if input.released {
            self.drag = None;
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let active = self.drag.map(|d| d.0).or(self.hover);
        for (i, h) in self.handles.iter().enumerate() {
            if let Some(p) = cx.project(h.target) {
                let col = if active == Some(i) { egui::Color32::YELLOW } else { egui::Color32::from_rgb(225, 60, 50) };
                painter.circle(p, HANDLE_PX, col, egui::Stroke::new(1.5, egui::Color32::from_gray(20)));
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
