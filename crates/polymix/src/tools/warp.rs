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
    /// Vertices the preview (a copy of the source, rewritten in place) has moved; the next
    /// rebuild puts them back first.
    moved: Vec<u32>,
    /// The BVH over the source (the object's own when it still has the source), and a copy of
    /// it refitted to the preview for picking on the warped surface. The copy lives as long as
    /// the preview: neither is dropped before the tool closes (accepting always closes it).
    bvh: Option<Arc<mm_core::Bvh>>,
    preview_bvh: Option<mm_core::Bvh>,
    moves: PreviewMoves,
}

const HANDLE_PX: f32 = 7.0;

impl WarpTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, selection) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.mesh.clone(), o.selection.iter().collect()),
            None => (None, Arc::new(Mesh::new()), Vec::new()),
        };
        let handle_size = 3.0 * source.average_edge_length();
        Self {
            object,
            source,
            selection,
            handles: Vec::new(),
            rigid: true,
            handle_size,
            preview: None,
            drag: None,
            hover: None,
            dirty: false,
            moved: Vec::new(),
            bvh: None,
            preview_bvh: None,
            moves: PreviewMoves::default(),
        }
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
        let moves = mm_edit::warp(&self.source, &self.selection, &handles, self.rigid);
        // The preview is a copy of the source made once; each rebuild puts back what the last
        // one moved and writes the new positions.
        if self.preview.is_none() {
            self.preview = Some((*self.source).clone());
            self.moved.clear();
            self.moves.reset();
        }
        let m = self.preview.as_mut().unwrap();
        let before = m.shape_version();
        let pos = m.positions_mut();
        for &v in &self.moved {
            pos[v as usize] = self.source.vertex(v);
        }
        for &(v, p) in &moves {
            pos[v as usize] = p;
        }
        m.bump_shape_version();
        let (after, limit) = (m.shape_version(), m.max_vid());
        let written = std::mem::replace(&mut self.moved, moves.iter().map(|x| x.0).collect());
        self.moves.record(before, after, written.into_iter().chain(moves.iter().map(|x| x.0)), limit);
        self.dirty = false;
    }

    /// The BVH over the source: the object's own while it still has the source mesh.
    fn source_bvh(&mut self, cx: &mut ToolCx) -> Arc<mm_core::Bvh> {
        if let Some(b) = &self.bvh {
            return b.clone();
        }
        let b = match self.object.and_then(|id| cx.scene.get_mut(id)) {
            Some(o) if Arc::ptr_eq(&o.mesh, &self.source) => o.bvh(),
            _ => Arc::new(mm_core::Bvh::build(&self.source)),
        };
        self.bvh = Some(b.clone());
        b
    }

    /// A BVH matching the preview: the source's refitted to the moved positions (same
    /// triangles), redone only when the preview changed.
    fn preview_bvh(&mut self, cx: &mut ToolCx) {
        let Some(m) = &self.preview else { return };
        if self.preview_bvh.as_ref().is_some_and(|b| b.is_current(m)) {
            return;
        }
        let mut b = match self.preview_bvh.take() {
            Some(b) if b.can_refit(m) => b,
            _ => {
                let s = self.source_bvh(cx);
                let m = self.preview.as_ref().unwrap();
                if s.can_refit(m) { (*s).clone() } else { mm_core::Bvh::build(m) }
            }
        };
        let m = self.preview.as_ref().unwrap();
        if !b.is_current(m) {
            b.refit(m);
        }
        self.preview_bvh = Some(b);
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
    fn vertex_at(&mut self, cx: &mut ToolCx, p: egui::Pos2) -> Option<u32> {
        let ray = cx.ray_at(p);
        let hit = if self.preview.is_some() {
            self.preview_bvh(cx);
            self.preview_bvh.as_ref()?.raycast(self.preview.as_ref()?, &ray)?
        } else {
            self.source_bvh(cx).raycast(&self.source, &ray)?
        };
        let m = self.current();
        let selected: mm_core::hash::FastSet<u32> = self.selection.iter().copied().collect();
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
                    let target = ray.origin + ray.dir * t;
                    // Holding the handle still changes nothing.
                    if target != self.handles[i].target || self.dirty || self.preview.is_none() {
                        self.handles[i].target = target;
                        self.rebuild();
                    }
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

    fn preview_moves(&self) -> Option<(u64, u64, &[u32])> {
        self.moves.get()
    }

    fn clear_preview_moves(&mut self) {
        let now = self.preview.as_ref().map(|m| m.shape_version());
        self.moves.clear(now);
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.commit(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn bits(p: DVec3) -> [u64; 3] {
        p.to_array().map(f64::to_bits)
    }

    /// The preview as the tool used to build it: a fresh copy of the source per rebuild.
    fn fresh_preview(t: &WarpTool) -> Mesh {
        let src = &t.source;
        let mut handles: Vec<(u32, DVec3)> = Vec::new();
        let mut taken = std::collections::HashSet::new();
        for h in &t.handles {
            let offset = h.target - src.vertex(h.vertex);
            for &v in &h.disc {
                if taken.insert(v) {
                    handles.push((v, src.vertex(v) + offset));
                }
            }
        }
        let mut m = (**src).clone();
        for (v, p) in mm_edit::warp(src, &t.selection, &handles, t.rigid) {
            m.set_vertex(v, p);
        }
        m
    }

    #[test]
    fn previews_and_picking_follow_the_warp() {
        let grid = mm_core::primitives::plane_grid(4.0, 4.0, 40, 40);
        let tris: Vec<u32> = grid.triangle_ids().filter(|&t| grid.tri_centroid(t).truncate().length() < 1.5).collect();
        let mut scene = Scene::default();
        let id = scene.add("grid", grid.clone());
        scene.get_mut(id).unwrap().selection = mm_core::TriSelection::from_tris(&grid, tris);
        scene.active = Some(id);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut t = WarpTool::new(&mut cx);
        t.act("add_handle", &json!({"at": [0.0, 0.0, 0.0]}), &mut cx).unwrap();
        t.act("add_handle", &json!({"at": [0.6, 0.4, 0.0]}), &mut cx).unwrap();
        t.rebuild();
        t.clear_preview_moves();
        let mut shadow = t.preview.as_ref().unwrap().positions().to_vec();
        let steps = [
            json!({"handle": 1, "to": [0.1, 0.0, 0.8]}),
            json!({"handle": 2, "to": [0.6, 0.5, -0.3]}),
            json!({"handle": 1, "to": [0.2, -0.1, 0.5]}),
        ];
        for (k, step) in steps.iter().enumerate() {
            t.rigid = k != 1;
            let from = t.preview.as_ref().unwrap().shape_version();
            t.act("move_handle", step, &mut cx).unwrap();
            let preview = t.preview.as_ref().unwrap();
            assert!(preview.positions().iter().zip(fresh_preview(&t).positions()).all(|(a, b)| bits(*a) == bits(*b)), "move {k}");
            let (f, to, verts) = t.preview_moves().unwrap();
            assert_eq!((f, to), (from, preview.shape_version()));
            for &v in verts {
                shadow[v as usize] = preview.vertex(v);
            }
            assert!(shadow.iter().zip(preview.positions()).all(|(a, b)| bits(*a) == bits(*b)), "move {k}");
            t.clear_preview_moves();
        }
        // Picking on the warped surface: the object's tree refitted to it, as good as a new one.
        t.preview_bvh(&mut cx);
        let preview = t.preview.as_ref().unwrap();
        let fresh = mm_core::Bvh::build(preview);
        let refitted = t.preview_bvh.as_ref().unwrap();
        assert!(refitted.is_current(preview));
        for i in 0..15 {
            for j in 0..15 {
                let ray = Ray::new(DVec3::new(-1.4 + 0.2 * i as f64, -1.4 + 0.2 * j as f64, 3.0), DVec3::new(0.05, -0.02, -1.0).normalize());
                let (a, b) = (refitted.raycast(preview, &ray), fresh.raycast(preview, &ray));
                assert_eq!(a.map(|h| (h.tid, h.t.to_bits())), b.map(|h| (h.tid, h.t.to_bits())));
            }
        }
        assert!(Arc::ptr_eq(t.bvh.as_ref().unwrap(), &cx.scene.get_mut(id).unwrap().bvh()), "the object's own tree");
        // Removing every handle puts the source back.
        t.act("clear", &json!({}), &mut cx).unwrap();
        assert!(t.preview.as_ref().unwrap().positions().iter().zip(grid.positions()).all(|(a, b)| bits(*a) == bits(*b)));
    }

    #[test]
    fn a_handle_held_still_does_not_rebuild() {
        let grid = mm_core::primitives::plane_grid(4.0, 4.0, 20, 20);
        let tris: Vec<u32> = grid.triangle_ids().collect();
        let mut scene = Scene::default();
        let id = scene.add("grid", grid.clone());
        scene.get_mut(id).unwrap().selection = mm_core::TriSelection::from_tris(&grid, tris);
        scene.active = Some(id);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut t = WarpTool::new(&mut cx);
        t.act("add_handle", &json!({"at": [0.0, 0.0, 0.0]}), &mut cx).unwrap();
        t.rebuild();
        let start = cx.project(t.handles[0].target).unwrap();
        let held = PointerInput { pos: Some(start + egui::vec2(20.0, -30.0)), down: true, ..Default::default() };
        t.pointer(&PointerInput { pos: Some(start), pressed: true, down: true, ..Default::default() }, &mut cx);
        t.pointer(&held, &mut cx);
        let version = t.preview.as_ref().unwrap().shape_version();
        t.pointer(&held, &mut cx);
        assert_eq!(t.preview.as_ref().unwrap().shape_version(), version);
    }
}
