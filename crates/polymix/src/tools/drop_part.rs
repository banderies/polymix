//! Drop Part (Meshmix): place a library part on the model — press where it should go and
//! drag to size it — adjust it, then Drop (Enter / A). Open parts are welded into the
//! surface (following its curvature unless that is turned off); solid parts are combined
//! by a boolean, appended, or added as a new object.

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use mm_edit::{Part, PartKind, Placement, StampFrame};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Combine {
    Union,
    Difference,
    Append,
    NewObject,
}

impl Combine {
    const ALL: [Combine; 4] = [Combine::Union, Combine::Difference, Combine::Append, Combine::NewObject];
    fn label(self) -> &'static str {
        match self {
            Combine::Union => "Boolean Union",
            Combine::Difference => "Boolean Difference",
            Combine::Append => "Append",
            Combine::NewObject => "New Object",
        }
    }
}

struct Spot {
    object: ObjectId,
    seed: u32,
    point: DVec3,
    normal: DVec3,
    /// Screen up when the part was placed (the part's in-plane orientation follows it).
    up: DVec3,
}

pub struct DropPartTool {
    part: Arc<Part>,
    spot: Option<Spot>,
    dragging: bool,
    /// Rim / bounding radius in model units.
    size: f64,
    rotation: f64,
    offset: f64,
    follow: bool,
    gap: f64,
    combine: Combine,
    /// Placed part for the preview, and the placement it was made for.
    preview: Option<(Placement, Mesh)>,
    error: Option<String>,
    /// A boolean drop, computing (no other part is dropped meanwhile).
    dropping: Option<Dropping>,
}

/// A part being combined with the object by a boolean, in the background.
struct Dropping {
    object: ObjectId,
    /// The object's mesh the boolean started from: the result applies only to it.
    source: Arc<Mesh>,
    label: String,
    /// Dropped with Enter (or Done pressed meanwhile): the tool closes once it is in.
    close: bool,
    job: Job<Mesh>,
}

impl DropPartTool {
    pub fn new(cx: &mut ToolCx, part: Arc<Part>) -> Self {
        let size = cx.scene.active().map_or(1.0, |o| o.bounds().diagonal() * 0.08);
        Self { part, spot: None, dragging: false, size, rotation: 0.0, offset: 0.0, follow: true, gap: 0.15, combine: Combine::Union, preview: None, error: None, dropping: None }
    }

    fn placement(&self) -> Option<Placement> {
        let s = self.spot.as_ref()?;
        let (sin, cos) = self.rotation.to_radians().sin_cos();
        let f = StampFrame::new(s.point, s.normal, s.up);
        let up = f.v * cos - f.u * sin;
        Some(Placement { frame: StampFrame::new(s.point, s.normal, up), scale: self.size, offset: self.offset, follow: self.follow })
    }

    fn refresh_preview(&mut self, cx: &mut ToolCx) {
        let Some(at) = self.placement() else {
            self.preview = None;
            return;
        };
        if self.preview.as_ref().is_some_and(|(p, _)| *p == at) {
            return;
        }
        let Some(o) = self.spot.as_ref().and_then(|s| cx.scene.get_mut(s.object)) else { return };
        let bvh = o.bvh();
        self.preview = Some((at, mm_edit::place_part(&self.part, &at, Some((&o.mesh, &bvh)))));
    }

    /// Drop the part. Returns whether it went in; false also while a boolean drop computes
    /// (it goes in when done, see [`DropPartTool::poll`]), and `close` then closes the tool
    /// afterwards.
    fn drop_part(&mut self, cx: &mut ToolCx, close: bool) -> bool {
        if let Some(d) = &mut self.dropping {
            d.close |= close;
            return false;
        }
        let (Some(spot), Some(at)) = (self.spot.as_ref(), self.placement()) else { return false };
        let (object, seed) = (spot.object, spot.seed);
        let label = format!("Drop {}", self.part.name);
        let result = match self.part.kind {
            PartKind::Open => {
                let Some(o) = cx.scene.get_mut(object) else { return false };
                // The tree the preview was placed with, rather than one built per drop.
                let bvh = o.bvh();
                let mut m = (*o.mesh).clone();
                mm_edit::part::drop_open_part_with(&mut m, &bvh, seed, &self.part, &at, self.gap).map(|_| m)
            }
            PartKind::Solid => {
                let Some(o) = cx.scene.get(object) else { return false };
                let placed = mm_edit::place_part(&self.part, &at, None);
                match self.combine {
                    Combine::NewObject => {
                        cx.history.push(label, cx.scene);
                        let id = cx.scene.add(self.part.name.clone(), placed);
                        cx.scene.active = Some(id);
                        *cx.status = format!("Dropped {} as a new object", self.part.name);
                        self.spot = None;
                        self.preview = None;
                        return true;
                    }
                    Combine::Append => {
                        let mut m = (*o.mesh).clone();
                        m.append_mesh(&placed, m.max_group_id() + 1);
                        Ok(m)
                    }
                    Combine::Union | Combine::Difference => {
                        let op = if self.combine == Combine::Union { mm_edit::BooleanOp::Union } else { mm_edit::BooleanOp::Difference };
                        let source = o.mesh.clone();
                        let a = source.clone();
                        let job = Job::spawn(label.clone(), cx.egui.clone(), move |_| mm_edit::mesh_boolean(&a, &placed, op).map(|r| r.mesh).map_err(|e| anyhow::anyhow!("{e}")));
                        self.error = None;
                        self.dropping = Some(Dropping { object, source, label, close, job });
                        return false;
                    }
                }
            }
        };
        self.finish(cx, object, label, result.map_err(|e| e.to_string()))
    }

    /// Put a dropped part's result in (or show why there is none). Returns whether it went in.
    fn finish(&mut self, cx: &mut ToolCx, object: ObjectId, label: String, result: Result<Mesh, String>) -> bool {
        match result {
            Ok(m) => {
                cx.history.push(label, cx.scene);
                if let Some(o) = cx.scene.get_mut(object) {
                    o.set_mesh(m);
                }
                *cx.status = format!("Dropped {}", self.part.name);
                self.error = None;
                self.spot = None;
                self.preview = None;
                true
            }
            Err(e) => {
                self.error = Some(e);
                false
            }
        }
    }

    /// Put a boolean drop in once it is done, if the object still has the mesh it was
    /// computed from. Returns true when the tool should close now.
    fn poll(&mut self, cx: &mut ToolCx) -> bool {
        let Some(r) = self.dropping.as_ref().and_then(|d| d.job.poll()) else { return false };
        let Some(d) = self.dropping.take() else { return false };
        if !cx.scene.get(d.object).is_some_and(|o| Arc::ptr_eq(&o.mesh, &d.source)) {
            self.error = Some("The object changed while the part was combined with it; drop it again.".into());
            return false;
        }
        self.finish(cx, d.object, d.label, r.map_err(|e| e.to_string())) && d.close
    }
}

impl Tool for DropPartTool {
    fn name(&self) -> &'static str {
        "Drop Part"
    }

    fn shows_groups(&self) -> bool {
        self.part.kind == PartKind::Open
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.dropping.as_ref().map(|d| (d.label.clone(), d.job.progress()))
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        if self.poll(cx) {
            return ToolOutcome::Close;
        }
        ui_util::section(ui, &format!("Drop Part: {}", self.part.name));
        ui.label(if self.part.kind == PartKind::Open { "Open part — welded into the surface" } else { "Solid part" });
        let span = cx.scene.active().map_or(10.0, |o| o.bounds().diagonal());
        // While a part is combined with the object, the next one waits.
        let dropping = self.dropping.is_some();
        ui.add_enabled_ui(!dropping, |ui| {
            ui.add(egui::Slider::new(&mut self.size, span * 0.005..=span * 0.5).logarithmic(true).text("size").max_decimals(3));
            ui.add(egui::Slider::new(&mut self.rotation, -180.0..=180.0).text("rotation").suffix("°"));
            match self.part.kind {
                PartKind::Open => {
                    ui.checkbox(&mut self.follow, "Follow surface").on_hover_text("Keep the part's shape relative to the surface under it (otherwise it is placed rigidly)");
                    ui.add(egui::Slider::new(&mut self.gap, 0.05..=0.5).text("blend width")).on_hover_text("Width of the band joining the part to the surface, as a fraction of its size");
                }
                PartKind::Solid => {
                    ui.add(egui::Slider::new(&mut self.offset, -1.0..=1.0).text("offset")).on_hover_text("Move the part along the normal (part radii)");
                    egui::ComboBox::from_label("Combine").selected_text(self.combine.label()).show_ui(ui, |ui| {
                        for c in Combine::ALL {
                            ui.selectable_value(&mut self.combine, c, c.label());
                        }
                    });
                }
            }
        });
        ui.small("Press on the model where the part goes and drag to size it. Drop (Enter) adds it; you can then place another.");
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        if dropping {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Dropping…");
            });
        }
        ui.add_space(6.0);
        let mut out = ToolOutcome::Continue;
        ui.horizontal(|ui| {
            if ui.add_enabled(self.spot.is_some() && !dropping, egui::Button::new("Drop")).clicked() {
                self.drop_part(cx, false);
            }
            if ui.button("Done").clicked() {
                // A part being dropped goes in first.
                match &mut self.dropping {
                    Some(d) => d.close = true,
                    None => out = ToolOutcome::Close,
                }
            }
        });
        self.refresh_preview(cx);
        out
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            if let Some((id, h)) = cx.pick(pos, cx.scene.active) {
                let o = cx.scene.get(id).unwrap();
                let n = o.mesh.triangle(h.tid).iter().map(|&v| o.mesh.vertex_normal(v)).sum::<DVec3>().normalize_or(o.mesh.tri_normal(h.tid));
                self.spot = Some(Spot { object: id, seed: h.tid, point: h.point, normal: n, up: cx.camera.up() });
                self.dragging = true;
                self.error = None;
            }
        }
        if self.dragging && input.down && !input.pressed {
            if let Some(s) = &self.spot {
                let ray = cx.ray_at(pos);
                let denom = ray.dir.dot(s.normal);
                if denom.abs() > 1e-9 {
                    let t = (s.point - ray.origin).dot(s.normal) / denom;
                    let d = ray.origin + ray.dir * t - s.point;
                    // Small wiggles keep the current size.
                    if d.length() > cx.world_per_point(s.point) * 6.0 {
                        self.size = d.length();
                    }
                }
            }
        }
        if input.released {
            self.dragging = false;
        }
        self.refresh_preview(cx);
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        let Some((_, m)) = &self.preview else { return };
        // Flat shading toward the viewer.
        let fwd = cx.camera.forward();
        let base = if self.part.kind == PartKind::Open { Vec4::new(1.0, 0.85, 0.55, 1.0) } else { Vec4::new(0.6, 0.8, 1.0, 1.0) };
        for t in m.triangle_ids() {
            let [a, b, c] = m.triangle(t).map(|v| m.vertex(v));
            let k = (0.35 + 0.65 * m.tri_normal(t).dot(-fwd).abs()) as f32;
            ov.tri(a, b, c, Vec4::new(base.x * k, base.y * k, base.z * k, 1.0));
        }
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.spot.is_some() || self.dropping.is_some() {
            return self.drop_part(cx, true);
        }
        true
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        // `combine:append`, `follow:off`, `rotate:30`, `size:4`.
        match cmd.split_once(':') {
            Some(("combine", c)) => self.combine = Combine::ALL.into_iter().find(|x| x.label().to_lowercase().contains(c)).unwrap_or(self.combine),
            Some(("follow", f)) => self.follow = f != "off",
            Some(("rotate", a)) => self.rotation = a.parse().unwrap_or(self.rotation),
            Some(("size", s)) => self.size = s.parse().unwrap_or(self.size),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// A sphere of radius 10 in a scene, with the tool open for the built-in part `name`,
    /// placed on top of the sphere.
    fn with_tool(name: &str, test: impl FnOnce(&mut DropPartTool, &mut ToolCx, ObjectId)) {
        let mut scene = Scene::default();
        let id = scene.add("ball", primitives::icosphere(10.0, 4));
        scene.active = Some(id);
        let (mut history, mut camera, mut status, ctx) = (History::default(), crate::camera::Camera::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let part = Arc::new(mm_edit::builtin_parts().into_iter().find(|p| p.name == name).unwrap());
        let mut tool = DropPartTool::new(&mut cx, part);
        let m = cx.scene.get(id).unwrap().mesh.clone();
        let seed = m.triangle_ids().max_by(|&a, &b| m.tri_centroid(a).z.total_cmp(&m.tri_centroid(b).z)).unwrap();
        tool.spot = Some(Spot { object: id, seed, point: m.tri_centroid(seed), normal: DVec3::Z, up: DVec3::Y });
        tool.size = 3.0;
        test(&mut tool, &mut cx, id);
    }

    /// Draw the panel while a drop computes: true if the panel closed the tool.
    fn run_panel(tool: &mut DropPartTool, cx: &mut ToolCx) -> bool {
        let ctx = cx.egui.clone();
        let start = std::time::Instant::now();
        while tool.dropping.is_some() {
            assert!(start.elapsed().as_secs() < 60, "the drop finishes");
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
            let mut closed = false;
            let _ = ctx.run_ui(raw, |ui| closed = matches!(tool.panel(ui, cx), ToolOutcome::Close));
            if closed {
                return true;
            }
            std::thread::sleep(std::time::Duration::from_millis(2));
        }
        false
    }

    #[test]
    fn a_boolean_drop_goes_in_from_the_background() {
        with_tool("Capsule", |tool, cx, id| {
            let source = cx.scene.get(id).unwrap().mesh.clone();
            let placed = mm_edit::place_part(&tool.part, &tool.placement().unwrap(), None);
            let want = mm_edit::mesh_boolean(&source, &placed, mm_edit::BooleanOp::Union).unwrap().mesh;
            // Enter: computed in the background, then applied, and the tool closes.
            assert!(!tool.accept(cx));
            assert!(tool.busy().is_some());
            assert!(!tool.drop_part(cx, false), "no second drop meanwhile");
            assert!(cx.history.labels().0.is_empty());
            assert!(run_panel(tool, cx));
            assert_eq!(cx.history.labels().0, vec!["Drop Capsule"]);
            assert_eq!(cx.scene.get(id).unwrap().mesh.to_arrays(), want.to_arrays());
            assert!(tool.spot.is_none() && tool.error.is_none());
        });
        // The Drop button keeps the tool open for the next part.
        with_tool("Capsule", |tool, cx, id| {
            tool.combine = Combine::Difference;
            assert!(!tool.drop_part(cx, false));
            assert!(!run_panel(tool, cx));
            assert_eq!(cx.history.labels().0, vec!["Drop Capsule"]);
            assert!(mm_core::measure::volume(&cx.scene.get(id).unwrap().mesh) < mm_core::measure::volume(&primitives::icosphere(10.0, 4)));
        });
    }

    #[test]
    fn an_open_part_drops_with_the_objects_tree() {
        with_tool("Dome", |tool, cx, id| {
            let source = cx.scene.get(id).unwrap().mesh.clone();
            let mut want = (*source).clone();
            let spot = tool.spot.as_ref().unwrap();
            mm_edit::drop_open_part(&mut want, spot.seed, &tool.part, &tool.placement().unwrap(), tool.gap).unwrap();
            assert!(tool.drop_part(cx, false), "open parts go in at once");
            assert_eq!(cx.history.labels().0, vec!["Drop Dome"]);
            assert_eq!(cx.scene.get(id).unwrap().mesh.to_arrays(), want.to_arrays());
        });
    }
}
