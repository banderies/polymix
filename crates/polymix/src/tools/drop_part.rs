//! Drop Part (Meshmix): place a library part on the model — press where it should go and
//! drag to size it — adjust it, then Drop (Enter / A). Open parts are welded into the
//! surface (following its curvature unless that is turned off); solid parts are combined
//! by a boolean, appended, or added as a new object.

use super::*;
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
}

impl DropPartTool {
    pub fn new(cx: &mut ToolCx, part: Arc<Part>) -> Self {
        let size = cx.scene.active().map_or(1.0, |o| o.bounds().diagonal() * 0.08);
        Self { part, spot: None, dragging: false, size, rotation: 0.0, offset: 0.0, follow: true, gap: 0.15, combine: Combine::Union, preview: None, error: None }
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

    /// Drop the part. Returns whether it went in.
    fn drop_part(&mut self, cx: &mut ToolCx) -> bool {
        let (Some(spot), Some(at)) = (self.spot.as_ref(), self.placement()) else { return false };
        let Some(o) = cx.scene.get(spot.object) else { return false };
        let label = format!("Drop {}", self.part.name);
        let result = match self.part.kind {
            PartKind::Open => {
                let mut m = (*o.mesh).clone();
                mm_edit::drop_open_part(&mut m, spot.seed, &self.part, &at, self.gap).map(|_| m)
            }
            PartKind::Solid => {
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
                        mm_edit::mesh_boolean(&o.mesh, &placed, op).map(|r| r.mesh)
                    }
                }
            }
        };
        match result {
            Ok(m) => {
                cx.history.push(label, cx.scene);
                if let Some(o) = cx.scene.get_mut(spot.object) {
                    o.set_mesh(m);
                }
                *cx.status = format!("Dropped {}", self.part.name);
                self.error = None;
                self.spot = None;
                self.preview = None;
                true
            }
            Err(e) => {
                self.error = Some(e.to_string());
                false
            }
        }
    }
}

impl Tool for DropPartTool {
    fn name(&self) -> &'static str {
        "Drop Part"
    }

    fn shows_groups(&self) -> bool {
        self.part.kind == PartKind::Open
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, &format!("Drop Part: {}", self.part.name));
        ui.label(if self.part.kind == PartKind::Open { "Open part — welded into the surface" } else { "Solid part" });
        let span = cx.scene.active().map_or(10.0, |o| o.bounds().diagonal());
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
        ui.small("Press on the model where the part goes and drag to size it. Drop (Enter) adds it; you can then place another.");
        if let Some(e) = &self.error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        ui.add_space(6.0);
        let mut out = ToolOutcome::Continue;
        ui.horizontal(|ui| {
            if ui.add_enabled(self.spot.is_some(), egui::Button::new("Drop")).clicked() {
                self.drop_part(cx);
            }
            if ui.button("Done").clicked() {
                out = ToolOutcome::Close;
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
        if self.spot.is_some() {
            return self.drop_part(cx);
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
