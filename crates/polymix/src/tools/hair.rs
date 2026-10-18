//! Grow Hair (Select > Edit > Grow Hair): strands on the selected faces (see
//! [`mm_edit::hair`]), previewed live; Accept adds them to the object as tapered tubes in a
//! new face group.

use super::*;
use crate::ui_util;
use mm_edit::HairParams;

pub struct HairTool {
    object: Option<ObjectId>,
    tris: Vec<u32>,
    params: HairParams,
    show_tubes: bool,
    /// Strands for the current parameters.
    curves: Option<(HairParams, Vec<Vec<DVec3>>)>,
}

impl HairTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, tris, diag) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.selection.to_vec(), o.bounds().diagonal()),
            None => (None, Vec::new(), 1.0),
        };
        let length = diag * 0.08;
        let down = match cx.camera.up_axis {
            crate::camera::UpAxis::Y => -DVec3::Y,
            crate::camera::UpAxis::Z => -DVec3::Z,
        };
        let params = HairParams { length, curl_pitch: length / 3.0, base_radius: length * 0.02, tip_radius: length * 0.006, down, ..Default::default() };
        Self { object, tris, params, show_tubes: false, curves: None }
    }

    fn refresh(&mut self, cx: &mut ToolCx) {
        if self.curves.as_ref().is_some_and(|(p, _)| *p == self.params) {
            return;
        }
        let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) else { return };
        let bvh = o.bvh();
        self.curves = Some((self.params, mm_edit::hair_curves(&o.mesh, &bvh, &self.tris, &self.params)));
    }
}

impl Tool for HairTool {
    fn name(&self) -> &'static str {
        "Grow Hair"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Grow Hair");
        if self.tris.is_empty() {
            ui.label("Select the faces to grow hair on first.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let p = &mut self.params;
        let len = p.length.max(1e-9);
        ui.add(egui::Slider::new(&mut p.count, 1..=5000).logarithmic(true).text("NumHairs"));
        ui.add(egui::Slider::new(&mut p.points, 2..=64).text("Points"));
        ui.add(egui::Slider::new(&mut p.length, len * 0.1..=len * 5.0).logarithmic(true).text("Length"));
        ui.add(egui::Slider::new(&mut p.curl_radius, 0.0..=len * 0.3).text("Curl Radius"));
        ui.add(egui::Slider::new(&mut p.curl_pitch, len * 0.02..=len * 2.0).logarithmic(true).text("Curl Pitch"));
        ui.add(egui::Slider::new(&mut p.hair_stiffness, 0.0..=1.0).text("HairStiff")).on_hover_text("Stiffness at the root");
        ui.add(egui::Slider::new(&mut p.tip_stiffness, 0.0..=1.0).text("TipStiff"));
        ui.add(egui::Slider::new(&mut p.gravity, 0.0..=1.0).text("Gravity"));
        ui.add(egui::Slider::new(&mut p.base_radius, len * 0.001..=len * 0.2).logarithmic(true).text("Base Radius"));
        ui.add(egui::Slider::new(&mut p.tip_radius, len * 0.001..=len * 0.2).logarithmic(true).text("Tip Radius"));
        ui.horizontal(|ui| {
            ui.checkbox(&mut p.collisions, "Obj Collisions").on_hover_text("Keep strands outside the object");
            ui.checkbox(&mut self.show_tubes, "Show Tubes");
            if ui.button("Reseed").clicked() {
                p.seed = p.seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
            }
        });
        self.refresh(cx);
        let n = self.curves.as_ref().map_or(0, |c| c.1.len());
        ui.small(format!("{n} strands on {} faces. Accept adds them as tubes in a new face group.", self.tris.len()));
        let (accept, cancel) = ui_util::accept_cancel(ui, n > 0);
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        let Some((p, curves)) = &self.curves else { return };
        let col = Vec4::new(0.95, 0.8, 0.45, 1.0);
        if self.show_tubes && curves.len() * p.points <= 40_000 {
            let tubes = mm_edit::hair_mesh(curves, p.base_radius, p.tip_radius, 6);
            let fwd = cx.camera.forward();
            for t in tubes.triangle_ids() {
                let [a, b, c] = tubes.tri_positions(t);
                let k = (0.35 + 0.65 * tubes.tri_normal(t).dot(-fwd).abs()) as f32;
                ov.tri(a, b, c, Vec4::new(col.x * k, col.y * k, col.z * k, 1.0));
            }
        } else {
            for c in curves {
                for w in c.windows(2) {
                    ov.line(w[0], w[1], col);
                }
            }
        }
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        let Some((p, curves)) = self.curves.clone() else { return true };
        if curves.is_empty() {
            return true;
        }
        let tubes = mm_edit::hair_mesh(&curves, p.base_radius, p.tip_radius, 6);
        cx.history.push("Grow Hair", cx.scene);
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            let g = o.mesh.max_group_id() + 1;
            o.mesh_mut().append_mesh(&tubes, g);
            let m = o.mesh.clone();
            o.selection.sync(&m);
            o.touch_selection();
        }
        *cx.status = format!("Grew {} strands", curves.len());
        true
    }
}
