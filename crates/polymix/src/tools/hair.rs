//! Grow Hair (Select > Edit > Grow Hair): strands on the selected faces (see
//! [`mm_edit::hair`]), previewed live; Accept adds them to the object as tapered tubes in a
//! new face group. Modes: NumHairs (a fixed count at random), Fill Density (an even spread
//! at so many per mm²) and Hair Patch (guide strands with the rest gathered into clumps).

use super::*;
use crate::ui_util;
use mm_edit::hair::{fill_count, fill_spacing, HairMode, MAX_HAIRS};
use mm_edit::HairParams;

pub struct HairTool {
    object: Option<ObjectId>,
    tris: Vec<u32>,
    /// Area of the selected faces.
    area: f64,
    params: HairParams,
    show_tubes: bool,
    /// Strands for the current parameters.
    curves: Option<(HairParams, Vec<Vec<DVec3>>)>,
    /// Show Tubes: the tubes around the strands, kept while the strands are.
    tubes: Option<TubeOverlay>,
}

/// The triangles Show Tubes draws, made once per set of strands rather than every frame.
struct TubeOverlay {
    /// The parameters the strands were grown with.
    params: HairParams,
    tris: Vec<[DVec3; 3]>,
    normals: Vec<DVec3>,
    /// The triangles' colours, shaded toward the view direction they were made for.
    shaded: Option<(DVec3, Vec<Vec4>)>,
}

impl HairTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, tris, diag) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.selection.to_vec(), o.bounds().diagonal()),
            None => (None, Vec::new(), 1.0),
        };
        let area = cx.scene.active().map_or(0.0, |o| mm_edit::hair::selection_area(&o.mesh, &tris));
        let length = diag * 0.08;
        let down = match cx.camera.up_axis {
            crate::camera::UpAxis::Y => -DVec3::Y,
            crate::camera::UpAxis::Z => -DVec3::Z,
        };
        // Fill Density starts at the default count's density.
        let density = if area > 0.0 { 200.0 / area } else { 1.0 };
        let params = HairParams { length, curl_pitch: length / 3.0, base_radius: length * 0.02, tip_radius: length * 0.006, down, density, ..Default::default() };
        Self { object, tris, area, params, show_tubes: false, curves: None, tubes: None }
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
    fn settings(&self) -> Option<serde_json::Value> {
        serde_json::to_value(self.params).ok()
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(schema_of::<HairParams>())
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let down = self.params.down;
        self.params = HairParams { down, ..merged(&self.params, changes).map_err(|e| format!("Grow Hair: {e}"))? };
        self.refresh(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        serde_json::json!({
            "object": self.object,
            "on_selected_triangles": self.tris.len(),
            "strands": self.curves.as_ref().filter(|(p, _)| *p == self.params).map(|(_, c)| c.len()),
            "next": if self.tris.is_empty() { "select the triangles to grow hair on first (close this tool, mesh_select, open it again)" } else { "accept (mesh_tool accept) to add the strands as tubes" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![Action::plain("reseed", "Scatter the strands anew")]
    }

    fn act(&mut self, action: &str, _args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "reseed" => {
                let p = &mut self.params;
                p.seed = p.seed.wrapping_mul(2862933555777941757).wrapping_add(3037000493);
                self.refresh(cx);
                Ok("Reseeded".into())
            }
            _ => Err(format!("Grow Hair has no action {action}")),
        }
    }

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
        ui.horizontal(|ui| {
            ui.selectable_value(&mut p.mode, HairMode::Count, "NumHairs").on_hover_text("A fixed number of strands at random points");
            ui.selectable_value(&mut p.mode, HairMode::FillDensity, "Fill Density").on_hover_text("Strands spread evenly over the selection, so many per unit area");
            ui.selectable_value(&mut p.mode, HairMode::HairPatch, "Hair Patch").on_hover_text("A few guide strands; the others follow them and gather into clumps");
        });
        match p.mode {
            HairMode::Count => {
                ui.add(egui::Slider::new(&mut p.count, 1..=5000).logarithmic(true).text("NumHairs"));
            }
            HairMode::FillDensity => {
                let a = self.area.max(1e-12);
                ui.add(egui::Slider::new(&mut p.density, 1.0 / a..=MAX_HAIRS as f64 / a).logarithmic(true).text("Density").suffix(" /mm²")).on_hover_text("Strands per unit area (per mm² for a model in millimetres)");
                let n = fill_count(self.area, p.density);
                let capped = if n == MAX_HAIRS { format!(" (at most {})", ui_util::format_count(MAX_HAIRS)) } else { String::new() };
                ui.small(format!("≈ {} strands{capped}, no two closer than {:.3}", ui_util::format_count(n), fill_spacing(p.density)));
            }
            HairMode::HairPatch => {
                ui.add(egui::Slider::new(&mut p.count, 1..=5000).logarithmic(true).text("NumHairs")).on_hover_text("All strands, guides included");
                ui.add(egui::Slider::new(&mut p.clumps, 1..=200).logarithmic(true).text("Clumps")).on_hover_text("Guide strands, spread evenly; each gathers the strands around it");
                ui.add(egui::Slider::new(&mut p.clumping, 0.0..=1.0).text("Clumping")).on_hover_text("How far strands are drawn toward their guide, more toward the tips (1: the tips meet)");
            }
        }
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
            if self.tubes.as_ref().is_none_or(|t| t.params != *p) {
                let tubes = mm_edit::hair_mesh(curves, p.base_radius, p.tip_radius, 6);
                let ids: Vec<u32> = tubes.triangle_ids().collect();
                let tris = ids.iter().map(|&t| tubes.tri_positions(t)).collect();
                let normals = ids.iter().map(|&t| tubes.tri_normal(t)).collect();
                self.tubes = Some(TubeOverlay { params: *p, tris, normals, shaded: None });
            }
            let Some(tubes) = &mut self.tubes else { return };
            // Flat shading toward the viewer, again only when the view turns.
            let fwd = cx.camera.forward();
            if tubes.shaded.as_ref().is_none_or(|(f, _)| *f != fwd) {
                let shade = |n: &DVec3| {
                    let k = (0.35 + 0.65 * n.dot(-fwd).abs()) as f32;
                    Vec4::new(col.x * k, col.y * k, col.z * k, 1.0)
                };
                tubes.shaded = Some((fwd, tubes.normals.iter().map(shade).collect()));
            }
            let Some((_, colours)) = &tubes.shaded else { return };
            for ([a, b, c], col) in tubes.tris.iter().zip(colours) {
                ov.tri(*a, *b, *c, *col);
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

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        // `mode:count|fill|patch`, `count:300`, `density:0.5`, `clumps:8`, `clumping:0.9`,
        // `tubes:on`.
        let p = &mut self.params;
        match cmd.split_once(':') {
            Some(("mode", m)) => {
                p.mode = match m {
                    "fill" => HairMode::FillDensity,
                    "patch" => HairMode::HairPatch,
                    _ => HairMode::Count,
                }
            }
            Some(("count", n)) => p.count = n.parse().unwrap_or(p.count),
            Some(("density", d)) => p.density = d.parse().unwrap_or(p.density),
            Some(("clumps", n)) => p.clumps = n.parse().unwrap_or(p.clumps),
            Some(("clumping", c)) => p.clumping = c.parse().unwrap_or(p.clumping),
            Some(("tubes", t)) => self.show_tubes = t == "on",
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Show Tubes as it was drawn: the tubes made and shaded anew every frame.
    fn tubes_every_frame(curves: &[Vec<DVec3>], p: &HairParams, fwd: DVec3) -> Vec<([f32; 3], [f32; 4])> {
        let col = Vec4::new(0.95, 0.8, 0.45, 1.0);
        let tubes = mm_edit::hair_mesh(curves, p.base_radius, p.tip_radius, 6);
        let mut ov = Overlay::default();
        for t in tubes.triangle_ids() {
            let [a, b, c] = tubes.tri_positions(t);
            let k = (0.35 + 0.65 * tubes.tri_normal(t).dot(-fwd).abs()) as f32;
            ov.tri(a, b, c, Vec4::new(col.x * k, col.y * k, col.z * k, 1.0));
        }
        ov.tris.iter().map(|v| (v.position, v.color)).collect()
    }

    #[test]
    fn show_tubes_draws_what_it_drew_every_frame() {
        let mut scene = Scene::default();
        let id = scene.add("ball", mm_core::primitives::icosphere(10.0, 3));
        scene.active = Some(id);
        let o = scene.get_mut(id).unwrap();
        let top: Vec<u32> = o.mesh.triangle_ids().filter(|&t| o.mesh.tri_centroid(t).z > 7.0).collect();
        for t in top {
            o.selection.insert(t);
        }
        let (mut history, mut camera, mut status, ctx) = (History::default(), crate::camera::Camera::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = HairTool::new(&mut cx);
        tool.params.count = 40;
        tool.show_tubes = true;
        let check = |tool: &mut HairTool, cx: &mut ToolCx| {
            tool.refresh(cx);
            let mut ov = Overlay::default();
            tool.overlay(&mut ov, cx);
            let (p, curves) = tool.curves.as_ref().unwrap();
            let drawn: Vec<([f32; 3], [f32; 4])> = ov.tris.iter().map(|v| (v.position, v.color)).collect();
            assert!(!drawn.is_empty());
            assert_eq!(drawn, tubes_every_frame(curves, p, cx.camera.forward()));
        };
        check(&mut tool, &mut cx);
        check(&mut tool, &mut cx);
        // Another view direction reshades them; other strands make other tubes.
        cx.camera.look_along(DVec3::new(1.0, -0.5, -0.3).normalize());
        check(&mut tool, &mut cx);
        tool.params.base_radius *= 1.7;
        check(&mut tool, &mut cx);
        tool.params.count = 25;
        check(&mut tool, &mut cx);
    }
}
