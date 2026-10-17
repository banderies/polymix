//! Surface sculpting brushes (Meshmixer "Sculpt > Brushes"): smooth, inflate, flatten,
//! drag. Brushes move vertices only, so the picking BVH is refitted rather than rebuilt.

use super::*;
use crate::ui_util;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brush {
    Smooth,
    RobustSmooth,
    Inflate,
    Flatten,
    Drag,
}

impl Brush {
    fn label(self) -> &'static str {
        match self {
            Brush::Smooth => "Smooth",
            Brush::RobustSmooth => "Robust smooth",
            Brush::Inflate => "Inflate",
            Brush::Flatten => "Flatten",
            Brush::Drag => "Drag",
        }
    }
}

pub struct SculptTool {
    brush: Brush,
    /// Brush radius in percent of the model size.
    size_pct: f32,
    strength: f64,
    stroke: Option<SculptStroke>,
    armed: bool,
    hover: Option<(DVec3, f64)>,
    /// Keep open-boundary vertices fixed.
    hold_boundary: bool,
}

struct SculptStroke {
    object: ObjectId,
    invert: bool,
    // Drag brush state.
    drag_verts: Vec<(u32, f64)>,
    drag_origin: Option<(DVec3, egui::Pos2)>,
    drag_start_pos: Vec<DVec3>,
}

impl SculptTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        Self { brush: Brush::RobustSmooth, size_pct: 4.0, strength: 0.5, stroke: None, armed: false, hover: None, hold_boundary: true }
    }
}

#[inline]
fn falloff(d: f64, r: f64) -> f64 {
    let x = (d / r).min(1.0);
    let t = 1.0 - x * x;
    t * t
}

/// Vertices within `r` of `c`, with falloff weights.
fn brush_vertices(mesh: &Mesh, bvh: &mm_core::Bvh, c: DVec3, r: f64, hold_boundary: bool) -> Vec<(u32, f64)> {
    let mut seen = std::collections::HashSet::new();
    let mut out = Vec::new();
    bvh.query_sphere(mesh, c, r, |t| {
        for v in mesh.triangle(t) {
            if seen.insert(v) {
                let d = mesh.vertex(v).distance(c);
                if d < r && !(hold_boundary && mesh.is_boundary_vertex(v)) {
                    out.push((v, falloff(d, r)));
                }
            }
        }
    });
    out
}

fn apply_dab(mesh: &mut Mesh, verts: &[(u32, f64)], brush: Brush, strength: f64, r: f64, invert: bool, c: DVec3) {
    let m: &Mesh = mesh;
    let updates: Vec<(u32, DVec3)> = match brush {
        Brush::Smooth | Brush::RobustSmooth => verts
            .par_iter()
            .map(|&(v, w)| {
                let p = m.vertex(v);
                let mut avg = DVec3::ZERO;
                let mut n = 0.0;
                for u in m.vertex_neighbors(v) {
                    avg += m.vertex(u);
                    n += 1.0;
                }
                if n == 0.0 {
                    return (v, p);
                }
                avg /= n;
                let mut delta = (avg - p) * (strength * w).min(1.0);
                if brush == Brush::RobustSmooth {
                    // Remove most of the normal component to limit shrinkage, keeping
                    // enough to flatten bumps.
                    let nrm = m.vertex_normal(v);
                    let dn = delta.dot(nrm);
                    delta -= nrm * dn * 0.5;
                }
                (v, p + delta)
            })
            .collect(),
        Brush::Inflate => {
            let s = if invert { -1.0 } else { 1.0 };
            verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + m.vertex_normal(v) * (s * strength * w * r * 0.05))).collect()
        }
        Brush::Flatten => {
            let mut nsum = DVec3::ZERO;
            let mut psum = DVec3::ZERO;
            let mut wsum = 0.0;
            for &(v, w) in verts {
                nsum += m.vertex_normal(v) * w;
                psum += m.vertex(v) * w;
                wsum += w;
            }
            if wsum == 0.0 {
                return;
            }
            let n = nsum.normalize_or_zero();
            let o = psum / wsum;
            let _ = c;
            verts
                .par_iter()
                .map(|&(v, w)| {
                    let p = m.vertex(v);
                    (v, p - n * (p - o).dot(n) * (strength * w * 0.5).min(1.0))
                })
                .collect()
        }
        Brush::Drag => Vec::new(),
    };
    let pos = mesh.positions_mut();
    for (v, p) in updates {
        pos[v as usize] = p;
    }
    mesh.bump_shape_version();
}

impl Tool for SculptTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Sculpt"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Sculpt brushes");
        ui.horizontal_wrapped(|ui| {
            for b in [Brush::RobustSmooth, Brush::Smooth, Brush::Inflate, Brush::Flatten, Brush::Drag] {
                ui.selectable_value(&mut self.brush, b, b.label());
            }
        });
        let r = cx.scene.active.map(|id| brush_radius(cx.scene, id, self.size_pct));
        brush_size_ui(ui, &mut self.size_pct, r);
        ui.horizontal(|ui| {
            ui.label("Strength");
            ui.add(egui::Slider::new(&mut self.strength, 0.01..=1.0));
        });
        ui.checkbox(&mut self.hold_boundary, "Hold open boundaries");
        ui.small("Drag on the surface. Cmd/Ctrl inverts Inflate. [ ] change size.");
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        let hover = cx.pick(pos, cx.scene.active);
        self.hover = hover.map(|(id, h)| (h.point, brush_radius(cx.scene, id, self.size_pct)));
        if input.pressed {
            self.armed = true;
        }
        if input.down && self.armed && self.stroke.is_none() {
            if let Some((id, _)) = hover {
                cx.scene.active = Some(id);
                cx.snapshot(&format!("Sculpt: {}", self.brush.label()));
                self.stroke = Some(SculptStroke { object: id, invert: input.command, drag_verts: Vec::new(), drag_origin: None, drag_start_pos: Vec::new() });
            }
        }
        if input.released {
            self.armed = false;
            self.stroke = None;
            return;
        }
        if !input.down {
            return;
        }
        let Some(stroke) = &mut self.stroke else { return };
        let brush = self.brush;
        let strength = self.strength;
        let hold = self.hold_boundary;
        if brush == Brush::Drag {
            if stroke.drag_origin.is_none() {
                let Some((_, h)) = hover else { return };
                let r = brush_radius(cx.scene, stroke.object, self.size_pct);
                let Some(o) = cx.scene.get_mut(stroke.object) else { return };
                let bvh = o.bvh();
                stroke.drag_verts = brush_vertices(&o.mesh, &bvh, h.point, r, hold);
                stroke.drag_start_pos = stroke.drag_verts.iter().map(|&(v, _)| o.mesh.vertex(v)).collect();
                stroke.drag_origin = Some((h.point, pos));
                return;
            }
            let (p0, s0) = stroke.drag_origin.unwrap();
            let wpp = cx.world_per_point(p0);
            let d = pos - s0;
            let delta = (cx.camera.right() * d.x as f64 - cx.camera.up() * d.y as f64) * wpp;
            let Some(o) = cx.scene.get_mut(stroke.object) else { return };
            if o.moved_vertices.is_empty() {
                o.moved_from = o.mesh.shape_version();
            }
            let m = o.mesh_mut();
            let pos_arr = m.positions_mut();
            for (k, &(v, w)) in stroke.drag_verts.iter().enumerate() {
                pos_arr[v as usize] = stroke.drag_start_pos[k] + delta * w;
            }
            m.bump_shape_version();
            o.moved_vertices.extend(stroke.drag_verts.iter().map(|&(v, _)| v));
            return;
        }
        let Some((_, h)) = hover else { return };
        let r = brush_radius(cx.scene, stroke.object, self.size_pct);
        let invert = stroke.invert;
        let Some(o) = cx.scene.get_mut(stroke.object) else { return };
        let bvh = o.bvh();
        let verts = brush_vertices(&o.mesh, &bvh, h.point, r, hold);
        if o.moved_vertices.is_empty() {
            o.moved_from = o.mesh.shape_version();
        }
        apply_dab(o.mesh_mut(), &verts, brush, strength, r, invert, h.point);
        o.moved_vertices.extend(verts.iter().map(|&(v, _)| v));
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        if let Some((c, r)) = self.hover {
            ov.brush = Some((c.as_vec3(), r as f32));
        }
    }

    fn keys(&mut self, i: &egui::InputState, _cx: &mut ToolCx) {
        if i.key_pressed(egui::Key::OpenBracket) {
            step_brush_size(&mut self.size_pct, false);
        }
        if i.key_pressed(egui::Key::CloseBracket) {
            step_brush_size(&mut self.size_pct, true);
        }
    }
}
