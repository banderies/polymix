//! Mesh Query (Analysis > Mesh Query): colour the surface by normal, position, triangle
//! quality or curvature, and click a triangle to read its id, group, shape and the
//! curvature at its nearest vertex.

use super::*;
use crate::ui_util;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ColorMode {
    VertexPosition,
    SurfaceNormal,
    AspectRatio,
    InternalAngle,
    GaussianCurvature,
    MeanCurvature,
}

impl ColorMode {
    const ALL: [ColorMode; 6] =
        [ColorMode::VertexPosition, ColorMode::SurfaceNormal, ColorMode::AspectRatio, ColorMode::InternalAngle, ColorMode::GaussianCurvature, ColorMode::MeanCurvature];
    fn label(self) -> &'static str {
        match self {
            ColorMode::VertexPosition => "Vertex Position",
            ColorMode::SurfaceNormal => "Surface Normal",
            ColorMode::AspectRatio => "Aspect Ratio",
            ColorMode::InternalAngle => "Internal Angle",
            ColorMode::GaussianCurvature => "Gaussian Curvature",
            ColorMode::MeanCurvature => "Mean Curvature",
        }
    }
    fn is_scalar(self) -> bool {
        !matches!(self, ColorMode::VertexPosition | ColorMode::SurfaceNormal)
    }
}

/// What was learned about the clicked triangle.
struct Query {
    tri: u32,
    group: i32,
    aspect: f64,
    min_angle: f64,
    max_angle: f64,
    edge: u32,
    vertex: u32,
    gauss: f64,
    mean: f64,
}

pub struct MeshQueryTool {
    object: Option<ObjectId>,
    mode: ColorMode,
    range: (f64, f64),
    /// Per-triangle values of the current scalar mode.
    values: Vec<f64>,
    /// Per-vertex curvatures (computed on demand).
    curvature: Option<(Vec<f64>, Vec<f64>)>,
    query: Option<Query>,
    dirty: bool,
}

/// Blue → cyan → green → yellow → red for `t` in [0, 1].
fn heat(t: f64) -> [f64; 3] {
    let t = t.clamp(0.0, 1.0);
    let stops = [[0.15, 0.25, 0.85], [0.1, 0.75, 0.9], [0.2, 0.8, 0.3], [0.95, 0.85, 0.2], [0.9, 0.2, 0.15]];
    let x = t * (stops.len() - 1) as f64;
    let i = (x.floor() as usize).min(stops.len() - 2);
    let f = x - i as f64;
    [0, 1, 2].map(|k| stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f)
}

fn pack(c: [f64; 3]) -> u32 {
    let b = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
    (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
}

impl MeshQueryTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, mode: ColorMode::SurfaceNormal, range: (0.0, 1.0), values: Vec::new(), curvature: None, query: None, dirty: true }
    }

    fn curvatures(&mut self, m: &Mesh) -> &(Vec<f64>, Vec<f64>) {
        self.curvature.get_or_insert_with(|| (mm_core::measure::gaussian_curvature(m), mm_core::measure::mean_curvature(m)))
    }

    /// Recompute the per-triangle values and, for scalar modes, an automatic range (2nd to
    /// 98th percentile).
    fn compute_values(&mut self, m: &Mesh) {
        let tris: Vec<u32> = m.triangle_ids().collect();
        self.values = vec![0.0; m.max_tid()];
        let per_vertex = |vals: &[f64], t: u32| m.triangle(t).iter().map(|&v| vals[v as usize]).sum::<f64>() / 3.0;
        match self.mode {
            ColorMode::AspectRatio => tris.iter().for_each(|&t| self.values[t as usize] = mm_core::measure::triangle_aspect_ratio(m, t)),
            ColorMode::InternalAngle => tris.iter().for_each(|&t| {
                self.values[t as usize] = mm_core::measure::triangle_angles(m, t).iter().fold(f64::INFINITY, |a, &b| a.min(b)).to_degrees()
            }),
            ColorMode::GaussianCurvature => {
                let (k, _) = self.curvatures(m).clone();
                tris.iter().for_each(|&t| self.values[t as usize] = per_vertex(&k, t));
            }
            ColorMode::MeanCurvature => {
                let (_, h) = self.curvatures(m).clone();
                tris.iter().for_each(|&t| self.values[t as usize] = per_vertex(&h, t));
            }
            _ => return,
        }
        let mut sorted: Vec<f64> = tris.iter().map(|&t| self.values[t as usize]).filter(|x| x.is_finite()).collect();
        sorted.sort_by(f64::total_cmp);
        if let (Some(_), n) = (sorted.first(), sorted.len()) {
            self.range = (sorted[n * 2 / 100], sorted[(n * 98 / 100).min(n - 1)]);
            if self.range.1 <= self.range.0 {
                self.range.1 = self.range.0 + 1e-9;
            }
        }
    }

    fn colors(&self, m: &Mesh) -> Vec<u32> {
        let b = m.bounds();
        let ext = b.extents().max(DVec3::splat(1e-300));
        let mut out = vec![0u32; m.max_tid()];
        for t in m.triangle_ids() {
            out[t as usize] = match self.mode {
                ColorMode::VertexPosition => {
                    let c = (m.tri_centroid(t) - b.min) / ext;
                    pack([c.x, c.y, c.z])
                }
                ColorMode::SurfaceNormal => {
                    let n = m.tri_normal(t) * 0.5 + DVec3::splat(0.5);
                    pack([n.x, n.y, n.z])
                }
                _ => {
                    let (lo, hi) = self.range;
                    pack(heat((self.values[t as usize] - lo) / (hi - lo)))
                }
            };
        }
        out
    }

    fn apply(&mut self, cx: &mut ToolCx) {
        self.dirty = false;
        let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) else { return };
        let m = o.mesh.clone();
        let colors = self.colors(&m);
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            o.colors = Some((crate::scene::next_stamp(), Arc::new(colors)));
        }
    }

    fn clear(&self, cx: &mut ToolCx) {
        if let Some(o) = self.object.and_then(|id| cx.scene.get_mut(id)) {
            o.colors = None;
        }
    }
}

impl Tool for MeshQueryTool {
    fn name(&self) -> &'static str {
        "Mesh Query"
    }

    fn copy_text(&self) -> Option<String> {
        let q = self.query.as_ref()?;
        Some(format!(
            "triangle {}\ngroup {}\naspect ratio {:.4}\nmin angle {:.3}\nmax angle {:.3}\nedge {}\nvertex {}\ngauss curvature {:.6}\nmean curvature {:.6}",
            q.tri, q.group, q.aspect, q.min_angle, q.max_angle, q.edge, q.vertex, q.gauss, q.mean
        ))
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Mesh Query");
        let Some(mesh) = self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let before = self.mode;
        egui::ComboBox::from_label("Color mode").selected_text(self.mode.label()).show_ui(ui, |ui| {
            for m in ColorMode::ALL {
                ui.selectable_value(&mut self.mode, m, m.label());
            }
        });
        if self.mode != before || self.values.len() != mesh.max_tid() {
            self.compute_values(&mesh);
            self.dirty = true;
        }
        ui_util::section(ui, "Selected triangle");
        let row = |ui: &mut egui::Ui, k: &str, v: String| {
            ui.horizontal(|ui| {
                ui.label(k);
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| ui.monospace(v));
            });
        };
        match &self.query {
            Some(q) => {
                row(ui, "Triangle ID", q.tri.to_string());
                row(ui, "FaceGroup ID", q.group.to_string());
                row(ui, "Aspect ratio", format!("{:.3}", q.aspect));
                row(ui, "Min angle", format!("{:.2}°", q.min_angle));
                row(ui, "Max angle", format!("{:.2}°", q.max_angle));
                ui_util::section(ui, "Nearest points");
                row(ui, "Edge ID", q.edge.to_string());
                row(ui, "Vertex ID", q.vertex.to_string());
                row(ui, "Gauss curvature", format!("{:.5}", q.gauss));
                row(ui, "Mean curvature", format!("{:.5}", q.mean));
            }
            None => {
                ui.weak("Click the surface.");
            }
        }
        ui_util::section(ui, "Color range");
        ui.add_enabled_ui(self.mode.is_scalar(), |ui| {
            let speed = ((self.range.1 - self.range.0).abs() * 0.01).max(1e-6);
            ui.horizontal(|ui| {
                ui.label("Min");
                self.dirty |= ui.add(egui::DragValue::new(&mut self.range.0).speed(speed).max_decimals(5)).changed();
                ui.label("Max");
                self.dirty |= ui.add(egui::DragValue::new(&mut self.range.1).speed(speed).max_decimals(5)).changed();
                if ui.small_button("Auto").clicked() {
                    self.compute_values(&mesh);
                    self.dirty = true;
                }
            });
        });
        if self.dirty {
            self.apply(cx);
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            self.clear(cx);
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        if !input.clicked {
            return;
        }
        let Some(pos) = input.pos else { return };
        let Some((id, hit)) = cx.pick(pos, self.object) else { return };
        let Some(m) = cx.scene.get(id).map(|o| o.mesh.clone()) else { return };
        let t = hit.tid;
        let tv = m.triangle(t);
        let angles = mm_core::measure::triangle_angles(&m, t).map(f64::to_degrees);
        let vertex = tv.into_iter().min_by(|&a, &b| m.vertex(a).distance(hit.point).total_cmp(&m.vertex(b).distance(hit.point))).unwrap_or(tv[0]);
        let seg = |e: u32| {
            let [a, b] = m.edge_vertices(e);
            let (pa, pb) = (m.vertex(a), m.vertex(b));
            let ab = pb - pa;
            let s = ((hit.point - pa).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
            (pa + ab * s).distance(hit.point)
        };
        let edge = m.tri_edges(t).into_iter().min_by(|&a, &b| seg(a).total_cmp(&seg(b))).unwrap_or(0);
        let (k, h) = self.curvatures(&m).clone();
        self.query = Some(Query {
            tri: t,
            group: m.tri_group(t),
            aspect: mm_core::measure::triangle_aspect_ratio(&m, t),
            min_angle: angles.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_angle: angles.iter().fold(0.0, |a, &b| a.max(b)),
            edge,
            vertex,
            gauss: k[vertex as usize],
            mean: h[vertex as usize],
        });
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn on_close(&mut self, cx: &mut ToolCx) {
        self.clear(cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_and_ranges() {
        let m = mm_core::primitives::icosphere(2.0, 3);
        let mut t = MeshQueryTool { object: None, mode: ColorMode::MeanCurvature, range: (0.0, 1.0), values: Vec::new(), curvature: None, query: None, dirty: true };
        t.compute_values(&m);
        // A sphere's mean curvature is 1/r everywhere, so the automatic range hugs 0.5.
        assert!((t.range.0 - 0.5).abs() < 0.05 && (t.range.1 - 0.5).abs() < 0.05, "{:?}", t.range);
        t.mode = ColorMode::SurfaceNormal;
        let c = t.colors(&m);
        // The triangle facing +X is red-dominant.
        let tx = m.triangle_ids().max_by(|&a, &b| m.tri_normal(a).x.total_cmp(&m.tri_normal(b).x)).unwrap();
        let rgb = c[tx as usize];
        assert!((rgb >> 16) & 0xff > 240 && (rgb >> 8) & 0xff < 160);
        assert_eq!(pack(heat(0.0)), pack([0.15, 0.25, 0.85]));
    }
}
