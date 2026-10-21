//! Mesh Query (Analysis > Mesh Query): colour the surface by normal, position, triangle
//! quality or curvature, and click a triangle to read its id, group, shape and the
//! curvature at its nearest vertex.

use super::*;
use crate::ui_util;
use rayon::prelude::*;
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
    fn key(self) -> &'static str {
        match self {
            ColorMode::VertexPosition => "vertex_position",
            ColorMode::SurfaceNormal => "surface_normal",
            ColorMode::AspectRatio => "aspect_ratio",
            ColorMode::InternalAngle => "internal_angle",
            ColorMode::GaussianCurvature => "gaussian_curvature",
            ColorMode::MeanCurvature => "mean_curvature",
        }
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
    /// Per-vertex curvatures (computed on demand), Gaussian and mean.
    curvature: Option<Arc<(Vec<f64>, Vec<f64>)>>,
    query: Option<Query>,
    dirty: bool,
    /// The scalar field's distribution and its worst triangles (agents).
    summary: Option<serde_json::Value>,
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

/// Triangle slots coloured by one parallel task.
const CHUNK: usize = 1 << 14;

/// The values that would stand at `ranks` (ascending) were `v` sorted by `total_cmp`, found
/// by selection rather than a full sort. `v` is reordered.
fn ranked(v: &mut [f64], ranks: &[usize]) -> Vec<f64> {
    let mut out = Vec::with_capacity(ranks.len());
    let mut lo = 0;
    for &k in ranks {
        // Selecting rank `lo` left the `lo` smallest values in front of it, so rank `k` is
        // rank `k - lo` of the rest.
        let (_, x, _) = v[lo..].select_nth_unstable_by(k - lo, f64::total_cmp);
        out.push(*x);
        lo = k;
    }
    out
}

impl MeshQueryTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, mode: ColorMode::SurfaceNormal, range: (0.0, 1.0), values: Vec::new(), curvature: None, query: None, dirty: true, summary: None }
    }

    fn curvatures(&mut self, m: &Mesh) -> Arc<(Vec<f64>, Vec<f64>)> {
        self.curvature.get_or_insert_with(|| Arc::new((mm_core::measure::gaussian_curvature(m), mm_core::measure::mean_curvature(m)))).clone()
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
                let c = self.curvatures(m);
                tris.iter().for_each(|&t| self.values[t as usize] = per_vertex(&c.0, t));
            }
            ColorMode::MeanCurvature => {
                let c = self.curvatures(m);
                tris.iter().for_each(|&t| self.values[t as usize] = per_vertex(&c.1, t));
            }
            _ => {
                self.summary = None;
                return;
            }
        }
        // The quantiles below and the automatic range, as they stand in the sorted values.
        let mut finite: Vec<f64> = tris.iter().map(|&t| self.values[t as usize]).filter(|x| x.is_finite()).collect();
        let n = finite.len();
        let q_rank = |f: f64| ((n.max(1) - 1) as f64 * f) as usize;
        let range_ranks = (n * 2 / 100, (n * 98 / 100).min(n.max(1) - 1));
        let mut ranks = vec![q_rank(0.0), q_rank(0.02), q_rank(0.5), q_rank(0.98), q_rank(1.0), range_ranks.0, range_ranks.1];
        ranks.sort_unstable();
        ranks.dedup();
        let at_rank = if n > 0 { ranked(&mut finite, &ranks) } else { Vec::new() };
        let sorted = |k: usize| at_rank[ranks.binary_search(&k).unwrap_or(0)];
        self.summary = (n > 0).then(|| {
            use polymix_workspace::agent::{round, vec3};
            let q = |f: f64| round(sorted(q_rank(f)), 6);
            // Worst: the most elongated (aspect ratio), the sharpest (internal angle), the
            // most curved (either sign).
            let badness = |t: u32| match self.mode {
                ColorMode::InternalAngle => -self.values[t as usize],
                ColorMode::AspectRatio => self.values[t as usize],
                _ => self.values[t as usize].abs(),
            };
            let mut worst: Vec<u32> = tris.iter().copied().filter(|&t| self.values[t as usize].is_finite()).collect();
            let k = worst.len().min(5);
            if k > 0 {
                worst.select_nth_unstable_by(k - 1, |&a, &b| badness(b).total_cmp(&badness(a)));
            }
            worst.truncate(k);
            worst.sort_by(|&a, &b| badness(b).total_cmp(&badness(a)));
            serde_json::json!({
                "triangles": n,
                "min": q(0.0), "p2": q(0.02), "median": q(0.5), "p98": q(0.98), "max": q(1.0),
                "worst": worst.iter().map(|&t| serde_json::json!({"triangle": t, "value": round(self.values[t as usize], 6), "centre_mm": vec3(m.tri_centroid(t).to_array(), 3)})).collect::<Vec<_>>(),
            })
        });
        if n > 0 {
            self.range = (sorted(range_ranks.0), sorted(range_ranks.1));
            if self.range.1 <= self.range.0 {
                self.range.1 = self.range.0 + 1e-9;
            }
        }
    }

    /// Per-triangle colours of the current mode. The triangle slots are independent of each
    /// other, so they are filled in parallel.
    fn colors(&self, m: &Mesh) -> Vec<u32> {
        let (mode, (lo, hi), values) = (self.mode, self.range, &self.values);
        // Only Vertex Position needs the box.
        let frame = (mode == ColorMode::VertexPosition).then(|| {
            let b = m.bounds();
            (b.min, b.extents().max(DVec3::splat(1e-300)))
        });
        let colour = |t: u32| match (mode, frame) {
            (ColorMode::VertexPosition, Some((min, ext))) => {
                let c = (m.tri_centroid(t) - min) / ext;
                pack([c.x, c.y, c.z])
            }
            (ColorMode::SurfaceNormal, _) => {
                let n = m.tri_normal(t) * 0.5 + DVec3::splat(0.5);
                pack([n.x, n.y, n.z])
            }
            _ => pack(heat((values[t as usize] - lo) / (hi - lo))),
        };
        let mut out = vec![0u32; m.max_tid()];
        out.par_chunks_mut(CHUNK).enumerate().for_each(|(k, chunk)| {
            for (i, out) in chunk.iter_mut().enumerate() {
                let t = (k * CHUNK + i) as u32;
                if m.is_triangle(t) {
                    *out = colour(t);
                }
            }
        });
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
        self.query_at(&m, hit.tid, hit.point);
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"color_mode": self.mode.key(), "range": [self.range.0, self.range.1]}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "color_mode": {"type": "string", "enum": ColorMode::ALL.map(|m| m.key()), "description": "What colours the surface: position, normal, triangle aspect ratio, smallest internal angle (degrees), Gaussian or mean curvature"},
            "range": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Values at the two ends of the colour scale (scalar modes)"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let mode = ColorMode::ALL.into_iter().find(|m| v["color_mode"] == m.key()).ok_or("unknown color_mode")?;
        let m = self.object.and_then(|id| cx.scene.get(id)).map(|o| o.mesh.clone()).ok_or("no active object")?;
        if mode != self.mode || self.values.len() != m.max_tid() {
            self.mode = mode;
            self.compute_values(&m);
        }
        if changes.get("range").is_some() {
            match v["range"].as_array().map(|r| r.iter().filter_map(serde_json::Value::as_f64).collect::<Vec<_>>()).as_deref() {
                Some([lo, hi]) if hi > lo => self.range = (*lo, *hi),
                _ => return Err("range must be [min, max] with max > min".into()),
            }
        }
        self.apply(cx);
        Ok(())
    }

    fn pending(&self) -> bool {
        self.dirty && self.object.is_some()
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::round;
        serde_json::json!({
            "object": self.object,
            "color_mode": self.mode.key(),
            "range": [round(self.range.0, 6), round(self.range.1, 6)],
            "distribution": self.summary,
            "queried_triangle": self.query.as_ref().map(|q| serde_json::json!({
                "triangle": q.tri, "group": q.group, "aspect_ratio": round(q.aspect, 4), "min_angle_deg": round(q.min_angle, 3), "max_angle_deg": round(q.max_angle, 3),
                "nearest_edge": q.edge, "nearest_vertex": q.vertex, "gaussian_curvature": round(q.gauss, 6), "mean_curvature": round(q.mean, 6),
            })),
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![
            Action::new("query_at", "Report the triangle at the surface point nearest `at` (as clicking it does)", serde_json::json!({"at": arg::point("A point in mm")})).required(&["at"]),
            Action::plain("compute", "Compute the colour field and its distribution now (also done while the panel is shown)"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let id = self.object.ok_or("no active object")?;
        let m = cx.scene.get(id).map(|o| o.mesh.clone()).ok_or("no active object")?;
        match action {
            "query_at" => {
                let o = cx.scene.get_mut(id).unwrap();
                let h = o.bvh().nearest(&m, DVec3::from_array(Args(args).point("at")?), f64::INFINITY).ok_or("no surface")?;
                self.query_at(&m, h.tid, h.point);
                Ok(format!("Triangle {}", h.tid))
            }
            "compute" => {
                self.compute_values(&m);
                self.apply(cx);
                Ok(format!("{} computed", self.mode.label()))
            }
            _ => Err(format!("Mesh Query has no action {action}")),
        }
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


impl MeshQueryTool {
    fn query_at(&mut self, m: &Mesh, t: u32, point: DVec3) {
        let hit = mm_core::bvh::RayHit { tid: t, t: 0.0, point, bary: DVec3::ZERO };
        let tv = m.triangle(t);
        let angles = mm_core::measure::triangle_angles(m, t).map(f64::to_degrees);
        let vertex = tv.into_iter().min_by(|&a, &b| m.vertex(a).distance(hit.point).total_cmp(&m.vertex(b).distance(hit.point))).unwrap_or(tv[0]);
        let seg = |e: u32| {
            let [a, b] = m.edge_vertices(e);
            let (pa, pb) = (m.vertex(a), m.vertex(b));
            let ab = pb - pa;
            let s = ((hit.point - pa).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
            (pa + ab * s).distance(hit.point)
        };
        let edge = m.tri_edges(t).into_iter().min_by(|&a, &b| seg(a).total_cmp(&seg(b))).unwrap_or(0);
        let c = self.curvatures(m);
        let (k, h) = (&c.0, &c.1);
        self.query = Some(Query {
            tri: t,
            group: m.tri_group(t),
            aspect: mm_core::measure::triangle_aspect_ratio(m, t),
            min_angle: angles.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            max_angle: angles.iter().fold(0.0, |a, &b| a.max(b)),
            edge,
            vertex,
            gauss: k[vertex as usize],
            mean: h[vertex as usize],
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn colours_and_ranges() {
        let m = mm_core::primitives::icosphere(2.0, 3);
        let mut t = MeshQueryTool { object: None, mode: ColorMode::MeanCurvature, range: (0.0, 1.0), values: Vec::new(), curvature: None, query: None, dirty: true, summary: None };
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

    /// The colouring as it was: one triangle after another, the box always measured.
    fn colors_serial(t: &MeshQueryTool, m: &Mesh) -> Vec<u32> {
        let b = m.bounds();
        let ext = b.extents().max(DVec3::splat(1e-300));
        let mut out = vec![0u32; m.max_tid()];
        for x in m.triangle_ids() {
            out[x as usize] = match t.mode {
                ColorMode::VertexPosition => {
                    let c = (m.tri_centroid(x) - b.min) / ext;
                    pack([c.x, c.y, c.z])
                }
                ColorMode::SurfaceNormal => {
                    let n = m.tri_normal(x) * 0.5 + DVec3::splat(0.5);
                    pack([n.x, n.y, n.z])
                }
                _ => {
                    let (lo, hi) = t.range;
                    pack(heat((t.values[x as usize] - lo) / (hi - lo)))
                }
            };
        }
        out
    }

    #[test]
    fn colours_and_quantiles_match_the_serial_sorted_ones() {
        // A bumpy sphere with a hole (dead triangle slots).
        let mut m = mm_core::primitives::icosphere(2.0, 4);
        m.transform(|p| p * (1.0 + 0.1 * (p.x * 3.0).sin() * (p.y * 2.0).cos()));
        let gone: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 1.6).collect();
        m.remove_triangles(gone);
        let mut t = MeshQueryTool { object: None, mode: ColorMode::SurfaceNormal, range: (0.0, 1.0), values: Vec::new(), curvature: None, query: None, dirty: true, summary: None };
        for mode in ColorMode::ALL {
            t.mode = mode;
            t.compute_values(&m);
            assert_eq!(t.colors(&m), colors_serial(&t, &m), "{mode:?}");
            if mode.is_scalar() {
                // The automatic range and the summary's quantiles, from a full sort.
                let mut sorted: Vec<f64> = m.triangle_ids().map(|x| t.values[x as usize]).filter(|x| x.is_finite()).collect();
                sorted.sort_by(f64::total_cmp);
                let n = sorted.len();
                let mut range = (sorted[n * 2 / 100], sorted[(n * 98 / 100).min(n - 1)]);
                if range.1 <= range.0 {
                    range.1 = range.0 + 1e-9;
                }
                assert_eq!(t.range, range, "{mode:?}");
                let q = |f: f64| serde_json::json!(polymix_workspace::agent::round(sorted[((n - 1) as f64 * f) as usize], 6));
                let s = t.summary.as_ref().unwrap();
                assert_eq!([&s["min"], &s["p2"], &s["median"], &s["p98"], &s["max"]], [&q(0.0), &q(0.02), &q(0.5), &q(0.98), &q(1.0)], "{mode:?}");
                // Another range from the panel recolours the same way.
                t.range = (range.0 * 0.5, range.1 * 1.5 + 0.1);
                assert_eq!(t.colors(&m), colors_serial(&t, &m), "{mode:?}");
            }
        }
    }

    #[test]
    fn selected_ranks_are_the_sorted_ones() {
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 33) as f64 / (1u64 << 31) as f64
        };
        for n in [1usize, 2, 3, 7, 50, 101, 1000] {
            // Many repeats, both zeros and negative values.
            let v: Vec<f64> = (0..n).map(|i| match i % 5 { 0 => 0.0, 1 => -0.0, 2 => (rnd() * 4.0).floor() - 2.0, _ => rnd() - 0.5 }).collect();
            let mut sorted = v.clone();
            sorted.sort_by(f64::total_cmp);
            let mut ranks = vec![0, n / 50, n / 2, n * 98 / 100, n - 1, (n * 2 / 100).min(n - 1)];
            ranks.sort_unstable();
            ranks.dedup();
            let got = ranked(&mut v.clone(), &ranks);
            let want: Vec<f64> = ranks.iter().map(|&k| sorted[k]).collect();
            assert_eq!(got.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), want.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), "{n}");
        }
    }
}
