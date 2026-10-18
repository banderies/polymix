//! Measure tool (Analysis > Measure): point-to-point distances, point coordinates, wall
//! thickness and gaps along the normal or an axis (min/max, interior/exterior), local
//! cylinder fits, and the vessel cross-section diameter. Clicked points can snap to
//! vertices, edges or face-group borders, and a measurement can be set to a target length
//! (scaling the object).

use super::*;
use crate::ui_util;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MeasureType {
    PointToPoint,
    PointCoordinates,
    MinInterior,
    MaxInterior,
    MinExterior,
    MaxExterior,
    FitCylinder,
    /// Equivalent diameter of the vessel cross-section at the click (CFD).
    VesselDiameter,
}

impl MeasureType {
    const ALL: [MeasureType; 8] = [
        MeasureType::PointToPoint,
        MeasureType::PointCoordinates,
        MeasureType::MinInterior,
        MeasureType::MaxInterior,
        MeasureType::MinExterior,
        MeasureType::MaxExterior,
        MeasureType::FitCylinder,
        MeasureType::VesselDiameter,
    ];
    fn label(self) -> &'static str {
        match self {
            MeasureType::PointToPoint => "Point to point",
            MeasureType::PointCoordinates => "Point coordinates",
            MeasureType::MinInterior => "Min interior (thickness)",
            MeasureType::MaxInterior => "Max interior",
            MeasureType::MinExterior => "Min exterior (gap)",
            MeasureType::MaxExterior => "Max exterior",
            MeasureType::FitCylinder => "Fit cylinder",
            MeasureType::VesselDiameter => "Vessel diameter",
        }
    }
    fn uses_direction(self) -> bool {
        matches!(self, MeasureType::MinInterior | MeasureType::MaxInterior | MeasureType::MinExterior | MeasureType::MaxExterior)
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Direction {
    Normal,
    Axis(usize),
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Snapping {
    None,
    Vertices,
    Edges,
    GroupBorders,
}

struct Measurement {
    object: ObjectId,
    a: DVec3,
    b: DVec3,
    /// The measured length (none for point coordinates).
    length: Option<f64>,
    label: String,
    ring: Vec<[DVec3; 2]>,
}

pub struct MeasureTool {
    kind: MeasureType,
    direction: Direction,
    snapping: Snapping,
    /// Fit Cylinder: radius of the surface patch that is fitted.
    fit_radius: f64,
    first: Option<(ObjectId, DVec3)>,
    pressed: Option<(egui::Pos2, ObjectId, DVec3)>,
    items: Vec<Measurement>,
    selected: Option<usize>,
    target_length: f64,
}

impl MeasureTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let diag = cx.scene.active().map(|o| o.bounds().diagonal()).unwrap_or(1.0);
        Self {
            kind: MeasureType::VesselDiameter,
            direction: Direction::Normal,
            snapping: Snapping::None,
            fit_radius: diag * 0.05,
            first: None,
            pressed: None,
            items: Vec::new(),
            selected: None,
            target_length: 0.0,
        }
    }

    /// Surface hit under the pointer, snapped as requested.
    fn hit(&self, cx: &mut ToolCx, pos: egui::Pos2) -> Option<(ObjectId, mm_core::bvh::RayHit)> {
        let (id, mut hit) = cx.pick(pos, None)?;
        let o = cx.scene.get(id)?;
        let m = &o.mesh;
        let tv = m.triangle(hit.tid);
        match self.snapping {
            Snapping::None => {}
            Snapping::Vertices => {
                hit.point = tv.map(|v| m.vertex(v)).into_iter().min_by(|a, b| a.distance(hit.point).total_cmp(&b.distance(hit.point)))?;
            }
            Snapping::Edges => {
                hit.point = (0..3).map(|j| closest_on_segment(hit.point, m.vertex(tv[j]), m.vertex(tv[(j + 1) % 3]))).min_by(|a, b| a.distance(hit.point).total_cmp(&b.distance(hit.point)))?;
            }
            Snapping::GroupBorders => {
                // Nearest group-border edge within a few screen points.
                let r = cx.world_per_point(hit.point) * 12.0;
                let o = cx.scene.get_mut(id)?;
                let bvh = o.bvh();
                let m = &o.mesh;
                let mut best: Option<(f64, DVec3)> = None;
                bvh.query_sphere(m, hit.point, r, |t| {
                    for e in m.tri_edges(t) {
                        let [a, b] = m.edge_triangles(e);
                        if b == mm_core::INVALID || m.tri_group(a) == m.tri_group(b) {
                            continue;
                        }
                        let [u, v] = m.edge_vertices(e);
                        let q = closest_on_segment(hit.point, m.vertex(u), m.vertex(v));
                        let d = q.distance(hit.point);
                        if d <= r && best.is_none_or(|x| d < x.0) {
                            best = Some((d, q));
                        }
                    }
                });
                if let Some((_, q)) = best {
                    hit.point = q;
                }
            }
        }
        Some((id, hit))
    }

    /// Ray hits along `dir` from `p` (beyond a small offset) on visible objects, nearest
    /// first; `only` restricts them to one object.
    fn hits_along(cx: &mut ToolCx, p: DVec3, dir: DVec3, only: Option<ObjectId>) -> Vec<DVec3> {
        let eps = 1e-6 * (1.0 + p.length());
        let ray = Ray::new(p + dir * eps, dir);
        let mut out = Vec::new();
        let mut hits = Vec::new();
        for o in cx.scene.objects.iter_mut() {
            if !o.visible || only.is_some_and(|id| id != o.id) {
                continue;
            }
            hits.clear();
            o.bvh().raycast_all(&o.mesh, &ray, &mut hits);
            out.extend(hits.iter().map(|h| (h.t, h.point)));
        }
        out.sort_by(|a, b| a.0.total_cmp(&b.0));
        out.into_iter().map(|x| x.1).collect()
    }

    fn measure_at(&mut self, cx: &mut ToolCx, id: ObjectId, hit: mm_core::bvh::RayHit) {
        let p = hit.point;
        let Some(o) = cx.scene.get(id) else { return };
        let n = o.mesh.tri_normal(hit.tid);
        let item = match self.kind {
            MeasureType::PointToPoint => match self.first.take() {
                None => {
                    self.first = Some((id, p));
                    return;
                }
                Some((_, a)) => {
                    let d = a.distance(p);
                    Measurement { object: id, a, b: p, length: Some(d), label: format!("↔ {d:.3}"), ring: Vec::new() }
                }
            },
            MeasureType::PointCoordinates => Measurement { object: id, a: p, b: p, length: None, label: format!("({:.3}, {:.3}, {:.3})", p.x, p.y, p.z), ring: Vec::new() },
            MeasureType::MinInterior | MeasureType::MaxInterior | MeasureType::MinExterior | MeasureType::MaxExterior => {
                let axis = match self.direction {
                    Direction::Normal => n,
                    Direction::Axis(i) => [DVec3::X, DVec3::Y, DVec3::Z][i] * if n.dot([DVec3::X, DVec3::Y, DVec3::Z][i]) < 0.0 { -1.0 } else { 1.0 },
                };
                let interior = matches!(self.kind, MeasureType::MinInterior | MeasureType::MaxInterior);
                let dir = if interior { -axis } else { axis };
                let hits = Self::hits_along(cx, p, dir, interior.then_some(id));
                let far = matches!(self.kind, MeasureType::MaxInterior | MeasureType::MaxExterior);
                let Some(&b) = (if far { hits.last() } else { hits.first() }) else {
                    *cx.status = "Nothing along that direction.".into();
                    return;
                };
                let d = p.distance(b);
                let word = if interior { "interior" } else { "exterior" };
                Measurement { object: id, a: p, b, length: Some(d), label: format!("{} {word} {d:.3}", if far { "max" } else { "min" }), ring: Vec::new() }
            }
            MeasureType::FitCylinder => {
                let Some(o) = cx.scene.get_mut(id) else { return };
                let bvh = o.bvh();
                let m = o.mesh.clone();
                // Patch: triangles within the fit radius, connected to the clicked one.
                let mut near = std::collections::HashSet::new();
                bvh.query_sphere(&m, p, self.fit_radius, |t| {
                    near.insert(t);
                });
                let mut patch = vec![hit.tid];
                let mut seen = std::collections::HashSet::from([hit.tid]);
                let mut i = 0;
                while i < patch.len() {
                    for nb in m.tri_neighbors(patch[i]) {
                        if nb != mm_core::INVALID && near.contains(&nb) && seen.insert(nb) {
                            patch.push(nb);
                        }
                    }
                    i += 1;
                }
                let Some(f) = mm_edit::fit_cylinder(&m, &patch) else {
                    *cx.status = "No cylinder fits there.".into();
                    return;
                };
                let c = f.axis_point(p);
                let (u, v) = mm_core::geom::orthonormal_basis(f.axis);
                let ring: Vec<[DVec3; 2]> = (0..48)
                    .map(|k| {
                        let t = |k: usize| k as f64 / 48.0 * std::f64::consts::TAU;
                        let at = |a: f64| c + (u * a.cos() + v * a.sin()) * f.radius;
                        [at(t(k)), at(t(k + 1))]
                    })
                    .collect();
                let d = 2.0 * f.radius;
                Measurement { object: id, a: c, b: p, length: Some(d), label: format!("⌀ {d:.3} (cylinder fit)"), ring }
            }
            MeasureType::VesselDiameter => {
                let wall = opposite_wall(cx.scene, id, p, hit.tid);
                let Some(o) = cx.scene.get_mut(id) else { return };
                let bvh = o.bvh();
                let mesh = o.mesh.clone();
                let section = mm_edit::estimate_vessel_cross_section(&mesh, Some(&bvh), p, None).ok();
                let ring = section
                    .as_ref()
                    .and_then(|x| mm_edit::nearest_intersection_curve(&mesh, x.plane, p))
                    .filter(|c| c.closed)
                    .map(|c| c.segments)
                    .unwrap_or_default();
                let eq = section.as_ref().and_then(|x| x.area).map(|a| 2.0 * (a / std::f64::consts::PI).sqrt());
                let wall_d = wall.map(|q| p.distance(q));
                let (label, length) = match (eq, wall_d) {
                    (Some(e), Some(w)) => (format!("⌀ {e:.3} (wall-to-wall {w:.3})"), e),
                    (Some(e), None) => (format!("⌀ {e:.3}"), e),
                    (None, Some(w)) => (format!("⌀ {w:.3} (wall-to-wall)"), w),
                    (None, None) => return,
                };
                Measurement { object: id, a: p, b: wall.unwrap_or(p), length: Some(length), label, ring }
            }
        };
        *cx.status = format!("{}: {}", self.kind.label(), item.label);
        self.items.push(item);
        self.selected = Some(self.items.len() - 1);
        if let Some(l) = self.items.last().and_then(|m| m.length) {
            self.target_length = l;
        }
    }

    /// Scale the measured object uniformly about its centre so the selected measurement
    /// becomes `target_length`.
    fn apply_target_length(&mut self, cx: &mut ToolCx) {
        let Some(m) = self.selected.and_then(|i| self.items.get(i)) else { return };
        let (Some(len), id) = (m.length, m.object) else { return };
        if !(len > 0.0 && self.target_length > 0.0) || cx.scene.get(id).is_none() {
            return;
        }
        let s = self.target_length / len;
        cx.history.push("Set target length", cx.scene);
        let Some(o) = cx.scene.get_mut(id) else { return };
        let c = o.bounds().center();
        o.mesh_mut().transform(|p| c + (p - c) * s);
        for m in self.items.iter_mut().filter(|m| m.object == id) {
            m.a = c + (m.a - c) * s;
            m.b = c + (m.b - c) * s;
            m.length = m.length.map(|l| l * s);
            for seg in &mut m.ring {
                *seg = seg.map(|q| c + (q - c) * s);
            }
            if let Some(l) = m.length {
                m.label = format!("{} → {l:.3}", m.label.split(" → ").next().unwrap_or(""));
            }
        }
        *cx.status = format!("Scaled by {s:.6}");
    }
}

fn closest_on_segment(p: DVec3, a: DVec3, b: DVec3) -> DVec3 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_squared().max(1e-300)).clamp(0.0, 1.0);
    a + ab * t
}

/// Distance across the surface from `p` along the inward normal (the "diameter" of a
/// tube at that point). Returns the opposite wall point.
pub fn opposite_wall(scene: &mut crate::scene::Scene, id: ObjectId, p: DVec3, tid: u32) -> Option<DVec3> {
    let o = scene.get_mut(id)?;
    let bvh = o.bvh();
    let n = o.mesh.tri_normal(tid);
    let eps = 1e-6 * (1.0 + p.length());
    let ray = Ray::new(p - n * eps, -n);
    bvh.raycast_filtered(&o.mesh, &ray, f64::INFINITY, |t| t != tid).map(|h| h.point)
}

impl Tool for MeasureTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn copy_text(&self) -> Option<String> {
        (!self.items.is_empty()).then(|| self.items.iter().map(|m| m.label.clone()).collect::<Vec<_>>().join("\n"))
    }

    fn name(&self) -> &'static str {
        "Measure"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Measure");
        egui::ComboBox::from_label("Type").selected_text(self.kind.label()).show_ui(ui, |ui| {
            for k in MeasureType::ALL {
                ui.selectable_value(&mut self.kind, k, k.label());
            }
        });
        ui.add_enabled_ui(self.kind.uses_direction(), |ui| {
            egui::ComboBox::from_label("Direction")
                .selected_text(match self.direction {
                    Direction::Normal => "Normal",
                    Direction::Axis(0) => "Axis X",
                    Direction::Axis(1) => "Axis Y",
                    Direction::Axis(_) => "Axis Z",
                })
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.direction, Direction::Normal, "Normal");
                    for (i, n) in ["Axis X", "Axis Y", "Axis Z"].iter().enumerate() {
                        ui.selectable_value(&mut self.direction, Direction::Axis(i), *n);
                    }
                });
        });
        egui::ComboBox::from_label("Snapping")
            .selected_text(match self.snapping {
                Snapping::None => "No Snapping",
                Snapping::Vertices => "Snap To Vertices",
                Snapping::Edges => "Snap To Edges",
                Snapping::GroupBorders => "Snap To Group Borders",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.snapping, Snapping::None, "No Snapping");
                ui.selectable_value(&mut self.snapping, Snapping::Vertices, "Snap To Vertices");
                ui.selectable_value(&mut self.snapping, Snapping::Edges, "Snap To Edges");
                ui.selectable_value(&mut self.snapping, Snapping::GroupBorders, "Snap To Group Borders");
            });
        if self.kind == MeasureType::FitCylinder {
            let max = cx.scene.active().map(|o| o.bounds().diagonal() * 0.3).unwrap_or(1.0).max(self.fit_radius);
            ui.add(egui::Slider::new(&mut self.fit_radius, 0.0..=max).text("fit radius")).on_hover_text("Size of the surface patch the cylinder is fitted to");
        }
        ui.small(match self.kind {
            MeasureType::PointToPoint => "Click two surface points, or drag from one to the other.",
            MeasureType::PointCoordinates => "Click the surface.",
            MeasureType::VesselDiameter => "Click a vessel wall: equivalent diameter of the cross-section perpendicular to the vessel (green ring) and the wall-to-wall distance.",
            MeasureType::FitCylinder => "Click the surface: a cylinder is fitted to the patch around the click.",
            _ => "Click the surface: the distance to the next (min) or last (max) surface along the direction, through the object (interior) or out of it (exterior).",
        });
        ui.separator();
        let mut remove = None;
        for (i, m) in self.items.iter().enumerate() {
            ui.horizontal(|ui| {
                if ui.radio(self.selected == Some(i), &m.label).clicked() {
                    self.selected = Some(i);
                    if let Some(l) = m.length {
                        self.target_length = l;
                    }
                }
                if ui.small_button("x").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.items.remove(i);
            self.selected = None;
        }
        let measurable = self.selected.and_then(|i| self.items.get(i)).is_some_and(|m| m.length.is_some());
        ui.add_enabled_ui(measurable, |ui| {
            ui.horizontal(|ui| {
                ui.label("Set target length");
                ui.add(egui::DragValue::new(&mut self.target_length).speed(0.01).range(0.0..=1e9).max_decimals(4));
                if ui.button("Apply").on_hover_text("Scale the object so the selected measurement has this length").clicked() {
                    self.apply_target_length(cx);
                }
            });
        });
        if !self.items.is_empty() && ui.button("Clear all").clicked() {
            self.items.clear();
            self.selected = None;
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.pressed = self.hit(cx, pos).map(|(id, h)| (pos, id, h.point));
        }
        if !input.released {
            return;
        }
        let Some((p0, id0, a)) = self.pressed.take() else { return };
        let Some((id, hit)) = self.hit(cx, pos) else { return };
        // Point to point by dragging from one point to another.
        if self.kind == MeasureType::PointToPoint && p0.distance(pos) > 5.0 {
            let d = a.distance(hit.point);
            self.items.push(Measurement { object: id0, a, b: hit.point, length: Some(d), label: format!("↔ {d:.3}"), ring: Vec::new() });
            self.selected = Some(self.items.len() - 1);
            self.target_length = d;
            self.first = None;
            *cx.status = format!("Distance: {d:.4}");
            return;
        }
        if p0.distance(pos) <= 5.0 {
            self.measure_at(cx, id, hit);
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        for (i, m) in self.items.iter().enumerate() {
            let c = if self.selected == Some(i) { Vec4::new(1.0, 0.6, 0.2, 1.0) } else { Vec4::new(1.0, 0.85, 0.2, 1.0) };
            ov.line(m.a, m.b, c);
            for s in &m.ring {
                ov.line(s[0], s[1], Vec4::new(0.3, 1.0, 0.6, 1.0));
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        for m in &self.items {
            if let Some(p) = cx.project((m.a + m.b) * 0.5) {
                painter.text(p, egui::Align2::LEFT_BOTTOM, &m.label, egui::FontId::proportional(14.0), egui::Color32::from_rgb(255, 220, 80));
            }
            for q in [m.a, m.b] {
                if let Some(p) = cx.project(q) {
                    painter.circle_filled(p, 3.0, egui::Color32::from_rgb(255, 220, 80));
                }
            }
        }
        if let Some((_, a)) = self.first {
            if let Some(p) = cx.project(a) {
                painter.circle_filled(p, 4.0, ui_util::ACCENT);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::Camera;
    use crate::scene::{History, Scene};
    use mm_core::primitives;

    /// Hit on the triangle of `id` closest to `p` facing `n`.
    fn hit_at(scene: &Scene, id: ObjectId, p: DVec3, n: DVec3) -> mm_core::bvh::RayHit {
        let m = &scene.get(id).unwrap().mesh;
        let tid = m.triangle_ids().filter(|&t| m.tri_normal(t).dot(n) > 0.99).min_by(|&a, &b| m.tri_centroid(a).distance(p).total_cmp(&m.tri_centroid(b).distance(p))).unwrap();
        mm_core::bvh::RayHit { tid, t: 0.0, point: p, bary: DVec3::ZERO }
    }

    #[test]
    fn interior_exterior_cylinder_and_target_length() {
        let mut scene = Scene::default();
        let a = scene.add("a", primitives::cube(2.0, 2));
        let mut small = primitives::cube(1.0, 2);
        small.transform(|p| p + DVec3::new(3.0, 0.0, 0.0));
        scene.add("b", small);
        let mut tube = primitives::cylinder(0.75, 6.0, 48, 12, false);
        tube.transform(|p| p + DVec3::new(0.0, 6.0, -3.0));
        let c = scene.add("c", tube);
        scene.active = Some(a);
        let (mut history, mut camera, mut status) = (History::default(), Camera::default(), String::new());
        let ctx = egui::Context::default();
        let mut cx = ToolCx {
            scene: &mut scene,
            history: &mut history,
            camera: &mut camera,
            status: &mut status,
            egui: &ctx,
            vp: Viewport { rect: egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0)) },
            brush_scroll: crate::navigation::BrushScroll::Shift,
        };
        let mut t = MeasureTool::new(&mut cx);
        let p = DVec3::new(1.0, 0.2, 0.3);
        let h = hit_at(cx.scene, a, p, DVec3::X);
        for (kind, want) in [(MeasureType::MinInterior, 2.0), (MeasureType::MinExterior, 1.5), (MeasureType::MaxExterior, 2.5), (MeasureType::MaxInterior, 2.0)] {
            t.kind = kind;
            t.measure_at(&mut cx, a, h);
            let got = t.items.last().unwrap().length.unwrap();
            assert!((got - want).abs() < 1e-9, "{kind:?}: {got} vs {want}");
        }
        // Along an axis rather than the normal: same here, since the face looks along +X.
        t.kind = MeasureType::MinInterior;
        t.direction = Direction::Axis(0);
        t.measure_at(&mut cx, a, h);
        assert!((t.items.last().unwrap().length.unwrap() - 2.0).abs() < 1e-9);
        // A cylinder fit on the tube gives its diameter.
        t.kind = MeasureType::FitCylinder;
        t.fit_radius = 1.5;
        let q = DVec3::new(0.75, 6.0, 0.0);
        let hc = hit_at(cx.scene, c, q, DVec3::X);
        t.measure_at(&mut cx, c, hc);
        let d = t.items.last().unwrap().length.unwrap();
        assert!((d - 1.5).abs() < 0.02, "cylinder diameter {d}");
        // Setting the first measurement's length scales its object.
        t.selected = Some(0);
        t.target_length = 4.0;
        t.apply_target_length(&mut cx);
        let e = cx.scene.get(a).unwrap().bounds().extents();
        assert!((e.x - 4.0).abs() < 1e-9, "{e}");
        assert!((t.items[0].length.unwrap() - 4.0).abs() < 1e-9);
    }
}
