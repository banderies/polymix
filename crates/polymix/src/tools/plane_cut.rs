//! Plane Cut (Meshmixer Edit > Plane Cut). Drag a line across the view to place the cut
//! plane, Shift+click a vessel to orient the plane perpendicular to it (cutting only that
//! branch), nudge / rotate numerically, then Accept. The half that will be removed is tinted
//! red.

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use glam::DQuat;
use mm_core::Plane;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutType {
    Discard,
    Slice,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FillType {
    None,
    /// Fewest triangles.
    Minimal,
    /// Remeshed cap; the cut edge stays as cut.
    Fixed,
    /// Remeshed cap, and the band across the cut is remeshed too so cap and side blend.
    Remeshed,
}

pub struct PlaneCutTool {
    object: Option<ObjectId>,
    source: Arc<Mesh>,
    origin: DVec3,
    normal: DVec3,
    half_size: f64,
    cut_type: CutType,
    fill: FillType,
    /// Slice mode: the far half gets its own face group.
    slice_groups: bool,
    /// Restrict the cut to the cross-section nearest this point (vessel mode).
    local_seed: Option<DVec3>,
    local_radius: f64,
    drag: Option<(egui::Pos2, egui::Pos2)>,
    job: Option<Job<(Mesh, String)>>,
    error: Option<String>,
    avg_edge: f64,
}

impl PlaneCutTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.mesh.clone()),
            None => (None, Arc::new(Mesh::new())),
        };
        let b = source.bounds();
        Self {
            object,
            origin: b.center(),
            normal: cx.camera.world_up(),
            half_size: (b.diagonal() * 0.6).max(1e-3),
            cut_type: CutType::Discard,
            fill: FillType::Remeshed,
            slice_groups: false,
            local_seed: None,
            local_radius: b.diagonal() * 0.15,
            drag: None,
            job: None,
            error: None,
            avg_edge: source.average_edge_length(),
            source,
        }
    }

    pub fn plane(&self) -> Plane {
        Plane::from_point_normal(self.origin, self.normal)
    }

    fn rotate(&mut self, axis: DVec3, deg: f64) {
        self.normal = (DQuat::from_axis_angle(axis.normalize(), deg.to_radians()) * self.normal).normalize();
    }

    /// Vessel mode: the plane square to the vessel at wall point `p`, cutting only that
    /// branch. The estimated axis points toward the branch tip, the side that is removed.
    fn square_to_vessel(&mut self, p: DVec3) -> Result<f64, String> {
        let x = mm_edit::estimate_vessel_cross_section(&self.source, None, p, None).map_err(|e| format!("{e}"))?;
        self.normal = x.axis;
        self.origin = x.center;
        self.local_radius = x.radius * 4.0;
        self.local_seed = Some(p);
        Ok(2.0 * x.radius)
    }

    fn key(c: CutType, f: FillType) -> (&'static str, &'static str) {
        let cut = if c == CutType::Discard { "discard" } else { "slice" };
        let fill = match f {
            FillType::None => "none",
            FillType::Minimal => "minimal",
            FillType::Fixed => "fixed",
            FillType::Remeshed => "remeshed",
        };
        (cut, fill)
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let src = self.source.clone();
        let plane = self.plane();
        let (cut_type, fill, seed, avg, groups) = (self.cut_type, self.fill, self.local_seed, self.avg_edge, self.slice_groups);
        self.error = None;
        self.job = Some(Job::spawn("Plane cut", cx.egui.clone(), move |_p| {
            let mut m = (*src).clone();
            let msg = run_cut(&mut m, plane, cut_type, fill, seed, avg, groups)?;
            Ok((m, msg))
        }));
    }
}

/// Perform the cut. The plane normal points toward the half that is discarded.
fn run_cut(m: &mut Mesh, plane: Plane, cut_type: CutType, fill: FillType, seed: Option<DVec3>, avg_edge: f64, slice_groups: bool) -> anyhow::Result<String> {
    let keep = match cut_type {
        CutType::Discard => mm_edit::KeepSide::Below,
        CutType::Slice => mm_edit::KeepSide::Both,
    };
    let cut_fill = match fill {
        FillType::None => mm_edit::CutFill::None,
        FillType::Minimal => mm_edit::CutFill::Flat,
        FillType::Fixed | FillType::Remeshed => mm_edit::CutFill::FlatRemeshed { target_edge_length: avg_edge },
    };
    let r = match seed {
        Some(seed) => mm_edit::local_plane_cut(m, plane, seed, keep, cut_fill)?.cut,
        None => mm_edit::plane_cut(m, plane, &mm_edit::PlaneCutParams { keep, fill: cut_fill, ..Default::default() })?,
    };
    if fill == FillType::Remeshed && !r.cap_triangles.is_empty() {
        // Remesh a band across each capped cut edge (cap groups keep the crease crisp).
        let mut band = mm_core::TriSelection::new(m);
        for lp in r.cut_loops.iter().zip(&r.loop_capped).filter(|(_, c)| **c).map(|(l, _)| l) {
            for &v in lp {
                if m.is_vertex(v) {
                    for t in m.vertex_triangles(v) {
                        band.insert(t);
                    }
                }
            }
        }
        band.expand(m);
        let band: Vec<u32> = band.iter().collect();
        if !band.is_empty() {
            let params = mm_remesh::RemeshParams {
                target_edge_length: avg_edge,
                iterations: 6,
                preserve_group_boundaries: true,
                region: Some(band),
                ..Default::default()
            };
            mm_remesh::remesh(m, &params, None)?;
        }
    }
    if cut_type == CutType::Slice && slice_groups {
        // The far half (above the plane) gets a new face group.
        let g = m.max_group_id() + 1;
        for comp in mm_core::topology::connected_components(m) {
            let c = comp.iter().map(|&t| m.tri_centroid(t)).sum::<DVec3>() / comp.len() as f64;
            if plane.signed_distance(c) > 0.0 {
                for t in comp {
                    m.set_tri_group(t, g);
                }
            }
        }
    }
    let capped = r.loop_capped.iter().filter(|c| **c).count();
    Ok(format!(
        "Plane cut: removed {} triangles, {} cut loop(s), {} capped",
        r.removed_triangles,
        r.cut_loops.len(),
        capped
    ))
}

impl Tool for PlaneCutTool {
    fn name(&self) -> &'static str {
        "Plane Cut"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        use polymix_workspace::agent::vec3;
        let (cut, fill) = Self::key(self.cut_type, self.fill);
        Some(serde_json::json!({
            "origin_mm": vec3(self.origin.to_array(), 5),
            "normal": vec3(self.normal.to_array(), 6),
            "cut_type": cut,
            "fill": fill,
            "slice_groups": self.slice_groups,
            "only_nearest_cross_section": self.local_seed.is_some(),
        }))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "origin_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "A point of the plane, mm"},
            "normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": "The plane's normal, pointing to the side that is removed (discard)"},
            "cut_type": {"type": "string", "enum": ["discard", "slice"], "description": "discard: remove the side the normal points to; slice: cut through and keep both"},
            "fill": {"type": "string", "enum": ["none", "minimal", "fixed", "remeshed"], "description": "How the cut is capped: none, minimal (fewest triangles), fixed (remeshed cap, cut edge kept), remeshed (cap blended into the side)"},
            "slice_groups": {"type": "boolean", "description": "slice: the far half gets its own face group"},
            "only_nearest_cross_section": {"type": "boolean", "description": "Cut only the cross-section nearest the origin (a vessel branch), not everything the plane crosses"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let a = Args(&v);
        self.origin = DVec3::from_array(a.point("origin_mm")?);
        self.normal = DVec3::from_array(a.point("normal")?).try_normalize().ok_or("normal cannot be zero")?;
        self.cut_type = if a.opt_str("cut_type") == Some("slice") { CutType::Slice } else { CutType::Discard };
        self.fill = match a.opt_str("fill") {
            Some("none") => FillType::None,
            Some("minimal") => FillType::Minimal,
            Some("fixed") => FillType::Fixed,
            _ => FillType::Remeshed,
        };
        self.slice_groups = a.opt_bool("slice_groups").unwrap_or(false);
        let local = a.opt_bool("only_nearest_cross_section").unwrap_or(false);
        if !local {
            self.local_seed = None;
        } else if self.local_seed.is_none() || changes.get("origin_mm").is_some() {
            self.local_seed = Some(self.origin);
        }
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::vec3;
        serde_json::json!({
            "object": self.object,
            "plane": {"origin_mm": vec3(self.origin.to_array(), 4), "normal_toward_removed_side": vec3(self.normal.to_array(), 5)},
            "only_near_mm": self.local_seed.map(|s| vec3(s.to_array(), 4)),
            "cutting": self.job.is_some(),
            "error": self.error,
            "next": "place the plane (settings origin_mm / normal, or square_to_vessel), then accept (mesh_tool accept) to cut",
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("square_to_vessel", "Put the plane square to the vessel through the wall point nearest `at`, cutting only that branch; the tip side is removed (as Shift+clicking the vessel does)", json!({"at": arg::point("A point on or near the vessel wall, mm")})).required(&["at"]),
            Action::new("move", "Move the plane along its normal", json!({"mm": arg::number("Distance in mm (positive toward the removed side)")})).required(&["mm"]),
            Action::new("tilt", "Turn the plane about an axis through its origin", json!({"axis": arg::point("Axis direction"), "degrees": arg::number("Angle, degrees")})).required(&["axis", "degrees"]),
            Action::plain("flip", "Remove the other side"),
            Action::plain("recentre", "Move the origin to the centre of the object's bounds"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        match action {
            "square_to_vessel" => {
                let id = self.object.ok_or("no active object")?;
                let o = cx.scene.get_mut(id).ok_or("no object")?;
                let p = o.bvh().nearest(&o.mesh, DVec3::from_array(a.point("at")?), f64::INFINITY).ok_or("no surface")?.point;
                let d = self.square_to_vessel(p)?;
                Ok(format!("Plane square to a ⌀ {d:.2} mm vessel; the tip side will be removed"))
            }
            "move" => {
                let d = a.f64("mm")?;
                self.origin += self.normal * d;
                if let Some(s) = &mut self.local_seed {
                    *s += self.normal * d;
                }
                Ok(format!("Plane moved {d} mm"))
            }
            "tilt" => {
                let axis = DVec3::from_array(a.point("axis")?);
                if axis.length_squared() < 1e-24 {
                    return Err("the axis cannot be zero".into());
                }
                self.rotate(axis, a.f64("degrees")?);
                Ok("Plane tilted".into())
            }
            "flip" => {
                self.normal = -self.normal;
                Ok("The other side will be removed".into())
            }
            "recentre" => {
                self.origin = self.source.bounds().center();
                Ok("Plane recentred".into())
            }
            _ => Err(format!("Plane Cut has no action {action}")),
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Plane Cut");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        ui.small("Drag across the view to place the plane. Shift+click a vessel to cut it square. The red side is removed.");
        ui.horizontal(|ui| {
            ui.label("Cut type");
            ui.selectable_value(&mut self.cut_type, CutType::Discard, "Cut (discard half)");
            ui.selectable_value(&mut self.cut_type, CutType::Slice, "Slice (keep both)");
        });
        egui::ComboBox::from_label("Fill type")
            .selected_text(match self.fill {
                FillType::None => "No Fill",
                FillType::Minimal => "Minimal Fill",
                FillType::Fixed => "Fixed Fill",
                FillType::Remeshed => "Remeshed Fill",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.fill, FillType::None, "No Fill");
                ui.selectable_value(&mut self.fill, FillType::Minimal, "Minimal Fill").on_hover_text("Fewest triangles");
                ui.selectable_value(&mut self.fill, FillType::Fixed, "Fixed Fill").on_hover_text("Remeshed cap; the cut edge stays as cut");
                ui.selectable_value(&mut self.fill, FillType::Remeshed, "Remeshed Fill").on_hover_text("Remeshed cap blended into the side");
            });
        if self.cut_type == CutType::Slice {
            ui.checkbox(&mut self.slice_groups, "Slice groups").on_hover_text("Give the far half its own face group");
        }
        let mut local = self.local_seed.is_some();
        if ui.checkbox(&mut local, "Only the nearest cross-section (vessel mode)").changed() {
            self.local_seed = if local { Some(self.origin) } else { None };
        }
        ui_util::section(ui, "Plane");
        let step = (self.half_size * 0.005).max(1e-5);
        let mut off = 0.0;
        ui.horizontal(|ui| {
            ui.label("Move along normal");
            ui.add(egui::DragValue::new(&mut off).speed(step).max_decimals(4));
            if ui.small_button("−").clicked() {
                off = -self.avg_edge.max(step);
            }
            if ui.small_button("+").clicked() {
                off = self.avg_edge.max(step);
            }
        });
        if off != 0.0 {
            self.origin += self.normal * off;
            if let Some(s) = &mut self.local_seed {
                *s += self.normal * off;
            }
        }
        let (u, v) = mm_core::geom::orthonormal_basis(self.normal);
        ui.horizontal(|ui| {
            ui.label("Tilt");
            for (lbl, axis, d) in [("↑", u, -2.0), ("↓", u, 2.0), ("←", v, -2.0), ("→", v, 2.0)] {
                if ui.small_button(lbl).clicked() {
                    self.rotate(axis, d);
                }
            }
        });
        ui.horizontal_wrapped(|ui| {
            if ui.button("Flip side").clicked() {
                self.normal = -self.normal;
            }
            if ui.button("Align to view").clicked() {
                self.normal = -cx.camera.forward();
            }
            for (n, a) in [("X", DVec3::X), ("Y", DVec3::Y), ("Z", DVec3::Z)] {
                if ui.button(n).clicked() {
                    self.normal = a;
                }
            }
            if ui.button("Re-center").clicked() {
                self.origin = self.source.bounds().center();
            }
        });
        ui.label(format!(
            "Origin ({:.2}, {:.2}, {:.2})  normal ({:.3}, {:.3}, {:.3})",
            self.origin.x, self.origin.y, self.origin.z, self.normal.x, self.normal.y, self.normal.z
        ));

        if let Some(job) = &self.job {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label("Cutting…");
            });
            if let Some(r) = job.poll() {
                self.job = None;
                match r {
                    Ok((mesh, msg)) => {
                        let Some(id) = self.object else { return ToolOutcome::Close };
                        if !cx.scene.get(id).map(|o| Arc::ptr_eq(&o.mesh, &self.source)).unwrap_or(false) {
                            *cx.status = "Plane cut cancelled: the object changed while cutting.".into();
                            return ToolOutcome::Close;
                        }
                        cx.history.push("Plane cut", cx.scene);
                        if let Some(o) = cx.scene.get_mut(id) {
                            o.set_mesh(mesh);
                        }
                        *cx.status = msg;
                        return ToolOutcome::Close;
                    }
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            }
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, self.job.is_none());
        if accept {
            self.start(cx);
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.clicked && input.shift {
            // Vessel mode: orient perpendicular to the clicked branch.
            if let Some((_, hit)) = cx.pick(pos, self.object) {
                let _ = self.square_to_vessel(hit.point);
                self.drag = None;
            }
            return;
        }
        if input.pressed && !input.shift {
            self.drag = Some((pos, pos));
        }
        if input.down {
            if let Some((a, _)) = self.drag {
                self.drag = Some((a, pos));
            }
        }
        if input.released {
            if let Some((a, b)) = self.drag.take() {
                if a.distance(b) > 8.0 {
                    // Plane containing the stroke and the view direction, at the current depth.
                    let depth_plane = Plane::from_point_normal(self.origin, cx.camera.forward());
                    let hit = |p: egui::Pos2| {
                        let r = cx.ray_at(p);
                        let t = -depth_plane.signed_distance(r.origin) / r.dir.dot(depth_plane.normal);
                        r.at(t)
                    };
                    let (pa, pb) = (hit(a), hit(b));
                    let n = (pb - pa).cross(cx.camera.forward()).normalize_or_zero();
                    if n != DVec3::ZERO {
                        self.normal = n;
                        self.origin = (pa + pb) * 0.5;
                        self.local_seed = None;
                    }
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        let pl = self.plane();
        let region = self.local_seed.map(|s| (s, self.local_radius));
        if self.cut_type == CutType::Discard {
            ov.cut_plane = Some((pl, region));
            ov.cut_target = self.object;
        }
        let size = if self.local_seed.is_some() { self.local_radius } else { self.half_size };
        ov.plane_patch(self.origin, self.normal, size, Vec4::new(0.95, 0.75, 0.3, 0.12), Vec4::new(1.0, 0.8, 0.3, 0.9));
        ov.line(self.origin, self.origin + self.normal * size * 0.3, Vec4::new(1.0, 0.3, 0.2, 1.0));
    }

    fn paint(&mut self, painter: &egui::Painter, _cx: &mut ToolCx) {
        if let Some((a, b)) = self.drag {
            painter.line_segment([a, b], egui::Stroke::new(2.0, ui_util::ACCENT));
        }
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.job.is_none() {
            self.start(cx);
        }
        false
    }

    fn keys(&mut self, i: &egui::InputState, _cx: &mut ToolCx) -> bool {
        if i.key_pressed(egui::Key::Q) {
            self.cut_type = if self.cut_type == CutType::Discard { CutType::Slice } else { CutType::Discard };
            return true;
        }
        false
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|j| ("Cutting".to_string(), j.progress()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn mean_edge(m: &Mesh) -> f64 {
        let (mut sum, mut n) = (0.0, 0);
        for t in m.triangle_ids() {
            let [a, b, c] = m.triangle(t).map(|v| m.vertex(v));
            sum += a.distance(b) + b.distance(c) + c.distance(a);
            n += 3;
        }
        sum / n as f64
    }

    #[test]
    fn fill_types_close_the_cut() {
        let src = primitives::icosphere(1.0, 4);
        let avg = mean_edge(&src);
        let plane = Plane::from_point_normal(DVec3::new(0.0, 0.0, 0.3), DVec3::new(0.2, 0.0, 1.0));
        for fill in [FillType::Minimal, FillType::Fixed, FillType::Remeshed] {
            let mut m = src.clone();
            run_cut(&mut m, plane, CutType::Discard, fill, None, avg, false).unwrap();
            assert!(m.is_closed(), "{fill:?} leaves the mesh closed");
            // Nothing is left above the plane, and the cap stays flat (its own group).
            let cap_group = m.max_group_id();
            for t in m.triangle_ids() {
                for v in m.triangle(t) {
                    assert!(plane.signed_distance(m.vertex(v)) < 1e-6, "{fill:?}: vertex above the plane");
                    if m.tri_group(t) == cap_group {
                        assert!(plane.signed_distance(m.vertex(v)).abs() < 1e-6, "{fill:?}: cap vertex off the plane");
                    }
                }
            }
        }
        // Remeshed Fill evens out the slivers the cut leaves along the rim.
        let rim_min_angle = |fill| {
            let mut m = src.clone();
            run_cut(&mut m, plane, CutType::Discard, fill, None, avg, false).unwrap();
            let cap_group = m.max_group_id();
            m.triangle_ids()
                .filter(|&t| m.tri_group(t) != cap_group && m.triangle(t).iter().any(|&v| plane.signed_distance(m.vertex(v)).abs() < 1e-6))
                .map(|t| {
                    let p = m.triangle(t).map(|v| m.vertex(v));
                    (0..3).map(|i| (p[(i + 1) % 3] - p[i]).angle_between(p[(i + 2) % 3] - p[i])).fold(f64::INFINITY, f64::min)
                })
                .fold(f64::INFINITY, f64::min)
        };
        let (fixed, remeshed) = (rim_min_angle(FillType::Fixed), rim_min_angle(FillType::Remeshed));
        assert!(remeshed > fixed, "rim min angle {:.1}° > {:.1}°", remeshed.to_degrees(), fixed.to_degrees());
        let mut none = src.clone();
        run_cut(&mut none, plane, CutType::Discard, FillType::None, None, avg, false).unwrap();
        assert!(!none.is_closed());
    }

    #[test]
    fn slice_groups_split_the_halves() {
        let src = primitives::icosphere(1.0, 3);
        let avg = mean_edge(&src);
        let plane = Plane::from_point_normal(DVec3::ZERO, DVec3::Z);
        let mut plain = src.clone();
        run_cut(&mut plain, plane, CutType::Slice, FillType::Fixed, None, avg, false).unwrap();
        let mut grouped = src.clone();
        run_cut(&mut grouped, plane, CutType::Slice, FillType::Fixed, None, avg, true).unwrap();
        let g = grouped.max_group_id();
        assert!(g > plain.max_group_id(), "a new group");
        for t in grouped.triangle_ids() {
            // Cap triangles lie on the plane; the rest tell the halves apart.
            let z = grouped.tri_centroid(t).z;
            if z.abs() > 1e-9 {
                assert_eq!(grouped.tri_group(t) == g, z > 0.0, "only the upper half gets the new group");
            }
        }
    }
}
