//! Symmetry plane of the brush tools (Sculpt and Select): every dab is mirrored across it.
//! The plane is free: X / Y / Z through the object's centre, through the middle of the view,
//! through three picked points, on the selection's best-fit plane, typed in, or dragged with
//! a gizmo while Set is on (Meshmixer's Symmetry → Set, with Accept / Cancel).

use crate::tools::*;
use crate::ui_util;
use glam::DQuat;
use mm_core::Aabb;
use polymix_workspace::agent::vec3;

/// Symmetry plane: through `point`, facing `normal` (unit).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SymPlane {
    pub point: DVec3,
    pub normal: DVec3,
}

impl SymPlane {
    /// Through `point`, facing `normal` (+X if that is zero).
    pub fn new(point: DVec3, normal: DVec3) -> Self {
        Self { point, normal: normal.normalize_or(DVec3::X) }
    }

    /// Across world axis `axis` (0..2) through the centre of `bounds`.
    pub fn axis(bounds: &Aabb, axis: usize) -> Self {
        let mut n = DVec3::ZERO;
        n[axis.min(2)] = 1.0;
        Self::new(bounds.center(), n)
    }

    pub fn signed(&self, p: DVec3) -> f64 {
        (p - self.point).dot(self.normal)
    }

    pub fn distance(&self, p: DVec3) -> f64 {
        self.signed(p).abs()
    }

    pub fn reflect(&self, p: DVec3) -> DVec3 {
        p - self.normal * (2.0 * self.signed(p))
    }

    pub fn reflect_dir(&self, v: DVec3) -> DVec3 {
        v - self.normal * (2.0 * v.dot(self.normal))
    }

    pub fn project(&self, p: DVec3) -> DVec3 {
        match self.axis_index() {
            // Exactly on an X / Y / Z plane.
            Some(k) => {
                let mut q = p;
                q[k] = self.point[k];
                q
            }
            None => p - self.normal * self.signed(p),
        }
    }

    /// Distance of the plane from the origin along its normal.
    pub fn offset(&self) -> f64 {
        self.point.dot(self.normal)
    }

    pub fn set_offset(&mut self, d: f64) {
        self.point += self.normal * (d - self.offset());
    }

    /// The world axis the plane is square to (the X / Y / Z presets).
    pub fn axis_index(&self) -> Option<usize> {
        (0..3).find(|&i| self.normal[i].abs() > 1.0 - 1e-9)
    }

    /// Through three points; `None` if they (nearly) lie on a line.
    pub fn through(a: DVec3, b: DVec3, c: DVec3) -> Option<Self> {
        let n = (b - a).cross(c - a);
        let size = (b - a).length_squared().max((c - a).length_squared()).max((c - b).length_squared());
        (size > 0.0 && n.length_squared() > 1e-10 * size * size).then(|| Self::new((a + b + c) / 3.0, n))
    }

    /// Least-squares plane of `points`; `None` for fewer than three or points on a line.
    pub fn fit(points: &[DVec3]) -> Option<Self> {
        if points.len() < 3 {
            return None;
        }
        let c = points.iter().sum::<DVec3>() / points.len() as f64;
        let mut m = [[0.0; 3]; 3];
        for p in points {
            let d = *p - c;
            for (i, row) in m.iter_mut().enumerate() {
                for (j, x) in row.iter_mut().enumerate() {
                    *x += d[i] * d[j];
                }
            }
        }
        let (vals, vecs) = mm_core::linalg::sym_eigen3(m);
        // Spread in two directions, not one.
        (vals[1] > 1e-10 * vals[2].max(1e-300)).then(|| Self::new(c, vecs[0]))
    }

    /// Through the middle of the view, splitting the screen into left and right halves.
    pub fn from_view(camera: &crate::camera::Camera) -> Self {
        Self::new(camera.target, camera.right())
    }

    /// Axes in the plane: the other two world axes for an X / Y / Z plane.
    pub fn basis(&self) -> (DVec3, DVec3) {
        match self.axis_index() {
            Some(k) => {
                let (i, j) = [(1, 2), (0, 2), (0, 1)][k];
                (DVec3::AXES[i], DVec3::AXES[j])
            }
            None => mm_core::geom::orthonormal_basis(self.normal),
        }
    }

    /// A rectangle on the plane covering `bounds` with a margin: its corners in order.
    pub fn patch(&self, bounds: &Aabb) -> [DVec3; 4] {
        let (u, v) = self.basis();
        let c = self.project(bounds.center());
        let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
        for k in 0..8 {
            let corner = DVec3::new(
                if k & 1 == 0 { bounds.min.x } else { bounds.max.x },
                if k & 2 == 0 { bounds.min.y } else { bounds.max.y },
                if k & 4 == 0 { bounds.min.z } else { bounds.max.z },
            );
            let q = DVec2::new((corner - c).dot(u), (corner - c).dot(v));
            lo = lo.min(q);
            hi = hi.max(q);
        }
        let pad = (hi - lo) * 0.04;
        let (lo, hi) = (lo - pad, hi + pad);
        [c + u * lo.x + v * lo.y, c + u * hi.x + v * lo.y, c + u * hi.x + v * hi.y, c + u * lo.x + v * hi.y]
    }
}

/// Gizmo handles of Set mode.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    /// Slide the plane along its normal.
    Move,
    /// Tilt the plane about one of its two in-plane axes.
    Tilt(usize),
}

#[derive(Clone, Copy, Debug)]
struct GizmoDrag {
    handle: Handle,
    start: egui::Pos2,
    plane: SymPlane,
    anchor: DVec3,
    axes: (DVec3, DVec3),
}

/// Set mode: the plane is being placed.
struct SymEdit {
    /// The plane when Set started (Cancel restores it).
    before: Option<SymPlane>,
    /// Clicks on the surface collect three points for the plane.
    picking: bool,
    picks: Vec<DVec3>,
    drag: Option<GizmoDrag>,
    hover: Option<Handle>,
}

/// Tilt snaps to this many degrees with Shift.
const TILT_SNAP_DEG: f64 = 15.0;

/// The symmetry state of a brush tool, with its Set mode.
pub struct Symmetry {
    pub enabled: bool,
    /// None until placed: then X through the centre of the object being brushed.
    plane: Option<SymPlane>,
    /// Object the plane was placed for: an X / Y / Z plane re-centres on another object.
    object: Option<ObjectId>,
    edit: Option<SymEdit>,
}

impl Symmetry {
    /// Off, with `plane` placed for `object` (or placed on first use).
    pub fn new(plane: Option<SymPlane>, object: Option<ObjectId>) -> Self {
        Self { enabled: false, plane, object, edit: None }
    }

    /// For the active object of `scene`: X through its centre.
    pub fn for_scene(scene: &Scene) -> Self {
        Self::new(scene.active().map(|o| SymPlane::axis(&o.bounds(), 0)), scene.active)
    }

    /// The plane to mirror across, when symmetry is on.
    pub fn active(&self) -> Option<SymPlane> {
        self.plane.filter(|_| self.enabled)
    }

    pub fn set_plane(&mut self, p: SymPlane) {
        self.plane = Some(p);
    }

    pub fn editing(&self) -> bool {
        self.edit.is_some()
    }

    /// The plane for brushing object `id` with bounds `bounds`: placed on first use, and an
    /// X / Y / Z plane follows the centre of a newly brushed object (a free plane stays).
    pub fn place(&mut self, id: ObjectId, bounds: &Aabb) -> SymPlane {
        if self.plane.is_none() || self.object != Some(id) {
            self.object = Some(id);
            match self.plane {
                Some(p) if p.axis_index().is_none() => {}
                p => self.plane = Some(SymPlane::axis(bounds, p.and_then(|p| p.axis_index()).unwrap_or(0))),
            }
        }
        self.plane.unwrap_or_else(|| SymPlane::axis(bounds, 0))
    }

    fn active_bounds(cx: &ToolCx) -> Option<(ObjectId, Aabb)> {
        cx.scene.active().map(|o| (o.id, o.bounds()))
    }

    /// X / Y / Z (`axis`) through the centre of the active object.
    pub fn preset(&mut self, cx: &ToolCx, axis: usize) {
        if let Some((id, b)) = Self::active_bounds(cx) {
            self.plane = Some(SymPlane::axis(&b, axis));
            self.object = Some(id);
        }
    }

    /// Back through the active object's centre, across its axis (or X for a free plane).
    pub fn reset(&mut self, cx: &ToolCx) {
        self.preset(cx, self.plane.and_then(|p| p.axis_index()).unwrap_or(0));
    }

    /// The best-fit plane of the active object's selected faces.
    pub fn fit_selection(&mut self, cx: &ToolCx) -> Result<(), String> {
        let o = cx.scene.active().ok_or("no active object")?;
        if o.selection.is_empty() {
            return Err("select faces first (e.g. a band along the seam)".into());
        }
        let mut verts: Vec<u32> = o.selection.iter().flat_map(|t| o.mesh.triangle(t)).collect();
        verts.sort_unstable();
        verts.dedup();
        let pts: Vec<DVec3> = verts.into_iter().map(|v| o.mesh.vertex(v)).collect();
        self.plane = Some(SymPlane::fit(&pts).ok_or("the selected faces lie on a line")?);
        self.object = Some(o.id);
        Ok(())
    }

    /// Through three points.
    pub fn through_points(&mut self, pts: &[DVec3]) -> Result<(), String> {
        let [a, b, c] = pts else { return Err("three points are needed".into()) };
        self.plane = Some(SymPlane::through(*a, *b, *c).ok_or("the three points lie on a line")?);
        Ok(())
    }

    /// Start Set mode.
    pub fn begin_edit(&mut self, cx: &ToolCx) {
        if let Some((id, b)) = Self::active_bounds(cx) {
            self.place(id, &b);
        }
        self.enabled = true;
        self.edit = Some(SymEdit { before: self.plane, picking: false, picks: Vec::new(), drag: None, hover: None });
    }

    /// Leave Set mode keeping the plane. False if not in Set mode.
    pub fn accept_edit(&mut self) -> bool {
        self.edit.take().is_some()
    }

    /// Leave Set mode restoring the plane. False if not in Set mode.
    pub fn cancel_edit(&mut self) -> bool {
        match self.edit.take() {
            Some(e) => {
                self.plane = e.before.or(self.plane);
                true
            }
            None => false,
        }
    }

    /// Panel rows: the toggle, X / Y / Z, Set and Reset, the plane's offset, and while
    /// setting, the ways to place it.
    pub fn ui(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx, label: &str) {
        if let Some((id, b)) = Self::active_bounds(cx) {
            self.place(id, &b);
        }
        enum Do {
            Preset(usize),
            Set,
            Reset,
            View,
            Selection,
            Pick,
            Accept,
            Cancel,
        }
        let mut todo = None;
        ui.horizontal(|ui| {
            if ui.checkbox(&mut self.enabled, label).on_hover_text("Mirror every dab across the symmetry plane").changed() && !self.enabled {
                self.cancel_edit();
            }
            if self.enabled {
                let axis = self.plane.and_then(|p| p.axis_index());
                for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                    if ui.selectable_label(axis == Some(i), *n).on_hover_text(format!("Plane across {n} through the object's centre")).clicked() {
                        todo = Some(Do::Preset(i));
                    }
                }
                if ui.selectable_label(self.editing(), "Set").on_hover_text("Place the plane freely: drag it, tilt it, pick points…").clicked() {
                    todo = Some(if self.editing() { Do::Accept } else { Do::Set });
                }
                if ui.button("Reset").on_hover_text("Back through the centre of the object").clicked() {
                    todo = Some(Do::Reset);
                }
            }
        });
        if !self.enabled {
            return;
        }
        let span = cx.scene.active().map_or(1.0, |o| o.bounds().diagonal());
        if let Some(p) = &mut self.plane {
            ui.horizontal(|ui| {
                ui.label("Plane at");
                let mut d = p.offset();
                if ui.add(egui::DragValue::new(&mut d).speed(span * 0.002).max_decimals(3)).on_hover_text("Distance from the origin along the plane's normal").changed() {
                    p.set_offset(d);
                }
                if p.axis_index().is_none() {
                    ui.small(format!("normal ({:.2}, {:.2}, {:.2})", p.normal.x, p.normal.y, p.normal.z));
                }
            });
        }
        if let (Some(e), Some(p)) = (&mut self.edit, &mut self.plane) {
            ui.small("Drag the arrow to slide the plane, the rings to tilt it (Shift: 15° steps).");
            ui.horizontal_wrapped(|ui| {
                if ui.button("From view").on_hover_text("Through the middle of the view, splitting it into left and right").clicked() {
                    todo = Some(Do::View);
                }
                if ui.selectable_label(e.picking, format!("Pick 3 points ({}/3)", e.picks.len())).on_hover_text("Click three points on the surface").clicked() {
                    todo = Some(Do::Pick);
                }
                if ui.button("From selection").on_hover_text("The best-fit plane of the selected faces (e.g. a band along the seam)").clicked() {
                    todo = Some(Do::Selection);
                }
            });
            ui.horizontal(|ui| {
                ui.label("Normal");
                let mut n = p.normal;
                let mut changed = false;
                for i in 0..3 {
                    changed |= ui.add(egui::DragValue::new(&mut n[i]).speed(0.01).range(-1.0..=1.0).max_decimals(3)).changed();
                }
                if changed && n.length_squared() > 1e-12 {
                    *p = SymPlane::new(p.point, n);
                }
            });
            let (accept, cancel) = ui_util::accept_cancel(ui, true);
            if accept {
                todo = Some(Do::Accept);
            } else if cancel {
                todo = Some(Do::Cancel);
            }
        }
        let result = match todo {
            None => Ok(()),
            Some(Do::Preset(i)) => {
                self.preset(cx, i);
                Ok(())
            }
            Some(Do::Set) => {
                self.begin_edit(cx);
                Ok(())
            }
            Some(Do::Reset) => {
                self.reset(cx);
                Ok(())
            }
            Some(Do::View) => {
                self.plane = Some(SymPlane::from_view(cx.camera));
                Ok(())
            }
            Some(Do::Selection) => self.fit_selection(cx),
            Some(Do::Pick) => {
                if let Some(e) = &mut self.edit {
                    e.picking = !e.picking;
                    e.picks.clear();
                }
                Ok(())
            }
            Some(Do::Accept) => {
                self.accept_edit();
                Ok(())
            }
            Some(Do::Cancel) => {
                self.cancel_edit();
                Ok(())
            }
        };
        if let Err(e) = result {
            *cx.status = format!("Symmetry: {e}");
        }
    }

    /// Gizmo centre (the active object's centre laid onto the plane), its size and the
    /// plane's in-plane axes.
    fn gizmo(&self, cx: &ToolCx) -> Option<(DVec3, f64, (DVec3, DVec3))> {
        let p = self.plane?;
        let anchor = Self::active_bounds(cx).map_or(p.point, |(_, b)| p.project(b.center()));
        Some((anchor, cx.world_per_point(anchor) * 80.0, p.basis()))
    }

    /// Ring of radius `r` around `axis` at `c`, on screen.
    fn ring(cx: &ToolCx, c: DVec3, axis: DVec3, r: f64) -> Vec<egui::Pos2> {
        let (u, v) = mm_core::geom::orthonormal_basis(axis);
        (0..=48).filter_map(|k| cx.project(c + (u * (k as f64 / 48.0 * std::f64::consts::TAU).cos() + v * (k as f64 / 48.0 * std::f64::consts::TAU).sin()) * r)).collect()
    }

    fn hit_test(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<Handle> {
        let (c3, len, (a, b)) = self.gizmo(cx)?;
        let n = self.plane?.normal;
        let mut best: Option<(f32, Handle)> = None;
        let mut consider = |d: f32, h: Handle, within: f32| {
            if d < within && best.is_none_or(|b| d < b.0) {
                best = Some((d, h));
            }
        };
        if let (Some(p), Some(q)) = (cx.project(c3 - n * len), cx.project(c3 + n * len)) {
            consider(dist_to_segment(pos, p, q), Handle::Move, 8.0);
        }
        for (i, axis) in [a, b].into_iter().enumerate() {
            for w in Self::ring(cx, c3, axis, len * 0.7).windows(2) {
                consider(dist_to_segment(pos, w[0], w[1]), Handle::Tilt(i), 6.0);
            }
        }
        best.map(|b| b.1)
    }

    fn drag_to(&mut self, d: GizmoDrag, pos: egui::Pos2, snap: bool, cx: &ToolCx) {
        let len = cx.world_per_point(d.anchor) * 80.0;
        let Some(c) = cx.project(d.anchor) else { return };
        let n = d.plane.normal;
        self.plane = Some(match d.handle {
            Handle::Move => {
                let Some(tip) = cx.project(d.anchor + n * len) else { return };
                let dir = tip - c;
                let px = dir.length().max(1e-3);
                let along = ((pos - d.start).dot(dir / px) / px) as f64 * len;
                SymPlane::new(d.plane.point + n * along, n)
            }
            Handle::Tilt(i) => {
                let axis = if i == 0 { d.axes.0 } else { d.axes.1 };
                let mut ang = ((pos - c).angle() - (d.start - c).angle()) as f64;
                // Screen angles run clockwise; flip when the axis points at the viewer.
                if axis.dot(-cx.camera.forward()) > 0.0 {
                    ang = -ang;
                }
                if snap {
                    let step = TILT_SNAP_DEG.to_radians();
                    ang = (ang / step).round() * step;
                }
                SymPlane::new(d.anchor, DQuat::from_axis_angle(axis, ang) * n)
            }
        });
    }

    /// Viewport input in Set mode (dragging the gizmo, picking points). Returns true when
    /// the input was used: in Set mode the brush does not paint.
    pub fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) -> bool {
        if self.edit.is_none() {
            return false;
        }
        let Some(pos) = input.pos else { return true };
        let hit = self.hit_test(cx, pos);
        let gizmo = self.gizmo(cx);
        let Some(e) = &mut self.edit else { return true };
        if e.drag.is_none() {
            e.hover = hit;
        }
        if input.pressed {
            if let (Some(h), Some((anchor, _, axes)), Some(plane)) = (hit, gizmo, self.plane) {
                e.drag = Some(GizmoDrag { handle: h, start: pos, plane, anchor, axes });
            } else if e.picking {
                if let Some((_, h)) = cx.pick(pos, cx.scene.active) {
                    e.picks.push(h.point);
                    if e.picks.len() == 3 {
                        let picks = std::mem::take(&mut e.picks);
                        e.picking = false;
                        *cx.status = match self.through_points(&picks) {
                            Ok(()) => "Symmetry plane through the three points".into(),
                            Err(err) => format!("Symmetry: {err}"),
                        };
                    }
                }
            }
        }
        if let Some(d) = self.edit.as_ref().and_then(|e| e.drag) {
            if input.down {
                self.drag_to(d, pos, input.shift, cx);
            }
        }
        if input.released {
            if let Some(e) = &mut self.edit {
                e.drag = None;
            }
        }
        true
    }

    /// The plane across the active object (brighter while setting).
    pub fn overlay(&self, ov: &mut Overlay, cx: &ToolCx) {
        let (Some(p), true) = (self.plane, self.enabled) else { return };
        let Some((_, b)) = Self::active_bounds(cx) else { return };
        let q = p.patch(&b);
        let k = if self.editing() { 2.0 } else { 1.0 };
        let (fill, edge) = (Vec4::new(0.3, 0.6, 1.0, 0.08 * k), Vec4::new(0.3, 0.6, 1.0, 0.7));
        ov.tri(q[0], q[1], q[2], fill);
        ov.tri(q[0], q[2], q[3], fill);
        for n in 0..4 {
            ov.line(q[n], q[(n + 1) % 4], edge);
        }
    }

    /// Set mode's gizmo and picked points.
    pub fn paint(&self, painter: &egui::Painter, cx: &ToolCx) {
        let Some(e) = &self.edit else { return };
        for p in &e.picks {
            if let Some(s) = cx.project(*p) {
                painter.circle(s, 4.0, ui_util::ACCENT, egui::Stroke::new(1.5, egui::Color32::from_gray(20)));
            }
        }
        let (Some((c3, len, (a, b))), Some(p)) = (self.gizmo(cx), self.plane) else { return };
        let active = e.drag.map(|d| d.handle).or(e.hover);
        let col = |h: Handle, base: egui::Color32| if active == Some(h) { egui::Color32::YELLOW } else { base };
        for (i, axis) in [a, b].into_iter().enumerate() {
            let hot = active == Some(Handle::Tilt(i));
            let c = col(Handle::Tilt(i), egui::Color32::from_rgb(120, 170, 255));
            painter.add(egui::Shape::line(Self::ring(cx, c3, axis, len * 0.7), egui::Stroke::new(if hot { 3.0 } else { 1.5 }, c)));
        }
        if let (Some(c), Some(tip), Some(back)) = (cx.project(c3), cx.project(c3 + p.normal * len), cx.project(c3 - p.normal * len)) {
            let hot = active == Some(Handle::Move);
            let s = egui::Stroke::new(if hot { 4.0 } else { 2.5 }, col(Handle::Move, egui::Color32::from_rgb(80, 150, 255)));
            painter.arrow(c, tip - c, s);
            painter.arrow(c, back - c, s);
            painter.circle_filled(c, 3.5, egui::Color32::from_gray(230));
        }
    }

    /// Agent settings (merged into the tool's own).
    pub fn settings(&self) -> serde_json::Map<String, serde_json::Value> {
        let p = self.plane.unwrap_or(SymPlane::new(DVec3::ZERO, DVec3::X));
        let v = serde_json::json!({
            "symmetry": self.enabled,
            "symmetry_axis": p.axis_index().map_or("free", |i| ["x", "y", "z"][i]),
            "symmetry_plane": {"point_mm": vec3(p.point.to_array(), 5), "normal": vec3(p.normal.to_array(), 6)},
        });
        v.as_object().cloned().unwrap_or_default()
    }

    /// JSON Schema properties of [`Symmetry::settings`].
    pub fn schema() -> serde_json::Map<String, serde_json::Value> {
        let v = serde_json::json!({
            "symmetry": {"type": "boolean", "description": "Mirror every dab across the symmetry plane"},
            "symmetry_axis": {"type": "string", "enum": ["x", "y", "z", "free"], "description": "x / y / z puts the plane across that axis through the object's centre; free: any plane (see symmetry_plane)"},
            "symmetry_plane": {"type": "object", "description": "The plane: a point on it and its normal (any length)", "properties": {
                "point_mm": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
                "normal": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3},
            }},
        });
        v.as_object().cloned().unwrap_or_default()
    }

    /// Apply agent settings: `merged` is the tool's settings with `changes` merged in.
    pub fn set_settings(&mut self, merged: &serde_json::Value, changes: &serde_json::Value, cx: &ToolCx) -> Result<(), String> {
        let a = Args(merged);
        self.enabled = a.opt_bool("symmetry").unwrap_or(self.enabled);
        match changes.get("symmetry_axis").and_then(|v| v.as_str()) {
            Some("free") | None => {}
            Some(axis) => {
                let i = ["x", "y", "z"].iter().position(|x| *x == axis).ok_or("symmetry_axis is x, y, z or free")?;
                self.preset(cx, i);
                return Ok(());
            }
        }
        if changes.get("symmetry_plane").is_some() {
            let p = Args(&merged["symmetry_plane"]);
            let (point, normal) = (DVec3::from_array(p.point("point_mm")?), DVec3::from_array(p.point("normal")?));
            if normal.length_squared() < 1e-18 {
                return Err("the symmetry plane's normal is zero".into());
            }
            self.plane = Some(SymPlane::new(point, normal));
            self.object = cx.scene.active;
        } else if let Some((id, b)) = Self::active_bounds(cx) {
            self.place(id, &b);
        }
        Ok(())
    }

    pub fn actions() -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("symmetry_through_points", "Turn symmetry on with the plane through three points", json!({"points": arg::points("Three points in mm, not on a line")})).required(&["points"]),
            Action::plain("symmetry_from_selection", "Turn symmetry on with the plane fitted to the selected faces (e.g. a band selected along the seam)"),
            Action::plain("symmetry_from_view", "Turn symmetry on with the plane through the middle of the view, splitting it into left and right"),
            Action::new("symmetry_reset", "Turn symmetry on with the plane through the active object's centre", json!({"axis": arg::choice(&["x", "y", "z"], "Across this axis (default: the plane's own axis, or x)")})),
        ]
    }

    /// Do a symmetry action; `None` if `action` is not one.
    pub fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Option<Result<String, String>> {
        let a = Args(args);
        let r = match action {
            "symmetry_through_points" => a.points("points").and_then(|p| self.through_points(&p.into_iter().map(DVec3::from_array).collect::<Vec<_>>())),
            "symmetry_from_selection" => self.fit_selection(cx),
            "symmetry_from_view" => {
                self.plane = Some(SymPlane::from_view(cx.camera));
                Ok(())
            }
            "symmetry_reset" => {
                match a.opt_str("axis") {
                    Some(x) => self.preset(cx, ["x", "y", "z"].iter().position(|k| *k == x).unwrap_or(0)),
                    None => self.reset(cx),
                }
                Ok(())
            }
            _ => return None,
        };
        Some(r.map(|()| {
            self.enabled = true;
            let p = self.plane.unwrap_or(SymPlane::new(DVec3::ZERO, DVec3::X));
            format!("Symmetry plane through ({:.3}, {:.3}, {:.3}), normal ({:.4}, {:.4}, {:.4})", p.point.x, p.point.y, p.point.z, p.normal.x, p.normal.y, p.normal.z)
        }))
    }
}

fn dist_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn free_plane_mirrors_points_and_directions() {
        let p = SymPlane::new(DVec3::new(1.0, 2.0, 3.0), DVec3::new(1.0, 1.0, 0.0));
        assert!((p.normal.length() - 1.0).abs() < 1e-12);
        for q in [DVec3::ZERO, DVec3::new(-3.0, 0.5, 7.0), DVec3::new(1.0, 2.0, 3.0)] {
            let m = p.reflect(q);
            assert!((p.reflect(m) - q).length() < 1e-12, "an involution");
            assert!((p.signed(m) + p.signed(q)).abs() < 1e-12, "the other side, as far");
            assert!(((m + q) * 0.5 - p.project(q)).length() < 1e-12, "the midpoint lies on the plane");
        }
        // A direction flips its normal part only.
        let v = DVec3::new(2.0, 0.0, 1.0);
        let w = p.reflect_dir(v);
        assert!((w.dot(p.normal) + v.dot(p.normal)).abs() < 1e-12 && (w - v).cross(p.normal).length() < 1e-12);
        // The axis presets as before: x → 2c − x.
        let b = Aabb::from_points([DVec3::new(-1.0, 0.0, 0.0), DVec3::new(3.0, 2.0, 2.0)]);
        let x = SymPlane::axis(&b, 0);
        assert_eq!(x.axis_index(), Some(0));
        assert!((x.reflect(DVec3::new(0.0, 5.0, 6.0)) - DVec3::new(2.0, 5.0, 6.0)).length() < 1e-12);
        assert_eq!(p.axis_index(), None);
        // Offset moves the plane along its normal.
        let mut o = p;
        o.set_offset(p.offset() + 2.0);
        assert!((o.signed(p.point) + 2.0).abs() < 1e-12 && o.normal == p.normal);
    }

    #[test]
    fn plane_from_points_selection_fit_and_view() {
        let (a, b, c) = (DVec3::new(0.0, 0.0, 1.0), DVec3::new(1.0, 0.0, 2.0), DVec3::new(0.0, 1.0, 1.0));
        let p = SymPlane::through(a, b, c).unwrap();
        assert!([a, b, c].iter().all(|&q| p.distance(q) < 1e-12));
        assert!(SymPlane::through(a, a * 2.0, a * 3.0).is_none(), "on a line");
        // A noisy ring of points in the plane x + y = 1.
        let n = DVec3::new(1.0, 1.0, 0.0).normalize();
        let (u, v) = mm_core::geom::orthonormal_basis(n);
        let pts: Vec<DVec3> = (0..40).map(|k| {
            let t = k as f64 * 0.37;
            DVec3::new(0.5, 0.5, 0.0) + (u * t.cos() + v * t.sin()) * 2.0 + n * (1e-4 * (k as f64 * 1.7).sin())
        }).collect();
        let f = SymPlane::fit(&pts).unwrap();
        assert!(f.normal.dot(n).abs() > 1.0 - 1e-6 && f.distance(DVec3::new(0.5, 0.5, 0.0)) < 1e-3);
        assert!(SymPlane::fit(&[a, b, a + (b - a) * 2.0, a + (b - a) * 3.0]).is_none());
        // From the view: the plane splits the screen down the middle.
        let cam = crate::camera::Camera { target: DVec3::new(1.0, 2.0, 3.0), ..Default::default() };
        let s = SymPlane::from_view(&cam);
        assert!(s.distance(cam.eye()) < 1e-9 && s.distance(cam.target + cam.up() * 5.0) < 1e-9);
        assert!((s.normal - cam.right()).length() < 1e-12);
    }

    #[test]
    fn x_plane_follows_a_new_object_a_free_one_stays() {
        let (b1, b2) = (Aabb::from_points([DVec3::ZERO, DVec3::splat(2.0)]), Aabb::from_points([DVec3::splat(10.0), DVec3::splat(12.0)]));
        let mut s = Symmetry::new(None, None);
        assert_eq!(s.place(1, &b1), SymPlane::axis(&b1, 0));
        s.set_plane(SymPlane::axis(&b1, 2));
        assert_eq!(s.place(2, &b2), SymPlane::axis(&b2, 2));
        let free = SymPlane::new(DVec3::ONE, DVec3::new(1.0, 2.0, 0.0));
        s.set_plane(free);
        assert_eq!(s.place(1, &b1), free);
        assert_eq!(s.active(), None, "off until enabled");
        s.enabled = true;
        assert_eq!(s.active(), Some(free));
    }
}
