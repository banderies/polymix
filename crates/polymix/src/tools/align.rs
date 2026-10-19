//! Align (Edit > Align): move and turn the active object so a source frame picked on it
//! (left-click) lands on a destination frame: the world origin, the ground plane, or a
//! surface picked with Shift+click (usually on another object). The source normal ends up
//! pointing into the destination, so the picked face rests on the ground or against the
//! picked surface; Flip reverses that.

use super::*;
use crate::ui_util;
use glam::DQuat;
use mm_core::geom::Ray;
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SourceType {
    /// Bounding-box centre, oriented by the clicked normal.
    CenterPoint,
    /// The object's extreme plane along the clicked normal, below its centre.
    BasePoint,
    /// The clicked point and its normal.
    SurfacePoint,
    /// A stroke drawn on the surface: its centroid and average normal.
    SurfaceScribble,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Destination {
    /// World origin with the given axis (0 = X, 1 = Y, 2 = Z) up.
    WorldOrigin(usize),
    /// The ground plane: only the height and tilt change.
    GroundPlane,
    /// A picked surface point and its normal.
    Surface(DVec3, DVec3),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transformation {
    TranslateOnly,
    TranslateAndRotate,
}

/// A point with a unit normal.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Frame {
    pub p: DVec3,
    pub n: DVec3,
}

/// The alignment `x ↦ r (x − pivot) + to` taking `src` onto `dst`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Alignment {
    pub r: DQuat,
    pub pivot: DVec3,
    pub to: DVec3,
}

impl Alignment {
    pub fn apply(&self, x: DVec3) -> DVec3 {
        self.r * (x - self.pivot) + self.to
    }
    pub fn invert(&self, y: DVec3) -> DVec3 {
        self.r.inverse() * (y - self.to) + self.pivot
    }
}

/// Destination frame for `dest`; `up` is the world up axis for the ground plane. The
/// destination normal points away from the surface the source should rest on.
fn dest_frame(dest: Destination, src: Frame, up: DVec3) -> Frame {
    match dest {
        Destination::WorldOrigin(axis) => Frame { p: DVec3::ZERO, n: -[DVec3::X, DVec3::Y, DVec3::Z][axis] },
        // The pivot keeps its horizontal position and drops to height 0.
        Destination::GroundPlane => Frame { p: src.p - up * src.p.dot(up), n: -up },
        Destination::Surface(p, n) => Frame { p, n: -n },
    }
}

/// The transform that takes `src` onto the destination frame.
pub fn align(src: Frame, dest: Destination, up: DVec3, mode: Transformation, flip: bool) -> Alignment {
    let d = dest_frame(dest, src, up);
    let n = if flip { -d.n } else { d.n };
    let r = match mode {
        Transformation::TranslateOnly => DQuat::IDENTITY,
        Transformation::TranslateAndRotate => DQuat::from_rotation_arc(src.n.normalize(), n.normalize()),
    };
    Alignment { r, pivot: src.p, to: d.p }
}

pub struct AlignTool {
    object: Option<ObjectId>,
    source: Arc<Mesh>,
    bvh: Arc<mm_core::Bvh>,
    frame: DQuat,
    source_type: SourceType,
    dest: Destination,
    mode: Transformation,
    flip: bool,
    /// Clicked source point and normal (object coordinates before alignment).
    picked: Option<Frame>,
    scribble: Vec<Frame>,
    pressed_at: Option<egui::Pos2>,
    preview: Option<Mesh>,
    alignment: Option<Alignment>,
    dirty: bool,
}

impl AlignTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let up = cx.camera.world_up();
        let axis = if up == DVec3::Z { 2 } else { 1 };
        let (object, source, bvh, frame) = match cx.scene.active {
            Some(id) => match cx.scene.get_mut(id) {
                Some(o) => (Some(id), o.mesh.clone(), o.bvh(), o.frame),
                None => (None, Arc::new(Mesh::new()), Arc::new(mm_core::Bvh::build(&Mesh::new())), DQuat::IDENTITY),
            },
            None => (None, Arc::new(Mesh::new()), Arc::new(mm_core::Bvh::build(&Mesh::new())), DQuat::IDENTITY),
        };
        Self {
            object,
            source,
            bvh,
            frame,
            source_type: SourceType::BasePoint,
            dest: Destination::WorldOrigin(axis),
            mode: Transformation::TranslateAndRotate,
            flip: false,
            picked: None,
            scribble: Vec::new(),
            pressed_at: None,
            preview: None,
            alignment: None,
            dirty: true,
        }
    }

    /// The source frame, if the current source type has what it needs.
    fn source_frame(&self, up: DVec3) -> Option<Frame> {
        let m = &*self.source;
        let c = m.bounds().center();
        let n = self.picked.map_or(-up, |f| f.n);
        match self.source_type {
            SourceType::CenterPoint => Some(Frame { p: c, n }),
            SourceType::BasePoint => {
                let reach = m.vertex_ids().map(|v| (m.vertex(v) - c).dot(n)).fold(f64::NEG_INFINITY, f64::max);
                (reach.is_finite()).then_some(Frame { p: c + n * reach, n })
            }
            SourceType::SurfacePoint => self.picked,
            SourceType::SurfaceScribble => {
                if self.scribble.is_empty() {
                    return None;
                }
                let p = self.scribble.iter().map(|f| f.p).sum::<DVec3>() / self.scribble.len() as f64;
                let n = self.scribble.iter().map(|f| f.n).sum::<DVec3>().normalize_or_zero();
                (n != DVec3::ZERO).then_some(Frame { p, n })
            }
        }
    }

    fn rebuild(&mut self, up: DVec3) {
        self.dirty = false;
        self.alignment = self.source_frame(up).map(|s| align(s, self.dest, up, self.mode, self.flip));
        self.preview = self.alignment.map(|a| {
            let mut m = (*self.source).clone();
            m.transform(|p| a.apply(p));
            m
        });
    }

    /// Hit on the active object as displayed, in its original coordinates.
    fn pick_source(&self, cx: &ToolCx, pos: egui::Pos2) -> Option<Frame> {
        let ray = cx.ray_at(pos);
        let ray = match self.alignment {
            Some(a) => Ray::new(a.invert(ray.origin), a.r.inverse() * ray.dir),
            None => ray,
        };
        let hit = self.bvh.raycast(&self.source, &ray)?;
        Some(Frame { p: hit.point, n: self.source.tri_normal(hit.tid) })
    }

    /// Hit on any other visible object.
    fn pick_dest(&self, cx: &mut ToolCx, pos: egui::Pos2) -> Option<Frame> {
        let ids: Vec<ObjectId> = cx.scene.objects.iter().filter(|o| o.visible && Some(o.id) != self.object).map(|o| o.id).collect();
        let mut best: Option<(f64, Frame)> = None;
        for id in ids {
            if let Some((_, h)) = cx.pick(pos, Some(id)) {
                let n = cx.scene.get(id).map(|o| o.mesh.tri_normal(h.tid)).unwrap_or(DVec3::Y);
                if best.as_ref().is_none_or(|b| h.t < b.0) {
                    best = Some((h.t, Frame { p: h.point, n }));
                }
            }
        }
        best.map(|b| b.1)
    }

    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        let (Some(id), Some(a)) = (self.object, self.alignment) else { return true };
        if !cx.scene.get(id).is_some_and(|o| Arc::ptr_eq(&o.mesh, &self.source)) {
            *cx.status = "Align cancelled: the object changed while the tool was open.".into();
            return true;
        }
        let Some(m) = self.preview.take() else { return true };
        cx.history.push("Align", cx.scene);
        if let Some(o) = cx.scene.get_mut(id) {
            let sel = o.selection.clone();
            o.mesh = Arc::new(m);
            o.frame = (a.r * self.frame).normalize();
            o.selection = sel;
            o.touch_selection();
        }
        true
    }
}

const SOURCES: [(&str, SourceType); 4] = [("center_point", SourceType::CenterPoint), ("base_point", SourceType::BasePoint), ("surface_point", SourceType::SurfacePoint), ("surface_scribble", SourceType::SurfaceScribble)];

impl AlignTool {
    /// The point of the active object (as it was when the tool opened) nearest `p`, with its
    /// normal.
    fn source_at(&self, p: DVec3) -> Option<Frame> {
        let h = self.bvh.nearest(&self.source, p, f64::INFINITY)?;
        Some(Frame { p: h.point, n: self.source.tri_normal(h.tid) })
    }
}

impl Tool for AlignTool {
    fn settings(&self) -> Option<serde_json::Value> {
        let source = SOURCES.iter().find(|(_, s)| *s == self.source_type).map_or("base_point", |(k, _)| *k);
        let destination = match self.dest {
            Destination::WorldOrigin(0) => "world_origin_x_up",
            Destination::WorldOrigin(1) => "world_origin_y_up",
            Destination::WorldOrigin(_) => "world_origin_z_up",
            Destination::GroundPlane => "ground_plane",
            Destination::Surface(..) => "surface_point",
        };
        Some(serde_json::json!({"source": source, "destination": destination, "rotate": self.mode == Transformation::TranslateAndRotate, "flip": self.flip}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "source": {"type": "string", "enum": SOURCES.map(|(k, _)| k), "description": "What on the object is aligned: its box centre or base (facing the picked normal, or down), a picked surface point, or the average of a scribble"},
            "destination": {"type": "string", "enum": ["world_origin_x_up", "world_origin_y_up", "world_origin_z_up", "ground_plane", "surface_point"], "description": "Where it goes: the origin with an axis up, onto the ground plane, or a point picked on another object (pick_destination)"},
            "rotate": {"type": "boolean", "description": "Turn the source's normal onto the destination's, not only move"},
            "flip": {"type": "boolean", "description": "Face the other way"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let a = Args(&v);
        self.source_type = SOURCES.iter().find(|(k, _)| Some(*k) == a.opt_str("source")).map(|(_, s)| *s).ok_or("unknown source")?;
        self.dest = match a.opt_str("destination") {
            Some("world_origin_x_up") => Destination::WorldOrigin(0),
            Some("world_origin_y_up") => Destination::WorldOrigin(1),
            Some("world_origin_z_up") => Destination::WorldOrigin(2),
            Some("ground_plane") => Destination::GroundPlane,
            Some("surface_point") if matches!(self.dest, Destination::Surface(..)) => self.dest,
            Some("surface_point") => return Err("pick the destination point with pick_destination".into()),
            _ => return Err("unknown destination".into()),
        };
        self.mode = if a.opt_bool("rotate").unwrap_or(true) { Transformation::TranslateAndRotate } else { Transformation::TranslateOnly };
        self.flip = a.opt_bool("flip").unwrap_or(false);
        self.rebuild(cx.camera.world_up());
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::vec3;
        let f = |f: &Frame| serde_json::json!({"at_mm": vec3(f.p.to_array(), 4), "normal": vec3(f.n.to_array(), 4)});
        serde_json::json!({
            "object": self.object,
            "picked_source": self.picked.as_ref().map(f),
            "scribble_points": self.scribble.len(),
            "destination_point": match self.dest { Destination::Surface(p, n) => Some(f(&Frame { p, n })), _ => None },
            "aligned": self.alignment.is_some(),
            "moves_centre_to_mm": self.alignment.map(|al| vec3(al.apply(self.source.bounds().center()).to_array(), 4)),
            "next": if self.alignment.is_some() { "accept (mesh_tool accept) to keep the alignment" } else { "pick_source (surface sources) or adjust the settings" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("pick_source", "Use the object's surface point nearest `at` (and its normal) as the source; with source surface_scribble, add it to the scribble", json!({"at": arg::point("A point on or near the object, mm (its position before aligning)")})).required(&["at"]),
            Action::new("pick_destination", "Align to the point of another visible object nearest `at` (and its normal); sets destination surface_point", json!({"at": arg::point("A point on or near another object, mm")})).required(&["at"]),
            Action::plain("clear", "Forget the picked source and the scribble"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        match action {
            "pick_source" => {
                let f = self.source_at(DVec3::from_array(a.point("at")?)).ok_or("the object has no surface")?;
                if self.source_type == SourceType::SurfaceScribble {
                    self.scribble.push(f);
                } else {
                    self.picked = Some(f);
                }
            }
            "pick_destination" => {
                let at = DVec3::from_array(a.point("at")?);
                let mut best: Option<(f64, Frame)> = None;
                for o in cx.scene.objects.iter_mut().filter(|o| o.visible && Some(o.id) != self.object) {
                    let bvh = o.bvh();
                    if let Some(h) = bvh.nearest(&o.mesh, at, f64::INFINITY) {
                        if best.as_ref().is_none_or(|b| h.dist_sq < b.0) {
                            best = Some((h.dist_sq, Frame { p: h.point, n: o.mesh.tri_normal(h.tid) }));
                        }
                    }
                }
                let (_, f) = best.ok_or("there is no other visible object")?;
                self.dest = Destination::Surface(f.p, f.n);
            }
            "clear" => {
                self.picked = None;
                self.scribble.clear();
            }
            _ => return Err(format!("Align has no action {action}")),
        }
        self.rebuild(cx.camera.world_up());
        Ok(if self.alignment.is_some() { "Previewing the alignment".into() } else { "Picked; the source needs a point".into() })
    }

    fn name(&self) -> &'static str {
        "Align"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Align");
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let up = cx.camera.world_up();
        let before = (self.source_type, self.dest, self.mode, self.flip);
        let src_name = |s: SourceType| match s {
            SourceType::CenterPoint => "Center Point",
            SourceType::BasePoint => "Base Point",
            SourceType::SurfacePoint => "Surface Point",
            SourceType::SurfaceScribble => "Surface Scribble",
        };
        egui::ComboBox::from_label("Source (left-click)").selected_text(src_name(self.source_type)).show_ui(ui, |ui| {
            for s in [SourceType::CenterPoint, SourceType::BasePoint, SourceType::SurfacePoint, SourceType::SurfaceScribble] {
                ui.selectable_value(&mut self.source_type, s, src_name(s));
            }
        });
        let dest_name = |d: Destination| match d {
            Destination::WorldOrigin(0) => "World Origin / X-up",
            Destination::WorldOrigin(1) => "World Origin / Y-up",
            Destination::WorldOrigin(_) => "World Origin / Z-up",
            Destination::GroundPlane => "Ground Plane",
            Destination::Surface(..) => "Surface Point (shift-click)",
        };
        egui::ComboBox::from_label("Destination (shift-click)").selected_text(dest_name(self.dest)).show_ui(ui, |ui| {
            for d in [Destination::WorldOrigin(1), Destination::WorldOrigin(2), Destination::WorldOrigin(0), Destination::GroundPlane] {
                ui.selectable_value(&mut self.dest, d, dest_name(d));
            }
        });
        egui::ComboBox::from_label("Transformation")
            .selected_text(match self.mode {
                Transformation::TranslateOnly => "Translate Only",
                Transformation::TranslateAndRotate => "Translate and Rotate",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.mode, Transformation::TranslateOnly, "Translate Only");
                ui.selectable_value(&mut self.mode, Transformation::TranslateAndRotate, "Translate and Rotate");
            });
        ui.checkbox(&mut self.flip, "Flip");
        ui.small(match self.source_type {
            SourceType::SurfaceScribble => "Draw on the object to choose the source surface; Shift+click another object for the destination.",
            _ => "Click the object to choose the source face; Shift+click another object for the destination.",
        });
        if (self.source_type, self.dest, self.mode, self.flip) != before {
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild(up);
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, self.alignment.is_some());
        if accept && self.commit(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.pressed_at = Some(pos);
            if self.source_type == SourceType::SurfaceScribble && !input.shift {
                self.scribble.clear();
            }
        }
        if input.down && self.source_type == SourceType::SurfaceScribble && !input.shift {
            if let Some(f) = self.pick_source(cx, pos) {
                self.scribble.push(f);
                self.dirty = true;
            }
        }
        if input.released {
            let clicked = self.pressed_at.take().is_some_and(|p| p.distance(pos) < 5.0);
            if clicked && input.shift {
                if let Some(f) = self.pick_dest(cx, pos) {
                    self.dest = Destination::Surface(f.p, f.n);
                    self.dirty = true;
                }
            } else if clicked && self.source_type != SourceType::SurfaceScribble {
                if let Some(f) = self.pick_source(cx, pos) {
                    self.picked = Some(f);
                    self.dirty = true;
                }
            }
        }
        if self.dirty {
            self.rebuild(cx.camera.world_up());
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let up = cx.camera.world_up();
        let Some(src) = self.source_frame(up) else { return };
        let len = cx.world_per_point(src.p) * 40.0;
        let shown = |p: DVec3| self.alignment.map_or(p, |a| a.apply(p));
        let arrow = |p: DVec3, n: DVec3, col: egui::Color32| {
            if let (Some(a), Some(b)) = (cx.project(p), cx.project(p + n * len)) {
                painter.circle_filled(a, 4.0, col);
                painter.arrow(a, b - a, egui::Stroke::new(2.0, col));
            }
        };
        let n = self.alignment.map_or(src.n, |a| a.r * src.n);
        arrow(shown(src.p), n, egui::Color32::from_rgb(230, 80, 70));
        if let Destination::Surface(p, n) = self.dest {
            arrow(p, n, egui::Color32::from_rgb(90, 200, 90));
        }
        for f in &self.scribble {
            if let Some(a) = cx.project(shown(f.p)) {
                painter.circle_filled(a, 1.5, egui::Color32::from_rgb(230, 80, 70));
            }
        }
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        Some((self.object?, self.preview.as_ref()?))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.commit(cx)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-9
    }

    #[test]
    fn face_lands_on_the_origin_facing_down() {
        let src = Frame { p: DVec3::new(3.0, 1.0, 2.0), n: DVec3::new(1.0, 1.0, 0.0).normalize() };
        let a = align(src, Destination::WorldOrigin(1), DVec3::Y, Transformation::TranslateAndRotate, false);
        assert!(close(a.apply(src.p), DVec3::ZERO));
        assert!(close(a.r * src.n, -DVec3::Y));
        assert!(close(a.invert(a.apply(DVec3::new(0.3, -2.0, 5.0))), DVec3::new(0.3, -2.0, 5.0)));
        let flipped = align(src, Destination::WorldOrigin(2), DVec3::Y, Transformation::TranslateAndRotate, true);
        assert!(close(flipped.r * src.n, DVec3::Z));
        let moved = align(src, Destination::WorldOrigin(1), DVec3::Y, Transformation::TranslateOnly, false);
        assert_eq!(moved.r, DQuat::IDENTITY);
        assert!(close(moved.apply(src.p), DVec3::ZERO));
    }

    #[test]
    fn base_point_rests_the_object_on_the_ground() {
        let mut m = mm_core::primitives::cube(1.0, 2);
        m.transform(|p| DVec3::new(3.0 * p.x, p.y, p.z) + DVec3::new(5.0, 2.0, -1.0));
        let source = Arc::new(m);
        let mut t = AlignTool {
            object: Some(1),
            bvh: Arc::new(mm_core::Bvh::build(&source)),
            source,
            frame: DQuat::IDENTITY,
            source_type: SourceType::BasePoint,
            dest: Destination::WorldOrigin(1),
            mode: Transformation::TranslateAndRotate,
            flip: false,
            picked: Some(Frame { p: DVec3::ZERO, n: DVec3::new(1.0, 0.0, 0.0) }),
            scribble: Vec::new(),
            pressed_at: None,
            preview: None,
            alignment: None,
            dirty: true,
        };
        t.rebuild(DVec3::Y);
        let out = t.preview.as_ref().unwrap();
        let b = out.bounds();
        // The +X end face is now the bottom, centred over the origin.
        assert!(b.min.y.abs() < 1e-9, "{}", b.min.y);
        assert!((b.extents().y - 3.0).abs() < 1e-9);
        assert!(b.center().x.abs() < 1e-9 && b.center().z.abs() < 1e-9);
    }

    #[test]
    fn ground_plane_keeps_the_footprint_and_surfaces_meet() {
        let src = Frame { p: DVec3::new(3.0, 1.5, 2.0), n: DVec3::new(0.0, -1.0, 0.2).normalize() };
        let a = align(src, Destination::GroundPlane, DVec3::Y, Transformation::TranslateAndRotate, false);
        assert!(close(a.apply(src.p), DVec3::new(3.0, 0.0, 2.0)));
        assert!(close(a.r * src.n, -DVec3::Y));
        // Against a picked surface the faces end up touching, normals opposed.
        let dst_p = DVec3::new(-1.0, 4.0, 0.5);
        let dst_n = DVec3::new(0.0, 0.0, 1.0);
        let b = align(src, Destination::Surface(dst_p, dst_n), DVec3::Y, Transformation::TranslateAndRotate, false);
        assert!(close(b.apply(src.p), dst_p));
        assert!(close(b.r * src.n, -dst_n));
    }
}
