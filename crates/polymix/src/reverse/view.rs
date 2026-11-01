// SPDX-License-Identifier: AGPL-3.0-only
//! The Reverse Engineer workspace's 3D view: the scan and the bodies drawn by Polymix's
//! renderer, the sketch painted over them on its plane, and what a click picks — a point on
//! the sketch's sheet, a height on the scan, an edge, a face, a body or a region. The view is
//! moved as the Mesh workspace's is (the same mouse bindings); while a sketch is open the
//! part turns to face its plane and is cut open there, as ScanRuler and CAD show a sketch.

use std::collections::HashMap;
use std::sync::Arc;

use egui::{Color32, PointerButton, Pos2, Stroke};
use glam::{DVec2, DVec3, Vec3, Vec4};
use mm_core::bvh::Bvh;
use mm_reverse::frame::SectionFrame;
use mm_reverse::model::{BodyId, FeatureId, FeatureKind};
use mm_reverse::regions::region_at;

use super::sketcher::{self, colors, paint_dimensions, Pick, SketchInput, Tool};
use super::{Mode, ReverseWorkspace};
use crate::camera::Camera;
use crate::navigation::NavAction;
use crate::renderer::{self, ColorVertex, GpuMesh, ObjectStyle};
use crate::tools::Viewport;

/// The colour of a body, of the bodies a feature box is changing, and of a cut's tool.
const BODY: Vec4 = Vec4::new(0.56, 0.66, 0.80, 1.0);
const BODY_PREVIEW: Vec4 = Vec4::new(0.45, 0.80, 0.55, 1.0);
const BODY_CUT: Vec4 = Vec4::new(0.90, 0.45, 0.40, 1.0);
/// The edges at rest: near-black #0c0f13 at 0.85.
const EDGE: Vec4 = Vec4::new(0.047, 0.059, 0.075, 0.85);
/// The colour of what a feature box has picked (#3cb371).
pub(crate) const PICKED: Color32 = Color32::from_rgb(0x3c, 0xb3, 0x71);
/// The colour of what a click would take on the bodies.
pub(crate) const BODY_HOVER: Color32 = Color32::WHITE;

/// What the view keeps between frames.
pub struct ViewState {
    pub camera: Camera,
    nav_drag: Option<PointerButton>,
    nav_anchor: Option<DVec3>,
    /// The view was fitted to something once.
    pub framed: bool,
    pub want_fit: bool,
    /// Fit the view when the next build comes in (a part opened).
    pub want_fit_bodies: bool,
    before_sketch: Option<crate::camera::ViewState>,
    pub panel_rect: Option<egui::Rect>,
    last_pointer: Option<Pos2>,
    scan_gpu: HashMap<u64, GpuMesh>,
    body_gpu: HashMap<BodyId, (usize, GpuMesh)>,
    /// The BVH of each body mesh a click is cast against, by the mesh (its address while the
    /// bodies hold it): a body shown and as it stood before the open feature are two meshes.
    body_bvh: HashMap<usize, Arc<Bvh>>,
    lines: Vec<ColorVertex>,
    dims: Vec<(mm_reverse::sketch::ConstraintId, egui::Rect)>,
    /// The body edge under the cursor (body, edge index in its topology), and the face.
    pub(crate) hover_edge: Option<(BodyId, usize)>,
    pub(crate) hover_face: Option<(BodyId, usize)>,
    /// The region of the open feature's sketch under the cursor.
    pub(crate) hover_region: Option<String>,
    /// The 3D mouse: whether it is moving the view, what it turns about (chosen as a move
    /// starts), the marker of that centre (fading once the move ends), and the centre of mass
    /// last worked out (the mesh it was for).
    space_moving: bool,
    space_centre: Option<DVec3>,
    space_marker: Option<(DVec3, std::time::Instant)>,
    com_cache: Option<(usize, DVec3)>,
}

impl Default for ViewState {
    fn default() -> Self {
        Self {
            camera: Camera::default(),
            nav_drag: None,
            nav_anchor: None,
            framed: false,
            want_fit: true,
            want_fit_bodies: true,
            before_sketch: None,
            panel_rect: None,
            last_pointer: None,
            scan_gpu: HashMap::new(),
            body_gpu: HashMap::new(),
            body_bvh: HashMap::new(),
            lines: Vec::new(),
            dims: Vec::new(),
            hover_edge: None,
            hover_face: None,
            hover_region: None,
            space_moving: false,
            space_centre: None,
            space_marker: None,
            com_cache: None,
        }
    }
}

impl ViewState {
    /// Turn the view to face a sketch plane (from its front: the side its normal points to),
    /// remembering the view to come back to.
    pub fn face_plane(&mut self, frame: &SectionFrame, flip: bool) {
        if self.before_sketch.is_none() {
            self.before_sketch = Some(self.camera.view_state());
        }
        let to_scene = |p: DVec3| if flip { crate::app::flip_zy_in(p) } else { p };
        let n = to_scene(frame.normal);
        let o = to_scene(frame.origin);
        // The target onto the plane, so the turn happens about where the sketch is.
        let t = self.camera.target;
        self.camera.target = t - n * (t - o).dot(n);
        let up = to_scene(frame.v);
        let f = -n;
        let right = f.cross(up).normalize_or_zero();
        if right.length_squared() > 0.5 {
            let r = glam::DQuat::from_mat3(&glam::DMat3::from_cols(right, up.normalize(), -f));
            let base = match self.camera.up_axis {
                crate::camera::UpAxis::Y => glam::DQuat::IDENTITY,
                crate::camera::UpAxis::Z => glam::DQuat::from_rotation_x(std::f64::consts::FRAC_PI_2),
            };
            self.camera.orient = base.inverse() * r;
        } else {
            self.camera.look_along(f);
        }
    }

    pub fn restore_after_sketch(&mut self) {
        if let Some(v) = self.before_sketch.take() {
            self.camera.set_view_state(v);
        }
    }
}

/// What the sketch box asks of the workspace.
pub enum SketchAction {
    None,
    Finish,
    /// Finish and extrude (or revolve) the region named, or every region.
    Extrude(Option<String>),
    Revolve(Option<String>),
    /// Move the plane to this offset.
    Offset(f64),
    /// Cut the scan this far off the plane for the slice. ---- SKETCH ----
    Level(f64),
}

/// What a ray hits: the scan, or a body's face.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Hit {
    Scan { point: DVec3 },
    Body { point: DVec3, body: BodyId, face: usize },
}

impl Hit {
    pub(crate) fn point(&self) -> DVec3 {
        match self {
            Hit::Scan { point } | Hit::Body { point, .. } => *point,
        }
    }
}

fn v3(p: DVec3) -> Vec3 {
    p.as_vec3()
}

fn seg_dist(p: Pos2, a: Pos2, b: Pos2) -> f32 {
    let ab = b - a;
    let t = if ab.length_sq() > 0.0 { ((p - a).dot(ab) / ab.length_sq()).clamp(0.0, 1.0) } else { 0.0 };
    (a + ab * t - p).length()
}

impl ReverseWorkspace {
    pub(crate) fn project(&self, vp: Viewport, p_part: DVec3) -> Option<Pos2> {
        let n = self.view.camera.project(self.to_scene(p_part), vp.aspect())?;
        (0.0..=1.0).contains(&n.z).then(|| vp.to_screen(n))
    }

    /// The ray under a screen point, in part coordinates.
    fn part_ray(&self, vp: Viewport, p: Pos2) -> (DVec3, DVec3) {
        let r = self.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
        (self.to_part(r.origin), self.to_part(r.dir).normalize_or_zero())
    }

    /// Where the ray under a screen point meets a sketch plane, on its sheet.
    pub(crate) fn sheet_point(&self, vp: Viewport, p: Pos2, frame: &SectionFrame) -> Option<DVec2> {
        let (o, d) = self.part_ray(vp, p);
        let denom = d.dot(frame.normal);
        if denom.abs() < 1e-12 {
            return None;
        }
        let t = (frame.origin - o).dot(frame.normal) / denom;
        Some(frame.to_sheet(o + d * t))
    }

    /// The nearest thing under a screen point, as the view shows it: a visible body (and its
    /// face) or the scan — what the view turns and zooms about.
    pub(crate) fn hit(&mut self, vp: Viewport, p: Pos2) -> Option<Hit> {
        self.hit_on(vp, p, false)
    }

    /// The nearest thing under a screen point that a click picks: a body of the pick set (see
    /// [`ReverseWorkspace::pick_set`]) and its face, or the scan.
    pub(crate) fn pick_hit(&mut self, vp: Viewport, p: Pos2) -> Option<Hit> {
        self.hit_on(vp, p, true)
    }

    fn hit_on(&mut self, vp: Viewport, p: Pos2, picking: bool) -> Option<Hit> {
        // No part under the sketch: nothing of it to hit.
        if self.vb_part_hidden() {
            return None;
        }
        let ray = self.view.camera.ray_from_ndc(vp.ndc(p), vp.aspect());
        let mut best: Option<(f64, Hit)> = None;
        if self.show_bodies {
            // ---- PICKLIST ---- a body not in play for the open box is not picked.
            let dimmed = if picking { self.dimmed_bodies() } else { Vec::new() };
            let meshes: Vec<(BodyId, Arc<mm_core::Mesh>)> = (if picking { self.pick_set() } else { &self.bodies }).iter().filter(|b| !self.hidden_bodies.contains(&b.id) && !dimmed.contains(&b.id)).map(|b| (b.id, b.mesh.clone())).collect();
            for (id, mesh) in meshes {
                let key = Arc::as_ptr(&mesh) as usize;
                let bvh = match self.view.body_bvh.get(&key) {
                    Some(bvh) => bvh.clone(),
                    None => {
                        let bvh = Arc::new(Bvh::build(&mesh));
                        self.view.body_bvh.insert(key, bvh.clone());
                        bvh
                    }
                };
                if let Some(h) = bvh.raycast(&mesh, &ray) {
                    if best.is_none_or(|(t, _)| h.t < t) {
                        let face = mesh.tri_group(h.tid).max(0) as usize;
                        best = Some((h.t, Hit::Body { point: self.to_part(h.point), body: id, face }));
                    }
                }
            }
        }
        if self.show_scan {
            let objects: Vec<super::ScanObject> = self.scan.iter().filter(|o| o.visible).cloned().collect();
            for o in &objects {
                if let Some(bvh) = self.scan_bvh(o) {
                    if let Some(h) = bvh.raycast(&o.mesh, &ray) {
                        if best.is_none_or(|(t, _)| h.t < t) {
                            best = Some((h.t, Hit::Scan { point: self.to_part(h.point) }));
                        }
                    }
                }
            }
        }
        best.map(|b| b.1)
    }

    // ---- the 3D mouse (as the Mesh workspace has it) ----

    /// Apply the 3D mouse's motion and buttons, with the Mesh workspace's settings.
    fn space_mouse(&mut self, ctx: &egui::Context, vp: Viewport) {
        let (motion, pressed) = crate::spacemouse::poll_shared();
        if !self.nav.space.enabled {
            return;
        }
        let dt = ctx.input(|i| i.stable_dt).clamp(0.001, 0.05) as f64;
        let moving = !motion.is_zero();
        if moving && !self.view.space_moving {
            // A move turns about one centre, chosen as it starts.
            self.view.space_centre = Some(self.space_move_centre(vp));
        }
        self.view.space_moving = moving;
        let pivot = self.view.space_centre.unwrap_or(self.view.camera.target);
        if crate::spacemouse::apply(&mut self.view.camera, motion, &self.nav.space, dt, vp.rect.height() as f64, pivot) {
            // Keep animating while the cap is deflected.
            ctx.request_repaint();
        }
        if moving && self.nav.space.show_centre && !self.nav.space.lock_rotation {
            self.view.space_marker = Some((pivot, std::time::Instant::now()));
        }
        for b in pressed {
            self.space_button(b);
        }
    }

    /// The meshes the view shows (scene coordinates): the scan's and the bodies'.
    fn shown_meshes(&self) -> Vec<&mm_core::Mesh> {
        let mut out: Vec<&mm_core::Mesh> = Vec::new();
        if self.show_scan {
            out.extend(self.scan.iter().filter(|o| o.visible).map(|o| &*o.mesh));
        }
        if self.show_bodies {
            out.extend(self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)).map(|b| &*b.mesh));
        }
        out
    }

    /// What a 3D mouse move starting now turns about: what is in view, or the centre of mass,
    /// as Preferences → 3D mouse → Centre says (see `crate::app::centre_in_view`).
    fn space_move_centre(&mut self, vp: Viewport) -> DVec3 {
        let meshes = self.shown_meshes();
        let mut bounds = mm_core::Aabb::EMPTY;
        for m in &meshes {
            let b = m.bounds();
            if !b.is_empty() {
                bounds.include(b.min);
                bounds.include(b.max);
            }
        }
        let in_view = match self.nav.space.centre {
            crate::spacemouse::Centre::View => crate::app::centre_in_view(&self.view.camera, vp, &bounds, &meshes),
            crate::spacemouse::Centre::Object => None,
        };
        if let Some(p) = in_view {
            return p;
        }
        // The centre of mass of the first mesh shown (the scan, else a body), kept per mesh.
        let first = meshes.first().map(|m| (*m as *const mm_core::Mesh as usize, *m));
        match first {
            Some((key, m)) => match self.view.com_cache {
                Some((k, c)) if k == key => c,
                _ => {
                    let c = mm_core::measure::center_of_mass(m);
                    self.view.com_cache = Some((key, c));
                    c
                }
            },
            None => self.view.camera.target,
        }
    }

    /// 3D mouse button `b` was pressed: what the Mesh workspace's settings give it to do, done
    /// here where it means something.
    fn space_button(&mut self, b: u32) {
        use crate::spacemouse::ButtonAction as A;
        let Some(action) = self.nav.space_buttons.iter().find(|(x, _)| *x == b).map(|(_, a)| a.clone()) else { return };
        let done = match action {
            // Held keys act while held, in the app's input (see `spacemouse::with_held_keys`).
            A::Nothing | A::Hold(_) => return,
            A::Key { key, modifiers } => {
                crate::spacemouse::press_key(key, modifiers);
                format!("key {}", crate::spacemouse::key_label(key, modifiers))
            }
            A::Fit => {
                self.view.want_fit = true;
                "fit the view".into()
            }
            A::View(k) => match crate::app::STANDARD_VIEWS.get(k) {
                Some((name, yaw, pitch)) => {
                    self.view.camera.set_view(*yaw, *pitch);
                    format!("{name} view")
                }
                None => return,
            },
            A::Undo | A::Redo => {
                let undo = action == A::Undo;
                match &mut self.mode {
                    Mode::Sketch(edit) => {
                        if undo {
                            edit.undo();
                        } else {
                            edit.redo();
                        }
                    }
                    Mode::Feature(_) => self.close_mode(),
                    Mode::Idle => {
                        if undo {
                            self.undo();
                        } else {
                            self.redo();
                        }
                    }
                }
                if undo { "Undo".into() } else { "Redo".into() }
            }
            A::Accept => {
                self.confirm();
                "Enter".into()
            }
            A::Cancel => {
                match &mut self.mode {
                    Mode::Sketch(edit) => edit.escape(),
                    Mode::Feature(_) => self.close_mode(),
                    Mode::Idle => {}
                }
                "Escape".into()
            }
            A::Hotbox | A::Tool(_) => "a Mesh workspace tool: it works in the Mesh tab".into(),
        };
        self.status = format!("3D mouse button {b}: {done}");
    }

    /// The 3D mouse's centre of rotation, while it moves the view and fading out after.
    fn paint_space_marker(&mut self, ctx: &egui::Context, painter: &egui::Painter, vp: Viewport) {
        let Some((p, last)) = self.view.space_marker else { return };
        const FADE: f32 = 0.1;
        let age = if self.view.space_moving { 0.0 } else { last.elapsed().as_secs_f32() };
        if age < FADE {
            if let Some(n) = self.view.camera.project(p, vp.aspect()).filter(|n| (0.0..=1.0).contains(&n.z)) {
                crate::app::paint_centre_marker(painter, vp.to_screen(n), 1.0 - age / FADE);
            }
            ctx.request_repaint();
        } else {
            self.view.space_marker = None;
        }
    }

    /// The bodies a click picks on: while a feature box is open, the bodies as they stood
    /// before its feature, once a build of them has come in (ScanRuler's `featuresToBuild`); the
    /// bodies shown otherwise. Not read off the mode, which a click's dispatch holds out of
    /// `self.mode` while it runs: `pick_bodies` is let go of whenever a box opens or closes.
    pub(crate) fn pick_set(&self) -> &[super::BodyView] {
        self.pick_bodies.as_deref().unwrap_or(&self.bodies)
    }

    /// A body of the pick set, by id: what a pick's fingerprint is read off.
    pub(crate) fn pick_view(&self, id: BodyId) -> Option<&super::BodyView> {
        self.pick_set().iter().find(|b| b.id == id)
    }

    /// Let go of the BVHs of meshes no body holds any more.
    pub(crate) fn forget_unshown_meshes(&mut self) {
        let live: std::collections::HashSet<usize> = self.bodies.iter().chain(self.pick_bodies.iter().flatten()).map(|b| Arc::as_ptr(&b.mesh) as usize).collect();
        self.view.body_bvh.retain(|k, _| live.contains(k));
    }

    /// The body edge nearest a screen point, within reach, on the pick set: (body, edge index in
    /// its topology). A seam — one face meeting itself — is no edge to pick.
    pub(crate) fn edge_at(&self, vp: Viewport, p: Pos2) -> Option<(BodyId, usize)> {
        let mut best: Option<((BodyId, usize), f32)> = None;
        // ---- PICKLIST ---- a body not in play for the open box is not picked.
        let dimmed = self.dimmed_bodies();
        for b in self.pick_set().iter().filter(|b| !self.hidden_bodies.contains(&b.id) && !dimmed.contains(&b.id)) {
            for (i, e) in b.topology.edges.iter().enumerate() {
                if e.faces.len() == 2 && e.faces[0] == e.faces[1] {
                    continue;
                }
                // Tangent edges left out are not picked either.
                if self.vb.edge_off(&b.topology, e.hash) {
                    continue;
                }
                let pts: Vec<Pos2> = e.polyline.iter().filter_map(|q| self.project(vp, *q)).collect();
                for w in pts.windows(2) {
                    let d = seg_dist(p, w[0], w[1]);
                    if d <= 8.0 && best.is_none_or(|(_, bd)| d < bd) {
                        best = Some(((b.id, i), d));
                    }
                }
            }
        }
        best.map(|b| b.0)
    }

    /// An edge of a body of the pick set, as drawn (part coordinates).
    pub(crate) fn edge_points(&self, edge: (BodyId, usize)) -> Option<&Vec<DVec3>> {
        self.pick_view(edge.0).and_then(|b| b.topology.edges.get(edge.1)).map(|e| &e.polyline)
    }

    /// A body shown, by id.
    pub(crate) fn body_view(&self, id: BodyId) -> Option<&super::BodyView> {
        self.bodies.iter().find(|b| b.id == id)
    }

    /// Fit the view to the scan and the bodies.
    fn fit_view(&mut self, vp: Viewport) {
        let mut b = mm_core::Aabb::EMPTY;
        for o in self.scan.iter().filter(|o| o.visible) {
            let ob = o.mesh.bounds();
            if !ob.is_empty() {
                b.include(ob.min);
                b.include(ob.max);
            }
        }
        for body in &self.bodies {
            let mb = body.mesh.bounds();
            if !mb.is_empty() {
                b.include(mb.min);
                b.include(mb.max);
            }
        }
        if b.is_empty() {
            // A blank stage sized for a part the size of a hand.
            b.include(DVec3::splat(-60.0));
            b.include(DVec3::splat(60.0));
        } else {
            self.view.framed = true;
        }
        self.view.camera.fit(&b, vp.aspect());
        self.view.want_fit = false;
    }

    pub(super) fn viewport(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame) {
        let rect = ui.available_rect_before_wrap();
        let vp = Viewport { rect };
        let response = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        let ctx = ui.ctx().clone();
        if self.view.want_fit {
            self.fit_view(vp);
        }
        // ---- SKETCH ---- the sheet turned (Align): the view swings round to it.
        if let Mode::Sketch(e) = &mut self.mode {
            if std::mem::take(&mut e.frame_turned) {
                let frame = e.frame;
                self.view.face_plane(&frame, self.flip);
            }
        }
        self.space_mouse(&ctx, vp);
        // The view bar's flags, and the section view's gizmo, which takes its drag first.
        self.sync_view_flags();
        self.section_gizmo_input(&ctx, vp, &response);

        // ---- navigation (as the Mesh workspace moves its view) ----
        let line_points = ctx.options(|o| o.input_options.line_scroll_speed);
        let (mods, (wheel, pinch), delta, hover) = ctx.input(|i| (i.modifiers, crate::navigation::wheel_zoom(&i.raw.events, line_points), i.pointer.delta(), i.pointer.hover_pos()));
        let in_rect = hover.is_some_and(|p| rect.contains(p)) && (response.hovered() || response.dragged());
        let nav = self.nav.bindings;
        let primary_nav = nav.starts_navigation(PointerButton::Primary, mods);
        for btn in [PointerButton::Secondary, PointerButton::Middle, PointerButton::Primary] {
            if response.drag_started_by(btn) && nav.starts_navigation(btn, mods) && !self.vb.grabs_pointer() {
                self.view.nav_drag = Some(btn);
                let origin = ctx.input(|i| i.pointer.press_origin());
                self.view.nav_anchor = origin.and_then(|p| self.hit(vp, p)).map(|h| self.to_scene(h.point()));
                break;
            }
        }
        let nav_active = self.view.nav_drag.is_some();
        if let Some(btn) = self.view.nav_drag {
            if response.dragged_by(btn) {
                let cam = &mut self.view.camera;
                match nav.drag_action(btn, mods) {
                    Some(NavAction::Tumble) => {
                        let pivot = self.view.nav_anchor.unwrap_or(cam.target);
                        cam.tumble_about(delta.x as f64, delta.y as f64, pivot);
                    }
                    Some(NavAction::Pan) => {
                        let anchor = self.view.nav_anchor.unwrap_or(cam.target);
                        cam.pan_at(delta.x as f64, delta.y as f64, rect.height() as f64, anchor);
                    }
                    Some(NavAction::Zoom) => cam.zoom((1.0 + delta.y as f64 * 0.005).max(0.05)),
                    None => {}
                }
            }
            if response.drag_stopped_by(btn) || !ctx.input(|i| i.pointer.button_down(btn)) {
                self.view.nav_drag = None;
                self.view.nav_anchor = None;
            }
        }
        // The wheel over a network's curve sets its strip's spans, not the zoom. ---- NET ----
        let wheel = if in_rect && wheel != 0.0 && super::editors::wheel(self, wheel) { 0.0 } else { wheel };
        if in_rect {
            let mut factor = 1.0f64;
            if wheel != 0.0 {
                factor *= (-wheel as f64 * 0.0025 * self.nav.zoom_speed as f64).exp();
            }
            if pinch != 1.0 {
                factor /= pinch as f64;
            }
            if factor != 1.0 {
                let aim = if self.nav.zoom_to_cursor { hover } else { Some(rect.center()) };
                let pivot = aim.and_then(|p| self.hit(vp, p)).map(|h| self.to_scene(h.point()));
                match pivot {
                    Some(p) => self.view.camera.zoom_about(factor, p),
                    None => self.view.camera.zoom(factor),
                }
            }
        }
        // The clip planes follow what is shown.
        let mut b = mm_core::Aabb::EMPTY;
        for o in self.scan.iter().filter(|o| o.visible) {
            let ob = o.mesh.bounds();
            if !ob.is_empty() {
                b.include(ob.min);
                b.include(ob.max);
            }
        }
        for body in &self.bodies {
            let mb = body.mesh.bounds();
            if !mb.is_empty() {
                b.include(mb.min);
                b.include(mb.max);
            }
        }
        if !b.is_empty() {
            self.view.camera.scene_radius = (b.diagonal() * 0.5).max(1e-6).max((self.view.camera.target - b.center()).length());
        } else {
            self.view.camera.scene_radius = 100.0;
        }

        // ---- the pointer ----
        let (pressed, down, released) = ctx.input(|i| (i.pointer.button_pressed(PointerButton::Primary), i.pointer.button_down(PointerButton::Primary), i.pointer.button_released(PointerButton::Primary)));
        let live = hover.filter(|_| in_rect || response.dragged());
        if live.is_some() {
            self.view.last_pointer = live;
        }
        let pos = if released && live.is_none() { self.view.last_pointer } else { live };
        let clicked = response.clicked_by(PointerButton::Primary) && !primary_nav;
        // ---- PLANE ---- the elements' layer: the element box's clicks on the scan, an element picked.
        let taken = self.elements_pointer(vp, pos, clicked, &ctx, mods);
        let clicked = clicked && !taken;
        // A middle click (not a drag) confirms, as Enter does.
        if response.clicked_by(PointerButton::Middle) && !mods.any() {
            self.confirm();
        }
        // A click on a dimension's number opens it for typing.
        let dim_click = clicked.then(|| pos.and_then(|p| self.view.dims.iter().find(|(_, r)| r.contains(p)).map(|d| d.0))).flatten();
        if let (Some(id), Mode::Sketch(edit)) = (dim_click, &mut self.mode) {
            if let Some(v) = edit.sketch.constraint(id).and_then(|c| c.rule.value()) {
                edit.edit_dim = Some((id, sketcher::fmt_value(v)));
                edit.selection = vec![Pick::Constraint(id)];
            }
        } else if !self.vb.grabs_pointer() {
            self.pointer_input(vp, pos, pressed && in_rect && !nav_active && !primary_nav, down && !nav_active && !primary_nav, released && !nav_active, clicked, response.double_clicked_by(PointerButton::Primary), mods);
        }

        // ---- SKETCH ---- the cursor over the sheet says what a click does.
        if let (true, Mode::Sketch(e)) = (in_rect && self.view.nav_drag.is_none(), &self.mode) {
            if let Some(icon) = e.cursor_icon() {
                ctx.set_cursor_icon(icon);
            }
        }

        // ---- render ----
        self.render(ui, frame, vp);

        // ---- painted on top ----
        let painter = ui.painter_at(rect);
        self.paint_overlays(&painter, vp);
        self.viewcmp_overlay(&ctx, vp);
        self.paint_space_marker(&ctx, &painter, vp);
        self.dimension_editor(&ctx, vp);
        self.axis_gizmo(&painter, rect);
        if self.view.nav_drag.is_some() || response.dragged() || response.is_pointer_button_down_on() {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn pointer_input(&mut self, vp: Viewport, pos: Option<Pos2>, pressed: bool, down: bool, released: bool, clicked: bool, _double_clicked: bool, mods: egui::Modifiers) {
        self.view.hover_edge = None;
        self.view.hover_face = None;
        self.view.hover_region = None;
        match std::mem::replace(&mut self.mode, Mode::Idle) {
            Mode::Sketch(mut edit) => {
                let sheet = pos.and_then(|p| self.sheet_point(vp, p, &edit.frame));
                let mm_per_px = match (sheet, pos) {
                    (Some(s), Some(_)) => {
                        let a = self.project(vp, edit.frame.to_world(s));
                        let b = self.project(vp, edit.frame.to_world(s + DVec2::X));
                        match (a, b) {
                            (Some(a), Some(b)) if (b - a).length() > 1e-3 => 1.0 / (b - a).length() as f64,
                            _ => 0.1,
                        }
                    }
                    _ => {
                        let _ = pos;
                        0.1
                    }
                };
                let frame = edit.frame;
                let unproject = |q: Pos2| self.sheet_point(vp, q, &frame);
                self.refresh_projectables(&mut edit);
                let input = SketchInput { pos: sheet, mm_per_px, pressed, down, released, clicked, shift: mods.shift, exact: mods.command, screen: pos, unproject: Some(&unproject) };
                edit.pointer(&input);
                self.mode = Mode::Sketch(edit);
            }
            Mode::Feature(mut edit) => {
                // ---- RND ---- a box's arrow, or a window dragged with Shift over the bodies,
                // takes the pointer before a click reaches the box.
                if let Some(changed) = super::editors::rnd_pointer(self, vp, pos, pressed, down, released, mods.shift, &mut edit) {
                    self.mode = Mode::Feature(edit);
                    if changed {
                        self.request_build();
                    }
                    return;
                }
                // ---- LOFT ---- a grip on the part (Move body's gizmo; the Extrude and Revolve boxes'
                // grips, EXT) takes the press on it and the drag after.
                let grip = super::editors::grip_input(self, vp, pos, pressed, down, released, mods.shift, &mut edit);
                let changed = match (grip, pos) {
                    (Some(changed), _) => changed,
                    (None, Some(p)) => self.feature_pointer(vp, p, clicked, &mut edit),
                    (None, None) => false,
                };
                self.mode = Mode::Feature(edit);
                if changed {
                    self.request_build();
                }
            }
            Mode::Idle => {
                // ---- PICKLIST ---- object-first picking, and the chips by the click (`picks.rs`).
                self.idle_pointer(vp, pos, pressed, down, released, clicked, mods);
            }
        }
    }

    /// The region of a sketch under the cursor, by key.
    #[allow(dead_code)] // the boxes that pick a sketch's regions; Extrude and Revolve have their own (EXT)
    pub(crate) fn region_of_sketch_at(&self, vp: Viewport, p: Pos2, sketch: FeatureId) -> Option<String> {
        let frame = self.sketch_frame(sketch)?;
        let sheet = self.sheet_point(vp, p, &frame)?;
        self.built.sketches.get(&sketch).and_then(|b| region_at(&b.regions, sheet).map(|r| r.key.clone()))
    }

    /// A click (or the hover) while a feature box is open, by the kind's own editor. Returns
    /// whether the feature changed.
    fn feature_pointer(&mut self, vp: Viewport, p: Pos2, clicked: bool, edit: &mut super::FeatureEdit) -> bool {
        super::editors::pick(self, vp, p, clicked, edit)
    }

    /// Enter, or a middle click: finish what is open.
    pub(super) fn confirm(&mut self) {
        if super::editors::confirm(self) {
            return; // ---- NET ----
        }
        match &mut self.mode {
            Mode::Sketch(edit) => {
                // A dimension being typed, or a drawing tool's clicks in progress (a chain of
                // lines), end first; then the sketch is finished.
                if edit.confirm_step() {
                    return;
                }
                self.finish_sketch();
            }
            Mode::Feature(_) => self.commit_feature(),
            Mode::Idle => {}
        }
    }

    pub(super) fn keys(&mut self, ctx: &egui::Context) {
        if ctx.egui_wants_keyboard_input() {
            return;
        }
        if super::editors::keys(self, ctx) {
            return; // ---- NET ----
        }
        let (esc, enter, del, undo, redo, mods) = ctx.input(|i| {
            (
                i.key_pressed(egui::Key::Escape),
                i.key_pressed(egui::Key::Enter),
                i.key_pressed(egui::Key::Delete) || i.key_pressed(egui::Key::Backspace),
                i.modifiers.command && !i.modifiers.shift && i.key_pressed(egui::Key::Z),
                i.modifiers.command && (i.modifiers.shift && i.key_pressed(egui::Key::Z) || i.key_pressed(egui::Key::Y)),
                i.modifiers,
            )
        });
        match &mut self.mode {
            Mode::Sketch(edit) => {
                if esc {
                    edit.escape();
                }
                if del {
                    edit.delete_key();
                }
                if undo {
                    edit.undo();
                }
                if redo {
                    edit.redo();
                }
                if !mods.any() {
                    for t in Tool::KEYED {
                        if let Some(k) = t.key() {
                            if ctx.input(|i| i.key_pressed(k)) {
                                edit.key_tool(t);
                            }
                        }
                    }
                }
                if enter {
                    self.confirm();
                }
            }
            Mode::Feature(_) => {
                if esc {
                    self.close_mode();
                    self.status = "Put away.".into();
                } else if enter {
                    self.confirm();
                }
                if undo {
                    self.close_mode();
                }
            }
            Mode::Idle => {
                if undo {
                    self.undo();
                }
                if redo {
                    self.redo();
                }
                // ---- PICKLIST ----
                self.idle_keys(ctx);
            }
        }
    }

    // ---- rendering ----

    fn render(&mut self, ui: &mut egui::Ui, frame: &eframe::Frame, vp: Viewport) {
        let rect = vp.rect;
        let ctx = ui.ctx().clone();
        let (Some(rs), Some(_)) = (frame.wgpu_render_state(), self.renderer.as_ref()) else {
            ui.painter().rect_filled(rect, 0.0, Color32::from_gray(30));
            ui.painter().text(rect.center(), egui::Align2::CENTER_CENTER, "wgpu renderer unavailable", egui::FontId::proportional(16.0), Color32::WHITE);
            return;
        };
        // ---- PLANE ---- the elements' surfaces and the hinted plane, see-through.
        let plane_surfaces = self.elements_surfaces();
        let sketch_frame = match &self.mode {
            Mode::Sketch(e) => Some(e.frame),
            _ => None,
        };
        // ---- PICKLIST ---- the picks, the coordinate planes and the overlaps (`picks.rs`).
        let pick_tris = self.pick_overlay_tris();
        let pick_lines = self.pick_lines();
        let dimmed = self.dimmed_bodies();
        // The bodies a feature box changes, and whether it cuts.
        let (preview_ids, cutting): (Vec<BodyId>, bool) = match &self.mode {
            Mode::Feature(edit) => {
                let ids = self.preview_feature.map(|fid| self.bodies.iter().filter(|b| b.id.feature == fid || self.built.touched.get(&fid).is_some_and(|t| t.contains(&b.id))).map(|b| b.id).collect()).unwrap_or_default();
                (ids, super::editors::cuts(&edit.kind))
            }
            _ => (Vec::new(), false),
        };
        // ---- DRAFT ----
        // A face being snapped: the scan is clicked next, so the bodies are a faint tint and
        // their edges, the scan solid under them.
        let faint = super::editors::bodies_faint(self);
        // Where the part is cut open (the sketch's plane or the section view's), whether it is
        // put away under the sketch, and how the bodies' edges meet.
        let cut = self.vb_cut();
        let part_hidden = self.vb_part_hidden();
        let joins = self.vb_edge_joins();
        let clip = self.vb_clip();
        let renderer = self.renderer.as_ref().expect("checked");
        let mut no_moves = Vec::new();
        let base_style = ObjectStyle {
            color: BODY,
            wireframe: false,
            flat: false,
            group_colors: false,
            dim: false,
            cut_tint: false,
            crease_deg: Some(30.0),
            group_normals: false,
            vertex_colors: false,
            uv_checker: false,
            texture_mode: 0,
            xray: false,
            cut_away: cut.is_some(),
            material: crate::materials::MATERIALS[0],
        };
        // The scan.
        let ids: Vec<u64> = self.scan.iter().map(|o| o.id).collect();
        self.view.scan_gpu.retain(|id, _| ids.contains(id));
        if self.show_scan {
            for o in self.scan.iter().filter(|o| o.visible) {
                let mut color = o.color;
                if self.scan_see_through && !faint {
                    color.w = color.w.min(0.35);
                }
                let style = ObjectStyle { color, crease_deg: None, wireframe: self.vb.scan_mesh, ..base_style };
                let g = self.view.scan_gpu.entry(o.id).or_insert_with(|| renderer.create_mesh(&rs.device));
                // The view bar's map (live, Deviation or curvature) as it painted the scan.
                let scalars = if !self.deviation.colors.is_empty() { self.deviation.colors.get(&o.id).map(|(stamp, c)| (*stamp, renderer::TriField::Rgb(c.as_slice()))) } else { None };
                let scalars = scalars.or_else(|| self.net.curvature_field(o.id, &self.mode)); // ---- NET ----
                renderer.sync_mesh(&rs.device, &rs.queue, g, &o.mesh, None, 0, &style, (&mut no_moves, 0), scalars, None, None);
            }
        }
        // The bodies.
        let body_ids: Vec<BodyId> = self.bodies.iter().map(|b| b.id).collect();
        self.view.body_gpu.retain(|id, _| body_ids.contains(id));
        // ---- LOFT ---- a Move's ghost: the bodies as they stood, carried where the gizmo puts them.
        let carried = super::editors::carried_bodies(self);
        if self.show_bodies {
            for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)) {
                let carry = carried.iter().find(|c| c.id == b.id);
                let mesh = carry.map_or(&b.mesh, |c| &c.mesh);
                let key = Arc::as_ptr(mesh) as usize;
                let mut color = if preview_ids.contains(&b.id) {
                    if cutting {
                        BODY_CUT
                    } else {
                        BODY_PREVIEW
                    }
                } else {
                    super::picks::vec4(self.body_color32(b.id), 1.0)
                };
                if self.vb.bodies_see_through {
                    color.w = 0.55;
                }
                if faint {
                    color.w = 0.18;
                }
                // ---- PICKLIST ---- a body not in play for the open box is faint.
                if dimmed.contains(&b.id) {
                    color.w = 0.15;
                }
                let style = ObjectStyle { color, ..base_style };
                let entry = self.view.body_gpu.entry(b.id).or_insert_with(|| (key, renderer.create_mesh(&rs.device)));
                if entry.0 != key {
                    *entry = (key, renderer.create_mesh(&rs.device));
                }
                renderer.sync_mesh(&rs.device, &rs.queue, &mut entry.1, mesh, None, 0, &style, (&mut no_moves, 0), None, None, None);
                if let Some(c) = carry {
                    renderer.sync_rigid_preview(&rs.queue, &mut entry.1, &style, c.model);
                }
            }
        }
        // ---- PLANE ----
        super::elements::sync_surfaces(&mut self.elements.gpu, renderer, rs, &plane_surfaces, base_style);
        // Lines: the bodies' edges, nudged towards the eye so they sit on their faces.
        let mut lines = std::mem::take(&mut self.view.lines);
        lines.clear();
        let eye = self.view.camera.eye();
        if self.show_bodies {
            for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id) && !part_hidden) {
                // ---- LOFT ---- a Move's ghost draws the edges it carries (the body as it stood
                // before the move, whose edge ids are not this body's): each drawn as it is.
                let carry = carried.iter().find(|c| c.id == b.id);
                let body_joins = if carry.is_some() { None } else { joins.get(&b.id).map(|j| j.as_ref()) };
                for (i, e) in carry.map_or(&b.edges, |c| &c.edges).iter().enumerate() {
                    // Tangent edges drawn, dimmed or left out as the view bar says; seams and
                    // splits never.
                    let hash = b.part_mesh.edge_ids.get(i).copied().unwrap_or(u32::MAX);
                    let Some(alpha) = self.vb.edge_alpha(body_joins, hash) else { continue };
                    let colour = Vec4::new(EDGE.x, EDGE.y, EDGE.z, EDGE.w * alpha);
                    for w in e.windows(2) {
                        // Not on the side the cut takes away.
                        let Some((a, z)) = super::viewbar::clip_segment(w[0], w[1], clip) else { continue };
                        for p in [a, z] {
                            let q = p + (eye - p) * 2e-4;
                            lines.push(ColorVertex::new(v3(q), colour));
                        }
                    }
                }
            }
        }
        // The round faces' outlines from where the camera stands.
        self.vb_silhouettes(&mut lines, EDGE);
        // Nothing on the part while the part is put away under the sketch.
        if !part_hidden {
            lines.extend_from_slice(&pick_lines);
            // Where a failed feature failed, as its diagnosis found it, red on the part.
            for (a, b) in self.failure_lines() {
                for p in [a, b] {
                    lines.push(ColorVertex::new(v3(p + (eye - p) * 4e-4), Vec4::new(0.878, 0.337, 0.31, 1.0)));
                }
            }
        }
        // The sketch plane's outline while sketching.
        if let Some(f) = sketch_frame {
            let extent = match &self.mode {
                Mode::Sketch(e) => e.chains.iter().flatten().copied().chain(e.sketch.entities.iter().flat_map(|x| mm_reverse::sketch::polyline(&x.shape, 8))).fold(30.0f64, |m, p| m.max(p.abs().max_element())) * 1.15,
                _ => 30.0,
            };
            let c = [DVec2::new(-extent, -extent), DVec2::new(extent, -extent), DVec2::new(extent, extent), DVec2::new(-extent, extent)];
            for i in 0..4 {
                lines.push(ColorVertex::new(v3(self.to_scene(f.to_world(c[i]))), Vec4::new(0.35, 0.8, 0.65, 0.8)));
                lines.push(ColorVertex::new(v3(self.to_scene(f.to_world(c[(i + 1) % 4]))), Vec4::new(0.35, 0.8, 0.65, 0.8)));
            }
        }
        // ---- PLANE ---- the elements' lines.
        self.element_lines(&mut lines);
        let ppp = ctx.pixels_per_point();
        let size = [(rect.width() * ppp).round().max(1.0) as u32, (rect.height() * ppp).round().max(1.0) as u32];
        let aspect = vp.aspect();
        let cam = &self.view.camera;
        let mut globals = renderer::globals(cam.view_proj_f32(aspect), cam.view().as_mat4(), cam.eye().as_vec3(), None, size[1] as f32);
        let bg = self.nav.background;
        globals.background = [bg[0], bg[1], bg[2], 1.0];
        if let Some((f, _)) = cut {
            // Cut open a hair in front of the plane, so what lies on it stays.
            let n = self.to_scene(f.normal);
            let o = self.to_scene(f.origin) + n * 1e-3;
            globals.cut_plane = [n.x as f32, n.y as f32, n.z as f32, n.dot(o) as f32];
        }
        // The caps where the part is cut (the view bar's); the faces picked, the coordinate
        // planes and what bodies share (PICKLIST's, `picks.rs`).
        let mut overlay = self.vb_overlay(&mut lines);
        if !part_hidden {
            overlay.extend_from_slice(&pick_tris);
        }
        let mut objs: Vec<(&GpuMesh, bool)> = Vec::new();
        if self.show_scan && !part_hidden {
            for o in self.scan.iter().filter(|o| o.visible) {
                if let Some(g) = self.view.scan_gpu.get(&o.id) {
                    objs.push((g, (self.scan_see_through && !faint) || o.color.w < 0.999));
                }
            }
            // ---- PLANE ---- the elements, drawn with the scan they were measured on.
            objs.extend(super::elements::surface_objects(&self.elements.gpu, &plane_surfaces));
        }
        if self.show_bodies && !part_hidden {
            for b in self.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id)) {
                if let Some((_, g)) = self.view.body_gpu.get(&b.id) {
                    objs.push((g, self.vb.bodies_see_through || faint || dimmed.contains(&b.id)));
                }
            }
        }
        let input = renderer::FrameInput { globals, objects: objs, lines: &lines, overlay_tris: &overlay };
        let r = self.renderer.as_mut().expect("checked");
        let tex = r.render(rs, size, &input);
        ui.painter().image(tex, rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        // Under a sketch the part is veiled toward the stage, so the sketch reads at full
        // strength over it.
        if sketch_frame.is_some() && !part_hidden {
            let b = |v: f32| (v.clamp(0.0, 1.0) * 255.0) as u8;
            ui.painter().rect_filled(rect, 0.0, Color32::from_rgba_unmultiplied(b(bg[0]), b(bg[1]), b(bg[2]), (0.58f32 * 255.0) as u8));
        }
        self.view.lines = lines;
    }

    fn paint_overlays(&mut self, painter: &egui::Painter, vp: Viewport) {
        self.view.dims.clear();
        // ---- PLANE ---- the elements' names, the picks and the marking, the hinted plane.
        self.paint_elements(painter, vp);
        // Sketches shown on the part (their eye on), the one open aside.
        let open = match &self.mode {
            Mode::Sketch(e) => e.feature,
            _ => None,
        };
        // The sketch the open feature box takes its profiles from is drawn though hidden.
        let feature_sketch = match &self.mode {
            Mode::Feature(edit) => edit.kind.source_sketches(),
            _ => Vec::new(),
        };
        for f in &self.part.features {
            let FeatureKind::Sketch(s) = &f.kind else { continue };
            let in_box = feature_sketch.contains(&f.id);
            if (!f.visible() && !in_box) || f.suppressed() || open == Some(f.id) || matches!(self.mode, Mode::Sketch(_)) {
                continue;
            }
            let frame = self.sketch_frame(f.id).unwrap_or(s.frame);
            let to_screen = |p: DVec2| self.project(vp, frame.to_world(p));
            for e in &s.sketch.entities {
                let pts: Vec<Pos2> = mm_reverse::sketch::polyline(&e.shape, 64).iter().filter_map(|p| to_screen(*p)).collect();
                if pts.len() >= 2 {
                    let c = if e.construction { colors::CONSTRUCTION } else { colors::FREE.gamma_multiply(0.8) };
                    painter.add(egui::Shape::line(pts, Stroke::new(1.2, c)));
                }
            }
        }
        // The open sketch.
        if let Mode::Sketch(edit) = &self.mode {
            let frame = edit.frame;
            let to_screen = |p: DVec2| self.project(vp, frame.to_world(p));
            let unproject = |q: Pos2| self.sheet_point(vp, q, &frame);
            super::sketch_paint::paint_sheet(edit, painter, &to_screen, &unproject, self.show_slice);
            self.view.dims = paint_dimensions(edit, painter, &to_screen);
        }
        // What the open feature box picked, by its own editor, and the edge under the cursor.
        if let Mode::Feature(edit) = &self.mode {
            super::editors::paint(self, painter, vp, edit);
            if let Some(e) = self.view.hover_edge.and_then(|e| self.edge_points(e)) {
                let pts: Vec<Pos2> = e.iter().filter_map(|q| self.project(vp, *q)).collect();
                painter.add(egui::Shape::line(pts, Stroke::new(3.0, BODY_HOVER)));
            }
            let face = self.view.hover_face.and_then(|(b, f)| self.pick_view(b).and_then(|v| v.topology.faces.get(f)).map(|x| x.point));
            if let Some(q) = face.and_then(|p| self.project(vp, p)) {
                painter.circle_stroke(q, 6.0, Stroke::new(2.0, BODY_HOVER));
            }
        }
        // ---- SURF ---- a trim that does not close: its open edges red on the part.
        super::editors::surf_common::paint_open_edges(self, painter, vp);
        // The deviation map's scale is the view bar's legend (viewbar.rs).
        // ---- PICKLIST ----
        self.paint_picks(painter, vp);
        if self.scan.is_empty() && self.part.features.is_empty() && matches!(self.mode, Mode::Idle) {
            painter.text(vp.rect.center(), egui::Align2::CENTER_CENTER, "Open a scan in the Mesh workspace to model over it,\nor sketch on a blank stage: choose a plane under Sketch on", egui::FontId::proportional(17.0), Color32::from_gray(150));
        }
    }

    /// The number of a dimension being typed, over where it is shown.
    fn dimension_editor(&mut self, ctx: &egui::Context, _vp: Viewport) {
        let Mode::Sketch(edit) = &mut self.mode else { return };
        let Some((id, mut text)) = edit.edit_dim.clone() else { return };
        let at = self.view.dims.iter().find(|d| d.0 == id).map(|d| d.1.left_top()).unwrap_or(ctx.content_rect().center());
        let mut done = None;
        egui::Area::new(egui::Id::NULL.with(("re_dim_edit", id))).fixed_pos(at + egui::vec2(-10.0, -30.0)).order(egui::Order::Foreground).show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(90.0));
                r.request_focus();
                if r.changed() {
                    edit.edit_dim = Some((id, text.clone()));
                }
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    done = Some(true);
                }
                if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    done = Some(false);
                }
                ui.weak("Enter keeps it; R6 or Ø12 turns a round one over");
            });
        });
        match done {
            Some(true) => {
                if edit.set_dimension(id, &text) {
                    edit.edit_dim = None;
                }
            }
            Some(false) => edit.edit_dim = None,
            None => {}
        }
    }

    fn axis_gizmo(&self, painter: &egui::Painter, rect: egui::Rect) {
        // Bottom right: the view bar has the bottom-left corner.
        let c = rect.right_bottom() + egui::vec2(-40.0, -40.0);
        let view = self.view.camera.view();
        // The part's axes (Z up in the part), as the scene shows them.
        for (axis, col, name) in [(DVec3::X, Color32::from_rgb(230, 80, 80), "X"), (DVec3::Y, Color32::from_rgb(90, 200, 90), "Y"), (DVec3::Z, Color32::from_rgb(90, 140, 240), "Z")] {
            let v = view.transform_vector3(self.to_scene(axis));
            let end = c + egui::vec2(v.x as f32, -v.y as f32) * 26.0;
            painter.line_segment([c, end], Stroke::new(2.0, col));
            painter.text(end, egui::Align2::CENTER_CENTER, name, egui::FontId::proportional(11.0), col);
        }
    }
}
