//! Transform (translate / rotate / scale), Face Transform and Soft Transform.

use super::*;
use crate::ui_util;
use glam::{DMat3, DMat4, DQuat, EulerRot};

/// Axes the Transform tool works in (Meshmixer's "Coordinate Space").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Space {
    World,
    /// The object's own frame: the axes turn with the rotations applied to it.
    Local,
}

/// Rotation snapping step while "Enable Snapping" is on.
const SNAP_ANGLE_DEG: f64 = 15.0;

/// Face Transform (T with a selection): only the selected faces move.
struct FaceRegion {
    tris: Vec<u32>,
    /// One pivot for all selected pieces; otherwise each connected piece turns and scales
    /// about its own centre.
    shared_frame: bool,
    /// The selection's border vertices stay put (the region stretches instead of the
    /// faces around it).
    fixed_boundary: bool,
    /// Moving vertices per piece, with the piece's pivot.
    groups: Vec<(Vec<u32>, DVec3)>,
    /// Soft Transform: the change fades out over a band instead of stopping at the border.
    soft: Option<SoftState>,
}

struct SoftState {
    kind: mm_edit::SoftKind,
    continuity: mm_edit::Continuity,
    falloff: f64,
    ignore_groups: bool,
    region: mm_edit::SoftRegion,
    /// Pivot per piece label of `region`.
    pivots: Vec<DVec3>,
}

/// Frame with Z along the area-weighted normal of `tris`.
fn region_frame(m: &Mesh, tris: &[u32]) -> DQuat {
    let n = tris.iter().map(|&t| m.tri_cross(t)).sum::<DVec3>();
    let n = if n.length_squared() > 0.0 { n.normalize() } else { DVec3::Z };
    let (u, _) = mm_core::geom::orthonormal_basis(n);
    let u = u.normalize();
    DQuat::from_mat3(&DMat3::from_cols(u, n.cross(u), n)).normalize()
}

pub struct TransformTool {
    object: Option<ObjectId>,
    region: Option<FaceRegion>,
    /// The object's mesh when the tool opened (to detect changes from elsewhere).
    original: std::sync::Arc<Mesh>,
    /// Mesh the parameters below apply to: the original, or the result baked when the
    /// coordinate space was switched.
    source: std::sync::Arc<Mesh>,
    /// Orientation of the source's local frame.
    frame: DQuat,
    space: Space,
    preview: Option<Mesh>,
    translate: DVec3,
    rotation: DQuat,
    /// Scale along the scale axes (the frame's axes in Local, world axes in World).
    scale: DVec3,
    uniform: bool,
    snap: bool,
    /// Snap the resulting coordinates / sizes to the grid instead of the change.
    absolute: bool,
    snap_step: f64,
    pivot: DVec3,
    /// Object pivot used as the centre (index into the object's pivots), else the box centre.
    pivot_choice: Option<usize>,
    /// The object's pivots when the tool opened.
    object_pivots: Vec<crate::scene::Pivot>,
    /// Transforms already baked into `source` (so pivots can follow on commit).
    baked: DMat4,
    /// Current size along each gizmo axis.
    sizes: DVec3,
    dirty: bool,
    drag: Option<GizmoDrag>,
    hover: Option<Handle>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Handle {
    Move(usize),
    Rotate(usize),
    Scale(usize),
    UniformScale,
}

#[derive(Clone, Copy, Debug)]
struct GizmoDrag {
    handle: Handle,
    start: egui::Pos2,
    translate: DVec3,
    rotation: DQuat,
    scale: DVec3,
    sizes: DVec3,
}

const AXES: [DVec3; 3] = [DVec3::X, DVec3::Y, DVec3::Z];
const AXIS_COLORS: [egui::Color32; 3] = [
    egui::Color32::from_rgb(230, 80, 80),
    egui::Color32::from_rgb(90, 200, 90),
    egui::Color32::from_rgb(90, 140, 240),
];

fn dist_to_segment(p: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let ab = b - a;
    let t = ((p - a).dot(ab) / ab.length_sq().max(1e-6)).clamp(0.0, 1.0);
    p.distance(a + ab * t)
}

fn snap_to(v: f64, step: f64) -> f64 {
    if step > 0.0 {
        (v / step).round() * step
    } else {
        v
    }
}

/// A round step (1, 2 or 5 × 10ⁿ) close to `x`.
fn nice_step(x: f64) -> f64 {
    if !(x > 0.0 && x.is_finite()) {
        return 1.0;
    }
    let p = 10f64.powf(x.log10().floor());
    [1.0, 2.0, 5.0, 10.0].into_iter().map(|k| k * p).min_by(|a, b| (a / x).ln().abs().total_cmp(&(b / x).ln().abs())).unwrap()
}

/// Extents of `m` along the unit axes `axes`, and the centre of that box.
fn extents_along(m: &Mesh, axes: [DVec3; 3]) -> (DVec3, DVec3) {
    extents_of(m.vertex_ids().map(|v| m.vertex(v)), axes)
}

/// Extents of `points` along the unit axes `axes`, and the centre of that box.
fn extents_of(points: impl Iterator<Item = DVec3>, axes: [DVec3; 3]) -> (DVec3, DVec3) {
    let mut lo = DVec3::splat(f64::INFINITY);
    let mut hi = DVec3::splat(f64::NEG_INFINITY);
    for p in points {
        let q = DVec3::new(p.dot(axes[0]), p.dot(axes[1]), p.dot(axes[2]));
        lo = lo.min(q);
        hi = hi.max(q);
    }
    if lo.x > hi.x {
        return (DVec3::ZERO, DVec3::ZERO);
    }
    let c = (lo + hi) * 0.5;
    (hi - lo, axes[0] * c.x + axes[1] * c.y + axes[2] * c.z)
}

impl TransformTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let (object, source, frame) = match cx.scene.active() {
            Some(o) => (Some(o.id), o.mesh.clone(), o.frame),
            None => (None, std::sync::Arc::new(Mesh::new()), DQuat::IDENTITY),
        };
        let mut t = Self {
            object,
            region: None,
            original: source.clone(),
            source,
            frame,
            space: Space::Local,
            preview: None,
            translate: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            scale: DVec3::ONE,
            uniform: false,
            snap: false,
            absolute: false,
            snap_step: 1.0,
            pivot: DVec3::ZERO,
            pivot_choice: None,
            object_pivots: Vec::new(),
            baked: DMat4::IDENTITY,
            sizes: DVec3::ZERO,
            dirty: true,
            drag: None,
            hover: None,
        };
        t.snap_step = nice_step(t.source.bounds().diagonal() / 100.0);
        t.object_pivots = cx.scene.active().map(|o| o.pivots.clone()).unwrap_or_default();
        t.reset_pivot();
        t
    }

    /// Face Transform: move only the active object's selected faces.
    pub fn new_faces(cx: &mut ToolCx) -> Self {
        let mut t = Self::new(cx);
        let tris: Vec<u32> = cx.scene.active().map(|o| o.selection.iter().collect()).unwrap_or_default();
        if !tris.is_empty() {
            t.frame = region_frame(&t.source, &tris);
            t.region = Some(FaceRegion { tris, shared_frame: true, fixed_boundary: false, groups: Vec::new(), soft: None });
            t.update_region();
        }
        t
    }

    /// Soft Transform: the selected faces move, and the change fades out over a falloff band.
    pub fn new_soft(cx: &mut ToolCx) -> Self {
        let mut t = Self::new_faces(cx);
        let Some(r) = t.region.as_mut() else { return t };
        let b = mm_core::Aabb::from_points(r.tris.iter().flat_map(|&tr| t.source.triangle(tr)).map(|v| t.source.vertex(v)));
        r.soft = Some(SoftState {
            kind: mm_edit::SoftKind::Linear,
            continuity: mm_edit::Continuity::SoftPosition,
            falloff: (0.5 * b.diagonal()).max(t.source.average_edge_length() * 2.0),
            ignore_groups: false,
            region: Default::default(),
            pivots: Vec::new(),
        });
        t.update_region();
        t
    }

    /// Recompute the moving vertices and pivots of a face region (after its options change
    /// or the source is baked).
    fn update_region(&mut self) {
        let f = self.scale_frame();
        let axes = AXES.map(|a| f * a);
        let m = &*self.source;
        let Some(r) = self.region.as_mut() else { return };
        let selected: std::collections::HashSet<u32> = r.tris.iter().copied().collect();
        let pieces: Vec<Vec<u32>> = if r.shared_frame {
            vec![r.tris.clone()]
        } else {
            let mut seen = std::collections::HashSet::new();
            let mut out = Vec::new();
            for &t in &r.tris {
                if !seen.insert(t) {
                    continue;
                }
                let mut piece = vec![t];
                let mut stack = vec![t];
                while let Some(x) = stack.pop() {
                    for n in m.tri_neighbors(x) {
                        if n != mm_core::INVALID && selected.contains(&n) && seen.insert(n) {
                            piece.push(n);
                            stack.push(n);
                        }
                    }
                }
                out.push(piece);
            }
            out
        };
        let fixed = r.fixed_boundary;
        r.groups = pieces
            .into_iter()
            .map(|piece| {
                let mut verts: Vec<u32> = piece.iter().flat_map(|&t| m.triangle(t)).collect();
                verts.sort_unstable();
                verts.dedup();
                if fixed {
                    verts.retain(|&v| m.vertex_triangles(v).iter().all(|t| selected.contains(t)));
                }
                let pivot = extents_of(verts.iter().map(|&v| m.vertex(v)), axes).1;
                (verts, pivot)
            })
            .collect();
        self.pivot = extents_of(r.groups.iter().flat_map(|g| g.0.iter()).map(|&v| m.vertex(v)), axes).1;
        if let Some(soft) = r.soft.as_mut() {
            soft.region = mm_edit::soft_region(m, &r.tris, soft.falloff, r.fixed_boundary, soft.ignore_groups, r.shared_frame);
            // Pivot per piece: the centre of its selected vertices.
            let pieces = soft.region.handle_piece.iter().chain(&soft.region.band_piece).max().map_or(0, |&p| p as usize + 1);
            let mut members: Vec<Vec<DVec3>> = vec![Vec::new(); pieces];
            let in_sel: std::collections::HashSet<u32> = r.tris.iter().flat_map(|&t| m.triangle(t)).collect();
            let all = soft.region.handle.iter().zip(&soft.region.handle_piece).chain(soft.region.band.iter().zip(&soft.region.band_piece));
            for (&v, &p) in all {
                if in_sel.contains(&v) {
                    members[p as usize].push(m.vertex(v));
                }
            }
            soft.pivots = members.into_iter().map(|pts| extents_of(pts.into_iter(), axes).1).collect();
        }
        self.dirty = true;
    }

    /// Axes the scale applies along, in the source's coordinates.
    fn scale_frame(&self) -> DQuat {
        match self.space {
            Space::Local => self.frame,
            Space::World => DQuat::IDENTITY,
        }
    }

    /// Gizmo axes: the frame's axes as currently rotated, or the world axes.
    fn axes(&self) -> [DVec3; 3] {
        match self.space {
            Space::Local => AXES.map(|a| (self.rotation * self.frame) * a),
            Space::World => AXES,
        }
    }

    fn reset_pivot(&mut self) {
        if self.region.is_some() {
            self.update_region();
            return;
        }
        let f = self.scale_frame();
        self.pivot = extents_along(&self.source, AXES.map(|a| f * a)).1;
        if let Some(p) = self.pivot_choice.and_then(|i| self.object_pivots.get(i)) {
            self.pivot = self.baked.transform_point3(p.position);
        }
    }

    fn linear(&self) -> DMat3 {
        let s = DMat3::from_diagonal(self.scale);
        match self.space {
            Space::Local => {
                let f = DMat3::from_quat(self.frame);
                DMat3::from_quat(self.rotation) * f * s * f.transpose()
            }
            Space::World => s * DMat3::from_quat(self.rotation),
        }
    }

    fn matrix(&self) -> DMat4 {
        DMat4::from_translation(self.pivot + self.translate) * DMat4::from_mat3(self.linear()) * DMat4::from_translation(-self.pivot)
    }

    fn rebuild(&mut self) {
        let mut mesh = (*self.source).clone();
        if let Some(soft) = self.region.as_ref().and_then(|r| r.soft.as_ref().map(|s| (s, r.shared_frame))) {
            let (soft, shared) = soft;
            let lin = self.linear();
            let (pivot, t) = (self.pivot, self.translate);
            let tf = |piece: u32, x: DVec3| {
                let c = if shared { pivot } else { soft.pivots.get(piece as usize).copied().unwrap_or(pivot) };
                c + t + lin * (x - c)
            };
            for (v, p) in mm_edit::soft_deform(&self.source, &soft.region, &tf, soft.kind, soft.continuity) {
                mesh.set_vertex(v, p);
            }
            self.sizes = extents_of(soft.region.handle.iter().map(|&v| mesh.vertex(v)), self.axes()).0;
        } else if let Some(r) = &self.region {
            let lin = self.linear();
            for (verts, pivot) in &r.groups {
                let pivot = if r.shared_frame { self.pivot } else { *pivot };
                for &v in verts {
                    mesh.set_vertex(v, pivot + self.translate + lin * (self.source.vertex(v) - pivot));
                }
            }
            let moving = r.groups.iter().flat_map(|g| g.0.iter()).map(|&v| mesh.vertex(v));
            self.sizes = extents_of(moving, self.axes()).0;
        } else {
            let m = self.matrix();
            mesh.transform(|p| m.transform_point3(p));
            if m.determinant() < 0.0 {
                mesh.reverse_orientation();
            }
            self.sizes = extents_along(&mesh, self.axes()).0;
        }
        self.preview = Some(mesh);
        self.dirty = false;
    }

    fn is_identity(&self) -> bool {
        self.translate == DVec3::ZERO && self.rotation.angle_between(DQuat::IDENTITY) < 1e-12 && self.scale == DVec3::ONE
    }

    /// Make the current result the new starting point (used when the coordinate space
    /// changes, since a pending scale means different things in the two spaces).
    fn bake(&mut self) {
        if self.dirty || self.preview.is_none() {
            self.rebuild();
        }
        if !self.is_identity() {
            self.baked = self.matrix() * self.baked;
            self.frame = (self.rotation * self.frame).normalize();
            self.source = std::sync::Arc::new(self.preview.clone().unwrap_or_default());
            self.translate = DVec3::ZERO;
            self.rotation = DQuat::IDENTITY;
            self.scale = DVec3::ONE;
        }
        self.reset_pivot();
        self.dirty = true;
    }

    fn commit(&mut self, cx: &mut ToolCx) -> bool {
        let Some(id) = self.object else { return true };
        let baked = !std::sync::Arc::ptr_eq(&self.source, &self.original);
        if self.is_identity() && !baked {
            return true;
        }
        if !cx.scene.get(id).map(|o| std::sync::Arc::ptr_eq(&o.mesh, &self.original)).unwrap_or(false) {
            *cx.status = "Transform cancelled: the object changed while the tool was open.".into();
            return true;
        }
        if self.dirty || self.preview.is_none() {
            self.rebuild();
        }
        cx.history.push(self.name(), cx.scene);
        let frame = (self.rotation * self.frame).normalize();
        let whole = self.region.is_none();
        let total = self.matrix() * self.baked;
        if let (Some(o), Some(m)) = (cx.scene.get_mut(id), self.preview.take()) {
            let sel = o.selection.clone();
            o.mesh = std::sync::Arc::new(m);
            if whole {
                o.frame = frame;
                // Pivots ride along with the object.
                let (_, rot, _) = total.to_scale_rotation_translation();
                for p in &mut o.pivots {
                    p.position = total.transform_point3(p.position);
                    p.frame = (rot * p.frame).normalize();
                }
            }
            o.selection = sel;
            o.touch_selection();
        }
        true
    }

    /// Set the size along gizmo axis `i` (all axes proportionally with uniform scaling).
    fn set_size(&mut self, i: usize, size: f64, base: DVec3, base_scale: DVec3) {
        if !(base[i] > 0.0 && size > 0.0) {
            return;
        }
        let f = size / base[i];
        if self.uniform {
            self.scale = base_scale * f;
        } else {
            self.scale = base_scale;
            self.scale[i] = base_scale[i] * f;
        }
        self.dirty = true;
    }

    fn snapped_size(&self, size: f64, base: f64) -> f64 {
        if !self.snap {
            return size;
        }
        let s = if self.absolute { snap_to(size, self.snap_step) } else { base + snap_to(size - base, self.snap_step) };
        if s > 0.0 {
            s
        } else {
            size
        }
    }

    /// Gizmo size in world units (constant on screen).
    fn gizmo_len(&self, cx: &ToolCx) -> f64 {
        cx.world_per_point(self.pivot + self.translate) * 90.0
    }

    fn hit_test(&self, cx: &ToolCx, p: egui::Pos2) -> Option<Handle> {
        let c3 = self.pivot + self.translate;
        let c = cx.project(c3)?;
        if (p - c).abs().max_elem() < 8.0 {
            return Some(Handle::UniformScale);
        }
        let len = self.gizmo_len(cx);
        let mut best: Option<(f32, Handle)> = None;
        for (i, a) in self.axes().iter().enumerate() {
            if let Some(q) = cx.project(c3 + *a * len * 1.18) {
                if (p - q).abs().max_elem() < 7.0 {
                    return Some(Handle::Scale(i));
                }
            }
            if let Some(tip) = cx.project(c3 + *a * len) {
                let d = dist_to_segment(p, c, tip);
                if d < 8.0 && best.map(|b| d < b.0).unwrap_or(true) {
                    best = Some((d, Handle::Move(i)));
                }
            }
            // Rotation ring of radius 0.7 * len around axis i.
            let (u, v) = mm_core::geom::orthonormal_basis(*a);
            let pts: Vec<egui::Pos2> = (0..=48)
                .filter_map(|k| {
                    let t = k as f64 / 48.0 * std::f64::consts::TAU;
                    cx.project(c3 + (u * t.cos() + v * t.sin()) * len * 0.7)
                })
                .collect();
            for w in pts.windows(2) {
                let d = dist_to_segment(p, w[0], w[1]);
                if d < 6.0 && best.map(|b| d < b.0).unwrap_or(true) {
                    best = Some((d, Handle::Rotate(i)));
                }
            }
        }
        best.map(|b| b.1)
    }

    fn drag_to(&mut self, cx: &ToolCx, p: egui::Pos2) {
        let Some(d) = self.drag else { return };
        let c3 = self.pivot + d.translate;
        let Some(c) = cx.project(c3) else { return };
        // Axes as they were when the drag started.
        let axes = {
            let now = self.rotation;
            self.rotation = d.rotation;
            let a = self.axes();
            self.rotation = now;
            a
        };
        let len = self.gizmo_len(cx);
        let screen_dir = |i: usize| -> Option<(egui::Vec2, f32)> {
            let tip = cx.project(c3 + axes[i] * len)?;
            let v = tip - c;
            let l = v.length().max(1e-3);
            Some((v / l, l))
        };
        match d.handle {
            Handle::Move(i) => {
                let Some((dir, px_len)) = screen_dir(i) else { return };
                let mut along = (p - d.start).dot(dir) as f64 / px_len as f64 * len;
                if self.snap {
                    along = if self.absolute {
                        let c0 = c3.dot(axes[i]);
                        snap_to(c0 + along, self.snap_step) - c0
                    } else {
                        snap_to(along, self.snap_step)
                    };
                }
                self.translate = d.translate + axes[i] * along;
            }
            Handle::Rotate(i) => {
                let a0 = (d.start - c).angle();
                let a1 = (p - c).angle();
                let mut ang = (a1 - a0) as f64;
                // Screen angles run clockwise; flip when the axis points away from the viewer.
                if axes[i].dot(-cx.camera.forward()) > 0.0 {
                    ang = -ang;
                }
                if self.snap {
                    ang = snap_to(ang, SNAP_ANGLE_DEG.to_radians());
                }
                self.rotation = DQuat::from_axis_angle(axes[i], ang) * d.rotation;
            }
            Handle::Scale(i) => {
                let Some((dir, _)) = screen_dir(i) else { return };
                let s0 = (d.start - c).dot(dir).abs().max(4.0) as f64;
                let s1 = (p - c).dot(dir) as f64;
                let size = d.sizes[i] * (s1 / s0).clamp(0.02, 50.0);
                let size = self.snapped_size(size, d.sizes[i]);
                self.set_size(i, size, d.sizes, d.scale);
            }
            Handle::UniformScale => {
                let f = (1.0 + (d.start.y - p.y) as f64 * 0.01).clamp(0.05, 20.0);
                // Snap through the largest dimension.
                let k = (0..3).max_by(|&a, &b| d.sizes[a].total_cmp(&d.sizes[b])).unwrap_or(0);
                let size = self.snapped_size(d.sizes[k] * f, d.sizes[k]);
                let f = if d.sizes[k] > 0.0 { size / d.sizes[k] } else { f };
                self.scale = d.scale * f;
            }
        }
        self.dirty = true;
    }
}

impl Tool for TransformTool {
    fn name(&self) -> &'static str {
        match &self.region {
            Some(r) if r.soft.is_some() => "Soft Transform",
            Some(_) => "Face Transform",
            None => "Transform",
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, self.name());
        if self.object.is_none() {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        if self.dirty {
            self.rebuild();
        }
        ui.small("Drag the arrows to move, the rings to rotate, the end boxes to scale an axis, the centre square to scale uniformly — or type values.");
        let mut changed = false;
        let before = self.space;
        egui::ComboBox::from_label("Coordinate space")
            .selected_text(match self.space {
                Space::Local => "Local Frame",
                Space::World => "World Frame",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.space, Space::Local, "Local Frame").on_hover_text("Axes turn with the object");
                ui.selectable_value(&mut self.space, Space::World, "World Frame");
            });
        if self.space != before {
            let now = self.space;
            self.space = before;
            self.bake();
            self.space = now;
            self.reset_pivot();
            self.rebuild();
        }
        if let Some(r) = &mut self.region {
            let mut region_changed = false;
            let mut redraw = false;
            if let Some(soft) = r.soft.as_mut() {
                use mm_edit::{Continuity, SoftKind};
                let kind_name = |k: SoftKind| match k {
                    SoftKind::Linear => "Linear Deformation",
                    SoftKind::Nonlinear => "Nonlinear Deformation",
                    SoftKind::Flat => "Flat Transition",
                    SoftKind::Stretchy => "Stretchy Transition",
                };
                egui::ComboBox::from_label("Deformation type").selected_text(kind_name(soft.kind)).show_ui(ui, |ui| {
                    for (k, tip) in [
                        (SoftKind::Linear, "Blend the change across the falloff band (fast)"),
                        (SoftKind::Nonlinear, "As-rigid-as-possible band: detail keeps its shape when the selection turns"),
                        (SoftKind::Flat, "Re-solve the band as a smooth surface (detail in the band is smoothed away)"),
                        (SoftKind::Stretchy, "Spread the movement smoothly, stretching the band's detail"),
                    ] {
                        redraw |= ui.selectable_value(&mut soft.kind, k, kind_name(k)).on_hover_text(tip).changed();
                    }
                });
                let cont_name = |c: Continuity| match c {
                    Continuity::Boundaries => "Boundaries",
                    Continuity::SoftPosition => "SoftPosition Boundary",
                    Continuity::SoftTangent => "SoftTangent",
                };
                ui.add_enabled_ui(soft.kind != SoftKind::Nonlinear, |ui| {
                    egui::ComboBox::from_label("Boundary continuity").selected_text(cont_name(soft.continuity)).show_ui(ui, |ui| {
                        for c in [Continuity::Boundaries, Continuity::SoftPosition, Continuity::SoftTangent] {
                            redraw |= ui.selectable_value(&mut soft.continuity, c, cont_name(c)).changed();
                        }
                    });
                });
                let max = (self.source.bounds().diagonal() * 0.5).max(soft.falloff);
                region_changed |= ui
                    .add(egui::Slider::new(&mut soft.falloff, 0.0..=max).text("falloff"))
                    .on_hover_text("Width of the transition band (along the surface)")
                    .changed();
                region_changed |= ui.checkbox(&mut soft.ignore_groups, "Ignore groups").on_hover_text("Let the band spread across face-group borders").changed();
            }
            ui.horizontal(|ui| {
                region_changed |= ui.checkbox(&mut r.shared_frame, "Shared frame").on_hover_text("Off: each selected piece turns and scales about its own centre").changed();
                region_changed |= ui.checkbox(&mut r.fixed_boundary, "Fixed boundary").on_hover_text("Keep the selection's border in place").changed();
            });
            if region_changed {
                self.update_region();
                self.rebuild();
            } else if redraw {
                self.dirty = true;
            }
        }
        let f = self.scale_frame();
        let step = (self.source.bounds().diagonal() * 0.002).max(1e-4);
        ui.label("Translate");
        let mut t = f.inverse() * self.translate;
        ui.horizontal(|ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                if ui.add(egui::DragValue::new(&mut t[i]).speed(step).max_decimals(3)).changed() {
                    self.translate = f * t;
                    changed = true;
                }
            }
        });
        ui.label("Rotate (degrees, XYZ)");
        let local_rot = f.inverse() * self.rotation * f;
        let (rx, ry, rz) = local_rot.to_euler(EulerRot::XYZ);
        // Round-off would show as "-0".
        let tidy = |r: f64| if r.abs() < 1e-9 { 0.0 } else { r.to_degrees() };
        let mut deg = [tidy(rx), tidy(ry), tidy(rz)];
        let mut rot_changed = false;
        ui.horizontal(|ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                rot_changed |= ui.add(egui::DragValue::new(&mut deg[i]).speed(0.5).range(-360.0..=360.0).max_decimals(2)).changed();
            }
        });
        if rot_changed {
            self.rotation = f * DQuat::from_euler(EulerRot::XYZ, deg[0].to_radians(), deg[1].to_radians(), deg[2].to_radians()) * f.inverse();
            changed = true;
        }
        ui.label("Scale");
        ui.horizontal(|ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                let mut s = self.scale[i];
                if ui.add(egui::DragValue::new(&mut s).speed(0.005).range(1e-6..=1e6).max_decimals(4)).changed() && self.scale[i] != 0.0 {
                    if self.uniform {
                        self.scale *= s / self.scale[i];
                    } else {
                        self.scale[i] = s;
                    }
                    changed = true;
                }
            }
        });
        ui.label("Size");
        let (sizes, scale) = (self.sizes, self.scale);
        ui.horizontal(|ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                let mut v = sizes[i];
                if ui.add(egui::DragValue::new(&mut v).speed(step).range(1e-9..=1e9).max_decimals(3)).changed() {
                    self.set_size(i, v, sizes, scale);
                    changed = true;
                }
            }
        });
        ui.checkbox(&mut self.uniform, "Uniform scaling");
        ui.horizontal(|ui| {
            ui.checkbox(&mut self.snap, "Enable snapping")
                .on_hover_text(format!("Gizmo moves and sizes snap to the step, rotations to {SNAP_ANGLE_DEG}°"));
            if self.snap {
                let speed = self.snap_step * 0.05;
                ui.add(egui::DragValue::new(&mut self.snap_step).speed(speed).range(1e-6..=1e6).max_decimals(4));
            }
        });
        if self.snap {
            ui.checkbox(&mut self.absolute, "Absolute coords/sizes").on_hover_text("Snap the resulting position and size to the grid, not the change");
        }
        ui.add_space(4.0);
        let whole = self.region.is_none();
        if whole && !self.object_pivots.is_empty() {
            let label = |c: Option<usize>| c.map_or("Box centre".to_string(), |i| format!("Pivot {}", i + 1));
            let before = self.pivot_choice;
            egui::ComboBox::from_label("Centre").selected_text(label(self.pivot_choice)).show_ui(ui, |ui| {
                ui.selectable_value(&mut self.pivot_choice, None, label(None));
                for i in 0..self.object_pivots.len() {
                    ui.selectable_value(&mut self.pivot_choice, Some(i), label(Some(i)));
                }
            });
            if self.pivot_choice != before {
                // Keep what has been done so far; turn and scale about the new centre from here.
                self.bake();
                changed = true;
            }
        }
        ui.horizontal_wrapped(|ui| {
            let b = self.source.bounds();
            if !whole {
                if ui.button("Reset").clicked() {
                    self.translate = DVec3::ZERO;
                    self.rotation = DQuat::IDENTITY;
                    self.scale = DVec3::ONE;
                    changed = true;
                }
                return;
            }
            if ui.button("Center at origin").on_hover_text("Move the pivot (box centre) to the origin").clicked() {
                self.translate = -self.pivot;
                changed = true;
            }
            if ui.button("Drop to ground").on_hover_text("Move the lowest point onto the ground plane").clicked() {
                let up = cx.camera.world_up();
                let low = if up == DVec3::Z { b.min.z } else { b.min.y };
                self.translate = -up * low;
                changed = true;
            }
            if ui.button("Align principal axes").on_hover_text("Rotate so the object's longest extent lies along X").clicked() {
                let mut tmp = (*self.source).clone();
                let m = mm_edit::align_to_principal_axes(&mut tmp);
                let (_, r, _) = m.to_scale_rotation_translation();
                self.rotation = r;
                changed = true;
            }
            if ui.button("Reset").clicked() {
                self.translate = DVec3::ZERO;
                self.rotation = DQuat::IDENTITY;
                self.scale = DVec3::ONE;
                changed = true;
            }
        });
        if changed {
            self.dirty = true;
        }
        if self.dirty {
            self.rebuild();
        }
        let (accept, cancel) = ui_util::accept_cancel(ui, true);
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
        if self.drag.is_none() {
            self.hover = self.hit_test(cx, pos);
        }
        if input.pressed {
            if let Some(h) = self.hit_test(cx, pos) {
                if self.dirty {
                    self.rebuild();
                }
                self.drag = Some(GizmoDrag {
                    handle: h,
                    start: pos,
                    translate: self.translate,
                    rotation: self.rotation,
                    scale: self.scale,
                    sizes: self.sizes,
                });
            }
        }
        if input.down && self.drag.is_some() {
            self.drag_to(cx, pos);
            if self.dirty {
                self.rebuild();
            }
        }
        if input.released {
            self.drag = None;
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if self.object.is_none() {
            return;
        }
        let c3 = self.pivot + self.translate;
        let Some(c) = cx.project(c3) else { return };
        let len = self.gizmo_len(cx);
        let active = self.drag.map(|d| d.handle).or(self.hover);
        let axes = self.axes();
        for (i, a) in axes.iter().enumerate() {
            let hot = active == Some(Handle::Rotate(i));
            let (u, v) = mm_core::geom::orthonormal_basis(*a);
            let pts: Vec<egui::Pos2> = (0..=64)
                .filter_map(|k| {
                    let t = k as f64 / 64.0 * std::f64::consts::TAU;
                    cx.project(c3 + (u * t.cos() + v * t.sin()) * len * 0.7)
                })
                .collect();
            let col = if hot { egui::Color32::YELLOW } else { AXIS_COLORS[i].gamma_multiply(0.7) };
            painter.add(egui::Shape::line(pts, egui::Stroke::new(if hot { 3.0 } else { 1.5 }, col)));
        }
        for (i, a) in axes.iter().enumerate() {
            let hot = active == Some(Handle::Move(i));
            if let Some(tip) = cx.project(c3 + *a * len) {
                let col = if hot { egui::Color32::YELLOW } else { AXIS_COLORS[i] };
                painter.arrow(c, tip - c, egui::Stroke::new(if hot { 4.0 } else { 2.5 }, col));
            }
            if let Some(q) = cx.project(c3 + *a * len * 1.18) {
                let hot = active == Some(Handle::Scale(i));
                let col = if hot { egui::Color32::YELLOW } else { AXIS_COLORS[i] };
                painter.rect_filled(egui::Rect::from_center_size(q, egui::vec2(8.0, 8.0)), 1.0, col);
            }
        }
        let sq = egui::Rect::from_center_size(c, egui::vec2(10.0, 10.0));
        painter.rect_filled(sq, 1.0, if active == Some(Handle::UniformScale) { egui::Color32::YELLOW } else { egui::Color32::from_gray(230) });
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
    use mm_core::primitives;

    /// A 2 × 1 × 1 box turned by `angle` about Z, with that turn as its frame.
    fn tool(angle: f64, space: Space) -> TransformTool {
        let frame = DQuat::from_rotation_z(angle);
        let mut m = primitives::cube(1.0, 1);
        m.transform(|p| frame * DVec3::new(2.0 * p.x, p.y, p.z));
        let source = std::sync::Arc::new(m);
        let mut t = TransformTool {
            object: Some(1),
            region: None,
            original: source.clone(),
            source,
            frame,
            space,
            preview: None,
            translate: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            scale: DVec3::ONE,
            uniform: false,
            snap: false,
            absolute: false,
            snap_step: 0.5,
            pivot: DVec3::ZERO,
            pivot_choice: None,
            object_pivots: Vec::new(),
            baked: DMat4::IDENTITY,
            sizes: DVec3::ZERO,
            dirty: true,
            drag: None,
            hover: None,
        };
        t.reset_pivot();
        t.rebuild();
        t
    }

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).abs().max_element() < 1e-9
    }

    #[test]
    fn local_frame_sizes_follow_the_object() {
        let mut t = tool(0.5, Space::Local);
        assert!(close(t.sizes, DVec3::new(2.0, 1.0, 1.0)), "{}", t.sizes);
        // Doubling local X stretches the box along its own long axis.
        let (sizes, scale) = (t.sizes, t.scale);
        t.set_size(0, 4.0, sizes, scale);
        t.rebuild();
        assert!(close(t.sizes, DVec3::new(4.0, 1.0, 1.0)), "{}", t.sizes);
        // In world space the same box measures its axis-aligned extents.
        let w = tool(0.5, Space::World);
        let e = w.source.bounds().extents();
        assert!(close(w.sizes, e));
    }

    #[test]
    fn switching_space_keeps_the_result() {
        let mut t = tool(0.4, Space::Local);
        t.rotation = DQuat::from_rotation_x(0.3);
        t.scale = DVec3::new(1.5, 1.0, 0.5);
        t.translate = DVec3::new(0.2, -0.1, 0.3);
        t.rebuild();
        let before = t.preview.clone().unwrap();
        t.bake();
        t.space = Space::World;
        t.reset_pivot();
        t.rebuild();
        let after = t.preview.as_ref().unwrap();
        assert!(before.vertex_ids().all(|v| close(before.vertex(v), after.vertex(v))));
        // The frame took the rotation; the parameters start over.
        assert!(t.frame.angle_between(DQuat::from_rotation_x(0.3) * DQuat::from_rotation_z(0.4)) < 1e-12);
        assert!(t.is_identity());
    }

    #[test]
    fn snapping_steps() {
        assert_eq!(nice_step(0.37), 0.5);
        assert_eq!(nice_step(1.6), 2.0);
        assert_eq!(nice_step(4.1), 5.0);
        assert_eq!(nice_step(7.2), 10.0);
        let mut t = tool(0.0, Space::World);
        t.snap = true;
        // Relative: the change is a multiple of the step; absolute: the result is.
        assert!((t.snapped_size(2.3, 2.0) - 2.5).abs() < 1e-12);
        t.absolute = true;
        assert!((t.snapped_size(2.3, 2.1) - 2.5).abs() < 1e-12);
        assert!((t.snapped_size(0.1, 2.0) - 0.1).abs() < 1e-12, "never snaps to zero");
    }
}

#[cfg(test)]
mod face_tests {
    use super::*;
    use mm_core::primitives;

    fn face_tool(m: Mesh, tris: Vec<u32>) -> TransformTool {
        let source = std::sync::Arc::new(m);
        let mut t = TransformTool {
            object: Some(1),
            region: None,
            original: source.clone(),
            frame: region_frame(&source, &tris),
            source,
            space: Space::Local,
            preview: None,
            translate: DVec3::ZERO,
            rotation: DQuat::IDENTITY,
            scale: DVec3::ONE,
            uniform: false,
            snap: false,
            absolute: false,
            snap_step: 0.5,
            pivot: DVec3::ZERO,
            pivot_choice: None,
            object_pivots: Vec::new(),
            baked: DMat4::IDENTITY,
            sizes: DVec3::ZERO,
            dirty: true,
            drag: None,
            hover: None,
        };
        t.region = Some(FaceRegion { tris, shared_frame: true, fixed_boundary: false, groups: Vec::new(), soft: None });
        t.update_region();
        t.rebuild();
        t
    }

    #[test]
    fn pull_selection_along_its_normal() {
        let m = primitives::plane_grid(4.0, 4.0, 8, 8);
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).truncate().length() < 1.0).collect();
        let selected: std::collections::HashSet<u32> = sel.iter().flat_map(|&t| m.triangle(t)).collect();
        for fixed in [false, true] {
            let mut t = face_tool(m.clone(), sel.clone());
            t.region.as_mut().unwrap().fixed_boundary = fixed;
            t.update_region();
            // Local Z is the selection's normal.
            assert!((t.frame * DVec3::Z - DVec3::Z).length() < 1e-9);
            t.translate = t.frame * DVec3::Z * 0.5;
            t.rebuild();
            let out = t.preview.as_ref().unwrap();
            for v in m.vertex_ids() {
                let z = out.vertex(v).z;
                let border = m.vertex_triangles(v).iter().any(|t| !sel.contains(t));
                let expect = if selected.contains(&v) && !(fixed && border) { 0.5 } else { 0.0 };
                assert!((z - expect).abs() < 1e-12, "fixed {fixed}: vertex {v} at {z}, want {expect}");
            }
        }
    }

    #[test]
    fn separate_frames_scale_each_piece_in_place() {
        let m = primitives::plane_grid(6.0, 2.0, 12, 4);
        let sel: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).x.abs() > 1.5).collect();
        let centre = |out: &Mesh, left: bool| {
            let pts: Vec<DVec3> = sel.iter().filter(|&&t| (m.tri_centroid(t).x < 0.0) == left).map(|&t| out.tri_centroid(t)).collect();
            pts.iter().sum::<DVec3>() / pts.len() as f64
        };
        let mut shared = face_tool(m.clone(), sel.clone());
        shared.scale = DVec3::splat(0.5);
        shared.rebuild();
        let mut apart = face_tool(m.clone(), sel.clone());
        apart.region.as_mut().unwrap().shared_frame = false;
        apart.update_region();
        apart.scale = DVec3::splat(0.5);
        apart.rebuild();
        let before = centre(&m, true);
        // Shared: the left piece moves toward the common centre; separate: it stays.
        assert!(centre(shared.preview.as_ref().unwrap(), true).x > before.x + 0.5);
        assert!((centre(apart.preview.as_ref().unwrap(), true) - before).length() < 1e-9);
    }
}
