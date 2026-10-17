//! Interactive tools. Each tool draws its own property panel, may consume viewport
//! input, may show a preview mesh, and ends with Accept / Cancel.

use crate::camera::Camera;
use crate::renderer::ColorVertex;
use crate::scene::{History, ObjectId, Scene};
use glam::{DVec2, DVec3, Vec3, Vec4};
use mm_core::bvh::RayHit;
use mm_core::{Mesh, Ray};

pub mod select;
pub mod transform;
pub mod inspector;
pub mod ops;
pub mod plane_cut;
pub mod vessel;
pub mod measure;
pub mod sculpt;
pub mod patches;
pub mod boolean;
pub mod deviation;
pub mod cfd_check;
pub mod centerline_tool;

/// Screen-space description of the 3D viewport.
#[derive(Clone, Copy, Debug)]
pub struct Viewport {
    pub rect: egui::Rect,
}

impl Viewport {
    pub fn aspect(&self) -> f64 {
        (self.rect.width() / self.rect.height().max(1.0)) as f64
    }
    pub fn ndc(&self, p: egui::Pos2) -> DVec2 {
        DVec2::new(
            ((p.x - self.rect.min.x) / self.rect.width() * 2.0 - 1.0) as f64,
            (1.0 - (p.y - self.rect.min.y) / self.rect.height() * 2.0) as f64,
        )
    }
    pub fn to_screen(self, ndc: DVec3) -> egui::Pos2 {
        egui::pos2(
            self.rect.min.x + ((ndc.x as f32 + 1.0) * 0.5) * self.rect.width(),
            self.rect.min.y + ((1.0 - ndc.y as f32) * 0.5) * self.rect.height(),
        )
    }
}

/// Overlay geometry a tool wants drawn this frame.
#[derive(Default)]
pub struct Overlay {
    pub lines: Vec<ColorVertex>,
    pub tris: Vec<ColorVertex>,
    pub brush: Option<(Vec3, f32)>,
    /// Plane whose positive side is tinted as "to be discarded", optionally limited to a sphere.
    pub cut_plane: Option<(mm_core::Plane, Option<(DVec3, f64)>)>,
    /// Object the cut tint applies to (None = every object).
    pub cut_target: Option<ObjectId>,
}

impl Overlay {
    pub fn line(&mut self, a: DVec3, b: DVec3, c: Vec4) {
        self.lines.push(ColorVertex::new(a.as_vec3(), c));
        self.lines.push(ColorVertex::new(b.as_vec3(), c));
    }
    pub fn tri(&mut self, a: DVec3, b: DVec3, c: DVec3, col: Vec4) {
        self.tris.push(ColorVertex::new(a.as_vec3(), col));
        self.tris.push(ColorVertex::new(b.as_vec3(), col));
        self.tris.push(ColorVertex::new(c.as_vec3(), col));
    }
    /// Filled square patch on a plane, with outline.
    pub fn plane_patch(&mut self, center: DVec3, normal: DVec3, half: f64, fill: Vec4, outline: Vec4) {
        let (u, v) = mm_core::geom::orthonormal_basis(normal);
        let p = [
            center + (-u - v) * half,
            center + (u - v) * half,
            center + (u + v) * half,
            center + (-u + v) * half,
        ];
        self.tri(p[0], p[1], p[2], fill);
        self.tri(p[0], p[2], p[3], fill);
        for i in 0..4 {
            self.line(p[i], p[(i + 1) % 4], outline);
        }
    }
}

/// Everything a tool may touch.
pub struct ToolCx<'a> {
    pub scene: &'a mut Scene,
    pub history: &'a mut History,
    pub camera: &'a mut Camera,
    pub status: &'a mut String,
    pub egui: &'a egui::Context,
    pub vp: Viewport,
}

impl ToolCx<'_> {
    pub fn ray_at(&self, p: egui::Pos2) -> Ray {
        self.camera.ray_from_ndc(self.vp.ndc(p), self.vp.aspect())
    }
    pub fn project(&self, p: DVec3) -> Option<egui::Pos2> {
        let n = self.camera.project(p, self.vp.aspect())?;
        if n.z < 0.0 || n.z > 1.0 {
            return None;
        }
        Some(self.vp.to_screen(n))
    }
    /// World-space length of one screen point at the depth of `p`.
    pub fn world_per_point(&self, p: DVec3) -> f64 {
        let d = (p - self.camera.eye()).dot(self.camera.forward()).max(1e-9);
        let h = self.vp.rect.height().max(1.0) as f64;
        if self.camera.orthographic {
            self.camera.pixel_size_at_target(h)
        } else {
            2.0 * d * (self.camera.fov_y * 0.5).tan() / h
        }
    }
    /// Closest hit among visible objects (optionally restricted to one object).
    pub fn pick(&mut self, p: egui::Pos2, only: Option<ObjectId>) -> Option<(ObjectId, RayHit)> {
        let ray = self.ray_at(p);
        pick_ray(self.scene, &ray, only)
    }
    pub fn snapshot(&mut self, label: &str) {
        self.history.push(label, self.scene);
    }
}

pub fn pick_ray(scene: &mut Scene, ray: &Ray, only: Option<ObjectId>) -> Option<(ObjectId, RayHit)> {
    let mut best: Option<(ObjectId, RayHit)> = None;
    for o in scene.objects.iter_mut() {
        if !o.visible || only.is_some_and(|id| id != o.id) {
            continue;
        }
        let bvh = o.bvh();
        if let Some(h) = bvh.raycast(&o.mesh, ray) {
            if best.as_ref().map(|(_, b)| h.t < b.t).unwrap_or(true) {
                best = Some((o.id, h));
            }
        }
    }
    best
}

/// Pointer state for the viewport this frame, filtered to what tools care about
/// (navigation gestures are removed before tools see the input).
#[derive(Clone, Debug, Default)]
pub struct PointerInput {
    pub pos: Option<egui::Pos2>,
    pub pressed: bool,
    pub down: bool,
    pub released: bool,
    pub clicked: bool,
    pub double_clicked: bool,
    pub shift: bool,
    pub command: bool,
}

pub enum ToolOutcome {
    Continue,
    Close,
}

pub trait Tool {
    fn name(&self) -> &'static str;
    /// Draw the property panel; return `Close` when the tool finished (accepted or cancelled).
    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome;
    /// Handle left-button interaction inside the viewport.
    fn pointer(&mut self, _input: &PointerInput, _cx: &mut ToolCx) {}
    /// Add 3D overlay geometry.
    fn overlay(&mut self, _ov: &mut Overlay, _cx: &mut ToolCx) {}
    /// Paint 2D overlays on top of the viewport.
    fn paint(&mut self, _painter: &egui::Painter, _cx: &mut ToolCx) {}
    /// Mesh to show instead of an object's own mesh while the tool is active.
    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        None
    }
    /// Keyboard shortcuts while the tool is active.
    fn keys(&mut self, _input: &egui::InputState, _cx: &mut ToolCx) {}
    /// Esc pressed. Return true to close the tool.
    fn cancel(&mut self, _cx: &mut ToolCx) -> bool {
        true
    }
    /// Enter pressed. Return true to close the tool (after committing its result).
    fn accept(&mut self, _cx: &mut ToolCx) -> bool {
        false
    }
    /// Called whenever the tool is removed (closed, replaced, or invalidated).
    fn on_close(&mut self, _cx: &mut ToolCx) {}
    /// Whether the tool stays valid when the scene changes underneath it (undo/redo,
    /// object deletion…). Tools holding previews computed from a mesh return false and are
    /// closed instead of committing stale results.
    fn survives_scene_change(&self) -> bool {
        false
    }
    /// Scripted command (automated tests), e.g. `auto` for the Inspector's Auto Repair All.
    fn script(&mut self, _cmd: &str, _cx: &mut ToolCx) {}
    /// A sub-tool this tool wants opened on top of itself.
    fn take_request(&mut self) -> Option<Box<dyn Tool>> {
        None
    }
    /// Whether the tool is computing something (shows a spinner, blocks other tools).
    fn busy(&self) -> Option<(String, f32)> {
        None
    }
}
