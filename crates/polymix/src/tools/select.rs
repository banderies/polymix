//! Face selection tool (Meshmixer "Select", hotkey S): sphere brush, lasso and laser lasso,
//! with the selection-based Edit / Modify / Deform commands in its panel.
//!
//! * Brush: drag on the surface. The brush has a fixed size on the model (zooming does not
//!   change what it covers).
//! * Lasso / Laser Lasso: a drag that starts on the background (or any drag in Lasso mode)
//!   draws a screen curve. A closed curve selects what is inside it; an open curve cuts
//!   across the model and selects the smaller side (see [`super::lasso`]).
//! * Double-click: on a triangle at an open boundary, selects the ring of triangles around
//!   that hole (ready for Erase & Fill); elsewhere, selects the connected part.

use super::lasso::ScreenCurve;
use super::*;
use crate::commands::{self, CmdCx, SelectCmd};
use crate::ui_util;
use mm_core::TriSelection;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectMode {
    Brush,
    Lasso,
    /// Click a face to select the connected faces facing (nearly) the same way.
    Wand,
}

pub struct SelectTool {
    pub mode: SelectMode,
    /// Brush radius in percent of the model size.
    pub size_pct: f32,
    /// Lasso / laser select through the model (occluded and back-facing faces too).
    pub through: bool,
    /// Brush shape (Meshmixer's Brush Mode); Unwrap measures along the surface.
    pub brush_shape: mm_core::BrushShape,
    /// The brush keeps to faces turned the same way as the one under it (leaving the far
    /// side of spikes and thin parts alone). Off, it selects everything its preview covers.
    pub visible_side_only: bool,
    /// The brush only selects faces whose outer side is outside the solid (skips faces
    /// buried inside other parts, e.g. overlapping tubes).
    pub restrict_exterior: bool,
    /// Mirror brush strokes across a plane (X / Y / Z through the object's centre, or free).
    pub symmetry: super::sculpt::Symmetry,
    /// Right-drag grow/shrink metric (Meshmixer's Expand Mode).
    pub expand_mode: ExpandMode,
    /// Brush spread and geodesic growth stop at edges sharper than this (degrees; 0 = off).
    pub crease_deg: f32,
    /// Magic wand: maximum angle (degrees) between a face's normal and the clicked face's.
    pub wand_deg: f32,
    exterior: Option<ExteriorCache>,
    stroke: Option<Stroke>,
    /// Screen curve being drawn (lasso or laser).
    curve: Option<Vec<egui::Pos2>>,
    /// Lasso Mode: finished curves on the surface, per object.
    surface: Option<(ObjectId, Vec<SurfaceCurve>)>,
    /// Lasso Mode: the curve being drawn on the surface, and its last screen position.
    surface_curve: Option<(SurfaceCurve, egui::Pos2)>,
    hover_brush: Option<(DVec3, f64)>,
    /// Selection before the latest click, so a double-click can undo its two brush dabs.
    before_click: Option<(ObjectId, TriSelection, f64)>,
    /// Right-drag expand/contract in progress.
    expand: Option<ExpandSession>,
    /// Sub-tool requested from the panel (opened by the app).
    pub request: Option<Box<dyn Tool>>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpandMode {
    /// Grow by surface distance.
    Geodesic,
    /// Grow across edges flatter than an angle that increases with the drag: fills smooth
    /// regions up to their creases.
    CreaseAngle,
    /// Grow into faces whose vertex colour is close to the selection's, with a tolerance
    /// that increases with the drag (geodesic on meshes without colours).
    VertexColor,
}

/// Mean vertex colour of triangle `t`.
fn tri_color(m: &Mesh, t: u32) -> DVec3 {
    m.triangle(t).iter().filter_map(|&v| m.vertex_color(v)).map(|c| DVec3::new(c[0] as f64, c[1] as f64, c[2] as f64)).sum::<DVec3>() / 3.0
}

/// Right-drag grow/shrink (Meshmixer's right-click-drag Expand Mode): drag right to grow the
/// selection, left to shrink it.
struct ExpandSession {
    object: ObjectId,
    original: TriSelection,
    /// Per triangle: surface distance (across faces) from the selection border, with
    /// creases as barriers.
    dist: Vec<f32>,
    /// World length per screen point at the press position.
    scale: f64,
    mode: ExpandMode,
    crease_cos: Option<f64>,
}

impl ExpandSession {
    fn new(mesh: &Mesh, object: ObjectId, original: TriSelection, scale: f64, mode: ExpandMode, crease_cos: Option<f64>) -> Self {
        use std::collections::BinaryHeap;
        #[derive(PartialEq)]
        struct Item(f64, u32);
        impl Eq for Item {}
        impl PartialOrd for Item {
            fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
                Some(self.cmp(o))
            }
        }
        impl Ord for Item {
            fn cmp(&self, o: &Self) -> std::cmp::Ordering {
                o.0.total_cmp(&self.0)
            }
        }
        // Dijkstra over the face graph from both sides of the border.
        let mut dist = vec![f64::INFINITY; mesh.max_tid()];
        let mut heap = BinaryHeap::new();
        for t in original.iter() {
            for n in mesh.tri_neighbors(t) {
                if n == mm_core::INVALID || original.contains(n) || mm_core::selection::is_crease(mesh, t, n, crease_cos) {
                    continue;
                }
                let half = 0.5 * mesh.tri_centroid(t).distance(mesh.tri_centroid(n));
                for x in [t, n] {
                    if half < dist[x as usize] {
                        dist[x as usize] = half;
                        heap.push(Item(half, x));
                    }
                }
            }
        }
        while let Some(Item(d, t)) = heap.pop() {
            if d > dist[t as usize] {
                continue;
            }
            let side = original.contains(t);
            let c = mesh.tri_centroid(t);
            for n in mesh.tri_neighbors(t) {
                // Stay on this side of the border; creases are barriers.
                if n == mm_core::INVALID || original.contains(n) != side || mm_core::selection::is_crease(mesh, t, n, crease_cos) {
                    continue;
                }
                let nd = d + c.distance(mesh.tri_centroid(n));
                if nd < dist[n as usize] {
                    dist[n as usize] = nd;
                    heap.push(Item(nd, n));
                }
            }
        }
        Self { object, original, dist: dist.into_iter().map(|d| d as f32).collect(), scale, mode, crease_cos }
    }

    /// Selection for a signed drag of `d` world units (positive grows).
    fn apply(&self, mesh: &Mesh, d: f64, out: &mut TriSelection) {
        *out = self.original.clone();
        if d > 0.0 && self.mode == ExpandMode::VertexColor && mesh.has_vertex_colors() {
            let n = self.original.len().max(1) as f64;
            let seed = self.original.iter().map(|t| tri_color(mesh, t)).sum::<DVec3>() / n;
            // RGB distance tolerance: 0.004 per screen point of drag.
            let tol = (d / self.scale.max(1e-12) * 0.004).min(1.8);
            let mut stack: Vec<u32> = self.original.iter().collect();
            while let Some(t) = stack.pop() {
                for n in mesh.tri_neighbors(t) {
                    if n != mm_core::INVALID && !out.contains(n) && !mm_core::selection::is_crease(mesh, t, n, self.crease_cos) && tri_color(mesh, n).distance(seed) <= tol {
                        out.insert(n);
                        stack.push(n);
                    }
                }
            }
            return;
        }
        if d > 0.0 && self.mode == ExpandMode::CreaseAngle {
            // 0.4 degrees per screen point of drag.
            let limit = (d / self.scale.max(1e-12) * 0.4).min(180.0).to_radians().cos();
            let mut stack: Vec<u32> = self.original.iter().collect();
            while let Some(t) = stack.pop() {
                for n in mesh.tri_neighbors(t) {
                    if n != mm_core::INVALID && !out.contains(n) && !mm_core::selection::is_crease(mesh, t, n, self.crease_cos) && mesh.tri_normal(t).dot(mesh.tri_normal(n)) >= limit {
                        out.insert(n);
                        stack.push(n);
                    }
                }
            }
            return;
        }
        let d = d as f32;
        for t in mesh.triangle_ids() {
            let within = self.dist[t as usize] <= d.abs();
            if d > 0.0 && within {
                out.insert(t);
            } else if d < 0.0 && within {
                out.remove(t);
            }
        }
    }
}

struct Stroke {
    object: ObjectId,
    deselect: bool,
    last: Option<DVec3>,
    /// The mesh (allocation and topology version) the selection was last synced with: the
    /// brush only adds and removes live triangles, so it stays in sync until they change.
    synced: Option<(usize, u64)>,
}

impl Default for SelectTool {
    fn default() -> Self {
        Self {
            mode: SelectMode::Brush,
            size_pct: 2.5,
            through: true,
            brush_shape: mm_core::BrushShape::Unwrap,
            visible_side_only: false,
            restrict_exterior: false,
            symmetry: super::sculpt::Symmetry::new(None, None),
            expand_mode: ExpandMode::Geodesic,
            crease_deg: 0.0,
            wand_deg: 20.0,
            exterior: None,
            stroke: None,
            curve: None,
            surface: None,
            surface_curve: None,
            hover_brush: None,
            before_click: None,
            expand: None,
            request: None,
        }
    }
}

/// Winding-number test for "exterior" faces, cached per mesh version.
struct ExteriorCache {
    object: ObjectId,
    version: u64,
    wn: std::sync::Arc<mm_solid::WindingNumber>,
    eps: f64,
    known: std::collections::HashMap<u32, bool>,
}

impl ExteriorCache {
    fn is_exterior(&mut self, mesh: &Mesh, t: u32) -> bool {
        if let Some(&e) = self.known.get(&t) {
            return e;
        }
        let p = mesh.tri_centroid(t) + mesh.tri_normal(t) * self.eps;
        let e = self.wn.eval(p).abs() < 0.5;
        self.known.insert(t, e);
        e
    }
}

/// A curve drawn on the surface: (triangle, point) samples.
type SurfaceCurve = Vec<(u32, DVec3)>;

/// Clicks closer together than this (seconds) may form a double-click.
const DOUBLE_CLICK_WINDOW: f64 = 0.6;

fn to_dvec(p: egui::Pos2) -> DVec2 {
    DVec2::new(p.x as f64, p.y as f64)
}

/// Is triangle `t` in view, not hidden behind other surface? Its orientation doesn't
/// matter: the inside of a cut-open model is visible too (on a closed surface the far side
/// is hidden anyway).
#[derive(Clone, Copy)]
struct ViewTest {
    eye: DVec3,
    fwd: DVec3,
    ortho: bool,
}

impl ViewTest {
    fn new(camera: &crate::camera::Camera) -> Self {
        Self { eye: camera.eye(), fwd: camera.forward(), ortho: camera.orthographic }
    }

    fn visible(&self, mesh: &Mesh, bvh: &mm_core::Bvh, t: u32) -> bool {
        let c = mesh.tri_centroid(t);
        let to_eye = if self.ortho { -self.fwd } else { (self.eye - c).normalize() };
        // Edge-on triangles can't be seen.
        if mesh.tri_normal(t).dot(to_eye).abs() < 1e-9 {
            return false;
        }
        // Occlusion: cast from the centroid toward the eye.
        let origin = c + to_eye * 1e-6 * (1.0 + c.length());
        let ray = Ray::new(origin, to_eye);
        let max_t = if self.ortho { f64::INFINITY } else { self.eye.distance(c) };
        bvh.raycast_filtered(mesh, &ray, max_t, |x| x != t).is_none()
    }
}

impl SelectTool {
    fn crease_cos(&self) -> Option<f64> {
        (self.crease_deg > 0.0).then(|| (self.crease_deg as f64).to_radians().cos())
    }

    fn paint_at(&mut self, cx: &mut ToolCx, pos: egui::Pos2) {
        let Some(stroke) = &mut self.stroke else { return };
        let ray = cx.ray_at(pos);
        let Some((_, hit)) = pick_ray(cx.scene, &ray, Some(stroke.object)) else {
            stroke.last = None;
            return;
        };
        let object = stroke.object;
        let r = brush_radius(cx.scene, object, self.size_pct);
        let Some(stroke) = &mut self.stroke else { return };
        let deselect = stroke.deselect;
        let last = stroke.last.replace(hit.point);
        let Some(o) = cx.scene.get_mut(object) else {
            self.stroke = None;
            return;
        };
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        // Versions are never reused, so the same key means the same live triangles.
        let key = (std::sync::Arc::as_ptr(&mesh) as usize, mesh.topology_version());
        if stroke.synced != Some(key) || o.selection.bits().len() != mesh.max_tid() {
            o.selection.sync(&mesh);
            stroke.synced = Some(key);
        }
        // Stamp along the segment from the previous sample so fast strokes stay continuous.
        let mut stamps = vec![(hit.tid, hit.point)];
        if let Some(prev) = last {
            let d = prev.distance(hit.point);
            let steps = (d / (r * 0.5)).floor() as usize;
            for k in 1..steps.min(64) {
                let p = prev.lerp(hit.point, k as f64 / steps as f64);
                if let Some(n) = bvh.nearest(&mesh, p, r) {
                    stamps.push((n.tid, n.point));
                }
            }
        }
        // Symmetry: also paint the mirror image of every stamp.
        if self.symmetry.enabled {
            let sym = self.symmetry.place(object, &bvh.bounds());
            let mirrored: Vec<(u32, DVec3)> = stamps.iter().filter_map(|&(_, p)| bvh.nearest(&mesh, sym.reflect(p), r).map(|n| (n.tid, n.point))).collect();
            stamps.extend(mirrored);
        }
        if self.restrict_exterior && self.exterior.as_ref().is_none_or(|e| e.object != object || e.version != mesh.shape_version()) {
            let eps = mesh.bounds().diagonal() * 1e-4;
            self.exterior = Some(ExteriorCache { object, version: mesh.shape_version(), wn: std::sync::Arc::new(mm_solid::WindingNumber::new(&mesh)), eps, known: Default::default() });
        }
        let (eye, fwd, ortho) = (cx.camera.eye(), cx.camera.forward(), cx.camera.orthographic);
        let visible_only = self.visible_side_only;
        let shape = self.brush_shape;
        let crease_cos = self.crease_cos();
        let exterior = if self.restrict_exterior { self.exterior.as_mut() } else { None };
        let exterior = std::cell::RefCell::new(exterior);
        let faces_eye = |t: u32| {
            let to_eye = if ortho { -fwd } else { eye - mesh.tri_centroid(t) };
            mesh.tri_normal(t).dot(to_eye) > 0.0
        };
        let mut changed = 0;
        for (tid, p) in stamps {
            // The brush takes everything its preview covers, including faces hidden from view
            // (the far side of a spike). With Visible Side Only it keeps to faces turned the
            // same way as the one under it: the outside as usual, or the inside where the model
            // is cut open, leaving the far wall of a thin part alone.
            let side = faces_eye(tid);
            let allow = |t: u32| {
                if visible_only && faces_eye(t) != side {
                    return false;
                }
                match exterior.borrow_mut().as_mut() {
                    Some(e) => e.is_exterior(&mesh, t),
                    None => true,
                }
            };
            let stamp = mm_core::BrushStamp { shape, seed: tid, center: p, normal: mesh.tri_normal(tid), radius: r, crease_cos };
            changed += o.selection.paint_brush(&mesh, &bvh, &stamp, allow, deselect);
        }
        if changed > 0 {
            o.touch_selection();
        }
    }

    /// Apply a finished lasso / laser curve to the active object.
    fn finish_curve(&mut self, cx: &mut ToolCx, deselect: bool) {
        let Some(pts) = self.curve.take() else { return };
        let pts: Vec<DVec2> = pts.into_iter().map(to_dvec).collect();
        let Some(curve) = ScreenCurve::new(&pts) else { return };
        let Some(id) = cx.scene.active.filter(|id| cx.scene.get(*id).is_some()) else { return };
        let vp = cx.vp;
        let view_proj = cx.camera.view_proj(vp.aspect());
        let view = ViewTest::new(cx.camera);
        let through = self.through;
        cx.snapshot(if curve.closed { "Lasso select" } else { "Laser select" });
        let o = cx.scene.get_mut(id).unwrap();
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        // Side of the curve for every triangle in front of the camera (by centroid), in id
        // order, with its area for the laser (worked out here, summed in order below).
        let laser = !curve.closed;
        let side: Vec<(u32, bool, f64)> = (0..mesh.max_tid() as u32)
            .into_par_iter()
            .filter(|&t| mesh.is_triangle(t))
            .filter_map(|t| {
                let clip = view_proj * mesh.tri_centroid(t).extend(1.0);
                if clip.w <= 0.0 {
                    return None;
                }
                let sp = vp.to_screen(clip.truncate() / clip.w);
                Some((t, curve.contains(to_dvec(sp)), if laser { mesh.tri_area(t) } else { 0.0 }))
            })
            .collect();
        // Closed: the inside. Open (laser): the side with less surface area.
        let pick_inside = curve.closed || {
            let (a_in, a_out) = side.iter().fold((0.0, 0.0), |(i, o), &(_, inside, a)| if inside { (i + a, o) } else { (i, o + a) });
            a_in <= a_out
        };
        let hits: Vec<u32> = side
            .par_iter()
            .filter(|&&(_, inside, _)| inside == pick_inside)
            .map(|&(t, _, _)| t)
            .filter(|&t| through || view.visible(&mesh, &bvh, t))
            .collect();
        o.sync_selection();
        for t in hits {
            if deselect {
                o.selection.remove(t);
            } else {
                o.selection.insert(t);
            }
        }
        o.touch_selection();
        *cx.status = format!("{} triangles selected", o.selection.len());
    }

    fn wand(&mut self, cx: &mut ToolCx, id: ObjectId, seed: u32, deselect: bool) {
        cx.snapshot("Magic wand");
        let crease = self.crease_cos();
        let Some(o) = cx.scene.get_mut(id) else { return };
        let mesh = o.mesh.clone();
        o.selection.sync(&mesh);
        for t in mm_core::selection::magic_wand(&mesh, seed, (self.wand_deg as f64).to_radians(), crease) {
            if deselect {
                o.selection.remove(t);
            } else {
                o.selection.insert(t);
            }
        }
        o.touch_selection();
        *cx.status = format!("{} triangles selected", o.selection.len());
    }

    /// Add a finished surface curve; close the loop when it ends near the first curve's start.
    fn finish_surface_curve(&mut self, cx: &mut ToolCx, pts: SurfaceCurve, deselect: bool) {
        let Some(id) = cx.scene.active else { return };
        let (_, curves) = self.surface.get_or_insert((id, Vec::new()));
        curves.push(pts);
        let first = curves[0][0].1;
        let last = curves.last().and_then(|c| c.last()).map(|p| p.1).unwrap_or(first);
        let n: usize = curves.iter().map(|c| c.len()).sum();
        let near = match (cx.project(first), cx.project(last)) {
            (Some(a), Some(b)) => a.distance(b) < 15.0,
            _ => false,
        };
        if n >= 6 && near {
            self.close_surface_loop(cx, deselect);
        } else {
            *cx.status = "Lasso: keep drawing (turn the view as needed); end near the start or press Enter to close".into();
        }
    }

    /// Join the surface curves into a loop and select (or deselect) the faces it encloses.
    fn close_surface_loop(&mut self, cx: &mut ToolCx, deselect: bool) {
        let Some((id, curves)) = self.surface.take() else { return };
        let Some(o) = cx.scene.get(id) else { return };
        let mesh = o.mesh.clone();
        let waypoints: Vec<u32> = curves.iter().flatten().filter(|(t, _)| mesh.is_triangle(*t)).map(|&(t, p)| surface_lasso::nearest_corner(&mesh, t, p)).collect();
        let faces = surface_lasso::close_loop(&mesh, &waypoints).map(|lp| surface_lasso::enclosed_faces(&mesh, &lp)).unwrap_or_default();
        if faces.is_empty() {
            *cx.status = "The lasso loop does not enclose a region.".into();
            return;
        }
        cx.snapshot("Lasso select");
        let o = cx.scene.get_mut(id).unwrap();
        o.sync_selection();
        for t in faces {
            if deselect {
                o.selection.remove(t);
            } else {
                o.selection.insert(t);
            }
        }
        o.touch_selection();
        *cx.status = format!("{} triangles selected", o.selection.len());
    }

    /// Add every triangle of the active object that is visible in the current view
    /// (Meshmixer: Modify > Select Visible).
    fn select_visible(&mut self, cx: &mut ToolCx) {
        let Some(id) = cx.scene.active.filter(|id| cx.scene.get(*id).is_some()) else { return };
        let view_proj = cx.camera.view_proj(cx.vp.aspect());
        let view = ViewTest::new(cx.camera);
        cx.snapshot("Select visible");
        let o = cx.scene.get_mut(id).unwrap();
        let bvh = o.bvh();
        let mesh = o.mesh.clone();
        let hits: Vec<u32> = mesh
            .triangle_ids()
            .collect::<Vec<_>>()
            .par_iter()
            .copied()
            .filter(|&t| {
                let clip = view_proj * mesh.tri_centroid(t).extend(1.0);
                if clip.w <= 0.0 {
                    return false;
                }
                let ndc = clip.truncate() / clip.w;
                ndc.x.abs() <= 1.0 && ndc.y.abs() <= 1.0 && view.visible(&mesh, &bvh, t)
            })
            .collect();
        o.selection.sync(&mesh);
        for t in hits {
            o.selection.insert(t);
        }
        o.touch_selection();
        *cx.status = format!("{} triangles selected", o.selection.len());
    }

    /// Double-click: the ring of triangles around the hole(s) at this triangle, or else the
    /// connected part.
    fn double_click(&mut self, cx: &mut ToolCx, pos: egui::Pos2, deselect: bool) {
        let Some((id, hit)) = cx.pick(pos, None) else { return };
        // Drop the brush dabs the two clicks just painted.
        let before = self.before_click.take().filter(|(bid, _, _)| *bid == id).map(|(_, sel, _)| sel);
        cx.snapshot("Select connected");
        cx.scene.active = Some(id);
        let o = cx.scene.get_mut(id).unwrap();
        let mesh = o.mesh.clone();
        if let Some(sel) = before {
            o.selection = sel;
        }
        o.selection.sync(&mesh);
        let ring = hole_ring(&mesh, hit.tid);
        let what = if ring.is_empty() { "connected part" } else { "hole border" };
        let tris = if ring.is_empty() { mm_core::topology::flood_fill_triangles(&mesh, &[hit.tid], |_| true) } else { ring };
        for t in tris {
            if deselect {
                o.selection.remove(t);
            } else {
                o.selection.insert(t);
            }
        }
        o.touch_selection();
        *cx.status = format!("Selected {what}: {} triangles selected", o.selection.len());
    }

    fn cmd(&mut self, cx: &mut ToolCx, f: impl FnOnce(&mut CmdCx)) {
        let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
        f(&mut c);
    }
}

impl Tool for SelectTool {
    fn brush_size(&mut self) -> Option<&mut f32> {
        Some(&mut self.size_pct)
    }

    fn settings(&self) -> Option<serde_json::Value> {
        let mode = match self.mode {
            SelectMode::Brush => "brush",
            SelectMode::Lasso => "lasso",
            SelectMode::Wand => "wand",
        };
        let mut v = serde_json::json!({
            "mode": mode, "size_percent": self.size_pct, "visible_side_only": self.visible_side_only, "restrict_exterior": self.restrict_exterior,
            "crease_deg": self.crease_deg, "wand_deg": self.wand_deg, "lasso_through": self.through,
        });
        if let Some(o) = v.as_object_mut() {
            o.extend(self.symmetry.settings());
        }
        Some(v)
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let mut v = serde_json::json!({"type": "object", "properties": {
            "mode": {"type": "string", "enum": ["brush", "lasso", "wand"], "description": "What clicking and dragging do (agents select with mesh_select, or wand_at here)"},
            "size_percent": {"type": "number", "description": "Brush radius as a percentage of the object's size"},
            "visible_side_only": {"type": "boolean", "description": "The brush keeps to faces turned the same way as the one under it"},
            "restrict_exterior": {"type": "boolean", "description": "Skip faces buried inside other parts of the mesh"},
            "crease_deg": {"type": "number", "description": "Brush spread and growth stop at edges sharper than this (degrees; 0 = off)"},
            "wand_deg": {"type": "number", "description": "Magic wand: largest angle between a face's normal and the picked face's (1..90)"},
            "lasso_through": {"type": "boolean", "description": "Lasso selects through the model"},
        }});
        if let Some(p) = v["properties"].as_object_mut() {
            p.extend(super::sculpt::Symmetry::schema());
        }
        Some(v)
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let a = Args(&v);
        self.mode = match a.opt_str("mode") {
            Some("lasso") => SelectMode::Lasso,
            Some("wand") => SelectMode::Wand,
            _ => SelectMode::Brush,
        };
        self.size_pct = (a.f64("size_percent")? as f32).clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1);
        self.visible_side_only = a.opt_bool("visible_side_only").unwrap_or(self.visible_side_only);
        self.restrict_exterior = a.opt_bool("restrict_exterior").unwrap_or(self.restrict_exterior);
        self.crease_deg = (a.f64("crease_deg")? as f32).clamp(0.0, 180.0);
        self.wand_deg = (a.f64("wand_deg")? as f32).clamp(1.0, 90.0);
        self.through = a.opt_bool("lasso_through").unwrap_or(self.through);
        self.symmetry.set_settings(&v, changes, cx)
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        serde_json::json!({
            "object": scene.active,
            "selected_triangles": scene.active().map(|o| o.selection.len()),
            "note": "mesh_selection describes the selection; mesh_select changes it; mesh_edit acts on it (discard, separate, …)",
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        let mut out = vec![Action::new(
            "wand_at",
            "Magic wand: add (or remove) the connected faces turned nearly the same way as the face nearest `at` (within wand_deg, stopping at creases)",
            json!({"at": arg::point("A point on or near the surface, mm"), "deselect": arg::boolean("Remove them from the selection instead")}),
        )
        .required(&["at"])];
        out.extend(super::sculpt::Symmetry::actions());
        out
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        if let Some(r) = self.symmetry.act(action, args, cx) {
            return r;
        }
        let a = Args(args);
        match action {
            "wand_at" => {
                let id = cx.scene.active.ok_or("no active object")?;
                let o = cx.scene.get_mut(id).ok_or("no object")?;
                let h = o.bvh().nearest(&o.mesh, DVec3::from_array(a.point("at")?), f64::INFINITY).ok_or("no surface")?;
                self.wand(cx, id, h.tid, a.opt_bool("deselect").unwrap_or(false));
                Ok(cx.status.clone())
            }
            _ => Err(format!("Select has no action {action}")),
        }
    }

    fn survives_scene_change(&self) -> bool {
        true
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        match cmd {
            "lasso" => self.mode = SelectMode::Lasso,
            "brush" => self.mode = SelectMode::Brush,
            "sphere" => self.brush_shape = mm_core::BrushShape::Sphere,
            "disc" => self.brush_shape = mm_core::BrushShape::SphereDisc,
            "unwrap" => self.brush_shape = mm_core::BrushShape::Unwrap,
            "wand" => self.mode = SelectMode::Wand,
            "backfaces" => self.visible_side_only = false,
            "visibleside" => self.visible_side_only = true,
            // `size:8` (percent of the model).
            c if c.starts_with("size:") => self.size_pct = c[5..].parse().unwrap_or(self.size_pct),
            _ => {}
        }
    }

    fn name(&self) -> &'static str {
        "Select"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Select");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, SelectMode::Brush, "Brush");
            ui.selectable_value(&mut self.mode, SelectMode::Lasso, "Lasso");
            ui.selectable_value(&mut self.mode, SelectMode::Wand, "Wand").on_hover_text("Magic wand: click a face to select the connected faces facing the same way");
        });
        match self.mode {
            SelectMode::Brush => {
                use mm_core::BrushShape;
                let label = |b: BrushShape| match b {
                    BrushShape::Sphere => "Sphere Brush",
                    BrushShape::SphereDisc => "SphereDisc Brush",
                    BrushShape::Unwrap => "Unwrap Brush",
                };
                egui::ComboBox::from_label("Brush mode").selected_text(label(self.brush_shape)).show_ui(ui, |ui| {
                    for b in [BrushShape::Sphere, BrushShape::SphereDisc, BrushShape::Unwrap] {
                        ui.selectable_value(&mut self.brush_shape, b, label(b)).on_hover_text(match b {
                            BrushShape::Sphere => "Everything inside the sphere",
                            BrushShape::SphereDisc => "The connected surface near the brush, not the far side of thin walls",
                            BrushShape::Unwrap => "A disc measured along the surface",
                        });
                    }
                });
                let r = cx.scene.active.map(|id| brush_radius(cx.scene, id, self.size_pct));
                brush_size_ui(ui, &mut self.size_pct, r);
                self.symmetry.ui(ui, cx, "Symmetry");
                ui_util::section(ui, "Filters");
                egui::ComboBox::from_label("Expand mode (Cmd+right-drag)")
                    .selected_text(match self.expand_mode {
                        ExpandMode::Geodesic => "Geodesic Distance",
                        ExpandMode::CreaseAngle => "Crease Angle",
                        ExpandMode::VertexColor => "Vertex Color",
                    })
                    .show_ui(ui, |ui| {
                        ui.selectable_value(&mut self.expand_mode, ExpandMode::Geodesic, "Geodesic Distance").on_hover_text("Grow by distance along the surface");
                        ui.selectable_value(&mut self.expand_mode, ExpandMode::CreaseAngle, "Crease Angle").on_hover_text("Grow across edges flatter than an angle that increases with the drag");
                        ui.selectable_value(&mut self.expand_mode, ExpandMode::VertexColor, "Vertex Color").on_hover_text("Grow into faces coloured like the selection; the drag sets the tolerance");
                    });
                ui.checkbox(&mut self.visible_side_only, "Visible side only").on_hover_text("Only select faces turned the same way as the one under the cursor, leaving the far side of spikes and thin parts alone. Off: the brush selects everything its preview covers, hidden faces too.");
                ui.checkbox(&mut self.restrict_exterior, "Restrict to exterior faces").on_hover_text("Skip faces buried inside other parts of the mesh (e.g. overlapping tubes)");
                ui.add(egui::Slider::new(&mut self.crease_deg, 0.0..=180.0).suffix("°").text("Crease angle threshold"))
                    .on_hover_text("The brush and Cmd+right-drag growth stop at edges sharper than this (0 = off)");
            }
            SelectMode::Wand => {
                ui.add(egui::Slider::new(&mut self.wand_deg, 1.0..=90.0).suffix("°").text("Angle threshold"))
                    .on_hover_text("Faces whose normal differs from the clicked face's by less than this are selected");
            }
            SelectMode::Lasso => {
                let curves = self.surface.as_ref().map(|(_, c)| c.len()).unwrap_or(0);
                ui.horizontal(|ui| {
                    ui.label(format!("{curves} curve(s)"));
                    if ui.add_enabled(curves > 0, egui::Button::new("Close loop")).on_hover_text("Enter").clicked() {
                        self.close_surface_loop(cx, false);
                    }
                    if ui.add_enabled(curves > 0, egui::Button::new("Clear")).on_hover_text("Esc").clicked() {
                        self.surface = None;
                    }
                });
            }
        }
        ui.checkbox(&mut self.through, "Lasso selects through the model")
            .on_hover_text("Include faces hidden behind the visible surface (Meshmixer's laser always cuts through). Turn off to select only what you can see.");
        let (n_sel, n_tri) = cx.scene.active().map(|o| (o.selection.len(), o.mesh.triangle_count())).unwrap_or((0, 0));
        ui.label(format!("{} / {} triangles selected", ui_util::format_count(n_sel), ui_util::format_count(n_tri)));
        ui.small(match self.mode {
            SelectMode::Brush => format!("Drag on the model to paint · Drag from the background to lasso (open curve = laser cut, selects the smaller side) · Shift: deselect · Size: [ ], {} or Cmd+middle-drag", cx.brush_scroll.scroll_label()),
            SelectMode::Lasso => "Draw curves on the surface, turning the view between them; end near the start (or press Enter) to select the enclosed faces · A drag from the background is the laser lasso · Shift: deselect".into(),
            SelectMode::Wand => "Click a face to select the connected faces facing the same way (within the angle) · Shift: deselect · A drag from the background is the laser lasso".into(),
        });
        ui.small("Double-click: hole border (on an open edge) or connected part · Cmd+right-drag: grow / shrink · S: clear");

        ui_util::section(ui, "Selection");
        ui.horizontal_wrapped(|ui| {
            let mut sc = |ui: &mut egui::Ui, label: &str, key: &str, c: SelectCmd, this: &mut Self| {
                if ui.button(label).on_hover_text(key).clicked() {
                    this.cmd(cx, |x| commands::selection_command(x, c));
                }
            };
            sc(ui, "All", "Ctrl/Cmd+A", SelectCmd::All, self);
            sc(ui, "Clear", "Esc", SelectCmd::Clear, self);
            sc(ui, "Invert", "I", SelectCmd::Invert, self);
            sc(ui, "Grow", ">", SelectCmd::Expand, self);
            sc(ui, "Shrink", "<", SelectCmd::Contract, self);
            sc(ui, "Connected", "E", SelectCmd::Connected, self);
            sc(ui, "Invert connected", "Shift+I", SelectCmd::InvertConnected, self);
            sc(ui, "To groups", "G: connected faces of the same groups", SelectCmd::ExpandToGroups, self);
            sc(ui, "Same groups", "All faces of the selected groups", SelectCmd::ExpandToSameGroups, self);
            sc(ui, "Optimize boundary", "O", SelectCmd::OptimizeBoundary, self);
            if ui.button("Visible").on_hover_text("Select every face visible in the view").clicked() {
                self.select_visible(cx);
            }
        });

        let has_sel = n_sel > 0;
        ui_util::section(ui, "Edit");
        ui.add_enabled_ui(has_sel, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Discard").on_hover_text("X").clicked() {
                    self.cmd(cx, commands::discard_selection);
                }
                if ui.button("Erase & Fill").on_hover_text("F").clicked() {
                    self.request = Some(Box::new(ops::EraseFillTool::new(cx)));
                }
                if ui.button("Extrude").on_hover_text("D").clicked() {
                    self.request = Some(Box::new(ops::ExtrudeTool::new(cx, ops::ExtrudeMode::Extrude)));
                }
                if ui.button("Extract").on_hover_text("Shift+D: new solid object from the selection").clicked() {
                    self.request = Some(Box::new(ops::ExtrudeTool::new(cx, ops::ExtrudeMode::Extract)));
                }
                if ui.button("Offset").on_hover_text("Ctrl/Cmd+D").clicked() {
                    self.request = Some(Box::new(ops::ExtrudeTool::new(cx, ops::ExtrudeMode::Offset)));
                }
                if ui.button("Bridge").on_hover_text("Cmd+B: remove two selected regions and join the openings with a tube").clicked() {
                    self.request = Some(Box::new(ops::RegionBridgeTool::new(cx)));
                }
                if ui.button("Tube Handle").on_hover_text("Like Bridge, with a curved tube standing up from each region").clicked() {
                    let mut t = ops::RegionBridgeTool::new(cx);
                    t.params = ops::RegionBridgeUi::tube_handle();
                    self.request = Some(Box::new(t));
                }
                if ui.button("Grow Hair").on_hover_text("Grow strands on the selected faces").clicked() {
                    self.request = Some(Box::new(super::hair::HairTool::new(cx)));
                }
                if ui.button("Join").on_hover_text("J: stitch the two open boundaries next to the selection").clicked() {
                    self.request = Some(Box::new(ops::JoinTool::new(cx)));
                }
                if ui.button("Weld Boundaries").on_hover_text("Weld the selection's open edges that lie close together").clicked() {
                    self.request = Some(Box::new(ops::WeldBoundariesTool::new(cx)));
                }
                if ui.button("Align To Target").on_hover_text("Move the object so the selected faces fit the target surface").clicked() {
                    self.request = Some(Box::new(ops::AlignToTargetTool::new(cx)));
                }
                if ui.button("Attract To Target").on_hover_text("Pull the selected faces onto the target surface").clicked() {
                    self.request = Some(Box::new(ops::AttractToTargetTool::new(cx)));
                }
                if ui.button("Fit Primitive").on_hover_text("P: fit a plane, cylinder, sphere, hull … to the selected faces").clicked() {
                    self.request = Some(Box::new(ops::FitPrimitiveTool::new(cx)));
                }
                if ui.button("Separate").clicked() {
                    self.cmd(cx, |x| commands::separate_selection(x, false));
                }
                if ui.button("Duplicate").clicked() {
                    self.cmd(cx, |x| commands::separate_selection(x, true));
                }
                if ui.button("Flip normals").clicked() {
                    self.cmd(cx, commands::flip_selection_normals);
                }
                if ui.button("Create face group").on_hover_text("Ctrl/Cmd+G").clicked() {
                    self.cmd(cx, |x| commands::create_face_group(x, false));
                }
                if ui.button("Clear face group").clicked() {
                    self.cmd(cx, |x| commands::create_face_group(x, true));
                }
            });
        });
        ui_util::section(ui, "Deform");
        ui.add_enabled_ui(has_sel, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Transform").on_hover_text("T: move the selected faces").clicked() {
                    self.request = Some(Box::new(super::transform::TransformTool::new_faces(cx)));
                }
                if ui.button("Soft Transform").on_hover_text("Shift+T: move the selection with a smooth falloff").clicked() {
                    self.request = Some(Box::new(super::transform::TransformTool::new_soft(cx)));
                }
                if ui.button("Warp").on_hover_text("Bend the selection with handles").clicked() {
                    self.request = Some(Box::new(super::warp::WarpTool::new(cx)));
                }
            });
        });
        ui_util::section(ui, "Modify");
        ui.add_enabled_ui(has_sel, |ui| {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Smooth").on_hover_text("Deform > Smooth (Cmd+F)").clicked() {
                    self.request = Some(Box::new(ops::SmoothTool::new(cx)));
                }
                if ui.button("Smooth Boundary").on_hover_text("Smooth the border of the selection (B)").clicked() {
                    self.request = Some(Box::new(ops::SmoothBoundaryTool::new(cx)));
                }
                if ui.button("Remesh").on_hover_text("R").clicked() {
                    self.request = Some(Box::new(ops::RemeshTool::new(cx)));
                }
                if ui.button("Reduce").on_hover_text("Shift+R").clicked() {
                    self.request = Some(Box::new(ops::ReduceTool::new(cx)));
                }
            });
        });
        ui_util::section(ui, "Convert To");
        ui.add_enabled_ui(has_sel, |ui| {
            ui.horizontal_wrapped(|ui| {
                use crate::parts::ConvertTo;
                for (label, hint, what) in [
                    ("Open Part", "Save the selection as a part that welds into surfaces along its border (Meshmix > My parts)", ConvertTo::OpenPart),
                    ("Solid Part", "Save a closed selection (a whole shell) as a solid part", ConvertTo::SolidPart),
                    ("Stamp", "Save the selection's outline as a Stamp stencil", ConvertTo::Stamp),
                ] {
                    if ui.button(label).on_hover_text(hint).clicked() {
                        if let Some(o) = cx.scene.active() {
                            *cx.status = match crate::parts::convert_selection(&o.mesh, &o.selection.to_vec(), what) {
                                Ok(msg) => msg,
                                Err(e) => format!("Convert To {label}: {e}"),
                            };
                        }
                    }
                }
            });
        });
        ui.add_space(10.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        // Setting the symmetry plane (a brush option) takes the pointer.
        if self.mode == SelectMode::Brush && self.symmetry.pointer(input, cx) {
            self.hover_brush = None;
            return;
        }
        let Some(pos) = input.pos else { return };
        let deselect = input.shift || input.command;
        if input.double_clicked {
            self.stroke = None;
            self.curve = None;
            self.double_click(cx, pos, deselect);
            return;
        }
        if input.pressed {
            let now = cx.egui.input(|i| i.time);
            let hit = cx.pick(pos, cx.scene.active).or_else(|| cx.pick(pos, None));
            // Remember the selection before this click unless it may be the second click of
            // a double-click (then keep the state from before the first one).
            let recent = self.before_click.as_ref().is_some_and(|(_, _, t)| now - t < DOUBLE_CLICK_WINDOW);
            if !recent {
                self.before_click = hit.as_ref().and_then(|(id, _)| cx.scene.get(*id).map(|o| (*id, o.selection.clone(), now)));
            }
            match (self.mode, hit) {
                (SelectMode::Brush, Some((id, _))) => {
                    cx.scene.active = Some(id);
                    cx.snapshot("Brush select");
                    self.stroke = Some(Stroke { object: id, deselect, last: None, synced: None });
                }
                // Magic wand: one click selects a region.
                (SelectMode::Wand, Some((id, h))) => {
                    cx.scene.active = Some(id);
                    self.wand(cx, id, h.tid, deselect);
                }
                // Lasso mode on the model: a curve on the surface (part of a multi-curve loop).
                (SelectMode::Lasso, Some((id, h))) => {
                    if self.surface.as_ref().is_some_and(|(o, _)| *o != id) {
                        self.surface = None;
                    }
                    cx.scene.active = Some(id);
                    self.surface_curve = Some((vec![(h.tid, h.point)], pos));
                }
                // A drag from the background (either mode): screen lasso / laser.
                _ => self.curve = Some(vec![pos]),
            }
        }
        if input.down {
            if self.stroke.is_some() {
                self.paint_at(cx, pos);
            }
            if let Some((pts, last)) = &mut self.surface_curve {
                if last.distance(pos) > 3.0 {
                    if let Some((_, h)) = cx.pick(pos, cx.scene.active) {
                        pts.push((h.tid, h.point));
                        *last = pos;
                    }
                }
            }
            if let Some(c) = &mut self.curve {
                if c.last().map(|l| l.distance(pos) > 2.0).unwrap_or(true) {
                    c.push(pos);
                }
            }
        }
        if input.released {
            if self.stroke.take().is_some() {
                if let Some(o) = cx.scene.active() {
                    *cx.status = format!("{} triangles selected", o.selection.len());
                }
            }
            if self.curve.is_some() {
                self.finish_curve(cx, deselect);
            }
            if let Some((pts, _)) = self.surface_curve.take() {
                self.finish_surface_curve(cx, pts, deselect);
            }
        }
        // Hover cursor (brush mode, not while drawing a curve).
        self.hover_brush = match (self.mode, &self.curve) {
            (SelectMode::Brush, None) => cx.pick(pos, cx.scene.active).map(|(id, h)| (h.point, brush_radius(cx.scene, id, self.size_pct))),
            _ => None,
        };
    }

    fn wheel(&mut self, delta: f32, _cx: &mut ToolCx) -> bool {
        if self.mode != SelectMode::Brush {
            return false;
        }
        self.size_pct = (self.size_pct * (delta * 0.004).exp()).clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1);
        true
    }

    fn claims_secondary_drag(&self, scene: &Scene) -> bool {
        scene.active().is_some_and(|o| !o.selection.is_empty())
    }

    fn secondary_drag(&mut self, phase: DragPhase, pos: egui::Pos2, total: egui::Vec2, cx: &mut ToolCx) {
        match phase {
            DragPhase::Start => {
                let Some(id) = cx.scene.active else { return };
                let at = cx.pick(pos, Some(id)).map(|(_, h)| h.point).unwrap_or(cx.camera.target);
                let scale = cx.world_per_point(at);
                cx.snapshot("Expand selection");
                let o = cx.scene.get_mut(id).unwrap();
                let mesh = o.mesh.clone();
                o.selection.sync(&mesh);
                self.expand = Some(ExpandSession::new(&mesh, id, o.selection.clone(), scale, self.expand_mode, self.crease_cos()));
            }
            DragPhase::Move | DragPhase::End => {
                let Some(session) = &self.expand else { return };
                let Some(o) = cx.scene.get_mut(session.object) else {
                    self.expand = None;
                    return;
                };
                let mesh = o.mesh.clone();
                let d = total.x as f64 * session.scale;
                session.apply(&mesh, d, &mut o.selection);
                o.touch_selection();
                *cx.status = if d > 0.0 && session.mode == ExpandMode::CreaseAngle {
                    format!("Grow across edges under {:.0}° · {} triangles selected", (d / session.scale.max(1e-12) * 0.4).min(180.0), o.selection.len())
                } else {
                    format!("{} by {:.3} · {} triangles selected", if d >= 0.0 { "Grow" } else { "Shrink" }, d.abs(), o.selection.len())
                };
                if phase == DragPhase::End {
                    self.expand = None;
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        if self.mode == SelectMode::Brush {
            self.symmetry.overlay(ov, cx);
        }
        let col = Vec4::new(1.0, 0.85, 0.2, 1.0);
        let drawing = self.surface_curve.as_ref().map(|(p, _)| p);
        for c in self.surface.iter().flat_map(|(_, cs)| cs.iter()).chain(drawing) {
            for w in c.windows(2) {
                ov.line(w[0].1, w[1].1, col);
            }
        }
        if let Some((c, r)) = self.hover_brush {
            ov.brush = Some((c.as_vec3(), r as f32));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if self.mode == SelectMode::Brush {
            self.symmetry.paint(painter, cx);
        }
        let Some(curve) = &self.curve else { return };
        if curve.len() < 2 {
            return;
        }
        let stroke = egui::Stroke::new(1.5, ui_util::ACCENT);
        painter.add(egui::Shape::line(curve.clone(), stroke));
        let pts: Vec<DVec2> = curve.iter().copied().map(to_dvec).collect();
        if super::lasso::is_closed(&pts) {
            painter.line_segment([*curve.last().unwrap(), curve[0]], stroke);
        } else if let Some(ext) = super::lasso::extensions(&pts, 4000.0) {
            // The laser continues straight past both ends.
            let faint = egui::Stroke::new(1.0, ui_util::ACCENT.gamma_multiply(0.45));
            for (a, b) in ext {
                painter.line_segment([egui::pos2(a.x as f32, a.y as f32), egui::pos2(b.x as f32, b.y as f32)], faint);
            }
        }
    }

    fn keys(&mut self, i: &egui::InputState, cx: &mut ToolCx) -> bool {
        use egui::Key;
        let cmd = i.modifiers.command;
        let shift = i.modifiers.shift;
        let mut run = |f: &mut dyn FnMut(&mut CmdCx)| {
            let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
            f(&mut c);
        };
        if i.key_pressed(Key::A) && cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::All));
        } else if i.key_pressed(Key::S) && !cmd && !shift {
            // S again while selecting starts over.
            run(&mut |x| commands::selection_command(x, SelectCmd::Clear));
        } else if i.key_pressed(Key::I) && !cmd {
            let c = if shift { SelectCmd::InvertConnected } else { SelectCmd::Invert };
            run(&mut |x| commands::selection_command(x, c));
        } else if i.key_pressed(Key::E) && !cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::Connected));
        } else if i.key_pressed(Key::G) && cmd {
            run(&mut |x| commands::create_face_group(x, shift));
        } else if i.key_pressed(Key::G) {
            run(&mut |x| commands::selection_command(x, SelectCmd::ExpandToGroups));
        } else if i.key_pressed(Key::O) && !cmd {
            run(&mut |x| commands::selection_command(x, SelectCmd::OptimizeBoundary));
        } else if i.key_pressed(Key::Period) || (i.key_pressed(Key::Equals) && !cmd) {
            run(&mut |x| commands::selection_command(x, SelectCmd::Expand));
        } else if i.key_pressed(Key::Comma) || (i.key_pressed(Key::Minus) && !cmd) {
            run(&mut |x| commands::selection_command(x, SelectCmd::Contract));
        } else if i.key_pressed(Key::X) || i.key_pressed(Key::Delete) || i.key_pressed(Key::Backspace) {
            run(&mut commands::discard_selection);
        } else if i.key_pressed(Key::OpenBracket) {
            step_brush_size(&mut self.size_pct, false);
        } else if i.key_pressed(Key::CloseBracket) {
            step_brush_size(&mut self.size_pct, true);
        } else if i.key_pressed(Key::F) && !cmd {
            self.request = Some(Box::new(ops::EraseFillTool::new(cx)));
        } else if i.key_pressed(Key::D) && !cmd && !shift {
            self.request = Some(Box::new(ops::ExtrudeTool::new(cx, ops::ExtrudeMode::Extrude)));
        } else if i.key_pressed(Key::D) && shift && !cmd {
            self.request = Some(Box::new(ops::ExtrudeTool::new(cx, ops::ExtrudeMode::Extract)));
        } else if i.key_pressed(Key::D) && cmd {
            self.request = Some(Box::new(ops::ExtrudeTool::new(cx, ops::ExtrudeMode::Offset)));
        } else if i.key_pressed(Key::R) && !cmd && shift {
            self.request = Some(Box::new(ops::ReduceTool::new(cx)));
        } else if i.key_pressed(Key::R) && !cmd {
            self.request = Some(Box::new(ops::RemeshTool::new(cx)));
        } else if i.key_pressed(Key::B) && !cmd {
            self.request = Some(Box::new(ops::SmoothBoundaryTool::new(cx)));
        } else if i.key_pressed(Key::F) && cmd {
            self.request = Some(Box::new(ops::SmoothTool::new(cx)));
        } else if i.key_pressed(Key::B) && cmd {
            self.request = Some(Box::new(ops::RegionBridgeTool::new(cx)));
        } else if i.key_pressed(Key::P) && !cmd && !shift {
            self.request = Some(Box::new(ops::FitPrimitiveTool::new(cx)));
        } else if i.key_pressed(Key::J) && !cmd && !shift {
            self.request = Some(Box::new(ops::JoinTool::new(cx)));
        } else if i.key_pressed(Key::Y) && !cmd && !shift {
            run(&mut |x| commands::separate_selection(x, false));
        } else if i.key_pressed(Key::T) && !cmd && cx.scene.active().is_some_and(|o| !o.selection.is_empty()) {
            // T with a selection transforms just the selected faces; Shift+T softly.
            let t = if shift { super::transform::TransformTool::new_soft(cx) } else { super::transform::TransformTool::new_faces(cx) };
            self.request = Some(Box::new(t));
        } else {
            return false;
        }
        true
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        if self.symmetry.accept_edit() {
            return false;
        }
        // Enter closes a Lasso Mode loop in progress.
        if self.surface.is_some() {
            self.close_surface_loop(cx, false);
        }
        false
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        // Esc leaves symmetry Set, drops lasso curves in progress, then clears the
        // selection, then exits.
        if self.symmetry.cancel_edit() {
            return false;
        }
        if self.surface.take().is_some() {
            return false;
        }
        if cx.scene.active().map(|o| !o.selection.is_empty()).unwrap_or(false) {
            let mut c = CmdCx { scene: cx.scene, history: cx.history, status: cx.status };
            commands::selection_command(&mut c, SelectCmd::Clear);
            return false;
        }
        true
    }

    fn take_request(&mut self) -> Option<Box<dyn Tool>> {
        self.request.take()
    }
}

/// Triangles touching the open boundary loop(s) that pass through triangle `t` — the ring
/// to erase and refill for a clean hole fill. Empty when `t` is not at an open boundary.
pub fn hole_ring(mesh: &Mesh, t: u32) -> Vec<u32> {
    let at_boundary: Vec<u32> = mesh.triangle(t).into_iter().filter(|&v| mesh.is_boundary_vertex(v)).collect();
    if at_boundary.is_empty() {
        return Vec::new();
    }
    let mut ring = Vec::new();
    for lp in mm_core::topology::boundary_loops(mesh) {
        if lp.vertices.iter().any(|v| at_boundary.contains(v)) {
            for &v in &lp.vertices {
                ring.extend(mesh.vertex_triangles(v));
            }
        }
    }
    ring.sort_unstable();
    ring.dedup();
    ring
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn exterior_filter_skips_buried_faces() {
        // Two overlapping closed spheres (like overlapping vessel segments).
        let mut m = primitives::icosphere(1.0, 3);
        let mut b = primitives::icosphere(1.0, 3);
        b.transform(|p| p + DVec3::new(1.0, 0.0, 0.0));
        m.append_mesh(&b, 0);
        let mut ext = ExteriorCache { object: 0, version: 0, wn: std::sync::Arc::new(mm_solid::WindingNumber::new(&m)), eps: 1e-4, known: Default::default() };
        let tris: Vec<u32> = m.triangle_ids().collect();
        let inside_other = |t: u32| {
            let c = m.tri_centroid(t);
            c.length() < 0.95 || (c - DVec3::new(1.0, 0.0, 0.0)).length() < 0.95
        };
        let mut buried = 0;
        for &t in &tris {
            if inside_other(t) {
                buried += 1;
                assert!(!ext.is_exterior(&m, t), "triangle {t} is buried");
            } else if m.tri_centroid(t).x < -0.5 || m.tri_centroid(t).x > 1.5 {
                assert!(ext.is_exterior(&m, t));
            }
        }
        assert!(buried > 50);
    }

    #[test]
    fn view_test_sees_the_front_half() {
        let m = primitives::icosphere(1.0, 3);
        let bvh = mm_core::Bvh::build(&m);
        let mut cam = crate::camera::Camera::default();
        cam.focus(&m.bounds());
        let view = ViewTest::new(&cam);
        let n = m.triangle_ids().filter(|&t| view.visible(&m, &bvh, t)).count();
        let frac = n as f64 / m.triangle_count() as f64;
        assert!((0.3..0.5).contains(&frac), "visible fraction {frac}");
        // On a closed surface the visible triangles face the eye.
        assert!(m.triangle_ids().filter(|&t| view.visible(&m, &bvh, t)).all(|t| m.tri_normal(t).dot(cam.eye() - m.tri_centroid(t)) > 0.0));
        // Cut open, the inside (turned away from the eye) is visible too.
        let mut bowl = m.clone();
        let front: Vec<u32> = bowl.triangle_ids().filter(|&t| bowl.tri_normal(t).dot(cam.eye() - bowl.tri_centroid(t)) > 0.0).collect();
        bowl.remove_triangles(front);
        let bvh = mm_core::Bvh::build(&bowl);
        let inside = bowl.triangle_ids().filter(|&t| view.visible(&bowl, &bvh, t)).count();
        assert!(inside as f64 > 0.3 * bowl.triangle_count() as f64, "{inside} of {}", bowl.triangle_count());
    }

    #[test]
    fn brush_paints_the_inside_of_a_cut_open_model() {
        // A bowl seen from its open side: every visible face is turned away from the eye.
        let mut m = primitives::icosphere(1.0, 4);
        let front: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.0).collect();
        m.remove_triangles(front);
        let mut scene = Scene::default();
        let id = scene.add("bowl", m);
        let mut history = crate::scene::History::default();
        let mut camera = crate::camera::Camera::default();
        // Looking down -z, into the opening.
        camera.set_view(0.0, 0.0);
        camera.focus(&scene.bounds());
        let mut status = String::new();
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = SelectTool { size_pct: 10.0, ..Default::default() };
        assert!(!tool.visible_side_only);
        let c = vp.rect.center();
        tool.pointer(&PointerInput { pos: Some(c), pressed: true, down: true, ..Default::default() }, &mut cx);
        tool.pointer(&PointerInput { pos: Some(c + egui::vec2(4.0, 0.0)), down: true, ..Default::default() }, &mut cx);
        tool.pointer(&PointerInput { pos: Some(c + egui::vec2(4.0, 0.0)), released: true, ..Default::default() }, &mut cx);
        let o = cx.scene.get(id).unwrap();
        assert!(o.selection.len() > 20, "the whole brush footprint, not one face: {}", o.selection.len());
        // Every selected face is on the inside (turned away from the eye), none on the far rim.
        let eye = cx.camera.eye();
        assert!(o.selection.iter().all(|t| o.mesh.tri_normal(t).dot(eye - o.mesh.tri_centroid(t)) < 0.0));
    }

    #[test]
    fn brush_takes_the_hidden_side_of_a_spike() {
        // A spike leaning 45° to the view (looking down −z): its far side faces away from the
        // eye. The brush covers the whole spike, so by default all of it is selected; with
        // Visible Side Only the far side is left alone.
        let mut m = primitives::icosphere(1.0, 4);
        let dir = DVec3::new(1.0, 0.0, 1.0).normalize();
        let tip = m.vertex_ids().max_by(|&a, &b| m.vertex(a).dot(dir).total_cmp(&m.vertex(b).dot(dir))).unwrap();
        let base = m.vertex(tip);
        m.set_vertex(tip, base * 2.5);
        let spike: Vec<u32> = m.vertex_triangles(tip).to_vec();
        let mut camera = crate::camera::Camera::default();
        camera.set_view(0.0, 0.0);
        let unselected = |visible_side_only: bool| {
            let mut scene = Scene::default();
            let id = scene.add("spiky", m.clone());
            let (mut history, mut camera, mut status) = (crate::scene::History::default(), camera.clone(), String::new());
            camera.focus(&scene.bounds());
            let ctx = egui::Context::default();
            let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
            let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
            let mut tool = SelectTool { size_pct: 40.0, visible_side_only, ..Default::default() };
            let at = cx.project(base * 1.75).unwrap();
            tool.pointer(&PointerInput { pos: Some(at), pressed: true, down: true, ..Default::default() }, &mut cx);
            tool.pointer(&PointerInput { pos: Some(at), released: true, ..Default::default() }, &mut cx);
            let o = cx.scene.get(id).unwrap();
            let eye = cx.camera.eye();
            let hidden = spike.iter().filter(|&&t| o.mesh.tri_normal(t).dot(eye - o.mesh.tri_centroid(t)) < 0.0).count();
            assert!(hidden > 0, "part of the spike faces away");
            spike.iter().filter(|&&t| !o.selection.contains(t)).count()
        };
        assert_eq!(unselected(false), 0, "the whole spike, as the preview shows");
        assert!(unselected(true) > 0, "Visible Side Only leaves the far side");
    }

    #[test]
    fn hole_ring_surrounds_the_hole() {
        // Open cylinder: double-clicking a triangle on the bottom rim selects that rim's ring.
        let m = primitives::cylinder(1.0, 2.0, 24, 6, false);
        let rim_tri = m.triangle_ids().find(|&t| m.triangle(t).iter().any(|&v| m.vertex(v).z == 0.0)).unwrap();
        let ring = hole_ring(&m, rim_tri);
        // One row of quads (2 triangles each) touches the bottom rim.
        assert_eq!(ring.len(), 48);
        assert!(ring.iter().all(|&t| m.triangle(t).iter().any(|&v| m.vertex(v).z == 0.0)));
        // A triangle away from both rims has no ring.
        let mid = m.triangle_ids().find(|&t| m.triangle(t).iter().all(|&v| m.vertex(v).z > 0.5 && m.vertex(v).z < 1.5)).unwrap();
        assert!(hole_ring(&m, mid).is_empty());
    }

    #[test]
    fn right_drag_grows_and_shrinks_by_surface_distance() {
        let m = primitives::plane_grid(10.0, 10.0, 20, 20);
        // Select the triangles within 2 units of the centre.
        let mut sel = TriSelection::new(&m);
        for t in m.triangle_ids() {
            if m.tri_centroid(t).length() < 2.0 {
                sel.insert(t);
            }
        }
        let n0 = sel.len();
        let session = ExpandSession::new(&m, 0, sel.clone(), 1.0, ExpandMode::Geodesic, None);
        let mut out = TriSelection::new(&m);
        session.apply(&m, 0.0, &mut out);
        assert_eq!(out.len(), n0);
        session.apply(&m, 1.0, &mut out);
        let grown = out.len();
        assert!(grown > n0);
        session.apply(&m, 2.0, &mut out);
        assert!(out.len() > grown, "a longer drag grows further");
        // Growth follows distance: nothing beyond ~2 + 2 units from the centre.
        assert!(out.iter().all(|t| m.tri_centroid(t).length() < 4.6));
        session.apply(&m, -0.8, &mut out);
        assert!(out.len() < n0 && !out.is_empty());
        assert!(out.iter().all(|t| sel.contains(t)), "shrinking only removes");
    }

    #[test]
    fn crease_angle_expand_and_barriers() {
        let m = primitives::cube(2.0, 6);
        let top: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_normal(t).z > 0.99).collect();
        let mut sel = TriSelection::new(&m);
        sel.insert(top[0]);
        let mut out = TriSelection::new(&m);
        // Crease Angle mode: a short drag (10°) fills the flat top face only…
        let crease = ExpandSession::new(&m, 0, sel.clone(), 1.0, ExpandMode::CreaseAngle, None);
        crease.apply(&m, 25.0, &mut out);
        assert_eq!(out.len(), top.len());
        // …a long one (100°) crosses the 90° edges.
        crease.apply(&m, 250.0, &mut out);
        assert_eq!(out.len(), m.triangle_count());
        // Geodesic growth with a 30° crease limit stays on the top face however far it grows.
        let limited = ExpandSession::new(&m, 0, sel, 1.0, ExpandMode::Geodesic, Some(30f64.to_radians().cos()));
        limited.apply(&m, 100.0, &mut out);
        assert_eq!(out.len(), top.len());
    }

    #[test]
    fn vertex_color_expand() {
        // Left half red, right half blue.
        let mut m = primitives::plane_grid(10.0, 4.0, 20, 8);
        for v in m.vertex_ids().collect::<Vec<_>>() {
            let c = if m.vertex(v).x < 0.0 { [0.9, 0.1, 0.1] } else { [0.1, 0.1, 0.9] };
            m.set_vertex_color(v, c);
        }
        let seed = m.triangle_ids().find(|&t| m.tri_centroid(t).x < -3.0).unwrap();
        let mut sel = TriSelection::new(&m);
        sel.insert(seed);
        let session = ExpandSession::new(&m, 0, sel, 1.0, ExpandMode::VertexColor, None);
        let mut out = TriSelection::new(&m);
        // A small tolerance fills the red half (triangles straddling x = 0 are mixed).
        session.apply(&m, 20.0, &mut out);
        assert!(out.len() > 10 && out.iter().all(|t| m.tri_centroid(t).x < 0.0));
        let red = m.triangle_ids().filter(|&t| m.triangle(t).iter().all(|&v| m.vertex(v).x < 0.0)).count();
        assert!(out.len() >= red, "{} >= {red}", out.len());
        // A large one takes everything.
        session.apply(&m, 450.0, &mut out);
        assert_eq!(out.len(), m.triangle_count());
    }

    #[test]
    fn erase_and_fill_of_the_ring_closes_the_hole() {
        use crate::tools::ops::{EraseFillSpec, OpInfo, OpSpec};
        let m = primitives::cylinder(1.0, 2.0, 24, 6, false);
        let rim_tri = m.triangle_ids().find(|&t| m.triangle(t).iter().any(|&v| m.vertex(v).z == 0.0)).unwrap();
        let ring = hole_ring(&m, rim_tri);
        let info = OpInfo { mesh: &m, selection: Some(&ring), avg_edge: m.average_edge_length(), bbox_diag: m.bounds().diagonal(), target: None };
        let out = EraseFillSpec::run(&m, Some(&ring), &EraseFillSpec::defaults(&info), &|_| true).unwrap();
        assert_eq!(mm_core::topology::boundary_loops(&out.mesh).len(), 1, "only the top rim stays open");
        out.mesh.check_validity().unwrap();
    }

    #[test]
    fn brush_selects_the_mirror_image_across_a_free_plane() {
        // A sphere turned so its mirror plane is tilted; brushing one spot selects its mirror
        // image too, with the plane set by agents or by three points.
        let rot = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.2, -0.4, 0.3);
        let mut m = primitives::icosphere(1.0, 4);
        m.transform(|p| rot * p);
        let plane = super::super::sculpt::SymPlane::new(DVec3::ZERO, rot * DVec3::X);
        let mut scene = Scene::default();
        let id = scene.add("ball", m);
        let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
        camera.look_along(-DVec3::Z);
        camera.focus(&scene.bounds());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = SelectTool { size_pct: 4.0, ..Default::default() };
        let n = plane.normal;
        tool.set_settings(&serde_json::json!({"symmetry": true, "symmetry_plane": {"point_mm": [0.0, 0.0, 0.0], "normal": [n.x, n.y, n.z]}}), &mut cx).unwrap();
        assert_eq!(tool.settings().unwrap()["symmetry_axis"], "free");
        // A spot facing the camera, off the plane.
        let spot = (DVec3::new(0.35, 0.25, 0.0) + DVec3::Z * (1.0 - 0.35f64.powi(2) - 0.25f64.powi(2)).sqrt()).normalize();
        assert!(plane.distance(spot) > 0.2);
        let at = cx.project(spot).unwrap();
        tool.pointer(&PointerInput { pos: Some(at), pressed: true, down: true, ..Default::default() }, &mut cx);
        tool.pointer(&PointerInput { pos: Some(at), released: true, ..Default::default() }, &mut cx);
        let o = cx.scene.get(id).unwrap();
        let near = |p: DVec3| o.mesh.triangle_ids().min_by(|&a, &b| o.mesh.tri_centroid(a).distance(p).total_cmp(&o.mesh.tri_centroid(b).distance(p))).unwrap();
        assert!(o.selection.contains(near(spot)) && o.selection.contains(near(plane.reflect(spot))));
        assert!(o.selection.iter().all(|t| {
            let c = o.mesh.tri_centroid(t);
            c.distance(spot).min(c.distance(plane.reflect(spot))) < 0.5
        }));
        // Three points on the plane place it too, X / Y / Z put it back across an axis.
        let (u, v) = mm_core::geom::orthonormal_basis(n);
        let pts = serde_json::json!({"points": [u.to_array(), v.to_array(), (-u).to_array()]});
        tool.act("symmetry_through_points", &pts, &mut cx).unwrap();
        assert_eq!(tool.settings().unwrap()["symmetry_axis"], "free");
        tool.set_settings(&serde_json::json!({"symmetry_axis": "z"}), &mut cx).unwrap();
        assert_eq!(tool.settings().unwrap()["symmetry_axis"], "z");
    }
}
