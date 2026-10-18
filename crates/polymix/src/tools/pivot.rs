//! Create Pivot (Edit > Create Pivot): click the model to drop a pivot — a point with a
//! coordinate frame attached to the object. Pivots move with the object in Transform, can be
//! its rotation / scale centre there, are drawn on top of the model (View > Show pivots) and
//! are kept in projects. "First Midpoint Along Normal" puts a pivot in the middle of a tube,
//! e.g. on a vessel's centreline.

use super::*;
use crate::scene::Pivot;
use crate::ui_util;
use glam::{DMat3, DQuat};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Placement {
    Vertex,
    Edge,
    EdgeMidpoint,
    FaceCenter,
    GroupCenter,
    GroupBorderCenter,
    BoundaryCenter,
    MidpointAlongRay,
    MidpointAlongNormal,
    BoundingBox,
}

impl Placement {
    const ALL: [Placement; 10] = [
        Placement::Vertex,
        Placement::Edge,
        Placement::EdgeMidpoint,
        Placement::FaceCenter,
        Placement::GroupCenter,
        Placement::GroupBorderCenter,
        Placement::BoundaryCenter,
        Placement::MidpointAlongRay,
        Placement::MidpointAlongNormal,
        Placement::BoundingBox,
    ];
    fn label(self) -> &'static str {
        match self {
            Placement::Vertex => "Snap To Vertex",
            Placement::Edge => "Snap To Edge",
            Placement::EdgeMidpoint => "Snap To Edge Midpoint",
            Placement::FaceCenter => "Snap To Face Center",
            Placement::GroupCenter => "Snap To Group Center",
            Placement::GroupBorderCenter => "Snap To Group Border Center",
            Placement::BoundaryCenter => "Snap To Boundary Center",
            Placement::MidpointAlongRay => "First Midpoint Along Ray",
            Placement::MidpointAlongNormal => "First Midpoint Along Normal",
            Placement::BoundingBox => "Bounding Box",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameMode {
    Geometry,
    EyeRay,
    World,
}

/// Position, frame z axis and (optionally) frame x axis for a click at `hit`.
fn place(m: &Mesh, bvh: &mm_core::Bvh, hit: &mm_core::bvh::RayHit, ray: &Ray, mode: Placement) -> Option<(DVec3, DVec3, Option<DVec3>)> {
    let t = hit.tid;
    let tv = m.triangle(t);
    let nt = m.tri_normal(t);
    let nearest_edge = || {
        m.tri_edges(t).into_iter().min_by(|&a, &b| {
            let d = |e: u32| {
                let [p, q] = m.edge_vertices(e).map(|v| m.vertex(v));
                let s = ((hit.point - p).dot(q - p) / (q - p).length_squared().max(1e-300)).clamp(0.0, 1.0);
                (p + (q - p) * s).distance(hit.point)
            };
            d(a).total_cmp(&d(b))
        })
    };
    let edge_normal = |e: u32| m.edge_triangles(e).into_iter().filter(|&x| x != mm_core::INVALID).map(|x| m.tri_normal(x)).sum::<DVec3>().normalize_or(nt);
    let group_tris = |g: i32| m.triangle_ids().filter(move |&x| m.tri_group(x) == g);
    let area_normal = |tris: &mut dyn Iterator<Item = u32>| tris.map(|x| m.tri_normal(x) * m.tri_area(x)).sum::<DVec3>().normalize_or(nt);
    Some(match mode {
        Placement::Vertex => {
            let v = *tv.iter().min_by(|&&a, &&b| m.vertex(a).distance(hit.point).total_cmp(&m.vertex(b).distance(hit.point)))?;
            (m.vertex(v), m.vertex_normal(v), None)
        }
        Placement::Edge | Placement::EdgeMidpoint => {
            let e = nearest_edge()?;
            let [p, q] = m.edge_vertices(e).map(|v| m.vertex(v));
            let pos = if mode == Placement::EdgeMidpoint {
                (p + q) * 0.5
            } else {
                let s = ((hit.point - p).dot(q - p) / (q - p).length_squared().max(1e-300)).clamp(0.0, 1.0);
                p + (q - p) * s
            };
            (pos, edge_normal(e), Some(q - p))
        }
        Placement::FaceCenter => (m.tri_centroid(t), nt, Some(m.vertex(tv[1]) - m.vertex(tv[0]))),
        Placement::GroupCenter => {
            let g = m.tri_group(t);
            let (mut c, mut a) = (DVec3::ZERO, 0.0);
            for x in group_tris(g) {
                c += m.tri_centroid(x) * m.tri_area(x);
                a += m.tri_area(x);
            }
            (c / a.max(1e-300), area_normal(&mut group_tris(g)), None)
        }
        Placement::GroupBorderCenter => {
            let g = m.tri_group(t);
            let (mut c, mut len) = (DVec3::ZERO, 0.0);
            for e in m.edge_ids() {
                let [a, b] = m.edge_triangles(e);
                let ga = m.tri_group(a);
                let gb = if b == mm_core::INVALID { None } else { Some(m.tri_group(b)) };
                if (ga == g || gb == Some(g)) && gb != Some(ga) {
                    let l = m.edge_length(e);
                    c += m.edge_midpoint(e) * l;
                    len += l;
                }
            }
            if len <= 0.0 {
                return None;
            }
            (c / len, area_normal(&mut group_tris(g)), None)
        }
        Placement::BoundaryCenter => {
            // The open boundary nearest to the click; its frame faces out of the opening.
            let loops = mm_core::topology::boundary_loops(m);
            let lp = loops.iter().min_by(|a, b| {
                let d = |l: &mm_core::topology::EdgeLoop| l.vertices.iter().map(|&v| m.vertex(v).distance(hit.point)).fold(f64::MAX, f64::min);
                d(a).total_cmp(&d(b))
            })?;
            let pts = lp.positions(m);
            let c = pts.iter().copied().sum::<DVec3>() / pts.len() as f64;
            let mut cov = [[0.0; 3]; 3];
            for p in &pts {
                mm_core::linalg::add_outer(&mut cov, *p - c, 1.0);
            }
            let mut n = mm_core::linalg::sym_eigen3(cov).1[0];
            let inner: DVec3 = lp.vertices.iter().flat_map(|&v| m.vertex_triangles(v)).map(|x| m.tri_centroid(x)).sum::<DVec3>();
            let inner = inner / (lp.vertices.iter().map(|&v| m.vertex_triangles(v).len()).sum::<usize>().max(1) as f64);
            if (c - inner).dot(n) < 0.0 {
                n = -n;
            }
            (c, n, None)
        }
        Placement::MidpointAlongRay => {
            let far = bvh.raycast_range(m, ray, hit.t + 1e-9, f64::INFINITY, |x| x != t)?;
            ((hit.point + far.point) * 0.5, -ray.dir, None)
        }
        Placement::MidpointAlongNormal => {
            // The smooth normal, so a faceted tube still gives its axis.
            let n = (m.vertex_normal(tv[0]) * hit.bary.x + m.vertex_normal(tv[1]) * hit.bary.y + m.vertex_normal(tv[2]) * hit.bary.z).normalize_or(nt);
            let inward = Ray::new(hit.point, -n);
            let far = bvh.raycast_range(m, &inward, 1e-9, f64::INFINITY, |x| x != t)?;
            ((hit.point + far.point) * 0.5, n, None)
        }
        Placement::BoundingBox => (m.bounds().center(), DVec3::Z, Some(DVec3::X)),
    })
}

/// Orientation whose z axis is `z` and x axis follows `x` where possible.
fn frame_from(z: DVec3, x: Option<DVec3>) -> DQuat {
    let z = z.normalize_or(DVec3::Z);
    let x = x.map(|x| x - z * x.dot(z)).filter(|x| x.length_squared() > 1e-18).map(|x| x.normalize()).unwrap_or_else(|| mm_core::geom::orthonormal_basis(z).0);
    DQuat::from_mat3(&DMat3::from_cols(x, z.cross(x), z)).normalize()
}

pub struct PivotTool {
    placement: Placement,
    frame: FrameMode,
    hover: Option<(ObjectId, Pivot)>,
}

impl PivotTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        Self { placement: Placement::FaceCenter, frame: FrameMode::Geometry, hover: None }
    }

    fn candidate(&self, cx: &mut ToolCx, pos: egui::Pos2) -> Option<(ObjectId, Pivot)> {
        let ray = cx.ray_at(pos);
        let (id, hit) = cx.pick(pos, cx.scene.active)?;
        let o = cx.scene.get_mut(id)?;
        let bvh = o.bvh();
        let (p, n, x) = place(&o.mesh, &bvh, &hit, &ray, self.placement)?;
        let frame = match self.frame {
            FrameMode::Geometry => frame_from(n, x),
            FrameMode::EyeRay => frame_from(-ray.dir, Some(cx.camera.right())),
            FrameMode::World => DQuat::IDENTITY,
        };
        Some((id, Pivot { position: p, frame }))
    }
}

/// Draw a pivot: a ball with its frame's axes (screen-space, on top of the model).
pub fn paint_pivot(painter: &egui::Painter, cx: &ToolCx, p: &Pivot, highlight: bool) {
    let Some(c) = cx.project(p.position) else { return };
    let len = cx.world_per_point(p.position) * 22.0;
    let cols = [egui::Color32::from_rgb(230, 80, 80), egui::Color32::from_rgb(90, 200, 90), egui::Color32::from_rgb(90, 130, 240)];
    for (axis, col) in [DVec3::X, DVec3::Y, DVec3::Z].into_iter().zip(cols) {
        if let Some(e) = cx.project(p.position + p.frame * axis * len) {
            painter.line_segment([c, e], egui::Stroke::new(2.0, col));
        }
    }
    let fill = if highlight { egui::Color32::from_rgb(255, 200, 60) } else { egui::Color32::from_rgb(235, 235, 235) };
    painter.circle(c, 5.0, fill, egui::Stroke::new(1.0, egui::Color32::from_gray(30)));
}

impl Tool for PivotTool {
    fn name(&self) -> &'static str {
        "Create Pivot"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Create Pivot");
        egui::ComboBox::from_label("Placement Mode").selected_text(self.placement.label()).show_ui(ui, |ui| {
            for p in Placement::ALL {
                ui.selectable_value(&mut self.placement, p, p.label());
            }
        });
        egui::ComboBox::from_label("Frame Mode")
            .selected_text(match self.frame {
                FrameMode::Geometry => "Geometry Frame",
                FrameMode::EyeRay => "Eye Ray Frame",
                FrameMode::World => "World Frame",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.frame, FrameMode::Geometry, "Geometry Frame").on_hover_text("z along the surface normal (or the opening's axis)");
                ui.selectable_value(&mut self.frame, FrameMode::EyeRay, "Eye Ray Frame").on_hover_text("z toward the viewer");
                ui.selectable_value(&mut self.frame, FrameMode::World, "World Frame");
            });
        ui.small("Click the model to drop a pivot. Transform can turn and scale about a pivot, and pivots move with their object.");
        if let Some(o) = cx.scene.active_mut() {
            ui.label(format!("{} pivot(s) on {}", o.pivots.len(), o.name));
            if !o.pivots.is_empty() && ui.button("Remove last pivot").clicked() {
                cx.history.push("Remove pivot", cx.scene);
                if let Some(o) = cx.scene.active_mut() {
                    o.pivots.pop();
                }
            }
        }
        ui.add_space(6.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        self.hover = self.candidate(cx, pos);
        if input.clicked {
            if let Some((id, p)) = self.hover {
                cx.history.push("Create Pivot", cx.scene);
                if let Some(o) = cx.scene.get_mut(id) {
                    o.pivots.push(p);
                    *cx.status = format!("Pivot {} on {}", o.pivots.len(), o.name);
                }
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if let Some((_, p)) = &self.hover {
            paint_pivot(painter, cx, p, true);
        }
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        // `mode:Snap To Boundary Center` (any unambiguous part of a label).
        if let Some(("mode", m)) = cmd.split_once(':') {
            let m = m.to_lowercase();
            self.placement = Placement::ALL.into_iter().find(|p| p.label().to_lowercase().contains(&m)).unwrap_or(self.placement);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn hit_from_above(m: &Mesh, bvh: &mm_core::Bvh, x: f64, y: f64) -> (mm_core::bvh::RayHit, Ray) {
        let ray = Ray::new(DVec3::new(x, y, 100.0), -DVec3::Z);
        (bvh.raycast(m, &ray).unwrap(), ray)
    }

    #[test]
    fn midpoints_find_the_middle_of_a_tube() {
        // A tube along x of radius 2: clicking its top puts the pivot on the axis.
        let path: Vec<DVec3> = (0..=10).map(|i| DVec3::new(-10.0 + 2.0 * i as f64, 0.0, 0.0)).collect();
        let tube = primitives::tube_along(&path, &[2.0; 11], 48);
        assert_eq!(mm_core::topology::boundary_loops(&tube).len(), 2);
        let bvh = mm_core::Bvh::build(&tube);
        let (hit, ray) = hit_from_above(&tube, &bvh, 1.0, 0.0);
        for mode in [Placement::MidpointAlongNormal, Placement::MidpointAlongRay] {
            let (p, _, _) = place(&tube, &bvh, &hit, &ray, mode).unwrap();
            assert!(p.z.abs() < 0.05 && p.y.abs() < 0.05, "{mode:?}: {p:?}");
        }
        // Boundary centre: the nearest opening, frame pointing out of it.
        let (hit, ray) = hit_from_above(&tube, &bvh, 9.5, 0.0);
        let (p, n, _) = place(&tube, &bvh, &hit, &ray, Placement::BoundaryCenter).unwrap();
        assert!((p - DVec3::new(10.0, 0.0, 0.0)).length() < 1e-6 || (p - DVec3::new(-10.0, 0.0, 0.0)).length() < 1e-6, "{p:?}");
        assert!(n.dot(p.normalize()) > 0.99, "{n:?}");
    }

    #[test]
    fn frames_are_right_handed() {
        let q = frame_from(DVec3::new(0.0, 1.0, 1.0), Some(DVec3::X));
        assert!((q * DVec3::Z - DVec3::new(0.0, 1.0, 1.0).normalize()).length() < 1e-9);
        assert!((q * DVec3::X - DVec3::X).length() < 1e-9);
        let (x, y, z) = (q * DVec3::X, q * DVec3::Y, q * DVec3::Z);
        assert!((x.cross(y) - z).length() < 1e-9);
    }
}
