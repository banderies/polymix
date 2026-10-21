//! Create Pivot (Edit > Create Pivot): click the model to drop a pivot — a point with a
//! coordinate frame attached to the object. Pivots move with the object in Transform and
//! Align, can be its rotation / scale centre there and Align's source or destination, are
//! drawn on top of the model (View > Show pivots) and are kept in projects. "First Midpoint
//! Along Normal" puts a pivot in the middle of a tube, e.g. on a vessel's centreline.
//!
//! "Existing Pivot" copies another pivot (on any object), "From Last Tool" the frame the last
//! tool worked in (Transform's centre and axes, where Align put its source). "Create Symmetric"
//! also drops each pivot's mirror image across the object's centre plane, and "Link To Target"
//! attaches new pivots to the target object (Set as Target), so they move with it.

use super::*;
use crate::scene::{Pivot, SceneObject};
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
    ExistingPivot,
    LastTool,
}

impl Placement {
    const ALL: [Placement; 12] = [
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
        Placement::ExistingPivot,
        Placement::LastTool,
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
            Placement::ExistingPivot => "Existing Pivot",
            Placement::LastTool => "From Last Tool",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FrameMode {
    Geometry,
    EyeRay,
    World,
}

/// Screen distance (points) within which a click picks a drawn pivot.
pub const PICK_POINTS: f32 = 12.0;

/// What a mesh is: the allocation it lives in and its shape version. Shape versions are never
/// reused, and the tool never edits the mesh, so the pair stays the same while the mesh does.
type MeshId = (usize, u64);

fn mesh_id(m: &Mesh) -> MeshId {
    (m as *const Mesh as usize, m.shape_version())
}

/// A face group of a mesh with a group version.
type GroupKey = (MeshId, u64, i32);

/// Whole-mesh sums the placement modes read, kept from one hover to the next (the pivot
/// under the pointer is placed every frame).
#[derive(Default)]
struct PlaceCache {
    /// For a face group: the sums of its triangles' area-weighted centroids, areas, and
    /// area-weighted normals.
    group: Option<(GroupKey, (DVec3, f64, DVec3))>,
    /// For a face group: the sums of its border's length-weighted midpoints and lengths.
    border: Option<(GroupKey, (DVec3, f64))>,
    /// The mesh's open boundaries.
    loops: Option<(MeshId, Vec<mm_core::topology::EdgeLoop>)>,
}

impl PlaceCache {
    /// Group `g`'s sums, added up in triangle order as Snap To Group Center always has.
    fn group(&mut self, m: &Mesh, g: i32) -> (DVec3, f64, DVec3) {
        let key = (mesh_id(m), m.group_version(), g);
        match self.group {
            Some((k, sums)) if k == key => sums,
            _ => {
                let (mut c, mut a, mut n) = (DVec3::ZERO, 0.0, DVec3::ZERO);
                for x in m.triangle_ids().filter(|&x| m.tri_group(x) == g) {
                    c += m.tri_centroid(x) * m.tri_area(x);
                    a += m.tri_area(x);
                    n += m.tri_normal(x) * m.tri_area(x);
                }
                self.group = Some((key, (c, a, n)));
                (c, a, n)
            }
        }
    }

    /// Group `g`'s border sums, in edge order.
    fn border(&mut self, m: &Mesh, g: i32) -> (DVec3, f64) {
        let key = (mesh_id(m), m.group_version(), g);
        match self.border {
            Some((k, sums)) if k == key => sums,
            _ => {
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
                self.border = Some((key, (c, len)));
                (c, len)
            }
        }
    }

    fn loops(&mut self, m: &Mesh) -> &[mm_core::topology::EdgeLoop] {
        let key = mesh_id(m);
        if self.loops.as_ref().is_none_or(|(k, _)| *k != key) {
            self.loops = Some((key, mm_core::topology::boundary_loops(m)));
        }
        self.loops.as_ref().map_or(&[], |(_, l)| l.as_slice())
    }
}

/// Position, frame z axis and (optionally) frame x axis for a click at `hit`.
#[cfg(test)]
fn place(m: &Mesh, bvh: &mm_core::Bvh, hit: &mm_core::bvh::RayHit, ray: &Ray, mode: Placement) -> Option<(DVec3, DVec3, Option<DVec3>)> {
    place_cached(m, bvh, hit, ray, mode, &mut PlaceCache::default())
}

/// [`place`], with the whole-mesh sums from `cache` (and kept there).
fn place_cached(m: &Mesh, bvh: &mm_core::Bvh, hit: &mm_core::bvh::RayHit, ray: &Ray, mode: Placement, cache: &mut PlaceCache) -> Option<(DVec3, DVec3, Option<DVec3>)> {
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
            let (c, a, n) = cache.group(m, m.tri_group(t));
            (c / a.max(1e-300), n.normalize_or(nt), None)
        }
        Placement::GroupBorderCenter => {
            let g = m.tri_group(t);
            let (c, len) = cache.border(m, g);
            if len <= 0.0 {
                return None;
            }
            (c / len, cache.group(m, g).2.normalize_or(nt), None)
        }
        Placement::BoundaryCenter => {
            // The open boundary nearest to the click; its frame faces out of the opening.
            let loops = cache.loops(m);
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
        Placement::ExistingPivot | Placement::LastTool => return None,
    })
}

/// Orientation whose z axis is `z` and x axis follows `x` where possible.
pub fn frame_from(z: DVec3, x: Option<DVec3>) -> DQuat {
    let z = z.normalize_or(DVec3::Z);
    let x = x.map(|x| x - z * x.dot(z)).filter(|x| x.length_squared() > 1e-18).map(|x| x.normalize()).unwrap_or_else(|| mm_core::geom::orthonormal_basis(z).0);
    DQuat::from_mat3(&DMat3::from_cols(x, z.cross(x), z)).normalize()
}

/// `p` mirrored across the plane through `point` with normal `n`. A mirror image of a
/// right-handed frame is left-handed, so the frame keeps the mirrored z (normal) and x axes
/// and takes y from them.
pub fn mirror_pivot(p: &Pivot, point: DVec3, n: DVec3) -> Pivot {
    let n = n.normalize_or(DVec3::X);
    let reflect = |v: DVec3| v - n * (2.0 * v.dot(n));
    let (x, z) = (reflect(p.frame * DVec3::X), reflect(p.frame * DVec3::Z));
    Pivot { position: point + reflect(p.position - point), frame: DQuat::from_mat3(&DMat3::from_cols(x, z.cross(x), z)).normalize() }
}

/// An object's symmetry plane across `axis` (0 = X, 1 = Y, 2 = Z): through the centre of its
/// box, square to that axis of the world or of the object's own frame (the turns Transform
/// applied to it). Returns a point on the plane and its normal.
pub fn symmetry_plane(o: &SceneObject, axis: usize, local: bool) -> (DVec3, DVec3) {
    let axes = [DVec3::X, DVec3::Y, DVec3::Z].map(|a| if local { o.frame * a } else { a });
    let m = &o.mesh;
    let (mut lo, mut hi) = (DVec3::splat(f64::INFINITY), DVec3::splat(f64::NEG_INFINITY));
    for v in m.vertex_ids() {
        let p = m.vertex(v);
        let q = DVec3::new(p.dot(axes[0]), p.dot(axes[1]), p.dot(axes[2]));
        lo = lo.min(q);
        hi = hi.max(q);
    }
    let c = if lo.x <= hi.x { (lo + hi) * 0.5 } else { DVec3::ZERO };
    (axes[0] * c.x + axes[1] * c.y + axes[2] * c.z, axes[axis.min(2)])
}

/// The pivot drawn nearest to `pos` (within [`PICK_POINTS`]) among the visible objects that
/// `accept`: its object, index and frame.
pub fn pivot_at(cx: &ToolCx, pos: egui::Pos2, accept: impl Fn(ObjectId) -> bool) -> Option<(ObjectId, usize, Pivot)> {
    let mut best: Option<(f32, (ObjectId, usize, Pivot))> = None;
    for o in cx.scene.objects.iter().filter(|o| o.visible && accept(o.id)) {
        for (i, p) in o.pivots.iter().enumerate() {
            let Some(s) = cx.project(p.position) else { continue };
            let d = s.distance(pos);
            if d <= PICK_POINTS && best.as_ref().is_none_or(|b| d < b.0) {
                best = Some((d, (o.id, i, *p)));
            }
        }
    }
    best.map(|b| b.1)
}

/// The pivot nearest to the point `at` among all objects' pivots.
fn pivot_nearest(scene: &Scene, at: DVec3) -> Option<(ObjectId, usize, Pivot)> {
    scene
        .objects
        .iter()
        .flat_map(|o| o.pivots.iter().enumerate().map(move |(i, p)| (o.id, i, *p)))
        .min_by(|a, b| a.2.position.distance(at).total_cmp(&b.2.position.distance(at)))
}

pub struct PivotTool {
    placement: Placement,
    frame: FrameMode,
    /// The pivot a click would drop now.
    hover: Option<Pivot>,
    /// New pivots attach to the target object (and move with it) instead of the active one.
    link_to_target: bool,
    /// Each new pivot comes with its mirror image across the symmetry plane.
    symmetric: bool,
    sym_axis: usize,
    /// The symmetry plane is square to an axis of the object's own frame, not the world's.
    sym_local: bool,
    cache: PlaceCache,
    /// The last symmetry plane, for the object, mesh, axis, space and object frame it is of
    /// (Create Symmetric shows the mirrored pivot every frame).
    sym: std::cell::Cell<Option<(SymKey, (DVec3, DVec3))>>,
}

type SymKey = (ObjectId, MeshId, usize, bool, [u64; 4]);

impl PivotTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        Self {
            placement: Placement::FaceCenter,
            frame: FrameMode::Geometry,
            hover: None,
            link_to_target: false,
            symmetric: false,
            sym_axis: 0,
            sym_local: false,
            cache: PlaceCache::default(),
            sym: std::cell::Cell::new(None),
        }
    }

    /// [`symmetry_plane`] of `o` for the tool's settings, worked out again only when
    /// something it depends on changed.
    fn symmetry_plane_of(&self, o: &SceneObject) -> (DVec3, DVec3) {
        let key = (o.id, mesh_id(&o.mesh), self.sym_axis, self.sym_local, o.frame.to_array().map(f64::to_bits));
        if let Some((_, plane)) = self.sym.get().filter(|(k, _)| *k == key) {
            return plane;
        }
        let plane = symmetry_plane(o, self.sym_axis, self.sym_local);
        self.sym.set(Some((key, plane)));
        plane
    }

    /// The frame for a pivot whose geometric frame is `geometry`, by the frame mode (`toward`
    /// is the direction to the viewer).
    fn oriented(&self, geometry: DQuat, toward: DVec3, right: DVec3) -> DQuat {
        match self.frame {
            FrameMode::Geometry => geometry,
            FrameMode::EyeRay => frame_from(toward, Some(right)),
            FrameMode::World => DQuat::IDENTITY,
        }
    }

    /// The pivot "From Last Tool" would drop, if a tool has recorded its frame.
    fn last_tool_pivot(&self, cx: &ToolCx) -> Option<Pivot> {
        let f = cx.scene.last_tool_frame.as_ref()?;
        Some(Pivot { position: f.pivot.position, frame: self.oriented(f.pivot.frame, -cx.camera.forward(), cx.camera.right()) })
    }

    fn candidate(&mut self, cx: &mut ToolCx, pos: egui::Pos2) -> Option<Pivot> {
        let ray = cx.ray_at(pos);
        let right = cx.camera.right();
        match self.placement {
            Placement::ExistingPivot => {
                let (_, _, p) = pivot_at(cx, pos, |_| true)?;
                Some(Pivot { position: p.position, frame: self.oriented(p.frame, -ray.dir, right) })
            }
            Placement::LastTool => self.last_tool_pivot(cx),
            mode => {
                let (id, hit) = cx.pick(pos, cx.scene.active)?;
                let o = cx.scene.get_mut(id)?;
                let bvh = o.bvh();
                let (p, n, x) = place_cached(&o.mesh, &bvh, &hit, &ray, mode, &mut self.cache)?;
                Some(Pivot { position: p, frame: self.oriented(frame_from(n, x), -ray.dir, right) })
            }
        }
    }

    /// The object new pivots attach to: the target with Link To Target (when one is set),
    /// otherwise the active object.
    fn owner(&self, scene: &Scene) -> Option<ObjectId> {
        let target = scene.target.filter(|&t| self.link_to_target && scene.get(t).is_some());
        target.or(scene.active.filter(|&a| scene.get(a).is_some()))
    }

    /// The symmetry plane: the active object's (the one the pivots are placed on), else the
    /// owner's.
    fn plane(&self, scene: &Scene) -> Option<(DVec3, DVec3)> {
        let o = scene.active().or_else(|| self.owner(scene).and_then(|id| scene.get(id)))?;
        Some(self.symmetry_plane_of(o))
    }

    /// Add `p` (and its mirror image with Create Symmetric) to the owner object.
    fn create(&self, cx: &mut ToolCx, p: Pivot) -> Result<String, String> {
        let owner = self.owner(cx.scene).ok_or("no active object")?;
        let twin = if self.symmetric { self.plane(cx.scene).map(|(c, n)| mirror_pivot(&p, c, n)) } else { None };
        cx.history.push(if twin.is_some() { "Create Symmetric Pivots" } else { "Create Pivot" }, cx.scene);
        let o = cx.scene.get_mut(owner).ok_or("the object is gone")?;
        o.pivots.push(p);
        let n = o.pivots.len();
        let at = |q: &Pivot| format!("[{:.3}, {:.3}, {:.3}]", q.position.x, q.position.y, q.position.z);
        Ok(match twin {
            Some(t) => {
                o.pivots.push(t);
                format!("Pivot {n} at {} and pivot {} (mirrored) at {} on {}", at(&p), n + 1, at(&t), o.name)
            }
            None => format!("Pivot {n} at {} on {}", at(&p), o.name),
        })
    }

    /// Add the mirror image of the active object's pivot `index` to that object.
    fn mirror(&self, cx: &mut ToolCx, index: usize) -> Result<String, String> {
        let o = cx.scene.active().ok_or("no active object")?;
        let p = *o.pivots.get(index).ok_or("the active object has no such pivot")?;
        let (c, n) = self.symmetry_plane_of(o);
        cx.history.push("Mirror pivot", cx.scene);
        let o = cx.scene.active_mut().ok_or("no active object")?;
        o.pivots.push(mirror_pivot(&p, c, n));
        Ok(format!("Pivot {} mirrors pivot {} on {}", o.pivots.len(), index + 1, o.name))
    }

    /// Move the active object's pivot `index` to the target object, so it follows the target.
    fn link(&self, cx: &mut ToolCx, index: usize) -> Result<String, String> {
        let id = cx.scene.active.ok_or("no active object")?;
        let target = cx.scene.target.filter(|&t| t != id && cx.scene.get(t).is_some()).ok_or("set another object as the target first")?;
        if cx.scene.get(id).is_none_or(|o| index >= o.pivots.len()) {
            return Err("the active object has no such pivot".into());
        }
        cx.history.push("Link pivot to target", cx.scene);
        let p = cx.scene.get_mut(id).ok_or("no active object")?.pivots.remove(index);
        let t = cx.scene.get_mut(target).ok_or("the target is gone")?;
        t.pivots.push(p);
        Ok(format!("The pivot now follows {} (its pivot {})", t.name, t.pivots.len()))
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

fn placement_key(p: Placement) -> String {
    p.label().trim_start_matches("Snap To ").to_lowercase().replace(' ', "_")
}

const AXIS_KEYS: [&str; 3] = ["x", "y", "z"];

impl Tool for PivotTool {
    fn name(&self) -> &'static str {
        "Create Pivot"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        let frame = match self.frame {
            FrameMode::Geometry => "geometry",
            FrameMode::EyeRay => "eye_ray",
            FrameMode::World => "world",
        };
        Some(serde_json::json!({
            "placement": placement_key(self.placement),
            "frame": frame,
            "link_to_target": self.link_to_target,
            "symmetric": self.symmetric,
            "symmetry_axis": AXIS_KEYS[self.sym_axis.min(2)],
            "symmetry_space": if self.sym_local { "local" } else { "world" },
        }))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "placement": {"type": "string", "enum": Placement::ALL.map(placement_key), "description": "Where the pivot goes relative to the picked point: the nearest vertex, edge point, edge midpoint, face centre, face-group centre, group-border centre, opening (boundary) centre, the middle of the object along the ray or the normal, or the bounding-box centre; existing_pivot copies the pivot nearest the point (on any object), from_last_tool the frame the last tool worked in (create_from_last_tool)"},
            "frame": {"type": "string", "enum": ["geometry", "eye_ray", "world"], "description": "The pivot's axes: z along the surface normal (or the opening's axis; the copied pivot's or tool's own axes for existing_pivot / from_last_tool), toward the viewer, or the world's"},
            "link_to_target": {"type": "boolean", "description": "New pivots attach to the target object (Set as Target) and move with it, instead of the active object"},
            "symmetric": {"type": "boolean", "description": "Each new pivot also gets its mirror image across the active object's symmetry plane"},
            "symmetry_axis": {"type": "string", "enum": AXIS_KEYS, "description": "The symmetry plane is square to this axis, through the object's centre"},
            "symmetry_space": {"type": "string", "enum": ["world", "local"], "description": "That axis is the world's or the object's own (turned with it by Transform)"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let a = Args(&v);
        self.placement = Placement::ALL.into_iter().find(|p| v["placement"] == placement_key(*p).as_str()).ok_or("unknown placement")?;
        self.frame = match a.opt_str("frame") {
            Some("geometry") => FrameMode::Geometry,
            Some("eye_ray") => FrameMode::EyeRay,
            Some("world") => FrameMode::World,
            _ => return Err("frame must be geometry, eye_ray or world".into()),
        };
        self.link_to_target = a.opt_bool("link_to_target").ok_or("link_to_target must be true or false")?;
        self.symmetric = a.opt_bool("symmetric").ok_or("symmetric must be true or false")?;
        self.sym_axis = AXIS_KEYS.iter().position(|k| a.opt_str("symmetry_axis") == Some(*k)).ok_or("symmetry_axis must be x, y or z")?;
        self.sym_local = match a.opt_str("symmetry_space") {
            Some("world") => false,
            Some("local") => true,
            _ => return Err("symmetry_space must be world or local".into()),
        };
        Ok(())
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::vec3;
        let o = scene.active();
        let frame = |p: &Pivot| serde_json::json!({"at_mm": vec3(p.position.to_array(), 4), "z_axis": vec3((p.frame * DVec3::Z).to_array(), 4), "x_axis": vec3((p.frame * DVec3::X).to_array(), 4)});
        serde_json::json!({
            "object": o.map(|o| o.id),
            "pivots": o.map(|o| o.pivots.iter().enumerate().map(|(i, p)| {
                let mut f = frame(p);
                f["pivot"] = serde_json::json!(i + 1);
                f
            }).collect::<Vec<_>>()),
            "target": scene.target,
            "new_pivots_go_on": self.owner(scene),
            "last_tool_frame": scene.last_tool_frame.as_ref().map(|f| {
                let mut v = frame(&f.pivot);
                v["tool"] = serde_json::json!(f.tool);
                v
            }),
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        let which = || arg::integer("Pivot number on the active object (1 = first; default the last)");
        vec![
            Action::new("create_at", "Drop a pivot by the placement setting at the surface point of the active object nearest `at` (as clicking there does); with placement existing_pivot, copy the pivot nearest `at` (on any object)", json!({"at": arg::point("A point on or near the surface (or the pivot to copy), mm"), "along": arg::point("The ray's direction for placement midpoint_along_ray (default: into the surface)")})).required(&["at"]),
            Action::plain("create_from_last_tool", "Drop a pivot at the frame the last tool worked in (Transform's centre and axes, where Align put its source)"),
            Action::new("mirror", "Add the mirror image of one of the active object's pivots across its symmetry plane (symmetry_axis / symmetry_space)", json!({"pivot": which()})),
            Action::new("link_to_target", "Move one of the active object's pivots to the target object, so it follows the target", json!({"pivot": which()})),
            Action::plain("remove_last", "Remove the active object's last pivot"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        let id = cx.scene.active.ok_or("no active object")?;
        let which = |cx: &ToolCx| -> Result<usize, String> {
            let n = cx.scene.get(id).map_or(0, |o| o.pivots.len());
            match a.opt_usize("pivot") {
                Some(i) if (1..=n).contains(&i) => Ok(i - 1),
                Some(i) => Err(format!("there is no pivot {i} (the active object has {n})")),
                None if n > 0 => Ok(n - 1),
                None => Err("the active object has no pivots".into()),
            }
        };
        match action {
            "create_at" => {
                let at = DVec3::from_array(a.point("at")?);
                let (toward, right) = (-cx.camera.forward(), cx.camera.right());
                let p = match self.placement {
                    Placement::LastTool => return Err("placement from_last_tool: use create_from_last_tool".into()),
                    Placement::ExistingPivot => {
                        let (_, _, p) = pivot_nearest(cx.scene, at).ok_or("there are no pivots to copy")?;
                        Pivot { position: p.position, frame: self.oriented(p.frame, toward, right) }
                    }
                    mode => {
                        let o = cx.scene.get_mut(id).unwrap();
                        let bvh = o.bvh();
                        let h = bvh.nearest(&o.mesh, at, f64::INFINITY).ok_or("no surface")?;
                        let n = o.mesh.tri_normal(h.tid);
                        let dir = a.opt_point("along").map(DVec3::from_array).and_then(|d| d.try_normalize()).unwrap_or(-n);
                        let ray = Ray::new(h.point - dir * 1e-3, dir);
                        let hit = mm_core::bvh::RayHit { tid: h.tid, t: 1e-3, point: h.point, bary: h.bary };
                        let (p, z, x) = place_cached(&o.mesh, &bvh, &hit, &ray, mode, &mut self.cache).ok_or("no pivot can be placed there in this placement mode")?;
                        Pivot { position: p, frame: self.oriented(frame_from(z, x), toward, right) }
                    }
                };
                self.create(cx, p)
            }
            "create_from_last_tool" => {
                let p = self.last_tool_pivot(cx).ok_or("no tool has recorded a frame yet (Transform and Align do when they finish)")?;
                self.create(cx, p)
            }
            "mirror" => {
                let i = which(cx)?;
                self.mirror(cx, i)
            }
            "link_to_target" => {
                let i = which(cx)?;
                self.link(cx, i)
            }
            "remove_last" => {
                if cx.scene.get(id).unwrap().pivots.is_empty() {
                    return Err("the active object has no pivots".into());
                }
                cx.history.push("Remove pivot", cx.scene);
                cx.scene.get_mut(id).unwrap().pivots.pop();
                Ok("Last pivot removed".into())
            }
            _ => Err(format!("Create Pivot has no action {action}")),
        }
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Create Pivot");
        egui::ComboBox::from_label("Placement Mode").selected_text(self.placement.label()).show_ui(ui, |ui| {
            for p in Placement::ALL {
                let tip = match p {
                    Placement::ExistingPivot => "Copy a pivot (on any object): click near it",
                    Placement::LastTool => "The frame the last tool worked in (Transform's centre and axes, where Align put its source)",
                    _ => "Click the model",
                };
                ui.selectable_value(&mut self.placement, p, p.label()).on_hover_text(tip);
            }
        });
        egui::ComboBox::from_label("Frame Mode")
            .selected_text(match self.frame {
                FrameMode::Geometry => "Geometry Frame",
                FrameMode::EyeRay => "Eye Ray Frame",
                FrameMode::World => "World Frame",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut self.frame, FrameMode::Geometry, "Geometry Frame").on_hover_text("z along the surface normal (or the opening's axis); a copied pivot's or the tool's own axes");
                ui.selectable_value(&mut self.frame, FrameMode::EyeRay, "Eye Ray Frame").on_hover_text("z toward the viewer");
                ui.selectable_value(&mut self.frame, FrameMode::World, "World Frame");
            });
        match self.placement {
            Placement::LastTool => match cx.scene.last_tool_frame.as_ref().map(|f| f.tool.clone()) {
                Some(tool) => {
                    ui.label(format!("Last tool: {tool}"));
                    if ui.button("Create at the last tool's frame").clicked() {
                        if let Some(p) = self.last_tool_pivot(cx) {
                            *cx.status = self.create(cx, p).unwrap_or_else(|e| e);
                        }
                    }
                }
                None => {
                    ui.small("No tool frame yet: Transform and Align record theirs when they finish.");
                }
            },
            Placement::ExistingPivot => {
                ui.small("Click near a pivot (on any object) to copy it.");
            }
            _ => {}
        }
        ui.checkbox(&mut self.link_to_target, "Link To Target").on_hover_text("New pivots attach to the target object (Set as Target) and move with it");
        if self.link_to_target {
            match cx.scene.target.and_then(|t| cx.scene.get(t)) {
                Some(t) => ui.small(format!("New pivots follow {}", t.name)),
                None => ui.small("No target set: new pivots go on the active object"),
            };
        }
        ui.checkbox(&mut self.symmetric, "Create Symmetric").on_hover_text("Each new pivot also gets its mirror image across the object's centre plane");
        if self.symmetric {
            ui.horizontal(|ui| {
                ui.label("Plane across");
                for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                    if ui.selectable_label(self.sym_axis == i, *n).clicked() {
                        self.sym_axis = i;
                    }
                }
                ui.checkbox(&mut self.sym_local, "Local frame").on_hover_text("The object's own axes (turned with it by Transform), not the world's");
            });
        }
        ui.small(if self.placement == Placement::LastTool {
            "Click the view (or the button) to drop the pivot. Transform can turn and scale about a pivot, Align can match pivots, and pivots move with their object."
        } else {
            "Click the model to drop a pivot. Transform can turn and scale about a pivot, Align can match pivots, and pivots move with their object."
        });
        if let Some(o) = cx.scene.active() {
            let (count, name) = (o.pivots.len(), o.name.clone());
            let can_link = cx.scene.target.is_some_and(|t| t != o.id && cx.scene.get(t).is_some());
            ui.label(format!("{count} pivot(s) on {name}"));
            if count > 0 {
                ui.horizontal_wrapped(|ui| {
                    let mut result = None;
                    if ui.button("Remove last pivot").clicked() {
                        cx.history.push("Remove pivot", cx.scene);
                        if let Some(o) = cx.scene.active_mut() {
                            o.pivots.pop();
                        }
                    }
                    if ui.button("Mirror last pivot").on_hover_text("Add its mirror image across the symmetry plane above").clicked() {
                        result = Some(self.mirror(cx, count - 1));
                    }
                    if ui.add_enabled(can_link, egui::Button::new("Link last pivot to target")).on_hover_text("Move it to the target object, so it follows the target").on_disabled_hover_text("Set another object as the target first").clicked() {
                        result = Some(self.link(cx, count - 1));
                    }
                    if let Some(r) = result {
                        *cx.status = r.unwrap_or_else(|e| e);
                    }
                });
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
            if let Some(p) = self.hover {
                match self.create(cx, p) {
                    Ok(msg) | Err(msg) => *cx.status = msg,
                }
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let shown = if self.placement == Placement::LastTool { self.last_tool_pivot(cx) } else { self.hover };
        if let Some(p) = shown {
            paint_pivot(painter, cx, &p, true);
            if self.symmetric {
                if let Some((c, n)) = self.plane(cx.scene) {
                    paint_pivot(painter, cx, &mirror_pivot(&p, c, n), true);
                }
            }
        }
        if self.placement == Placement::ExistingPivot {
            // Existing Pivot: ring the pivot a click would copy.
            if let Some(c) = self.hover.and_then(|p| cx.project(p.position)) {
                painter.circle_stroke(c, PICK_POINTS, egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 200, 60)));
            }
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

    /// The whole-mesh placements worked out from scratch, as every hover did.
    fn place_from_scratch(m: &Mesh, hit: &mm_core::bvh::RayHit, mode: Placement) -> Option<(DVec3, DVec3, Option<DVec3>)> {
        let t = hit.tid;
        let nt = m.tri_normal(t);
        let group_tris = |g: i32| m.triangle_ids().filter(move |&x| m.tri_group(x) == g);
        let area_normal = |tris: &mut dyn Iterator<Item = u32>| tris.map(|x| m.tri_normal(x) * m.tri_area(x)).sum::<DVec3>().normalize_or(nt);
        Some(match mode {
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
            _ => unreachable!(),
        })
    }

    #[test]
    fn cached_placements_are_the_ones_worked_out_from_scratch() {
        // A wavy tube along x with four face groups around it and two openings.
        let path: Vec<DVec3> = (0..=20).map(|i| DVec3::new(-10.0 + i as f64, (i as f64 * 0.7).sin(), 0.0)).collect();
        let mut tube = primitives::tube_along(&path, &[2.0; 21], 32);
        let ids: Vec<u32> = tube.triangle_ids().collect();
        for t in ids {
            let c = tube.tri_centroid(t);
            tube.set_tri_group(t, (c.y > 0.0) as i32 + 2 * (c.x > 0.0) as i32);
        }
        let bvh = mm_core::Bvh::build(&tube);
        let mut cache = PlaceCache::default();
        let modes = [Placement::GroupCenter, Placement::GroupBorderCenter, Placement::BoundaryCenter];
        for k in 0..60 {
            let (x, y) = (-9.5 + 0.33 * k as f64, if k % 2 == 0 { 0.4 } else { -0.6 });
            let (hit, ray) = hit_from_above(&tube, &bvh, x, y);
            // The same hit twice in a row (the cache answers the second), then the next mode.
            for mode in modes.into_iter().chain(modes).chain([modes[k % 3]]) {
                assert_eq!(place_cached(&tube, &bvh, &hit, &ray, mode, &mut cache), place_from_scratch(&tube, &hit, mode), "{mode:?} at {x}, {y}");
            }
        }
        // Another mesh (or the same one with other groups) is not answered from the cache.
        let (hit, ray) = hit_from_above(&tube, &bvh, 3.0, 0.4);
        let before = place_cached(&tube, &bvh, &hit, &ray, Placement::GroupCenter, &mut cache);
        let mut regrouped = tube.clone();
        regrouped.set_tri_group(hit.tid, 9);
        let after = place_cached(&regrouped, &bvh, &hit, &ray, Placement::GroupCenter, &mut cache);
        assert_ne!(before, after);
        assert_eq!(after, place_from_scratch(&regrouped, &hit, Placement::GroupCenter));
    }

    #[test]
    fn the_symmetry_plane_is_kept_until_something_it_depends_on_changes() {
        with_scene(|cx, a, _| {
            let mut tool = PivotTool::new(cx);
            let check = |tool: &PivotTool, cx: &ToolCx| {
                let o = cx.scene.get(a).unwrap();
                assert_eq!(tool.plane(cx.scene), Some(symmetry_plane(o, tool.sym_axis, tool.sym_local)));
            };
            check(&tool, cx);
            check(&tool, cx);
            for (axis, local) in [(1, false), (1, true), (2, true)] {
                tool.sym_axis = axis;
                tool.sym_local = local;
                check(&tool, cx);
            }
            cx.scene.get_mut(a).unwrap().frame = DQuat::from_rotation_z(0.4);
            check(&tool, cx);
            cx.scene.get_mut(a).unwrap().mesh_mut().transform(|p| p * 1.5 + DVec3::X);
            check(&tool, cx);
        });
    }

    #[test]
    fn frames_are_right_handed() {
        let q = frame_from(DVec3::new(0.0, 1.0, 1.0), Some(DVec3::X));
        assert!((q * DVec3::Z - DVec3::new(0.0, 1.0, 1.0).normalize()).length() < 1e-9);
        assert!((q * DVec3::X - DVec3::X).length() < 1e-9);
        let (x, y, z) = (q * DVec3::X, q * DVec3::Y, q * DVec3::Z);
        assert!((x.cross(y) - z).length() < 1e-9);
    }

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-9
    }

    /// A scene of two boxes (`a` active, `b` beside it) with a tool context.
    fn with_scene(test: impl FnOnce(&mut ToolCx, ObjectId, ObjectId)) {
        let mut scene = Scene::default();
        let a = scene.add("a", primitives::cube(2.0, 2));
        let mut m = primitives::cube(1.0, 2);
        m.transform(|p| p + DVec3::new(5.0, 0.0, 0.0));
        let b = scene.add("b", m);
        scene.active = Some(a);
        let (mut history, mut camera, mut status) = (History::default(), crate::camera::Camera::default(), String::new());
        let ctx = egui::Context::default();
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        test(&mut cx, a, b);
    }

    #[test]
    fn existing_pivot_copies_a_pivot_onto_the_active_object() {
        with_scene(|cx, a, b| {
            let src = Pivot { position: DVec3::new(5.0, 0.5, 0.2), frame: DQuat::from_rotation_y(0.6) * DQuat::from_rotation_z(0.3) };
            cx.scene.get_mut(b).unwrap().pivots.push(src);
            let mut t = PivotTool::new(cx);
            t.set_settings(&serde_json::json!({"placement": "existing_pivot"}), cx).unwrap();
            let msg = t.act("create_at", &serde_json::json!({"at": [5.1, 0.4, 0.3]}), cx).unwrap();
            assert!(msg.contains("on a"), "{msg}");
            let got = cx.scene.get(a).unwrap().pivots.clone();
            assert_eq!(got.len(), 1);
            assert!(close(got[0].position, src.position) && got[0].frame.angle_between(src.frame) < 1e-6);
            // World frame mode keeps the position and drops the orientation.
            t.set_settings(&serde_json::json!({"frame": "world"}), cx).unwrap();
            t.act("create_at", &serde_json::json!({"at": [5.0, 0.5, 0.2]}), cx).unwrap();
            assert_eq!(cx.scene.get(a).unwrap().pivots[1].frame, DQuat::IDENTITY);
            assert!(t.act("create_from_last_tool", &serde_json::json!({}), cx).is_err(), "no tool frame yet");
        });
    }

    #[test]
    fn mirrored_pivots_reflect_position_and_frame() {
        let p = Pivot { position: DVec3::new(1.5, 2.0, -0.5), frame: DQuat::from_rotation_x(0.4) * DQuat::from_rotation_z(0.9) };
        let (c, n) = (DVec3::new(0.5, 0.0, 0.0), DVec3::X);
        let q = mirror_pivot(&p, c, n);
        assert!(close(q.position, DVec3::new(-0.5, 2.0, -0.5)));
        let flip = |v: DVec3| DVec3::new(-v.x, v.y, v.z);
        assert!(close(q.frame * DVec3::Z, flip(p.frame * DVec3::Z)), "the normal is mirrored");
        assert!(close(q.frame * DVec3::X, flip(p.frame * DVec3::X)));
        let (x, y, z) = (q.frame * DVec3::X, q.frame * DVec3::Y, q.frame * DVec3::Z);
        assert!(close(x.cross(y), z), "still right-handed");
        assert!(close(mirror_pivot(&q, c, n).position, p.position));
        // The tool's Create Symmetric drops the twin across the object's centre plane.
        with_scene(|cx, a, _| {
            let mut t = PivotTool::new(cx);
            t.set_settings(&serde_json::json!({"placement": "vertex", "symmetric": true, "symmetry_axis": "x"}), cx).unwrap();
            t.act("create_at", &serde_json::json!({"at": [0.9, 0.9, 1.1]}), cx).unwrap();
            let pv = cx.scene.get(a).unwrap().pivots.clone();
            assert_eq!(pv.len(), 2);
            let centre = cx.scene.get(a).unwrap().bounds().center();
            assert!(close(pv[1].position, DVec3::new(2.0 * centre.x - pv[0].position.x, pv[0].position.y, pv[0].position.z)), "{pv:?}");
            assert!(close(pv[1].frame * DVec3::Z, flip(pv[0].frame * DVec3::Z)));
            // Mirror an existing pivot across Y in the object's own (turned) frame.
            cx.scene.get_mut(a).unwrap().frame = DQuat::from_rotation_z(std::f64::consts::FRAC_PI_2);
            t.set_settings(&serde_json::json!({"symmetry_axis": "y", "symmetry_space": "local"}), cx).unwrap();
            t.act("mirror", &serde_json::json!({"pivot": 1}), cx).unwrap();
            let m = cx.scene.get(a).unwrap().pivots[2];
            // Local Y is world -X here: the same plane as before.
            assert!(close(m.position, pv[1].position), "{m:?}");
        });
    }

    #[test]
    fn linked_pivots_follow_their_target() {
        with_scene(|cx, a, b| {
            cx.scene.target = Some(b);
            let mut t = PivotTool::new(cx);
            t.set_settings(&serde_json::json!({"placement": "face_center", "link_to_target": true}), cx).unwrap();
            assert_eq!(t.state(cx.scene)["new_pivots_go_on"], b);
            // Placed on the active object's surface, attached to the target.
            t.act("create_at", &serde_json::json!({"at": [0.3, 0.2, 1.0]}), cx).unwrap();
            assert!(cx.scene.get(a).unwrap().pivots.is_empty());
            let placed = cx.scene.get(b).unwrap().pivots[0];
            assert!((placed.position.z - 1.0).abs() < 1e-9, "on a's top face: {placed:?}");
            // Moving the target carries the pivot; moving the active object does not.
            for (id, shift) in [(b, DVec3::new(1.0, 2.0, 3.0)), (a, DVec3::new(-4.0, 0.0, 0.0))] {
                cx.scene.active = Some(id);
                let mut tr = super::super::transform::TransformTool::new(cx);
                tr.set_settings(&serde_json::json!({"translate_mm": shift.to_array()}), cx).unwrap();
                assert!(tr.accept(cx));
            }
            let moved = cx.scene.get(b).unwrap().pivots[0];
            assert!(close(moved.position, placed.position + DVec3::new(1.0, 2.0, 3.0)), "{moved:?}");
            // An existing pivot can be linked afterwards, keeping where it is.
            cx.scene.active = Some(a);
            cx.scene.get_mut(a).unwrap().pivots.push(Pivot { position: DVec3::ONE, frame: DQuat::IDENTITY });
            t.act("link_to_target", &serde_json::json!({}), cx).unwrap();
            assert!(cx.scene.get(a).unwrap().pivots.is_empty());
            assert!(close(cx.scene.get(b).unwrap().pivots[1].position, DVec3::ONE));
        });
    }

    #[test]
    fn from_last_tool_uses_the_transform_frame() {
        with_scene(|cx, a, _| {
            let mut tr = super::super::transform::TransformTool::new(cx);
            tr.set_settings(&serde_json::json!({"translate_mm": [0.0, 0.0, 2.0], "rotate_deg": [0.0, 0.0, 30.0]}), cx).unwrap();
            assert!(tr.accept(cx));
            let f = cx.scene.last_tool_frame.clone().expect("Transform records its frame");
            assert_eq!(f.tool, "Transform");
            assert!(close(f.pivot.position, DVec3::new(0.0, 0.0, 2.0)), "{f:?}");
            assert!(f.pivot.frame.angle_between(DQuat::from_rotation_z(30f64.to_radians())) < 1e-6);
            let mut t = PivotTool::new(cx);
            t.set_settings(&serde_json::json!({"placement": "from_last_tool"}), cx).unwrap();
            t.act("create_from_last_tool", &serde_json::json!({}), cx).unwrap();
            assert_eq!(cx.scene.get(a).unwrap().pivots, vec![f.pivot]);
            // The recorded frame survives undoing the pivot and the transform.
            assert!(cx.history.undo(cx.scene).is_some() && cx.history.undo(cx.scene).is_some());
            assert!(cx.scene.get(a).unwrap().pivots.is_empty());
            assert_eq!(cx.scene.last_tool_frame.as_ref(), Some(&f));
        });
    }
}
