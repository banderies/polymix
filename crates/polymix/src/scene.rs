//! Scene objects and undo/redo history.
//!
//! Meshes are held in `Arc`s and mutated through `Arc::make_mut`, so a history snapshot
//! costs nothing until an object is edited, and then only that object is copied.

use glam::Vec4;
use mm_core::{Aabb, Bvh, Mesh, TriSelection};
use std::sync::Arc;

pub type ObjectId = u64;

static STAMP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
static NEXT_OBJECT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Globally unique stamp (never repeats, so caches can't confuse restored state).
pub fn next_stamp() -> u64 {
    STAMP.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

#[derive(Clone)]
pub struct SceneObject {
    pub id: ObjectId,
    pub name: String,
    pub mesh: Arc<Mesh>,
    pub selection: TriSelection,
    pub visible: bool,
    pub color: Vec4,
    /// Bumped whenever the selection or face groups change (drives GPU flag uploads).
    pub selection_stamp: u64,
    /// Vertices moved since the last GPU sync without any topology change (lets brushes
    /// update only the affected part of the GPU buffers). Cleared by the renderer.
    pub moved_vertices: Vec<u32>,
    /// Mesh shape version the moves in `moved_vertices` were applied on top of.
    pub moved_from: u64,
    /// Per-triangle values in [0, 1] shown with a colormap (e.g. deviation analysis),
    /// with a stamp that changes whenever the values do.
    pub scalars: Option<(u64, Arc<Vec<f32>>)>,
    /// Per-triangle display colours (`0xRRGGBB`, e.g. Mesh Query); take precedence over
    /// `scalars`.
    pub colors: Option<(u64, Arc<Vec<u32>>)>,
    /// Optional names for face groups (CFD boundary patches: wall, inlet, outlet_1, …).
    pub group_names: std::collections::HashMap<i32, String>,
    /// Orientation of the object's local frame: the rotations applied by Transform so far
    /// (Transform's "Local Frame" axes).
    pub frame: glam::DQuat,
    /// Index into [`crate::materials::MATERIALS`].
    pub material: usize,
    /// Texture coordinates (Unwrap); see [`SceneObject::uvs`].
    pub uvs: Option<Arc<UvSet>>,
    /// Pivots attached to the object (Create Pivot).
    pub pivots: Vec<Pivot>,
    /// An image mapped onto the object through its UVs (View → Color → Texture).
    pub texture: Option<Arc<TextureImage>>,
    /// A reference (File → Import reference…): drawn see-through as a guide, not picked in the
    /// view nor exported with the visible objects, until made editable.
    pub reference: bool,
    bvh: Option<Arc<Bvh>>,
}

/// An RGBA image for an object's texture.
pub struct TextureImage {
    pub width: u32,
    pub height: u32,
    /// Rows from the top, 4 bytes per pixel.
    pub rgba: Vec<u8>,
    /// Unique per image (GPU cache key).
    pub stamp: u64,
    /// The file it came from.
    pub path: Option<std::path::PathBuf>,
}

impl TextureImage {
    /// Read an image file (PNG, JPEG, BMP, TGA).
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let mut img = image::open(path).map_err(|e| anyhow::anyhow!("{}: {e}", path.display()))?.to_rgba8();
        // GPUs take textures of 8192 pixels a side at least; keep well within that.
        let (w, h) = img.dimensions();
        if w.max(h) > 4096 {
            let f = 4096.0 / w.max(h) as f64;
            img = image::imageops::resize(&img, ((w as f64 * f) as u32).max(1), ((h as f64 * f) as u32).max(1), image::imageops::FilterType::Triangle);
        }
        let (width, height) = img.dimensions();
        Ok(Self { width, height, rgba: img.into_raw(), stamp: next_stamp(), path: Some(path.to_path_buf()) })
    }

    /// The colour (0–1, as stored) at texture coordinates `uv` (repeating; v up), bilinear.
    pub fn sample(&self, uv: [f32; 2]) -> [f32; 3] {
        let (w, h) = (self.width.max(1) as usize, self.height.max(1) as usize);
        let x = uv[0].rem_euclid(1.0) * w as f32 - 0.5;
        let y = (1.0 - uv[1]).rem_euclid(1.0) * h as f32 - 0.5;
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let at = |i: f32, j: f32| {
            let (i, j) = ((i as i64).rem_euclid(w as i64) as usize, (j as i64).rem_euclid(h as i64) as usize);
            let k = (j * w + i) * 4;
            [0, 1, 2].map(|c| self.rgba.get(k + c).copied().unwrap_or(0) as f32 / 255.0)
        };
        let (a, b, c, d) = (at(x0, y0), at(x0 + 1.0, y0), at(x0, y0 + 1.0), at(x0 + 1.0, y0 + 1.0));
        [0, 1, 2].map(|k| (a[k] * (1.0 - fx) + b[k] * fx) * (1.0 - fy) + (c[k] * (1.0 - fx) + d[k] * fx) * fy)
    }
}

/// A point with a coordinate frame attached to an object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pivot {
    pub position: glam::DVec3,
    pub frame: glam::DQuat,
}

/// The frame a tool last worked in (Create Pivot's "From Last Tool"): see
/// [`Scene::record_tool_frame`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolFrame {
    /// The tool's name, e.g. "Transform".
    pub tool: String,
    pub pivot: Pivot,
}

/// UVs of each triangle's corners (indexed by triangle id). They belong to one mesh
/// topology: edits that move vertices keep them, edits that change triangles drop them.
pub struct UvSet {
    pub topo: u64,
    pub corners: Vec<[[f32; 2]; 3]>,
    /// Unique per set (GPU cache key).
    pub stamp: u64,
}

impl UvSet {
    pub fn new(mesh: &Mesh, corners: Vec<[[f32; 2]; 3]>) -> Self {
        Self { topo: mesh.topology_version(), corners, stamp: next_stamp() }
    }
}

impl SceneObject {
    pub fn new(id: ObjectId, name: impl Into<String>, mesh: Mesh) -> Self {
        let selection = TriSelection::new(&mesh);
        Self {
            id,
            name: name.into(),
            mesh: Arc::new(mesh),
            selection,
            visible: true,
            color: default_color(id),
            selection_stamp: next_stamp(),
            group_names: Default::default(),
            moved_vertices: Vec::new(),
            moved_from: 0,
            scalars: None,
            colors: None,
            frame: glam::DQuat::IDENTITY,
            material: 0,
            uvs: None,
            pivots: Vec::new(),
            reference: false,
            texture: None,
            bvh: None,
        }
    }

    /// Mutable access to the mesh (copy-on-write if shared with the undo history).
    pub fn mesh_mut(&mut self) -> &mut Mesh {
        Arc::make_mut(&mut self.mesh)
    }

    /// Replace the mesh, keeping (and remapping) nothing of the old selection.
    pub fn set_mesh(&mut self, mesh: Mesh) {
        self.selection = TriSelection::new(&mesh);
        self.mesh = Arc::new(mesh);
        self.touch_selection();
    }

    /// The UVs, if they still match the mesh's triangles.
    pub fn uvs(&self) -> Option<&UvSet> {
        self.uvs.as_deref().filter(|u| u.topo == self.mesh.topology_version())
    }

    pub fn touch_selection(&mut self) {
        self.selection_stamp = next_stamp();
    }

    /// `self.selection.sync(&self.mesh)`, without its walk over every triangle when nothing
    /// is selected: an empty selection only needs its length to follow the triangle ids, and
    /// stays all clear. The result is the same as `sync`'s.
    pub fn sync_selection(&mut self) {
        let n = self.mesh.max_tid();
        let len = self.selection.bits().len();
        if !self.selection.is_empty() || len > n {
            self.selection.sync(&self.mesh);
        } else if len < n {
            // Inserting the last id grows the bits (all clear) to the id range; it then goes.
            let last = n as u32 - 1;
            self.selection.insert(last);
            self.selection.remove(last);
        }
    }

    /// BVH for picking, rebuilt lazily after geometry changes.
    pub fn bvh(&mut self) -> Arc<Bvh> {
        match &mut self.bvh {
            Some(b) if b.is_current(&self.mesh) => {}
            Some(b) if b.can_refit(&self.mesh) => Arc::make_mut(b).refit(&self.mesh),
            _ => self.bvh = Some(Arc::new(Bvh::build(&self.mesh))),
        }
        self.bvh.clone().unwrap()
    }

    /// The BVH if it matches the mesh as it is now, without building or refitting it (for
    /// code that only has `&self`, such as tool panels).
    pub fn bvh_if_current(&self) -> Option<&Arc<Bvh>> {
        self.bvh.as_ref().filter(|b| b.is_current(&self.mesh))
    }

    /// A handle on the current BVH for a background job, if there is one (otherwise the job
    /// builds its own from the mesh).
    pub fn cached_bvh(&self) -> Option<Arc<Bvh>> {
        self.bvh_if_current().cloned()
    }

    /// Take a BVH built elsewhere (e.g. by a background job) over all of the mesh's
    /// triangles. It is kept only if it matches the mesh as it is now and the object has no
    /// current BVH yet: versions are unique, so a BVH built for an earlier state, or for
    /// another mesh, is turned away. Returns whether it was kept.
    pub fn install_bvh(&mut self, bvh: impl Into<Arc<Bvh>>) -> bool {
        let bvh = bvh.into();
        // A tree over only some triangles would answer queries (and bounds) for those only.
        let whole = bvh.tris.len() == self.mesh.triangle_count();
        if !bvh.is_current(&self.mesh) || !whole || self.bvh_if_current().is_some() {
            return false;
        }
        self.bvh = Some(bvh);
        true
    }

    pub fn bounds(&self) -> Aabb {
        self.mesh.bounds()
    }
}

fn default_color(id: ObjectId) -> Vec4 {
    // Meshmixer-like neutral clay, with gentle variation for additional objects.
    const PALETTE: [[f32; 3]; 6] = [
        [0.78, 0.76, 0.74],
        [0.62, 0.74, 0.86],
        [0.80, 0.68, 0.62],
        [0.66, 0.80, 0.66],
        [0.82, 0.78, 0.56],
        [0.74, 0.66, 0.82],
    ];
    let c = PALETTE[(id as usize).wrapping_sub(1) % PALETTE.len()];
    Vec4::new(c[0], c[1], c[2], 1.0)
}

#[derive(Clone, Default)]
pub struct Scene {
    pub objects: Vec<SceneObject>,
    pub active: Option<ObjectId>,
    /// The target surface (Meshmixer's Set as Target) for Attract / Align To Target and
    /// "Project to Target" options.
    pub target: Option<ObjectId>,
    /// The frame the last tool that records one worked in (not saved in projects).
    pub last_tool_frame: Option<ToolFrame>,
    /// History::undo / redo made this restored scene coherent already, knowing the scene it
    /// replaced: the [`Scene::after_restore`] their callers make next has nothing to do.
    restored: bool,
}

/// A target surface ready for nearest-point queries.
#[derive(Clone)]
pub struct TargetSurface {
    pub id: ObjectId,
    pub mesh: Arc<Mesh>,
    pub bvh: Arc<Bvh>,
}

impl PartialEq for TargetSurface {
    fn eq(&self, o: &Self) -> bool {
        self.id == o.id && Arc::ptr_eq(&self.mesh, &o.mesh)
    }
}

impl Scene {
    pub fn add(&mut self, name: impl Into<String>, mesh: Mesh) -> ObjectId {
        // Ids come from a global counter so an id is never reused, even after undo.
        let id = NEXT_OBJECT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.objects.push(SceneObject::new(id, name, mesh));
        self.active = Some(id);
        id
    }

    pub fn get(&self, id: ObjectId) -> Option<&SceneObject> {
        self.objects.iter().find(|o| o.id == id)
    }
    pub fn get_mut(&mut self, id: ObjectId) -> Option<&mut SceneObject> {
        self.objects.iter_mut().find(|o| o.id == id)
    }
    pub fn active(&self) -> Option<&SceneObject> {
        self.active.and_then(|id| self.get(id))
    }
    pub fn active_mut(&mut self) -> Option<&mut SceneObject> {
        let id = self.active?;
        self.get_mut(id)
    }
    /// The target surface, unless it is `except` (an object can't be its own target).
    pub fn target_surface(&mut self, except: Option<ObjectId>) -> Option<TargetSurface> {
        let id = self.target.filter(|&t| Some(t) != except)?;
        let o = self.get_mut(id)?;
        Some(TargetSurface { id, mesh: o.mesh.clone(), bvh: o.bvh() })
    }

    pub fn remove(&mut self, id: ObjectId) {
        if self.target == Some(id) {
            self.target = None;
        }
        self.objects.retain(|o| o.id != id);
        if self.active == Some(id) {
            self.active = self.objects.last().map(|o| o.id);
        }
    }
    pub fn bounds(&self) -> Aabb {
        let mut b = Aabb::EMPTY;
        for o in self.objects.iter().filter(|o| o.visible) {
            b = b.union(&o.bounds());
        }
        b
    }
    pub fn is_empty(&self) -> bool {
        self.objects.is_empty()
    }

    /// Remember the frame a tool just worked in, for Create Pivot's "From Last Tool": its
    /// centre and axes (Transform), where the source frame landed (Align), or a point on its
    /// plane with z along the normal (Plane Cut, Mirror). Tools call this when they commit.
    pub fn record_tool_frame(&mut self, tool: &str, position: glam::DVec3, frame: glam::DQuat) {
        if position.is_finite() && frame.is_finite() {
            self.last_tool_frame = Some(ToolFrame { tool: tool.to_string(), pivot: Pivot { position, frame: frame.normalize() } });
        }
    }
}

/// Linear undo history of whole-scene snapshots.
pub struct History {
    undo: Vec<(String, Scene)>,
    redo: Vec<(String, Scene)>,
    /// Rough memory budget for stored snapshots (bytes).
    budget: usize,
    /// Bumped by every push, undo and redo: a cheap "the scene changed" signal (autosave).
    generation: u64,
    /// An agent is making the changes: their steps are labelled "Agent: …".
    pub by_agent: bool,
}

impl Default for History {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), budget: 4 << 30, generation: 0, by_agent: false }
    }
}

fn mesh_bytes(m: &Mesh) -> usize {
    // Vertex: pos + edge list + flag; triangle: verts + edges + group; edge: verts + tris.
    m.max_vid() * 64 + m.max_tid() * 32 + m.max_eid() * 20
}

impl History {
    /// Record the scene state *before* an edit labelled `label`.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Forget every undo and redo step (Meshmixer: Discard Undo History), freeing their memory.
    pub fn clear(&mut self) {
        self.undo.clear();
        self.redo.clear();
    }

    /// Count a change that adds no undo step of its own (a later part of a gesture whose
    /// first change pushed the step), so autosave still sees the final state.
    pub fn touch(&mut self) {
        self.generation += 1;
    }

    pub fn push(&mut self, label: impl Into<String>, scene: &Scene) {
        self.generation += 1;
        let label = label.into();
        let label = if self.by_agent { format!("Agent: {label}") } else { label };
        self.undo.push((label, scene.clone()));
        self.redo.clear();
        self.enforce_budget();
    }

    fn enforce_budget(&mut self) {
        // Count each distinct mesh allocation once.
        loop {
            let mut seen = std::collections::HashSet::new();
            let mut total = 0usize;
            for (_, s) in self.undo.iter().chain(self.redo.iter()) {
                for o in &s.objects {
                    if seen.insert(Arc::as_ptr(&o.mesh) as usize) {
                        total += mesh_bytes(&o.mesh);
                    }
                    total += o.selection.bits().len();
                }
            }
            if total <= self.budget || self.undo.len() <= 1 {
                break;
            }
            self.undo.remove(0);
        }
    }

    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty()
    }
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty()
    }
    pub fn undo_label(&self) -> Option<&str> {
        self.undo.last().map(|(l, _)| l.as_str())
    }
    pub fn redo_label(&self) -> Option<&str> {
        self.redo.last().map(|(l, _)| l.as_str())
    }
    /// Every undo and redo step's label, oldest first (the last is undone / redone next).
    pub fn labels(&self) -> (Vec<&str>, Vec<&str>) {
        (self.undo.iter().map(|(l, _)| l.as_str()).collect(), self.redo.iter().map(|(l, _)| l.as_str()).collect())
    }

    pub fn undo(&mut self, scene: &mut Scene) -> Option<String> {
        self.generation += 1;
        let (label, prev) = self.undo.pop()?;
        let cur = std::mem::replace(scene, prev);
        scene.carry_metadata(&cur);
        scene.after_restore_from(&cur);
        scene.restored = true;
        self.redo.push((label.clone(), cur));
        Some(label)
    }

    pub fn redo(&mut self, scene: &mut Scene) -> Option<String> {
        self.generation += 1;
        let (label, next) = self.redo.pop()?;
        let cur = std::mem::replace(scene, next);
        scene.carry_metadata(&cur);
        scene.after_restore_from(&cur);
        scene.restored = true;
        self.undo.push((label.clone(), cur));
        Some(label)
    }

}

/// The same selection (the derived `==`, without walking two empty ones).
fn same_selection(a: &TriSelection, b: &TriSelection) -> bool {
    a.len() == b.len() && a.bits().len() == b.bits().len() && (a.is_empty() || a.bits() == b.bits())
}

impl Scene {
    /// Make sure selections and stamps are coherent after restoring a snapshot. After
    /// History::undo / redo there is nothing left to do: they restore with
    /// [`Scene::after_restore_from`] themselves.
    pub fn after_restore(&mut self) {
        if std::mem::take(&mut self.restored) {
            return;
        }
        for o in &mut self.objects {
            o.selection.sync(&o.mesh);
            o.touch_selection();
            o.moved_vertices.clear();
            o.scalars = None;
            o.colors = None;
        }
    }

    /// [`Scene::after_restore`] for a snapshot that took the place of `replaced`. An object
    /// whose mesh and selection are the very ones it had there keeps that selection stamp,
    /// so what was made from them (the GPU's per-face flags) is still good; the others get
    /// a new one. Selections are synced either way (cheaply when empty).
    pub fn after_restore_from(&mut self, replaced: &Scene) {
        for o in &mut self.objects {
            o.sync_selection();
            match replaced.get(o.id).filter(|r| Arc::ptr_eq(&r.mesh, &o.mesh) && same_selection(&r.selection, &o.selection)) {
                Some(r) => o.selection_stamp = r.selection_stamp,
                None => o.touch_selection(),
            }
            o.moved_vertices.clear();
            o.scalars = None;
            o.colors = None;
        }
    }

    /// Keep user-facing metadata that is not part of the edit history (names, colours,
    /// visibility, patch names) when switching to an undo/redo snapshot.
    fn carry_metadata(&mut self, from: &Scene) {
        self.target = from.target.filter(|&t| self.get(t).is_some());
        self.last_tool_frame = from.last_tool_frame.clone();
        for o in &mut self.objects {
            if let Some(src) = from.get(o.id) {
                o.name = src.name.clone();
                o.color = src.color;
                o.material = src.material;
                o.visible = src.visible;
                for (g, n) in &src.group_names {
                    o.group_names.insert(*g, n.clone());
                }
            }
        }
    }
}

#[cfg(test)]
mod history_tests {
    use super::*;

    #[test]
    fn generation_counts_changes() {
        let mut h = History::default();
        let mut s = Scene::default();
        let g0 = h.generation();
        h.push("a", &s);
        s.add("x", mm_core::primitives::cube(1.0, 1));
        assert!(h.generation() > g0);
        let g1 = h.generation();
        h.undo(&mut s);
        assert!(h.generation() > g1, "undo counts too");
        assert!(h.can_redo());
        h.clear();
        assert!(!h.can_undo() && !h.can_redo());
        h.by_agent = true;
        h.push("Remesh", &s);
        assert_eq!(h.undo_label(), Some("Agent: Remesh"));
        let g2 = h.generation();
        let undo_steps = h.labels().0.len();
        h.touch();
        assert!(h.generation() > g2, "a coalesced change counts");
        assert_eq!(h.labels().0.len(), undo_steps, "without an undo step");
    }

    #[test]
    fn undo_keeps_the_stamps_of_objects_it_did_not_change() {
        let mut s = Scene::default();
        let a = s.add("a", mm_core::primitives::icosphere(1.0, 2));
        let b = s.add("b", mm_core::primitives::cube(1.0, 2));
        let c = s.add("c", mm_core::primitives::cube(2.0, 1));
        s.get_mut(c).unwrap().selection.insert(3);
        let mut h = History::default();
        h.push("edit", &s);
        let stamps = |s: &Scene| [a, b, c].map(|id| s.get(id).unwrap().selection_stamp);
        let before = stamps(&s);
        // a's mesh changes, c's selection changes, b stays as it was.
        s.get_mut(a).unwrap().mesh_mut().set_vertex(0, glam::DVec3::splat(2.0));
        let o = s.get_mut(c).unwrap();
        o.selection.insert(5);
        o.touch_selection();
        for o in &mut s.objects {
            o.moved_vertices.push(1);
            o.scalars = Some((next_stamp(), Arc::new(vec![0.5; o.mesh.max_tid()])));
            o.colors = Some((next_stamp(), Arc::new(vec![0; o.mesh.max_tid()])));
        }
        let edited = stamps(&s);
        h.undo(&mut s);
        let undone = stamps(&s);
        assert_ne!(undone[0], edited[0], "a's mesh changed back");
        assert_eq!(undone[1], edited[1], "b is the same");
        assert_ne!(undone[2], edited[2], "c's selection changed back");
        assert!([undone[0], undone[2]].iter().all(|x| !before.contains(x) && !edited.contains(x)), "fresh stamps");
        assert!(s.objects.iter().all(|o| o.moved_vertices.is_empty() && o.scalars.is_none() && o.colors.is_none()));
        assert_eq!(s.get(c).unwrap().selection.to_vec(), vec![3]);
        // The caller's after_restore has nothing left to do; called on its own it renews all.
        s.after_restore();
        assert_eq!(stamps(&s), undone);
        s.after_restore();
        assert!(stamps(&s).iter().zip(&undone).all(|(x, y)| x != y));
        // Redo: b keeps its stamp again.
        let restamped = stamps(&s);
        h.redo(&mut s);
        s.after_restore();
        let redone = stamps(&s);
        assert_eq!(redone[1], restamped[1]);
        assert!(redone[0] != restamped[0] && redone[2] != restamped[2]);
        assert_eq!(s.get(c).unwrap().selection.to_vec(), vec![3, 5]);
    }

    #[test]
    fn restoring_syncs_selections_as_before() {
        // A snapshot whose selection is out of step with its mesh (longer, holding a removed
        // triangle) or shorter than the mesh's triangle ids: synced as the full walk does.
        let mut m = mm_core::primitives::icosphere(1.0, 2);
        let mut s = Scene::default();
        let a = s.add("a", m.clone());
        let b = s.add("b", m.clone());
        m.remove_triangles([7, 8]);
        let grown = {
            let mut g = m.clone();
            g.append_mesh(&mm_core::primitives::cube(0.2, 1), 1);
            g
        };
        {
            let o = s.get_mut(a).unwrap();
            o.mesh = Arc::new(m.clone());
            o.selection.insert(7);
            o.selection.insert(9);
            let o = s.get_mut(b).unwrap();
            o.mesh = Arc::new(grown.clone());
        }
        let snapshot = s.clone();
        let mut want = snapshot.clone();
        want.after_restore();
        let mut h = History::default();
        h.push("x", &s);
        s.get_mut(a).unwrap().set_mesh(mm_core::primitives::cube(1.0, 1));
        h.undo(&mut s);
        for (o, w) in s.objects.iter().zip(&want.objects) {
            assert_eq!(o.selection, w.selection, "object {}", o.id);
            assert_eq!(o.selection.bits().len(), o.mesh.max_tid());
        }
    }
}

#[cfg(test)]
mod object_tests {
    use super::*;
    use glam::DVec3;

    fn bits(b: Aabb) -> [u64; 6] {
        let (lo, hi) = (b.min.to_array(), b.max.to_array());
        [lo[0], lo[1], lo[2], hi[0], hi[1], hi[2]].map(f64::to_bits)
    }

    /// The bounds without any cache: every corner of every live triangle, in order.
    fn walked(m: &Mesh) -> Aabb {
        Aabb::from_points(m.triangle_ids().flat_map(|t| m.tri_positions(t)))
    }

    #[test]
    fn a_stale_bvh_is_told_from_a_current_one() {
        let mut sphere = mm_core::primitives::icosphere(2.0, 3);
        sphere.transform(|p| p * DVec3::new(1.0, 0.5, 3.0) + DVec3::new(4.0, -1.25, 0.25));
        let mut o = SceneObject::new(1, "sphere", sphere);
        assert!(o.bvh_if_current().is_none() && o.cached_bvh().is_none());
        assert_eq!(bits(o.bounds()), bits(walked(&o.mesh)), "no BVH");
        let _ = o.bvh();
        assert!(o.bvh_if_current().is_some() && o.cached_bvh().is_some());
        // Dabs: the mesh moves (its kept bounds and the tree go stale), then the BVH is refitted.
        for step in 0..20u32 {
            let v = (step * 37) % o.mesh.max_vid() as u32;
            let m = o.mesh_mut();
            let p = m.vertex(v);
            m.set_vertex(v, p + DVec3::new(0.3, -0.2, 0.1) * step as f64);
            assert!(o.mesh.cached_bounds().is_none(), "step {step}");
            assert!(o.bvh_if_current().is_none() && o.cached_bvh().is_none(), "step {step}: stale tree");
            assert_eq!(bits(o.bounds()), bits(walked(&o.mesh)), "step {step}");
            assert!(o.mesh.cached_bounds().is_some(), "step {step}: kept for the next call");
            let b = o.bvh();
            assert!(Arc::ptr_eq(o.bvh_if_current().expect("refitted"), &b), "step {step}");
        }
    }

    #[test]
    fn install_bvh_keeps_only_a_current_whole_tree() {
        let mut o = SceneObject::new(1, "cube", mm_core::primitives::cube(1.0, 2));
        let all = Bvh::build(&o.mesh);
        let some: Vec<u32> = o.mesh.triangle_ids().take(5).collect();
        assert!(!o.install_bvh(Bvh::build_subset(&o.mesh, &some)), "part of the triangles");
        // Built for an older state of the mesh.
        let old = Bvh::build(&o.mesh);
        o.mesh_mut().set_vertex(0, DVec3::splat(3.0));
        assert!(!o.install_bvh(old), "stale");
        assert!(!o.install_bvh(all), "stale too");
        assert!(o.bvh_if_current().is_none());
        let fresh = Arc::new(Bvh::build(&o.mesh));
        assert!(o.install_bvh(fresh.clone()));
        assert!(Arc::ptr_eq(o.bvh_if_current().unwrap(), &fresh));
        assert!(Arc::ptr_eq(&o.bvh(), &fresh), "used as it is");
        assert!(!o.install_bvh(Bvh::build(&o.mesh)), "already has a current one");
        assert!(Arc::ptr_eq(&o.cached_bvh().unwrap(), &fresh));
    }
}
