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
    /// Optional names for face groups (CFD boundary patches: wall, inlet, outlet_1, …).
    pub group_names: std::collections::HashMap<i32, String>,
    bvh: Option<Arc<Bvh>>,
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

    pub fn touch_selection(&mut self) {
        self.selection_stamp = next_stamp();
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
    pub fn remove(&mut self, id: ObjectId) {
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
}

/// Linear undo history of whole-scene snapshots.
pub struct History {
    undo: Vec<(String, Scene)>,
    redo: Vec<(String, Scene)>,
    /// Rough memory budget for stored snapshots (bytes).
    budget: usize,
}

impl Default for History {
    fn default() -> Self {
        Self { undo: Vec::new(), redo: Vec::new(), budget: 4 << 30 }
    }
}

fn mesh_bytes(m: &Mesh) -> usize {
    // Vertex: pos + edge list + flag; triangle: verts + edges + group; edge: verts + tris.
    m.max_vid() * 64 + m.max_tid() * 32 + m.max_eid() * 20
}

impl History {
    /// Record the scene state *before* an edit labelled `label`.
    pub fn push(&mut self, label: impl Into<String>, scene: &Scene) {
        self.undo.push((label.into(), scene.clone()));
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

    pub fn undo(&mut self, scene: &mut Scene) -> Option<String> {
        let (label, prev) = self.undo.pop()?;
        let cur = std::mem::replace(scene, prev);
        scene.carry_metadata(&cur);
        self.redo.push((label.clone(), cur));
        Some(label)
    }

    pub fn redo(&mut self, scene: &mut Scene) -> Option<String> {
        let (label, next) = self.redo.pop()?;
        let cur = std::mem::replace(scene, next);
        scene.carry_metadata(&cur);
        self.undo.push((label.clone(), cur));
        Some(label)
    }

}

impl Scene {
    /// Make sure selections and stamps are coherent after restoring a snapshot.
    pub fn after_restore(&mut self) {
        for o in &mut self.objects {
            o.selection.sync(&o.mesh);
            o.touch_selection();
            o.moved_vertices.clear();
            o.scalars = None;
        }
    }

    /// Keep user-facing metadata that is not part of the edit history (names, colours,
    /// visibility, patch names) when switching to an undo/redo snapshot.
    fn carry_metadata(&mut self, from: &Scene) {
        for o in &mut self.objects {
            if let Some(src) = from.get(o.id) {
                o.name = src.name.clone();
                o.color = src.color;
                o.visible = src.visible;
                for (g, n) in &src.group_names {
                    o.group_names.insert(*g, n.clone());
                }
            }
        }
    }
}
