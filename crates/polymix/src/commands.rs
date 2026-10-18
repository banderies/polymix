//! Immediate (non-interactive) commands shared by menus, hotkeys and tool panels.
//! Every command records an undo snapshot before changing the scene.

use crate::scene::{History, ObjectId, Scene};
use mm_core::{topology, TriSelection};

pub struct CmdCx<'a> {
    pub scene: &'a mut Scene,
    pub history: &'a mut History,
    pub status: &'a mut String,
}

impl CmdCx<'_> {
    fn active_with_selection(&mut self) -> Option<ObjectId> {
        let o = self.scene.active_mut()?;
        // Drop ids of triangles that no longer exist before acting on the selection.
        let m = o.mesh.clone();
        o.selection.sync(&m);
        if o.selection.is_empty() {
            *self.status = "Nothing selected.".into();
            return None;
        }
        Some(o.id)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectCmd {
    All,
    Clear,
    Invert,
    Expand,
    Contract,
    Connected,
    InvertConnected,
    /// Connected triangles of the same groups (G).
    ExpandToGroups,
    /// Every triangle of the selected groups, connected or not.
    ExpandToSameGroups,
    OptimizeBoundary,
}

pub fn selection_command(cx: &mut CmdCx, cmd: SelectCmd) {
    let Some(id) = cx.scene.active else {
        return;
    };
    cx.history.push(format!("Select: {cmd:?}"), cx.scene);
    let o = cx.scene.get_mut(id).unwrap();
    let m = o.mesh.clone();
    let sel = &mut o.selection;
    sel.sync(&m);
    match cmd {
        SelectCmd::All => *sel = TriSelection::all(&m),
        SelectCmd::Clear => sel.clear(),
        SelectCmd::Invert => sel.invert(&m),
        SelectCmd::Expand => sel.expand(&m),
        SelectCmd::Contract => sel.contract(&m),
        SelectCmd::Connected => sel.expand_to_connected(&m),
        SelectCmd::InvertConnected => {
            let mut connected = sel.clone();
            connected.expand_to_connected(&m);
            let cur = sel.clone();
            sel.clear();
            for t in connected.iter() {
                if !cur.contains(t) {
                    sel.insert(t);
                }
            }
        }
        SelectCmd::ExpandToGroups => sel.expand_to_connected_groups(&m),
        SelectCmd::ExpandToSameGroups => sel.expand_to_groups(&m),
        SelectCmd::OptimizeBoundary => sel.optimize_boundary(&m, 8),
    }
    o.touch_selection();
    *cx.status = format!("{} triangles selected", o.selection.len());
}

/// Delete the selected triangles (Meshmixer "Discard", hotkey X).
pub fn discard_selection(cx: &mut CmdCx) {
    let Some(id) = cx.active_with_selection() else { return };
    cx.history.push("Discard", cx.scene);
    let o = cx.scene.get_mut(id).unwrap();
    let tris = o.selection.to_vec();
    let n = tris.len();
    o.mesh_mut().remove_triangles(tris);
    o.selection = TriSelection::new(&o.mesh);
    o.touch_selection();
    if o.mesh.triangle_count() == 0 {
        cx.scene.remove(id);
    }
    *cx.status = format!("Discarded {n} triangles");
}

/// Move the selection into a new object.
pub fn separate_selection(cx: &mut CmdCx, keep_original: bool) {
    let Some(id) = cx.active_with_selection() else { return };
    cx.history.push(if keep_original { "Duplicate selection" } else { "Separate" }, cx.scene);
    let o = cx.scene.get_mut(id).unwrap();
    let tris = o.selection.to_vec();
    let (part, _) = o.mesh.submesh(&tris);
    let name = format!("{} ({})", o.name, if keep_original { "copy" } else { "part" });
    if !keep_original {
        o.mesh_mut().remove_triangles(tris);
    }
    o.selection = TriSelection::new(&o.mesh);
    o.touch_selection();
    let empty = o.mesh.triangle_count() == 0;
    let new_id = cx.scene.add(name, part);
    if empty {
        cx.scene.remove(id);
    }
    cx.scene.active = Some(new_id);
    *cx.status = "Created new object from selection".into();
}

pub fn flip_selection_normals(cx: &mut CmdCx) {
    let Some(id) = cx.scene.active else { return };
    cx.history.push("Flip normals", cx.scene);
    let o = cx.scene.get_mut(id).unwrap();
    let m = o.mesh.clone();
    o.selection.sync(&m);
    let tris: Vec<u32> = if o.selection.is_empty() { o.mesh.triangle_ids().collect() } else { o.selection.to_vec() };
    let m = o.mesh_mut();
    for t in tris {
        m.reverse_triangle(t);
    }
    *cx.status = "Flipped normals".into();
}

pub fn create_face_group(cx: &mut CmdCx, clear: bool) {
    let Some(id) = cx.active_with_selection() else { return };
    cx.history.push(if clear { "Clear face group" } else { "Create face group" }, cx.scene);
    let o = cx.scene.get_mut(id).unwrap();
    let tris = o.selection.to_vec();
    let g = if clear { 0 } else { o.mesh.max_group_id() + 1 };
    let m = o.mesh_mut();
    for t in tris {
        m.set_tri_group(t, g);
    }
    m.bump_shape_version();
    o.touch_selection();
    *cx.status = if clear { "Cleared face group".into() } else { format!("Created face group {g}") };
}

/// Split every connected component of the active object into its own object.
pub fn separate_shells(cx: &mut CmdCx) {
    let Some(id) = cx.scene.active else { return };
    let o = cx.scene.get(id).unwrap();
    let comps = topology::connected_components(&o.mesh);
    if comps.len() < 2 {
        *cx.status = "Object has a single shell.".into();
        return;
    }
    cx.history.push("Separate shells", cx.scene);
    let o = cx.scene.get(id).unwrap();
    let base = o.name.clone();
    let mut parts: Vec<(usize, mm_core::Mesh)> = comps.iter().map(|c| (c.len(), o.mesh.submesh(c).0)).collect();
    parts.sort_by_key(|p| std::cmp::Reverse(p.0));
    cx.scene.remove(id);
    let n = parts.len();
    let mut first = None;
    for (i, (_, m)) in parts.into_iter().enumerate() {
        let nid = cx.scene.add(format!("{base} shell {}", i + 1), m);
        first.get_or_insert(nid);
    }
    cx.scene.active = first;
    *cx.status = format!("Separated into {n} shells");
}

/// Generate Complex: the cells of the visible closed objects (every region covered by a
/// particular set of them), as one new object with a face group per cell.
pub fn generate_complex(cx: &mut CmdCx) {
    let solids: Vec<(ObjectId, mm_core::Mesh)> = cx.scene.objects.iter().filter(|o| o.visible && o.mesh.is_closed() && o.mesh.triangle_count() > 0).map(|o| (o.id, (*o.mesh).clone())).collect();
    if solids.len() < 2 {
        *cx.status = "Generate Complex needs at least two visible closed objects that overlap.".into();
        return;
    }
    if solids.len() > 6 {
        *cx.status = "Generate Complex works on up to six visible closed objects.".into();
        return;
    }
    let meshes: Vec<mm_core::Mesh> = solids.iter().map(|s| s.1.clone()).collect();
    match mm_edit::generate_complex(&meshes) {
        Ok(cells) => {
            cx.history.push("Generate Complex", cx.scene);
            let mut m = mm_core::Mesh::new();
            for (i, c) in cells.iter().enumerate() {
                let start = m.max_tid() as u32;
                m.append_mesh(c, 0);
                for t in start..m.max_tid() as u32 {
                    if m.is_triangle(t) {
                        m.set_tri_group(t, i as i32);
                    }
                }
            }
            let n = cells.len();
            for (id, _) in &solids {
                if let Some(o) = cx.scene.get_mut(*id) {
                    o.visible = false;
                }
            }
            let id = cx.scene.add("Complex", m);
            cx.scene.active = Some(id);
            *cx.status = format!("Generate Complex: {n} cells (one face group each); the inputs are hidden");
        }
        Err(e) => *cx.status = format!("Generate Complex: {e}"),
    }
}

/// Split Complex: one object per face group of the active object.
pub fn split_complex(cx: &mut CmdCx) {
    let Some(id) = cx.scene.active else { return };
    let o = cx.scene.get(id).unwrap();
    let mut groups: std::collections::BTreeMap<i32, Vec<u32>> = Default::default();
    for t in o.mesh.triangle_ids() {
        groups.entry(o.mesh.tri_group(t)).or_default().push(t);
    }
    if groups.len() < 2 {
        *cx.status = "Split Complex: the object has a single face group.".into();
        return;
    }
    cx.history.push("Split Complex", cx.scene);
    let o = cx.scene.get(id).unwrap();
    let base = o.name.clone();
    let parts: Vec<(i32, mm_core::Mesh)> = groups.into_iter().map(|(g, tris)| (g, o.mesh.submesh(&tris).0)).collect();
    cx.scene.remove(id);
    let n = parts.len();
    let mut first = None;
    for (g, m) in parts {
        let nid = cx.scene.add(format!("{base} cell {g}"), m);
        first.get_or_insert(nid);
    }
    cx.scene.active = first;
    *cx.status = format!("Split Complex: {n} objects");
}

/// Merge all visible objects into the active one. Every object's group 0 (its wall) stays
/// group 0; other face groups get fresh ids, and their patch names come along.
pub fn combine_visible(cx: &mut CmdCx) {
    let ids: Vec<ObjectId> = cx.scene.objects.iter().filter(|o| o.visible).map(|o| o.id).collect();
    if ids.len() < 2 {
        *cx.status = "Need at least two visible objects to combine.".into();
        return;
    }
    cx.history.push("Combine", cx.scene);
    let target = cx.scene.active.filter(|a| ids.contains(a)).unwrap_or(ids[0]);
    let others: Vec<(mm_core::Mesh, std::collections::HashMap<i32, String>)> = ids
        .iter()
        .filter(|&&i| i != target)
        .filter_map(|&i| cx.scene.get(i).map(|o| ((*o.mesh).clone(), o.group_names.clone())))
        .collect();
    let Some(t) = cx.scene.get_mut(target) else { return };
    let mut next_group = t.mesh.max_group_id() + 1;
    for (mut m, names) in others {
        let mut remap = std::collections::HashMap::new();
        let tris: Vec<u32> = m.triangle_ids().collect();
        for tri in tris {
            let g = m.tri_group(tri);
            if g != 0 {
                let ng = *remap.entry(g).or_insert_with(|| {
                    next_group += 1;
                    next_group - 1
                });
                m.set_tri_group(tri, ng);
            }
        }
        t.mesh_mut().append_mesh(&m, 0);
        for (g, n) in names {
            if let Some(ng) = remap.get(&g) {
                t.group_names.insert(*ng, n);
            }
        }
    }
    t.selection = TriSelection::new(&t.mesh);
    t.touch_selection();
    for i in ids {
        if i != target {
            cx.scene.remove(i);
        }
    }
    cx.scene.active = Some(target);
    *cx.status = "Combined objects".into();
}

/// Make `id` the target surface (`None` clears it).
pub fn set_target(cx: &mut CmdCx, id: Option<ObjectId>) {
    match id {
        Some(_) if cx.scene.objects.len() < 2 => {
            *cx.status = "A target surface needs a second object in the scene.".into();
        }
        Some(id) => {
            cx.scene.target = Some(id);
            let name = cx.scene.get(id).map(|o| o.name.clone()).unwrap_or_default();
            *cx.status = format!("{name} is the target surface");
        }
        None => {
            cx.scene.target = None;
            *cx.status = "Target cleared".into();
        }
    }
}

pub fn duplicate_object(cx: &mut CmdCx) {
    if cx.scene.active().is_none() {
        return;
    }
    cx.history.push("Duplicate object", cx.scene);
    let o = cx.scene.active().unwrap();
    let (name, mesh, frame) = (format!("{} copy", o.name), (*o.mesh).clone(), o.frame);
    let id = cx.scene.add(name, mesh);
    if let Some(o) = cx.scene.get_mut(id) {
        o.frame = frame;
    }
}

pub fn delete_object(cx: &mut CmdCx, id: ObjectId) {
    cx.history.push("Delete object", cx.scene);
    cx.scene.remove(id);
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn generate_then_split_complex() {
        let mut scene = Scene::default();
        let mut history = History::default();
        let mut status = String::new();
        let a = primitives::cube(2.0, 1);
        let mut b = primitives::cube(2.0, 1);
        b.transform(|p| p + glam::DVec3::new(1.0, 0.5, 0.0));
        scene.add("a", a);
        scene.add("b", b);
        generate_complex(&mut CmdCx { scene: &mut scene, history: &mut history, status: &mut status });
        let complex = scene.active().unwrap();
        assert_eq!(complex.name, "Complex", "{status}");
        let groups: std::collections::BTreeSet<i32> = complex.mesh.triangle_ids().map(|t| complex.mesh.tri_group(t)).collect();
        assert_eq!(groups.len(), 3);
        assert_eq!(scene.objects.iter().filter(|o| o.visible).count(), 1);
        split_complex(&mut CmdCx { scene: &mut scene, history: &mut history, status: &mut status });
        let cells: Vec<_> = scene.objects.iter().filter(|o| o.name.contains("cell")).collect();
        assert_eq!(cells.len(), 3);
        assert!(cells.iter().all(|o| o.mesh.is_closed()));
    }
}
