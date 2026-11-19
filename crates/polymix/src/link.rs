//! Parts in the Assembly tab and the Mesh objects they were sent from (Send to Assembly).
//!
//! The Mesh and Assembly tabs have scenes of their own. An Assembly part is a copy of a Mesh
//! object, placed on its own: moving, turning or scaling either one leaves the other where it
//! is, while edits of the shape in the Mesh tab (brushes, remeshing, booleans, cuts) reach the
//! part, which keeps its place.
//!
//! Meshes hold their own world positions (objects have no transforms), so what changed is
//! told by comparing. A Mesh object that was sent keeps the moves, turns and scales made to it
//! since ([`Placed`]): when it holds another mesh, the same triangles with every vertex moved by
//! one affine map (not a mirror, which makes a left part a right one) is a new place for it;
//! anything else is an edit. A part remembers the object's mesh it last took and the object's
//! place then ([`Link`]); with the map from that mesh to its own (where the Assembly tab put
//! it), the object's mesh as it is now is put where the part is, which changes nothing after a
//! move in the Mesh tab and brings an edit made there.
//!
//! [`sync`] compares after each change in either tab. An undo in the Assembly tab brings back
//! a part with the mesh and place it last took then, so the edits made in the Mesh tab since
//! come back to it at the next sync, wherever the object was moved meanwhile; an undo in the
//! Mesh tab brings back the object with its place of the time.

use crate::scene::{ObjectId, Scene, SceneObject};
use glam::{DMat3, DVec3};
use mm_core::{Mesh, TriSelection};
use std::sync::Arc;

/// Where an Assembly part came from: the Mesh object, its mesh as the part last took it, and
/// that object's place then ([`Placed::map`]).
#[derive(Clone)]
pub struct Link {
    pub source: ObjectId,
    pub seen: Arc<Mesh>,
    pub seen_place: Affine,
}

/// A Mesh object sent to the Assembly tab: the moves, turns and scales made to it since it was
/// first sent (`map`, taking where it was then to where it is), as known when it held `mesh`.
#[derive(Clone)]
pub struct Placed {
    pub map: Affine,
    pub mesh: Arc<Mesh>,
}

/// An affine map `p ↦ m p + t`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Affine {
    pub m: DMat3,
    pub t: DVec3,
}

impl Affine {
    pub const IDENTITY: Affine = Affine { m: DMat3::IDENTITY, t: DVec3::ZERO };

    pub fn apply(&self, p: DVec3) -> DVec3 {
        self.m * p + self.t
    }

    /// `self` after `first`: `p ↦ self(first(p))`.
    pub fn after(&self, first: &Affine) -> Affine {
        Affine { m: self.m * first.m, t: self.m * first.t + self.t }
    }

    pub fn inverse(&self) -> Affine {
        let m = self.m.inverse();
        Affine { m, t: -(m * self.t) }
    }

    /// Moves no point of `mesh` by more than `eps`.
    pub fn is_identity_on(&self, mesh: &Mesh, eps: f64) -> bool {
        let b = mesh.bounds();
        if mesh.vertex_count() == 0 {
            return true;
        }
        (0..8).all(|i| {
            let c = DVec3::new(if i & 1 == 0 { b.min.x } else { b.max.x }, if i & 2 == 0 { b.min.y } else { b.max.y }, if i & 4 == 0 { b.min.z } else { b.max.z });
            (self.apply(c) - c).length() <= eps
        })
    }

    /// The matrix by columns, then the shift (as projects write it).
    pub fn to_array(&self) -> [f64; 12] {
        let m = self.m.to_cols_array();
        [m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8], self.t.x, self.t.y, self.t.z]
    }

    pub fn from_array(a: &[f64; 12]) -> Affine {
        Affine { m: DMat3::from_cols_array(&[a[0], a[1], a[2], a[3], a[4], a[5], a[6], a[7], a[8]]), t: DVec3::new(a[9], a[10], a[11]) }
    }
}

/// `a bᵀ`.
fn outer(a: DVec3, b: DVec3) -> DMat3 {
    DMat3::from_cols(a * b.x, a * b.y, a * b.z)
}

/// The affine map that best takes the vertices of `from` onto the same vertices of `to` (least
/// squares), and the largest distance it leaves a vertex from its place. `None` when the two do
/// not have the same vertices. For points in a plane (or along a line) the map across it is not
/// set by them: a turn and an even scale is fitted instead.
pub fn fit(from: &Mesh, to: &Mesh) -> Option<(Affine, f64)> {
    let n = from.vertex_count();
    if n == 0 || n != to.vertex_count() || from.max_vid() != to.max_vid() {
        return None;
    }
    let (pf, pt) = (from.positions(), to.positions());
    let (mut cf, mut ct) = (DVec3::ZERO, DVec3::ZERO);
    for v in from.vertex_ids() {
        if !to.is_vertex(v) {
            return None;
        }
        cf += pf[v as usize];
        ct += pt[v as usize];
    }
    cf /= n as f64;
    ct /= n as f64;
    let (mut sxx, mut syx) = (DMat3::ZERO, DMat3::ZERO);
    for v in from.vertex_ids() {
        let (x, y) = (pf[v as usize] - cf, pt[v as usize] - ct);
        sxx += outer(x, x);
        syx += outer(y, x);
    }
    let tr = sxx.x_axis.x + sxx.y_axis.y + sxx.z_axis.z;
    let m = if tr <= 0.0 {
        DMat3::IDENTITY
    } else if sxx.determinant() > 1e-9 * (tr / 3.0).powi(3) {
        syx * sxx.inverse()
    } else {
        let r = mm_core::linalg::closest_rotation(syx);
        let rt = syx * r.transpose();
        let s = (rt.x_axis.x + rt.y_axis.y + rt.z_axis.z) / tr;
        r * s
    };
    let a = Affine { m, t: ct - m * cf };
    let worst = from.vertex_ids().map(|v| (a.apply(pf[v as usize]) - pt[v as usize]).length()).fold(0.0, f64::max);
    Some((a, worst))
}

/// The same triangles, with the same vertices.
fn same_triangles(a: &Mesh, b: &Mesh) -> bool {
    if a.topology_version() == b.topology_version() {
        return true;
    }
    a.max_tid() == b.max_tid() && a.triangle_count() == b.triangle_count() && a.triangle_ids().all(|t| b.is_triangle(t) && a.triangle(t) == b.triangle(t))
}

/// Within this share of a part's size, a vertex is where a map puts it.
const TOLERANCE: f64 = 1e-7;

/// `to` is `from` moved, turned or scaled as a whole (a new place, not an edit): the map, if
/// so. A mirror is not one: it makes a left part a right one.
pub fn placement(from: &Mesh, to: &Mesh) -> Option<Affine> {
    if !same_triangles(from, to) {
        return None;
    }
    let (a, worst) = fit(from, to)?;
    let size = from.bounds().diagonal().max(to.bounds().diagonal()).max(1e-12);
    (worst <= TOLERANCE * size && a.m.determinant() > 0.0).then_some(a)
}

/// Where the Assembly tab put a part: the map from the mesh it was made from (`seen`) to its
/// own. When the two no longer have the same vertices (which the Assembly tab's tools never
/// change), the shift between their middles.
pub fn pose(seen: &Mesh, part: &Mesh) -> Affine {
    match fit(seen, part) {
        Some((a, _)) => a,
        None => Affine { m: DMat3::IDENTITY, t: part.bounds().center() - seen.bounds().center() },
    }
}

/// `mesh` placed by `pose`: shared when that moves nothing.
pub fn posed(mesh: &Arc<Mesh>, pose: &Affine) -> Arc<Mesh> {
    if moves_nothing(mesh, pose) {
        return mesh.clone();
    }
    Arc::new(place((**mesh).clone(), pose))
}

/// `pose` leaves every point of `mesh` where it is (to a trillionth of its size).
pub fn moves_nothing(mesh: &Mesh, pose: &Affine) -> bool {
    pose.is_identity_on(mesh, 1e-12 * mesh.bounds().diagonal().max(1e-12))
}

/// `mesh` with every vertex moved by `pose` (turned inside out back again by a mirror).
pub fn place(mut mesh: Mesh, pose: &Affine) -> Mesh {
    mesh.transform(|p| pose.apply(p));
    if pose.m.determinant() < 0.0 {
        mesh.reverse_orientation();
    }
    mesh
}

/// The map placing `source` exactly where `part` is (the same triangles, every vertex within a
/// billionth of the size), if there is one: a part stored as that in a project is made again
/// from its Mesh object on loading.
pub fn exact_pose(source: &Mesh, part: &Mesh) -> Option<Affine> {
    if !same_triangles(source, part) {
        return None;
    }
    let (a, worst) = fit(source, part)?;
    let size = source.bounds().diagonal().max(part.bounds().diagonal()).max(1e-12);
    (worst <= 1e-9 * size && a.m.determinant() > 0.0).then_some(a)
}

/// Give `part` a new mesh in place of its own (an edit from the Mesh tab), with what belongs to
/// the mesh (face groups and their names, UVs) from `source`.
fn take_mesh(part: &mut SceneObject, mesh: Arc<Mesh>, source: &SceneObject) {
    part.selection = TriSelection::new(&mesh);
    part.mesh = mesh;
    part.touch_selection();
    part.moved_vertices.clear();
    part.scalars = None;
    part.colors = None;
    part.group_names = source.group_names.clone();
    part.uvs = source.uvs.clone().filter(|u| u.topo == part.mesh.topology_version());
}

/// Every vertex of `from`, moved by `map`, is where it is in `to` (the same triangles).
fn maps_onto(from: &Mesh, map: &Affine, to: &Mesh) -> bool {
    if !same_triangles(from, to) || from.max_vid() != to.max_vid() {
        return false;
    }
    let tol = TOLERANCE * to.bounds().diagonal().max(1e-12);
    let (pf, pt) = (from.positions(), to.positions());
    from.vertex_ids().all(|v| to.is_vertex(v) && (map.apply(pf[v as usize]) - pt[v as usize]).length() <= tol)
}

/// Bring a sent Mesh object's place up to date with its mesh: a move, turn or scale since it
/// was last compared is added to it (an edit leaves it as it is). Returns the place.
fn update_place(o: &mut SceneObject) -> Affine {
    let p = o.placed.get_or_insert_with(|| Placed { map: Affine::IDENTITY, mesh: o.mesh.clone() });
    if !Arc::ptr_eq(&p.mesh, &o.mesh) {
        if let Some(a) = placement(&p.mesh, &o.mesh) {
            p.map = a.after(&p.map);
        }
        p.mesh = o.mesh.clone();
    }
    p.map
}

/// Bring the Assembly tab's parts (`asm`) up to date with the Mesh tab's objects (`mesh`): first
/// the places of the objects that were sent (a move, turn or scale since they were last
/// compared), then each part whose object was edited takes the edit, where the part is.
/// Returns those parts.
pub fn sync(mesh: &mut Scene, asm: &mut Scene) -> Vec<ObjectId> {
    for o in mesh.objects.iter_mut().filter(|o| o.placed.is_some()) {
        update_place(o);
    }
    let mut edited = Vec::new();
    for part in asm.objects.iter_mut() {
        let Some(link) = &part.link else { continue };
        let Some(src) = mesh.get(link.source) else { continue };
        let place = src.placed.as_ref().map_or(Affine::IDENTITY, |p| p.map);
        if Arc::ptr_eq(&src.mesh, &link.seen) && link.seen_place == place {
            continue;
        }
        // The object's mesh as it is now, put where the part is: the part's map from the mesh
        // it last took, after undoing the moves made in the Mesh tab since.
        let to_part = pose(&link.seen, &part.mesh).after(&link.seen_place).after(&place.inverse());
        if !maps_onto(&src.mesh, &to_part, &part.mesh) {
            take_mesh(part, posed(&src.mesh, &to_part), src);
            edited.push(part.id);
        }
        part.link = Some(Link { source: src.id, seen: src.mesh.clone(), seen_place: place });
    }
    edited
}

/// Copy the Mesh objects `ids` into the Assembly tab (`asm`) as parts, where they are, each
/// linked to its object. Returns the new parts.
pub fn send(mesh: &mut Scene, ids: &[ObjectId], asm: &mut Scene) -> Vec<ObjectId> {
    let mut made = Vec::new();
    for &id in ids {
        let Some(o) = mesh.get_mut(id) else { continue };
        let place = update_place(o);
        let copies = asm.objects.iter().filter(|p| p.link.as_ref().is_some_and(|l| l.source == id)).count();
        let name = if copies == 0 { o.name.clone() } else { format!("{} ({})", o.name, copies + 1) };
        let pid = asm.add_shared(name, o.mesh.clone());
        let p = asm.get_mut(pid).unwrap();
        copy_look(p, o);
        p.link = Some(Link { source: id, seen: o.mesh.clone(), seen_place: place });
        made.push(pid);
    }
    made
}

/// Copy the visible parts of the Assembly tab (`asm`), as they are posed, into the Mesh tab
/// (`mesh`) as objects of their own (not linked). Returns the new objects.
pub fn send_posed(asm: &Scene, mesh: &mut Scene) -> Vec<ObjectId> {
    let mut made = Vec::new();
    for p in asm.objects.iter().filter(|p| p.visible && !p.reference) {
        let id = mesh.add_shared(format!("{} (posed)", p.name), p.mesh.clone());
        copy_look(mesh.get_mut(id).unwrap(), p);
        made.push(id);
    }
    made
}

/// What `to` takes of `from` on being sent to the other tab: its look and what belongs to its
/// mesh.
fn copy_look(to: &mut SceneObject, from: &SceneObject) {
    to.color = from.color;
    to.material = from.material;
    to.frame = from.frame;
    to.group_names = from.group_names.clone();
    to.uvs = from.uvs.clone();
    to.texture = from.texture.clone();
    to.pivots = from.pivots.clone();
    to.reference = from.reference;
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::DQuat;
    use mm_core::primitives;

    fn moved(m: &Mesh, f: impl Fn(DVec3) -> DVec3) -> Mesh {
        let mut m = m.clone();
        m.transform(f);
        m
    }

    /// A Mesh tab with a cube and an Assembly tab with the cube sent to it.
    fn sent() -> (Scene, Scene, ObjectId, ObjectId) {
        let mut mesh = Scene::default();
        let id = mesh.add("cube", primitives::cube(10.0, 3));
        let mut asm = Scene::default();
        let part = send(&mut mesh, &[id], &mut asm)[0];
        (mesh, asm, id, part)
    }

    #[test]
    fn moves_turns_and_scales_are_placements_edits_and_mirrors_are_not() {
        let m = primitives::cube(10.0, 3);
        let turn = DQuat::from_rotation_z(0.7) * DQuat::from_rotation_x(-0.3);
        for (what, f) in [
            ("move", Box::new(|p: DVec3| p + DVec3::new(4.0, -2.0, 30.0)) as Box<dyn Fn(DVec3) -> DVec3>),
            ("turn", Box::new(move |p| turn * p + DVec3::X)),
            ("scale", Box::new(|p| DVec3::new(2.0 * p.x, 0.5 * p.y, 3.0 * p.z))),
        ] {
            let a = placement(&m, &moved(&m, &f)).unwrap_or_else(|| panic!("{what}"));
            let c = DVec3::new(1.0, 2.0, 3.0);
            assert!((a.apply(c) - f(c)).length() < 1e-9, "{what}");
        }
        assert!(placement(&m, &moved(&m, |p| DVec3::new(-p.x, p.y, p.z))).is_none(), "mirror");
        let mut dent = m.clone();
        let v = dent.vertex_ids().next().unwrap();
        dent.set_vertex(v, dent.vertex(v) * 0.9);
        assert!(placement(&m, &dent).is_none(), "one vertex moved");
        let mut fewer = m.clone();
        let t = fewer.triangle_ids().next().unwrap();
        fewer.remove_triangle(t, false).unwrap();
        assert!(placement(&m, &fewer).is_none(), "a triangle less, the same vertices");
        // A flat part: its turn and size are still found.
        let plate = primitives::plane_grid(20.0, 10.0, 4, 4);
        assert!(placement(&plate, &moved(&plate, move |p| turn * p * 1.5 + DVec3::Y)).is_some());
    }

    #[test]
    fn a_move_in_mesh_leaves_the_part_where_it_is() {
        let (mut mesh, mut asm, id, part) = sent();
        let before = asm.get(part).unwrap().mesh.clone();
        mesh.get_mut(id).unwrap().mesh_mut().transform(|p| DQuat::from_rotation_y(1.0) * p * 2.0 + DVec3::splat(50.0));
        assert!(sync(&mut mesh, &mut asm).is_empty());
        assert!(Arc::ptr_eq(&asm.get(part).unwrap().mesh, &before), "the part is untouched");
        // Its next edit is still told apart from that move.
        let o = mesh.get_mut(id).unwrap();
        let v = o.mesh.vertex_ids().next().unwrap();
        let p = o.mesh.vertex(v);
        o.mesh_mut().set_vertex(v, p + DVec3::splat(3.0));
        assert_eq!(sync(&mut mesh, &mut asm), vec![part]);
        // In the part's place: the move undone, so the pull is turned back and halved.
        let got = &asm.get(part).unwrap().mesh;
        let pull = DQuat::from_rotation_y(-1.0) * DVec3::splat(3.0) / 2.0;
        assert!((got.vertex(v) - (before.vertex(v) + pull)).length() < 1e-9, "the dent reached the part");
        let w = got.vertex_ids().nth(7).unwrap();
        assert!((got.vertex(w) - before.vertex(w)).length() < 1e-9, "the rest where it was, not where the Mesh object went");
    }

    #[test]
    fn an_edit_in_mesh_reaches_the_part_where_it_is_posed() {
        let (mut mesh, mut asm, id, part) = sent();
        // Posed in Assembly: turned and moved.
        let pose = |p: DVec3| DQuat::from_rotation_z(0.5) * p + DVec3::new(0.0, 40.0, 0.0);
        asm.get_mut(part).unwrap().mesh_mut().transform(pose);
        // Sculpted in Mesh: one vertex pulled out.
        let o = mesh.get_mut(id).unwrap();
        let v = o.mesh.vertex_ids().next().unwrap();
        let pulled = o.mesh.vertex(v) * 1.5;
        o.mesh_mut().set_vertex(v, pulled);
        assert_eq!(sync(&mut mesh, &mut asm), vec![part]);
        let p = asm.get(part).unwrap();
        assert!((p.mesh.vertex(v) - pose(pulled)).length() < 1e-9, "the pulled vertex, posed");
        let w = p.mesh.vertex_ids().nth(5).unwrap();
        assert!((p.mesh.vertex(w) - pose(mesh.get(id).unwrap().mesh.vertex(w))).length() < 1e-9, "the rest posed as before");
        assert!(Arc::ptr_eq(&p.link.as_ref().unwrap().seen, &mesh.get(id).unwrap().mesh));
        assert!(sync(&mut mesh, &mut asm).is_empty(), "nothing more to take");
    }

    #[test]
    fn a_remesh_in_mesh_reaches_the_part_and_an_unposed_part_shares_the_mesh() {
        let (mut mesh, mut asm, id, part) = sent();
        let finer = primitives::cube(10.0, 6);
        mesh.get_mut(id).unwrap().set_mesh(finer);
        assert_eq!(sync(&mut mesh, &mut asm), vec![part]);
        let (p, src) = (asm.get(part).unwrap(), mesh.get(id).unwrap());
        assert!(Arc::ptr_eq(&p.mesh, &src.mesh), "not moved in Assembly: the very mesh");
        assert_eq!(p.selection.bits().len(), p.mesh.max_tid());
    }

    #[test]
    fn undo_in_assembly_keeps_the_edits_made_in_mesh() {
        let (mut mesh, mut asm, id, part) = sent();
        let mut history = crate::scene::History::default();
        history.push("Pose", &asm);
        asm.get_mut(part).unwrap().mesh_mut().transform(|p| p + DVec3::X * 20.0);
        let edit = primitives::icosphere(6.0, 3);
        mesh.get_mut(id).unwrap().set_mesh(edit);
        sync(&mut mesh, &mut asm);
        assert_eq!(asm.get(part).unwrap().mesh.triangle_count(), mesh.get(id).unwrap().mesh.triangle_count());
        // Undoing the pose brings back the cube it was then; the sync gives it the sphere again,
        // unposed.
        history.undo(&mut asm);
        assert_eq!(sync(&mut mesh, &mut asm), vec![part]);
        assert!(Arc::ptr_eq(&asm.get(part).unwrap().mesh, &mesh.get(id).unwrap().mesh));
    }

    #[test]
    fn undo_in_assembly_after_a_move_and_an_edit_in_mesh_keeps_the_part_in_its_place() {
        let (mut mesh, mut asm, id, part) = sent();
        let mut asm_history = crate::scene::History::default();
        asm_history.push("Pose", &asm);
        let posed_by = |p: DVec3| p + DVec3::X * 30.0;
        asm.get_mut(part).unwrap().mesh_mut().transform(posed_by);
        // Moved and scaled in Mesh, then the moved object edited (a vertex pulled out).
        let moved_by = |p: DVec3| p * 2.0 + DVec3::Y * 50.0;
        mesh.get_mut(id).unwrap().mesh_mut().transform(moved_by);
        assert!(sync(&mut mesh, &mut asm).is_empty(), "a move is not an edit");
        let o = mesh.get_mut(id).unwrap();
        let v = o.mesh.vertex_ids().next().unwrap();
        let pulled = o.mesh.vertex(v) + DVec3::Z * 4.0;
        o.mesh_mut().set_vertex(v, pulled);
        assert_eq!(sync(&mut mesh, &mut asm), vec![part]);
        // In the part: the pull, scaled back to the part's size, where the part is posed.
        let unmoved = |p: DVec3| (p - DVec3::Y * 50.0) / 2.0;
        assert!((asm.get(part).unwrap().mesh.vertex(v) - posed_by(unmoved(pulled))).length() < 1e-9);
        // Undoing the pose: the part where it was sent, with the edit, at its own size.
        asm_history.undo(&mut asm);
        assert_eq!(sync(&mut mesh, &mut asm), vec![part]);
        let p = asm.get(part).unwrap();
        assert!((p.mesh.vertex(v) - unmoved(pulled)).length() < 1e-9, "{}", p.mesh.vertex(v));
        let w = p.mesh.vertex_ids().nth(9).unwrap();
        assert!((p.mesh.vertex(w) - unmoved(mesh.get(id).unwrap().mesh.vertex(w))).length() < 1e-9);
        // Undoing the move in Mesh (its history restores its place of the time) moves no part.
        let mut mesh_history = crate::scene::History::default();
        mesh_history.push("Transform", &mesh);
        mesh.get_mut(id).unwrap().mesh_mut().transform(|p| p - DVec3::Z * 70.0);
        assert!(sync(&mut mesh, &mut asm).is_empty());
        mesh_history.undo(&mut mesh);
        assert!(sync(&mut mesh, &mut asm).is_empty());
        assert!((asm.get(part).unwrap().mesh.vertex(v) - unmoved(pulled)).length() < 1e-9);
    }

    #[test]
    fn sending_again_right_after_a_move_starts_the_new_part_where_the_object_is() {
        let (mut mesh, mut asm, id, part) = sent();
        mesh.get_mut(id).unwrap().mesh_mut().transform(|p| p + DVec3::Y * 20.0);
        // No sync between the move and the second send.
        let again = send(&mut mesh, &[id], &mut asm)[0];
        assert!(sync(&mut mesh, &mut asm).is_empty(), "nothing moves");
        assert!(Arc::ptr_eq(&asm.get(again).unwrap().mesh, &mesh.get(id).unwrap().mesh));
        assert!(asm.get(part).unwrap().bounds().center().length() < 1e-9, "the first part where it was sent");
    }

    #[test]
    fn a_part_keeps_its_mesh_when_its_object_is_deleted() {
        let (mut mesh, mut asm, id, part) = sent();
        let before = asm.get(part).unwrap().mesh.clone();
        mesh.remove(id);
        assert!(sync(&mut mesh, &mut asm).is_empty());
        assert!(Arc::ptr_eq(&asm.get(part).unwrap().mesh, &before));
    }

    #[test]
    fn sending_twice_makes_a_second_copy_and_posed_parts_go_back_as_objects() {
        let (mut mesh, mut asm, id, part) = sent();
        let again = send(&mut mesh, &[id], &mut asm)[0];
        assert_ne!(again, part);
        assert_eq!(asm.get(again).unwrap().name, "cube (2)");
        asm.get_mut(again).unwrap().mesh_mut().transform(|p| p + DVec3::Z * 15.0);
        asm.get_mut(part).unwrap().visible = false;
        let mut mesh2 = mesh.clone();
        let made = send_posed(&asm, &mut mesh2);
        assert_eq!(made.len(), 1, "the visible parts");
        let o = mesh2.get(made[0]).unwrap();
        assert_eq!(o.name, "cube (2) (posed)");
        assert!(o.link.is_none() && Arc::ptr_eq(&o.mesh, &asm.get(again).unwrap().mesh));
    }

    #[test]
    fn affine_maps_round_trip_as_numbers() {
        let a = Affine { m: DMat3::from_cols_array(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]), t: DVec3::new(-1.0, 0.5, 9.0) };
        assert_eq!(Affine::from_array(&a.to_array()), a);
    }
}
