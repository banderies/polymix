// SPDX-License-Identifier: AGPL-3.0-only
//! The components of the body list: named folders the bodies are sorted into, one inside
//! another, so an assembly's parts can be kept apart — a motor's in one, a housing's in another.
//! A component holds its bodies by id and names the component it is in; a body in none stands at
//! the head of the list. They are the list's alone: nothing is built from them, and a body keeps
//! its id wherever it is put. Rows picked in the list — several with Ctrl or Shift — move
//! together. Plain data in and out.
//!
//! Ported from ScanRuler (`plugins/reverse/core/bodyFolders.ts`), © Stefan Hermann
//! (CNC Kitchen), AGPL-3.0-only. The folders themselves are `model::BodyFolder`.

use std::collections::HashSet;

use crate::model::{BodyFolder, BodyId};

/// What is moved between components: bodies, components, or both.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FolderMove {
    pub bodies: Vec<BodyId>,
    pub folders: Vec<u32>,
}

/// Rows picked in the body list: bodies and components, by id. The same shape as a move.
pub type ListPick = FolderMove;

impl FolderMove {
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty() && self.folders.is_empty()
    }

    pub fn len(&self) -> usize {
        self.bodies.len() + self.folders.len()
    }
}

/// A pick as it moves: a component inside another picked one, or a body in one, goes along with
/// it rather than on its own.
pub fn outermost(folders: &[BodyFolder], pick: &ListPick) -> ListPick {
    let mut inside: HashSet<u32> = HashSet::new();
    for id in &pick.folders {
        for f in folders_within(folders, *id) {
            if f != *id {
                inside.insert(f);
            }
        }
    }
    let kept: Vec<u32> = pick.folders.iter().copied().filter(|id| !inside.contains(id)).collect();
    let carried: HashSet<BodyId> = kept.iter().flat_map(|id| bodies_within(folders, *id)).collect();
    FolderMove { bodies: pick.bodies.iter().copied().filter(|b| !carried.contains(b)).collect(), folders: kept }
}

/// Every body a pick holds: its own, and every body in its components at any depth — in the
/// order they come.
pub fn bodies_of_pick(folders: &[BodyFolder], pick: &ListPick) -> Vec<BodyId> {
    let mut out: Vec<BodyId> = Vec::new();
    let mut add = |b: BodyId| {
        if !out.contains(&b) {
            out.push(b);
        }
    };
    for b in &pick.bodies {
        add(*b);
    }
    for id in &pick.folders {
        for b in bodies_within_ordered(folders, *id) {
            add(b);
        }
    }
    out
}

/// The component a body is in, or `None`.
pub fn folder_of(folders: &[BodyFolder], body: BodyId) -> Option<&BodyFolder> {
    folders.iter().find(|f| f.bodies.contains(&body))
}

/// The components from the head of the list down to this one, it last; empty for none.
pub fn folder_path(folders: &[BodyFolder], id: Option<u32>) -> Vec<&BodyFolder> {
    let mut path: Vec<&BodyFolder> = Vec::new();
    let mut seen: HashSet<u32> = HashSet::new();
    let mut at = id.and_then(|id| folders.iter().find(|f| f.id == id));
    while let Some(f) = at {
        if !seen.insert(f.id) {
            break;
        }
        path.insert(0, f);
        at = f.parent.and_then(|p| folders.iter().find(|g| g.id == p));
    }
    path
}

/// The components from the head of the list down to this one, by name: "Housing › Motor".
pub fn folder_path_text(folders: &[BodyFolder], id: Option<u32>) -> String {
    folder_path(folders, id).iter().map(|f| f.name.as_str()).collect::<Vec<_>>().join(" › ")
}

/// A component and every one inside it, at any depth.
pub fn folders_within(folders: &[BodyFolder], id: u32) -> HashSet<u32> {
    let mut out: HashSet<u32> = HashSet::from([id]);
    let mut grew = true;
    while grew {
        grew = false;
        for f in folders {
            if let Some(p) = f.parent {
                if out.contains(&p) && !out.contains(&f.id) {
                    out.insert(f.id);
                    grew = true;
                }
            }
        }
    }
    out
}

/// Every body in a component and in those inside it.
pub fn bodies_within(folders: &[BodyFolder], id: u32) -> HashSet<BodyId> {
    bodies_within_ordered(folders, id).into_iter().collect()
}

fn bodies_within_ordered(folders: &[BodyFolder], id: u32) -> Vec<BodyId> {
    let within = folders_within(folders, id);
    let mut out: Vec<BodyId> = Vec::new();
    for f in folders.iter().filter(|f| within.contains(&f.id)) {
        for b in &f.bodies {
            if !out.contains(b) {
                out.push(*b);
            }
        }
    }
    out
}

/// The name a new component takes: Component n, one past the highest n.
pub fn next_folder_name(folders: &[BodyFolder]) -> String {
    let n = folders.iter().filter_map(|f| f.name.strip_prefix("Component ").and_then(|t| if !t.is_empty() && t.chars().all(|c| c.is_ascii_digit()) { t.parse::<u64>().ok() } else { None })).max().unwrap_or(0);
    format!("Component {}", n + 1)
}

pub fn next_folder_id(folders: &[BodyFolder]) -> u32 {
    folders.iter().map(|f| f.id).max().unwrap_or(0) + 1
}

/// The component the things moved share, or `None` when they stand in different places — or
/// all at the head of the list: where a component made of them goes.
pub fn common_folder(folders: &[BodyFolder], what: &FolderMove) -> Option<u32> {
    let places: Vec<Option<u32>> = what.bodies.iter().map(|b| folder_of(folders, *b).map(|f| f.id)).chain(what.folders.iter().map(|id| folders.iter().find(|f| f.id == *id).and_then(|f| f.parent))).collect();
    match places.first() {
        Some(first) if places.iter().all(|p| p == first) => *first,
        _ => None,
    }
}

/// Bodies and components put into a component — or, with `None`, at the head of the list — out
/// of wherever they were. The same list when nothing moves (compare it with what was given);
/// `None` when it cannot be: into a component that is not there, or a component into itself or
/// into one inside it.
pub fn move_into(folders: &[BodyFolder], what: &FolderMove, target: Option<u32>) -> Option<Vec<BodyFolder>> {
    if let Some(t) = target {
        if !folders.iter().any(|f| f.id == t) {
            return None;
        }
        for id in &what.folders {
            if folders_within(folders, *id).contains(&t) {
                return None;
            }
        }
    }
    let moved: HashSet<u32> = what.folders.iter().copied().collect();
    let bodies: Vec<BodyId> = {
        let mut seen: Vec<BodyId> = Vec::new();
        for b in &what.bodies {
            if !seen.contains(b) {
                seen.push(*b);
            }
        }
        seen
    };
    Some(
        folders
            .iter()
            .map(|f| {
                let mut out = f.clone();
                if moved.contains(&f.id) && f.parent != target {
                    out.parent = target;
                }
                let is_target = Some(f.id) == target;
                let mut kept: Vec<BodyId> = out.bodies.iter().copied().filter(|b| !bodies.contains(b) || is_target).collect();
                if is_target {
                    for b in &bodies {
                        if !kept.contains(b) {
                            kept.push(*b);
                        }
                    }
                }
                out.bodies = kept;
                out
            })
            .collect(),
    )
}

/// A component taken away: the bodies and components it held go up to where it stood.
pub fn without_folder(folders: &[BodyFolder], id: u32) -> Vec<BodyFolder> {
    let Some(gone) = folders.iter().find(|f| f.id == id).cloned() else { return folders.to_vec() };
    folders
        .iter()
        .filter(|f| f.id != id)
        .map(|f| {
            let mut f = f.clone();
            if f.parent == Some(id) {
                f.parent = gone.parent;
            }
            if Some(f.id) == gone.parent {
                for b in &gone.bodies {
                    if !f.bodies.contains(b) {
                        f.bodies.push(*b);
                    }
                }
            }
            f
        })
        .collect()
}

/// The components as a file holds them, checked: ids each once, a parent among them (or none),
/// and no ring of parents. The message names what is wrong.
pub fn validate_folders(folders: &[BodyFolder]) -> Result<(), String> {
    let mut ids: HashSet<u32> = HashSet::new();
    for f in folders {
        if !ids.insert(f.id) {
            return Err(format!("reverse.bodyFolders: the component id {} is used twice", f.id));
        }
    }
    for f in folders {
        if let Some(p) = f.parent {
            if !ids.contains(&p) {
                return Err(format!("reverse.bodyFolders: {} is in a component that is not there ({p})", f.name));
            }
            let path = folder_path(folders, Some(f.id));
            if path.first().is_none_or(|head| head.parent.is_some()) {
                return Err(format!("reverse.bodyFolders: {} is in a ring of components", f.name));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(n: u32) -> BodyId {
        BodyId::new(n)
    }

    fn folder(id: u32, name: &str, parent: Option<u32>, bodies: &[u32]) -> BodyFolder {
        BodyFolder { id, name: name.into(), parent, bodies: bodies.iter().map(|n| b(*n)).collect() }
    }

    /// Housing › Motor, and Fasteners beside Housing.
    fn tree() -> Vec<BodyFolder> {
        vec![folder(1, "Housing", None, &[10]), folder(2, "Motor", Some(1), &[11, 12]), folder(3, "Fasteners", None, &[13])]
    }

    fn sorted<T: Ord + Clone>(s: impl IntoIterator<Item = T>) -> Vec<T> {
        let mut v: Vec<T> = s.into_iter().collect();
        v.sort();
        v
    }

    fn pick(bodies: &[u32], folders: &[u32]) -> ListPick {
        FolderMove { bodies: bodies.iter().map(|n| b(*n)).collect(), folders: folders.to_vec() }
    }

    #[test]
    fn finds_where_a_body_is_the_path_and_what_a_component_holds() {
        let t = tree();
        assert_eq!(folder_of(&t, b(12)).map(|f| f.name.as_str()), Some("Motor"));
        assert!(folder_of(&t, b(99)).is_none());
        assert_eq!(folder_path(&t, Some(2)).iter().map(|f| f.name.as_str()).collect::<Vec<_>>(), vec!["Housing", "Motor"]);
        assert!(folder_path(&t, None).is_empty());
        assert_eq!(folder_path_text(&t, Some(2)), "Housing › Motor");
        assert_eq!(sorted(folders_within(&t, 1)), vec![1, 2]);
        assert_eq!(sorted(bodies_within(&t, 1)), vec![b(10), b(11), b(12)]);
        assert_eq!(next_folder_name(&t), "Component 1");
        let mut more = t.clone();
        more.push(folder(4, "Component 7", None, &[]));
        assert_eq!(next_folder_name(&more), "Component 8");
    }

    #[test]
    fn moves_bodies_and_components_and_refuses_a_ring() {
        let t = tree();
        // A body out of Motor into Fasteners: in one, out of the other.
        let moved = move_into(&t, &pick(&[11], &[]), Some(3)).expect("a move");
        assert_eq!(moved[1].bodies, vec![b(12)]);
        assert_eq!(moved[2].bodies, vec![b(13), b(11)]);
        // To the head of the list: in none.
        assert!(folder_of(&move_into(&t, &pick(&[10], &[]), None).expect("a move"), b(10)).is_none());
        // Motor beside Housing.
        assert_eq!(move_into(&t, &pick(&[], &[2]), None).expect("a move")[1].parent, None);
        // Housing into Motor, which is inside it — or into itself: refused.
        assert!(move_into(&t, &pick(&[], &[1]), Some(2)).is_none());
        assert!(move_into(&t, &pick(&[], &[1]), Some(1)).is_none());
        assert!(move_into(&t, &pick(&[10], &[]), Some(42)).is_none());
        // Where it is already: nothing moves, the same list comes back.
        assert_eq!(move_into(&t, &pick(&[11], &[]), Some(2)).expect("a move"), t);
        assert_eq!(move_into(&t, &pick(&[], &[2]), Some(1)).expect("a move"), t);
    }

    #[test]
    fn takes_a_component_away_with_what_it_held_moved_up() {
        let t = tree();
        let left = without_folder(&t, 1);
        assert_eq!(left.iter().map(|f| f.id).collect::<Vec<_>>(), vec![2, 3]);
        // Motor stands at the head now; Housing's own body is in none.
        assert_eq!(left[0].parent, None);
        assert!(folder_of(&left, b(10)).is_none());
        // Motor taken away: its bodies go into Housing.
        assert_eq!(without_folder(&t, 2)[0].bodies, vec![b(10), b(11), b(12)]);
    }

    #[test]
    fn moves_rows_picked_together_as_their_outermost() {
        let t = tree();
        // Housing picked with Motor inside it and a body of each: Housing alone, and Fasteners'
        // body, which no picked component holds.
        assert_eq!(outermost(&t, &pick(&[10, 11, 13], &[2, 1])), pick(&[13], &[1]));
        assert_eq!(outermost(&t, &pick(&[11, 13], &[])), pick(&[11, 13], &[]));
        assert_eq!(sorted(bodies_of_pick(&t, &pick(&[13], &[2]))), vec![b(11), b(12), b(13)]);
        // Several picked rows dragged onto a component go in together.
        let moved = move_into(&t, &outermost(&t, &pick(&[10, 13], &[])), Some(2)).expect("a move");
        assert_eq!(moved[1].bodies, vec![b(11), b(12), b(10), b(13)]);
        assert!(moved[2].bodies.is_empty());
    }

    #[test]
    fn a_component_made_of_bodies_goes_where_they_all_stand() {
        let t = tree();
        assert_eq!(common_folder(&t, &pick(&[11, 12], &[])), Some(2));
        assert_eq!(common_folder(&t, &pick(&[11, 13], &[])), None);
        assert_eq!(common_folder(&t, &pick(&[99, 98], &[])), None);
    }

    #[test]
    fn reads_a_files_components_refusing_a_missing_parent_or_a_ring() {
        assert!(validate_folders(&tree()).is_ok());
        assert!(validate_folders(&[folder(1, "A", Some(5), &[])]).is_err());
        assert!(validate_folders(&[folder(1, "A", Some(2), &[]), folder(2, "B", Some(1), &[])]).is_err());
        assert!(validate_folders(&[folder(1, "A", Some(1), &[])]).is_err());
        assert!(validate_folders(&[folder(1, "A", None, &[]), folder(1, "B", None, &[])]).is_err());
        // A file's components with a body id that is no number are refused by the part's reading.
        let text = r#"{"version":2,"features":[],"nextId":1,"bodyFolders":[{"id":1,"name":"A","parent":null,"bodies":["x"]}]}"#;
        assert!(crate::model::Part::from_json(text).is_err());
    }

    #[test]
    fn keeps_the_components_in_the_part_file() {
        let mut part = crate::model::Part::default();
        part.body_folders = vec![folder(1, "Component 1", None, &[3]), folder(2, "Component 2", Some(1), &[1])];
        let back = crate::model::Part::from_json(&part.to_json()).expect("read back");
        assert_eq!(back.body_folders, part.body_folders);
        // A ring in the file: refused.
        let mut bad = crate::model::Part::default();
        bad.body_folders = vec![folder(1, "A", Some(1), &[])];
        let err = crate::model::Part::from_json(&bad.to_json()).expect_err("a ring");
        assert!(err.contains("bodyFolders"), "{err}");
    }
}
