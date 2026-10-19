//! The Mesh workspace's agent tools: the scene, one object's statistics and the selection.
//!
//! The UI thread only snapshots (meshes are `Arc`s); counting, bounds and topology run later on
//! a worker thread, so a multi-million-triangle scene never stalls a frame. Lengths are in
//! millimetres, as everywhere in Polymix.

use super::{MeshWorkspace, UpAxis};
use crate::scene::{ObjectId, Scene, SceneObject};
use mm_core::{measure, topology, Mesh, TriSelection};
use polymix_workspace::agent::{arg_usize, round, vec3, AgentReply, AgentTool};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

pub fn tools() -> Vec<AgentTool> {
    let object = json!({"object": {"type": ["integer", "string"], "description": "Object id (from mesh_scene) or name; default: the active object"}});
    let mut detail = object.clone();
    detail["limit"] = json!({"type": "integer", "description": "Most open boundaries, shells and face groups to list (default 20, up to 500)"});
    vec![
        AgentTool::new(
            "mesh_scene",
            "The Mesh workspace: every object (id, name, visibility, triangle and vertex counts, open edges, bounds in mm, selected triangles, face groups), which is active and which is the target, the open tool and whether it is computing, the status line, recent undo / redo steps, background jobs, camera and display options.",
            json!({}),
        ),
        AgentTool::new(
            "mesh_object",
            "One Mesh object in detail: counts, bounds, surface area (mm²), enclosed volume (mm³) and orientation when closed, edge lengths, mesh validity, shells, open boundaries (holes and vessel ends: edge count, perimeter, centre), defects (bowtie vertices, degenerate triangles, inconsistently oriented edges), face groups and boundary-patch names, pivots.",
            detail,
        ),
        AgentTool::new(
            "mesh_selection",
            "The selected triangles of a Mesh object: count, area, bounds, how many separate regions, the length of their border, and the face groups they cover.",
            object,
        ),
    ]
}

/// An object as the tools read it.
struct Snap {
    id: ObjectId,
    name: String,
    visible: bool,
    mesh: Arc<Mesh>,
    selection: TriSelection,
    group_names: HashMap<i32, String>,
    color: [f32; 4],
    material: usize,
    pivots: Vec<[f64; 3]>,
    uvs: bool,
}

impl Snap {
    fn of(o: &SceneObject) -> Self {
        Self {
            id: o.id,
            name: o.name.clone(),
            visible: o.visible,
            mesh: o.mesh.clone(),
            selection: o.selection.clone(),
            group_names: o.group_names.clone(),
            color: o.color.to_array(),
            material: o.material,
            pivots: o.pivots.iter().map(|p| p.position.to_array()).collect(),
            uvs: o.uvs().is_some(),
        }
    }
}

pub fn call(ws: &mut MeshWorkspace, tool: &str, args: &Value) -> AgentReply {
    match tool {
        "mesh_scene" => scene(ws),
        "mesh_object" | "mesh_selection" => {
            let o = match find(&ws.scene, args) {
                Ok(o) => o,
                Err(e) => return AgentReply::error(e),
            };
            let mut snap = Snap::of(o);
            snap.selection.sync(&snap.mesh);
            let limit = arg_usize(args, "limit", 20, 1, 500);
            if tool == "mesh_object" { AgentReply::later(move || Ok(object(&snap, limit))) } else { AgentReply::later(move || Ok(selection(&snap))) }
        }
        _ => AgentReply::error(format!("the Mesh workspace has no tool {tool}")),
    }
}

/// The object an agent named (by id or name), or the active one.
fn find<'a>(scene: &'a Scene, args: &Value) -> Result<&'a SceneObject, String> {
    if scene.is_empty() {
        return Err("The Mesh workspace has no objects (open a mesh first).".into());
    }
    let listing = || scene.objects.iter().map(|o| format!("{} \"{}\"", o.id, o.name)).collect::<Vec<_>>().join(", ");
    match args.get("object") {
        None | Some(Value::Null) => scene.active().ok_or_else(|| format!("No object is active; name one: {}", listing())),
        Some(Value::Number(n)) => n.as_u64().and_then(|id| scene.get(id)).ok_or_else(|| format!("No object with id {n}; the objects are {}", listing())),
        Some(Value::String(s)) => {
            if let Some(o) = scene.objects.iter().find(|o| o.name == *s) {
                return Ok(o);
            }
            let q = s.to_lowercase();
            let like: Vec<&SceneObject> = scene.objects.iter().filter(|o| o.name.to_lowercase().contains(&q)).collect();
            match like.as_slice() {
                [o] => Ok(o),
                [] => Err(format!("No object named like \"{s}\"; the objects are {}", listing())),
                _ => Err(format!("Several objects are named like \"{s}\"; use an id: {}", listing())),
            }
        }
        Some(other) => Err(format!("object must be an id or a name, not {other}")),
    }
}

fn scene(ws: &MeshWorkspace) -> AgentReply {
    let objects: Vec<Snap> = ws.scene.objects.iter().map(Snap::of).collect();
    let (active, target) = (ws.scene.active, ws.scene.target);
    let (undo, redo) = ws.history.labels();
    let busy = ws.tools.last().and_then(|t| t.busy());
    let mut jobs: Vec<String> = Vec::new();
    jobs.extend(ws.import_job.as_ref().map(|j| format!("{} ({:.0}%)", j.label, j.progress() * 100.0)));
    jobs.extend(ws.export_job.as_ref().map(|j| format!("{} ({:.0}%)", j.label, j.progress() * 100.0)));
    jobs.extend(ws.project_job.as_ref().map(|(_, j)| format!("{} ({:.0}%)", j.label, j.progress() * 100.0)));
    let waiting = if ws.recovery.is_some() {
        Some("the recovery prompt (unsaved work from a session that did not exit cleanly)")
    } else if ws.pending_import.is_some() {
        Some("the import options for the files being opened")
    } else {
        None
    };
    let c = &ws.camera;
    let v = &ws.view;
    let fixed = json!({
        "units": "mm",
        "active_object": active,
        "target_object": target,
        "tool": {
            "stack": ws.tools.iter().map(|t| t.name()).collect::<Vec<_>>(),
            "computing": busy.map(|(what, p)| json!({"what": what, "progress": round(p as f64, 2)})),
        },
        "status": ws.status,
        "history": {"undo": undo.iter().rev().take(15).collect::<Vec<_>>(), "redo": redo.iter().rev().take(15).collect::<Vec<_>>(), "undo_steps": undo.len(), "redo_steps": redo.len()},
        "project_file": ws.project_path.as_ref().and_then(|p| p.file_name()).map(|n| n.to_string_lossy().to_string()),
        "background_jobs": jobs,
        "waiting_for_user": waiting,
        "camera": {
            "orthographic": c.orthographic,
            "target_mm": vec3(c.target.to_array(), 3),
            "eye_mm": vec3(c.eye().to_array(), 3),
            "distance_mm": round(c.distance, 3),
            "up_axis": if c.up_axis == UpAxis::Z { "z" } else { "y" },
        },
        "view": {"wireframe": v.wireframe, "normals": v.normals.key(), "colors": v.colors.key(), "boundaries": v.boundaries, "grid": v.show_grid},
    });
    AgentReply::later(move || {
        let mut out = fixed;
        out["objects"] = objects.iter().map(|o| {
            let m = &o.mesh;
            let groups: std::collections::HashSet<i32> = m.triangle_ids().map(|t| m.tri_group(t)).collect();
            let open = m.boundary_edge_ids().count();
            json!({
                "id": o.id,
                "name": o.name,
                "visible": o.visible,
                "active": Some(o.id) == active,
                "target": Some(o.id) == target,
                "triangles": m.triangle_count(),
                "vertices": m.vertex_count(),
                "open_edges": open,
                "closed": open == 0,
                "bounds_mm": bounds(m),
                "selected_triangles": o.selection.len(),
                "face_groups": groups.len(),
            })
        }).collect();
        Ok(out)
    })
}

fn bounds(m: &Mesh) -> Value {
    let b = m.bounds();
    if m.triangle_count() == 0 {
        return Value::Null;
    }
    json!({"min": vec3(b.min.to_array(), 4), "max": vec3(b.max.to_array(), 4), "size": vec3((b.max - b.min).to_array(), 4)})
}

fn object(o: &Snap, limit: usize) -> Value {
    let m = &o.mesh;
    let stats = measure::stats(m);
    let b = m.bounds();
    let diag = (b.max - b.min).length();
    let defects = topology::find_defects(m, 1e-12 * diag * diag);
    let (mut lo, mut hi) = (f64::INFINITY, 0.0f64);
    for e in m.edge_ids() {
        let l = m.edge_length(e);
        lo = lo.min(l);
        hi = hi.max(l);
    }
    let closed = stats.boundary_edges == 0;
    let mut shells: Vec<(usize, f64)> = defects.components.iter().map(|c| (c.len(), c.iter().map(|&t| m.tri_area(t)).sum())).collect();
    shells.sort_by(|a, b| b.0.cmp(&a.0));
    let mut loops: Vec<Value> = defects.boundary_loops.iter().map(|l| json!({"edges": l.len(), "perimeter_mm": round(l.perimeter(m), 3), "centre_mm": vec3(l.centroid(m).to_array(), 3)})).collect();
    loops.sort_by(|a, b| b["perimeter_mm"].as_f64().partial_cmp(&a["perimeter_mm"].as_f64()).unwrap_or(std::cmp::Ordering::Equal));
    let mut groups: BTreeMap<i32, usize> = BTreeMap::new();
    for t in m.triangle_ids() {
        *groups.entry(m.tri_group(t)).or_default() += 1;
    }
    let mut groups: Vec<(i32, usize)> = groups.into_iter().collect();
    groups.sort_by(|a, b| b.1.cmp(&a.1));
    let [r, g, bl, _] = o.color.map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8);
    json!({
        "id": o.id,
        "name": o.name,
        "visible": o.visible,
        "units": "mm",
        "triangles": stats.triangles,
        "vertices": stats.vertices,
        "edges": stats.edges,
        "bounds_mm": bounds(m),
        "area_mm2": round(stats.area, 4),
        // The signed volume only means something for a closed surface; negative = inside out.
        "volume_mm3": closed.then(|| round(stats.volume.abs(), 4)),
        "orientation": if !closed { "open surface" } else if stats.volume >= 0.0 { "outward" } else { "inward (inside out)" },
        "edge_length_mm": {"min": round(if lo.is_finite() { lo } else { 0.0 }, 5), "mean": round(stats.avg_edge_length, 5), "max": round(hi, 5)},
        "closed": closed,
        "valid": match m.check_validity() { Ok(()) => json!(true), Err(e) => json!(e) },
        "shells": {"count": shells.len(), "largest": shells.iter().take(limit).map(|(n, a)| json!({"triangles": n, "area_mm2": round(*a, 3)})).collect::<Vec<_>>()},
        "open_boundaries": {"count": loops.len(), "open_edges": stats.boundary_edges, "largest": loops.into_iter().take(limit).collect::<Vec<_>>()},
        "defects": {"bowtie_vertices": defects.bowtie_vertices.len(), "degenerate_triangles": defects.degenerate_triangles.len(), "inconsistently_oriented_edges": defects.inconsistent_edges.len()},
        "face_groups": {"count": groups.len(), "largest": groups.iter().take(limit).map(|(g, n)| json!({"group": g, "name": o.group_names.get(g), "triangles": n})).collect::<Vec<_>>()},
        "selected_triangles": o.selection.len(),
        "color": format!("#{r:02x}{g:02x}{bl:02x}"),
        "material": o.material,
        "pivots_mm": o.pivots.iter().map(|p| vec3(*p, 4)).collect::<Vec<_>>(),
        "has_uvs": o.uvs,
    })
}

fn selection(o: &Snap) -> Value {
    let m = &o.mesh;
    let sel = &o.selection;
    let n = sel.len();
    let mut area = 0.0;
    let mut b = mm_core::Aabb::EMPTY;
    let mut groups: BTreeMap<i32, usize> = BTreeMap::new();
    for t in sel.iter() {
        area += m.tri_area(t);
        for p in m.tri_positions(t) {
            b.include(p);
        }
        *groups.entry(m.tri_group(t)).or_default() += 1;
    }
    // Separate regions: edge-connected pieces of the selection.
    let mut seen = vec![false; m.max_tid()];
    let mut regions = 0;
    let mut stack = Vec::new();
    for t in sel.iter() {
        if seen[t as usize] {
            continue;
        }
        regions += 1;
        seen[t as usize] = true;
        stack.push(t);
        while let Some(u) = stack.pop() {
            for w in m.tri_neighbors(u) {
                if w != mm_core::INVALID && !seen[w as usize] && sel.contains(w) {
                    seen[w as usize] = true;
                    stack.push(w);
                }
            }
        }
    }
    let border: f64 = sel.boundary_edges(m).iter().map(|&e| m.edge_length(e)).sum();
    json!({
        "id": o.id,
        "name": o.name,
        "selected_triangles": n,
        "of_triangles": m.triangle_count(),
        "area_mm2": round(area, 4),
        "bounds_mm": (n > 0).then(|| json!({"min": vec3(b.min.to_array(), 4), "max": vec3(b.max.to_array(), 4), "size": vec3((b.max - b.min).to_array(), 4)})),
        "regions": regions,
        "border_length_mm": round(border, 3),
        "face_groups": groups.iter().map(|(g, k)| json!({"group": g, "name": o.group_names.get(g), "triangles": k})).collect::<Vec<_>>(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn scene() -> Scene {
        let mut s = Scene::default();
        s.add("cube", primitives::cube(10.0, 2));
        let mut open = primitives::cube(4.0, 2);
        let top: Vec<u32> = open.triangle_ids().filter(|&t| open.tri_normal(t).z > 0.9).collect();
        open.remove_triangles(top);
        s.add("open box", open);
        s
    }

    #[test]
    fn finds_objects_by_id_and_name() {
        let s = scene();
        let id = s.objects[0].id;
        assert_eq!(find(&s, &json!({})).unwrap().name, "open box", "the last added is active");
        assert_eq!(find(&s, &json!({"object": id})).unwrap().name, "cube");
        assert_eq!(find(&s, &json!({"object": "CUBE"})).unwrap().name, "cube");
        assert!(find(&s, &json!({"object": 9999})).err().unwrap().contains("\"cube\""));
        assert!(find(&Scene::default(), &json!({})).is_err());
    }

    #[test]
    fn object_statistics() {
        let s = scene();
        let cube = object(&Snap::of(&s.objects[0]), 20);
        assert_eq!(cube["closed"], true);
        assert_eq!(cube["orientation"], "outward");
        assert_eq!(cube["volume_mm3"], 1000.0);
        assert_eq!(cube["area_mm2"], 600.0);
        assert_eq!(cube["shells"]["count"], 1);
        assert_eq!(cube["valid"], true);
        assert_eq!(cube["bounds_mm"]["size"], json!([10.0, 10.0, 10.0]));
        let open = object(&Snap::of(&s.objects[1]), 20);
        assert_eq!(open["closed"], false);
        assert_eq!(open["volume_mm3"], Value::Null);
        assert_eq!(open["open_boundaries"]["count"], 1);
        assert_eq!(open["open_boundaries"]["largest"][0]["perimeter_mm"], 16.0);
    }

    #[test]
    fn selection_regions() {
        let s = scene();
        let mut snap = Snap::of(&s.objects[0]);
        let m = snap.mesh.clone();
        // Two opposite faces of the cube: two regions.
        let faces: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_normal(t).x.abs() > 0.9).collect();
        snap.selection = TriSelection::from_tris(&m, faces.iter().copied());
        let r = selection(&snap);
        assert_eq!(r["selected_triangles"], faces.len());
        assert_eq!(r["regions"], 2);
        assert_eq!(r["area_mm2"], 200.0);
        assert_eq!(r["border_length_mm"], 80.0);
    }
}
