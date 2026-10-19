//! The Mesh workspace's agent tools. Reading: the scene, one object's statistics, the
//! selection, the open tool and the tool catalog. Changing: selection, objects, edits, undo,
//! view, tools and files, each turned into [`MeshCommand`]s run by
//! [`MeshWorkspace::execute`] like the menus' and hotkeys'.
//!
//! The UI thread only snapshots (meshes are `Arc`s); counting, bounds and topology run later on
//! a worker thread, so a multi-million-triangle scene never stalls a frame. Lengths are in
//! millimetres, as everywhere in Polymix.

use super::command::{boundary_loops, MeshCommand, Shape, STANDARD_VIEWS, TOOLS};
use super::{ColorMode, ExportScope, MeshWorkspace, NormalMode, UpAxis};
use crate::commands::SelectCmd;
use crate::scene::{ObjectId, Scene, SceneObject};
use glam::DVec3;
use mm_core::{measure, topology, Mesh, TriSelection};
use polymix_workspace::agent::{arg_str, arg_usize, round, vec3, AgentReply, AgentResult, AgentTool, ToolKind};
use serde_json::{json, Value};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

fn point(description: &str) -> Value {
    json!({"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3, "description": description})
}

fn choice(values: &[&str], description: &str) -> Value {
    json!({"type": "string", "enum": values, "description": description})
}

const SELECT_ACTIONS: [(&str, SelectCmd); 10] = [
    ("all", SelectCmd::All),
    ("clear", SelectCmd::Clear),
    ("invert", SelectCmd::Invert),
    ("grow", SelectCmd::Expand),
    ("shrink", SelectCmd::Contract),
    ("connected", SelectCmd::Connected),
    ("invert_connected", SelectCmd::InvertConnected),
    ("grow_to_groups", SelectCmd::ExpandToGroups),
    ("same_groups", SelectCmd::ExpandToSameGroups),
    ("optimize_border", SelectCmd::OptimizeBoundary),
];

pub fn tools() -> Vec<AgentTool> {
    let object = json!({"object": {"type": ["integer", "string"], "description": "Object id (from mesh_scene) or name; default: the active object"}});
    let mut detail = object.clone();
    detail["limit"] = json!({"type": "integer", "description": "Most open boundaries, shells and face groups to list (default 20, up to 500)"});
    let select_actions: Vec<&str> = SELECT_ACTIONS.iter().map(|(n, _)| *n).chain(["groups", "boundary", "near"]).collect();
    let tool_names: Vec<&str> = TOOLS.iter().map(|t| t.name).collect();
    let views: Vec<String> = ["fit".to_string()].into_iter().chain(STANDARD_VIEWS.iter().map(|(n, _, _)| n.to_lowercase())).chain(["orbit".to_string()]).collect();
    let views: Vec<&str> = views.iter().map(String::as_str).collect();
    let normals: Vec<&str> = NormalMode::ALL.iter().map(|m| m.key()).collect();
    let colors: Vec<&str> = ColorMode::ALL.iter().map(|m| m.key()).collect();
    let shapes: Vec<String> = Shape::ALL.iter().map(|s| s.label().to_lowercase()).collect();
    let shapes: Vec<&str> = shapes.iter().map(String::as_str).collect();
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
            object.clone(),
        ),
        AgentTool::new(
            "mesh_tool_state",
            "The tools open in the Mesh workspace and the top one's state: whether it is computing, its settings (operation tools such as Remesh, Reduce, Smooth, Make Solid) and what its panel reports (result, errors). Interactive tools have no settings here: read their panel with app_ui.",
            json!({"schema": {"type": "boolean", "description": "Include the JSON Schema of the settings, with each setting's meaning and range"}}),
        ),
        AgentTool::new(
            "mesh_tools",
            "The Mesh tools that mesh_tool can open: name, title and purpose; with `tool`, that tool's settings schema (operation tools only). Tools worked by clicking say so.",
            json!({"tool": choice(&tool_names, "One tool to describe in full")}),
        ),
        AgentTool::new(
            "mesh_select",
            "Change the selection of the active object (or `object`, which becomes active): all, clear, invert, grow, shrink, connected (expand to the connected parts), invert_connected, grow_to_groups, same_groups, optimize_border; groups (`groups`: face group numbers), boundary (`boundary`: an open boundary's number from mesh_object, selecting the triangles along that hole's rim), near (`at` and `radius`: triangles within the radius of a point). Operation tools act on the selection when there is one. Undoable.",
            json!({
                "action": choice(&select_actions, "What to do"),
                "object": object["object"],
                "groups": {"type": "array", "items": {"type": "integer"}, "description": "Face group numbers (action groups)"},
                "boundary": {"type": "integer", "description": "Open boundary number from mesh_object (action boundary)"},
                "at": point("Point in mm (action near)"),
                "radius": {"type": "number", "description": "Radius in mm (action near)"},
                "add": {"type": "boolean", "description": "Add to the selection instead of replacing it (groups, boundary, near)"}
            }),
        )
        .required(&["action"])
        .kind(ToolKind::Change),
        AgentTool::new(
            "mesh_objects",
            "Activate, show, hide or rename an object, or set / clear the target surface (used by Attract / Align To Target and Deviation).",
            json!({
                "action": choice(&["activate", "show", "hide", "rename", "set_target", "clear_target"], "What to do"),
                "object": object["object"],
                "name": {"type": "string", "description": "New name (action rename)"}
            }),
        )
        .required(&["action"])
        .kind(ToolKind::Change),
        AgentTool::new(
            "mesh_edit",
            "Edit the scene: discard_selection (delete the selected triangles), separate_selection (the selection becomes a new object; `keep_original` keeps it in the old one too), flip_normals (of the selection), new_face_group (the selection becomes a face group), duplicate, delete_object, separate_shells (one object per shell), combine_visible, add_shape (`shape`), clear_scene. Acts on the active object or `object`. Undoable (mesh_history).",
            json!({
                "action": choice(&["discard_selection", "separate_selection", "flip_normals", "new_face_group", "duplicate", "delete_object", "separate_shells", "combine_visible", "add_shape", "clear_scene"], "What to do"),
                "object": object["object"],
                "keep_original": {"type": "boolean", "description": "separate_selection: keep the triangles in the original object too"},
                "shape": choice(&shapes, "Shape to add (action add_shape)")
            }),
        )
        .required(&["action"])
        .kind(ToolKind::Destructive),
        AgentTool::new(
            "mesh_history",
            "Undo or redo steps (yours are labelled \"Agent: …\").",
            json!({
                "action": choice(&["undo", "redo"], "Undo or redo"),
                "steps": {"type": "integer", "description": "How many steps (default 1, up to 50)"}
            }),
        )
        .required(&["action"])
        .kind(ToolKind::Change),
        AgentTool::new(
            "mesh_view",
            "Change the Mesh view (for screenshots): `view` fit (frame the visible objects), front, back, left, right, top, bottom, or orbit (`yaw` and `pitch` in degrees); and display options. Give any of them.",
            json!({
                "view": choice(&views, "Where to look from"),
                "yaw": {"type": "number", "description": "Degrees around the up axis (view orbit; 0 = front)"},
                "pitch": {"type": "number", "description": "Degrees up (view orbit; 90 = top)"},
                "wireframe": {"type": "boolean"},
                "normals": choice(&normals, "Shading normals"),
                "colors": choice(&colors, "What colours the surface (group = face groups)"),
                "boundaries": {"type": "boolean", "description": "Highlight open boundaries"},
                "grid": {"type": "boolean"},
                "orthographic": {"type": "boolean"},
                "up": choice(&["y", "z"], "Up axis (z for medical / RAS data)")
            }),
        )
        .kind(ToolKind::Change),
        AgentTool::new(
            "mesh_tool",
            "Work a Mesh tool. open (`tool` from mesh_tools, optionally `settings`; the open tool is applied first, as in the app), set (`settings`: only the keys to change, see mesh_tool_state / mesh_tools), accept (apply the result and close; a tool still computing applies when done), cancel (close without applying), click (`at`: the surface nearest that point is clicked as a user would, for tools worked by clicking; `double`, `shift`). Operation tools preview live and act on the selection if there is one. Call app_wait after open, set, accept or click, then read the result with mesh_tool_state, mesh_scene or app_screenshot. Accepted results are undoable.",
            json!({
                "action": choice(&["open", "set", "accept", "cancel", "click"], "What to do"),
                "tool": choice(&tool_names, "Tool to open (action open)"),
                "object": object["object"],
                "settings": {"type": "object", "description": "Settings to change (actions open and set)"},
                "at": point("Point in mm on or near the surface (action click)"),
                "double": {"type": "boolean", "description": "Double-click (action click)"},
                "shift": {"type": "boolean", "description": "Hold Shift (action click)"}
            }),
        )
        .required(&["action"])
        .kind(ToolKind::Destructive),
        AgentTool::new(
            "mesh_file",
            "Files: open (`paths`: meshes, appended to the scene without prompts, or a .polymix project, which replaces it), export (`path`; format from the extension: .stl, .obj, .ply, .off, .3mf, .amf, .dae, .wrl, .smesh; `scope` active, visible or selection), save_project (`path` ending in .polymix). Absolute paths only; existing files are kept unless `overwrite`. Runs in the background: app_wait, then app_status or mesh_scene.",
            json!({
                "action": choice(&["open", "export", "save_project"], "What to do"),
                "paths": {"type": "array", "items": {"type": "string"}, "description": "Absolute paths (action open)"},
                "path": {"type": "string", "description": "Absolute path (actions export, save_project)"},
                "scope": choice(&["active", "visible", "selection"], "What to export (default active)"),
                "overwrite": {"type": "boolean", "description": "Replace an existing file"}
            }),
        )
        .required(&["action"])
        .kind(ToolKind::Destructive),
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

pub fn call(ws: &mut MeshWorkspace, tool: &str, args: &Value, ctx: &egui::Context) -> AgentReply {
    match tool {
        "mesh_scene" => scene(ws),
        "mesh_tool_state" => AgentReply::Now(Ok(json!({"open_tools": ws.tools.iter().map(|t| t.name()).collect::<Vec<_>>(), "tool": current_tool(ws, args.get("schema").and_then(Value::as_bool).unwrap_or(false))}))),
        "mesh_tools" => AgentReply::Now(catalog(arg_str(args, "tool"))),
        "mesh_select" | "mesh_objects" | "mesh_edit" | "mesh_history" | "mesh_view" | "mesh_tool" | "mesh_file" => AgentReply::Now(act(ws, tool, args, ctx)),
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

/// The top tool's state for agents.
fn current_tool(ws: &MeshWorkspace, schema: bool) -> Value {
    let Some(t) = ws.tools.last() else { return Value::Null };
    let mut v = json!({
        "name": t.name(),
        "computing": t.busy().map(|(what, p)| json!({"what": what, "progress": round(p as f64, 2)})),
        "settings": t.settings(),
        "report": t.report(),
    });
    if schema {
        v["settings_schema"] = json!(t.settings_schema());
    }
    v
}

fn catalog(tool: Option<&str>) -> AgentResult {
    match tool {
        Some(name) => {
            let t = TOOLS.iter().find(|t| t.name == name).ok_or_else(|| format!("No tool {name} (see mesh_tools)"))?;
            Ok(json!({"name": t.name, "title": t.title, "about": t.about, "settings_schema": crate::tools::ops::settings_schema(t.name)}))
        }
        None => Ok(json!({"tools": TOOLS.iter().map(|t| json!({"name": t.name, "title": t.title, "about": t.about, "typed_settings": crate::tools::ops::settings_schema(t.name).is_some()})).collect::<Vec<_>>()})),
    }
}

/// Run a changing tool's commands (as the agent: undo steps say so).
fn act(ws: &mut MeshWorkspace, tool: &str, args: &Value, ctx: &egui::Context) -> AgentResult {
    let commands = parse(ws, tool, args)?;
    let vp = ws.vp_prev(ctx);
    ws.history.by_agent = true;
    ws.agent_quiet_frames = 0;
    if tool == "mesh_file" && arg_str(args, "action") == Some("open") {
        ws.agent_import = true;
    }
    let mut done: Vec<String> = Vec::new();
    for c in commands {
        match ws.execute(c, ctx, vp) {
            Ok(m) => done.push(m),
            Err(e) => {
                ws.status = format!("Agent: {e}");
                return Err(if done.is_empty() { e } else { format!("{}; then: {e}", done.join("; ")) });
            }
        }
    }
    let msg = done.join("; ");
    ws.status = format!("Agent: {msg}");
    ctx.request_repaint();
    let mut out = json!({"done": msg});
    match tool {
        "mesh_select" => out["selected_triangles"] = json!(ws.scene.active().map(|o| o.selection.len())),
        "mesh_tool" => out["tool"] = current_tool(ws, false),
        _ => {}
    }
    if !ws.agent_quiet() {
        out["pending"] = json!("Still working (computing, clicking or loading): call app_wait, then read the result.");
    }
    Ok(out)
}

/// The commands for a changing tool's arguments.
fn parse(ws: &MeshWorkspace, tool: &str, args: &Value) -> Result<Vec<MeshCommand>, String> {
    use MeshCommand as C;
    let action = arg_str(args, "action");
    let flag = |k: &str| args.get(k).and_then(Value::as_bool);
    let number = |k: &str| args.get(k).and_then(Value::as_f64);
    let need = |k: &str| -> Result<&Value, String> { args.get(k).filter(|v| !v.is_null()).ok_or_else(|| format!("{} needs `{k}`", action.unwrap_or(tool))) };
    // `object` given: that object (it becomes active first where the command acts on the active one).
    let named = || -> Result<Option<ObjectId>, String> { if args.get("object").is_some_and(|v| !v.is_null()) { find(&ws.scene, args).map(|o| Some(o.id)) } else { Ok(None) } };
    let object = || -> Result<ObjectId, String> { find(&ws.scene, args).map(|o| o.id) };
    let activate = || -> Result<Vec<MeshCommand>, String> { Ok(named()?.map(C::Activate).into_iter().collect()) };
    let unknown = || Err(format!("{tool}: unknown action {:?}", action.unwrap_or("")));
    let path = |k: &str| -> Result<PathBuf, String> {
        let p = PathBuf::from(need(k)?.as_str().ok_or_else(|| format!("`{k}` must be a path"))?);
        if !p.is_absolute() {
            return Err(format!("{} is not an absolute path", p.display()));
        }
        if p.exists() && !flag("overwrite").unwrap_or(false) {
            return Err(format!("{} exists: pass overwrite: true to replace it", p.display()));
        }
        Ok(p)
    };
    let add = flag("add").unwrap_or(false);
    Ok(match tool {
        "mesh_select" => {
            let mut v = activate()?;
            v.push(match action {
                Some("groups") => C::SelectGroups { groups: need("groups")?.as_array().ok_or("`groups` must be a list of numbers")?.iter().filter_map(|g| g.as_i64()).map(|g| g as i32).collect(), add },
                Some("boundary") => C::SelectBoundary { boundary: need("boundary")?.as_u64().ok_or("`boundary` must be a number")? as usize, add },
                Some("near") => C::SelectNear { at: vec3_arg(need("at")?)?, radius: number("radius").filter(|r| *r > 0.0).ok_or("near needs a positive `radius` (mm)")?, add },
                Some(a) => match SELECT_ACTIONS.iter().find(|(n, _)| *n == a) {
                    Some((_, c)) => C::Select(*c),
                    None => return unknown(),
                },
                None => return unknown(),
            });
            v
        }
        "mesh_objects" => match action {
            Some("activate") => vec![C::Activate(object()?)],
            Some("show") => vec![C::SetVisible(object()?, true)],
            Some("hide") => vec![C::SetVisible(object()?, false)],
            Some("rename") => vec![C::Rename(object()?, need("name")?.as_str().unwrap_or_default().to_string())],
            Some("set_target") => vec![C::SetTarget(Some(object()?))],
            Some("clear_target") => vec![C::SetTarget(None)],
            _ => return unknown(),
        },
        "mesh_edit" => {
            let on_active = |c: MeshCommand| -> Result<Vec<MeshCommand>, String> {
                let mut v = activate()?;
                v.push(c);
                Ok(v)
            };
            match action {
                Some("discard_selection") => on_active(C::Discard)?,
                Some("separate_selection") => on_active(C::Separate { keep_original: flag("keep_original").unwrap_or(false) })?,
                Some("flip_normals") => on_active(C::FlipNormals)?,
                Some("new_face_group") => on_active(C::CreateFaceGroup)?,
                Some("duplicate") => on_active(C::Duplicate)?,
                Some("separate_shells") => on_active(C::SeparateShells)?,
                Some("delete_object") => vec![C::Delete(named()?.ok_or("delete_object needs `object`")?)],
                Some("combine_visible") => vec![C::CombineVisible],
                Some("clear_scene") => vec![C::ClearScene],
                Some("add_shape") => {
                    let name = need("shape")?.as_str().unwrap_or_default();
                    vec![C::AddShape(*Shape::ALL.iter().find(|s| s.label().eq_ignore_ascii_case(name)).ok_or_else(|| format!("No shape {name}"))?)]
                }
                _ => return unknown(),
            }
        }
        "mesh_history" => {
            let c = match action {
                Some("undo") => C::Undo,
                Some("redo") => C::Redo,
                _ => return unknown(),
            };
            vec![c; arg_usize(args, "steps", 1, 1, 50)]
        }
        "mesh_view" => {
            let mut v = Vec::new();
            match arg_str(args, "view") {
                None => {}
                Some("fit") => v.push(C::Focus),
                Some("orbit") => v.push(C::Orbit { yaw_deg: number("yaw").unwrap_or(0.0), pitch_deg: number("pitch").unwrap_or(0.0) }),
                Some(name) => v.push(C::StandardView(STANDARD_VIEWS.iter().position(|(n, _, _)| n.eq_ignore_ascii_case(name)).ok_or_else(|| format!("No view {name}"))?)),
            }
            if let Some(b) = flag("wireframe") {
                v.push(C::Wireframe(b));
            }
            if let Some(k) = arg_str(args, "normals") {
                v.push(C::Normals(NormalMode::from_key(k).ok_or_else(|| format!("No normals mode {k}"))?));
            }
            if let Some(k) = arg_str(args, "colors") {
                v.push(C::Colors(ColorMode::from_key(k).ok_or_else(|| format!("No colour mode {k}"))?));
            }
            if let Some(b) = flag("boundaries") {
                v.push(C::Boundaries(b));
            }
            if let Some(b) = flag("grid") {
                v.push(C::Grid(b));
            }
            if let Some(b) = flag("orthographic") {
                v.push(C::Orthographic(b));
            }
            match arg_str(args, "up") {
                Some("z") => v.push(C::Up(UpAxis::Z)),
                Some("y") => v.push(C::Up(UpAxis::Y)),
                Some(o) => return Err(format!("up must be y or z, not {o}")),
                None => {}
            }
            if v.is_empty() {
                return Err("mesh_view: give `view` or a display option".into());
            }
            v
        }
        "mesh_tool" => match action {
            Some("open") => {
                let name = need("tool")?.as_str().unwrap_or_default();
                if !TOOLS.iter().any(|t| t.name == name) {
                    return Err(format!("No tool {name} (see mesh_tools)"));
                }
                let mut v = activate()?;
                v.push(C::OpenTool(name.to_string()));
                if let Some(s) = args.get("settings").filter(|s| !s.is_null()) {
                    v.push(C::ToolSettings(s.clone()));
                }
                v
            }
            Some("set") => vec![C::ToolSettings(need("settings")?.clone())],
            Some("accept") => vec![C::AcceptTool],
            Some("cancel") => vec![C::CancelTool],
            Some("click") => vec![C::Click { at: vec3_arg(need("at")?)?, double: flag("double").unwrap_or(false), shift: flag("shift").unwrap_or(false) }],
            _ => return unknown(),
        },
        "mesh_file" => match action {
            Some("open") => {
                let paths: Vec<PathBuf> = need("paths")?.as_array().ok_or("`paths` must be a list")?.iter().filter_map(Value::as_str).map(PathBuf::from).collect();
                if let Some(p) = paths.iter().find(|p| !p.is_absolute()) {
                    return Err(format!("{} is not an absolute path", p.display()));
                }
                vec![C::Open(paths)]
            }
            Some("export") => {
                let scope = match arg_str(args, "scope").unwrap_or("active") {
                    "active" => ExportScope::Active,
                    "visible" => ExportScope::AllVisible,
                    "selection" => ExportScope::Selection,
                    o => return Err(format!("scope must be active, visible or selection, not {o}")),
                };
                vec![C::Export { path: path("path")?, scope }]
            }
            Some("save_project") => {
                let p = path("path")?;
                if !p.extension().is_some_and(|e| e.eq_ignore_ascii_case(crate::project::EXTENSION)) {
                    return Err(format!("A project file ends in .{}", crate::project::EXTENSION));
                }
                vec![C::SaveProject(p)]
            }
            _ => return unknown(),
        },
        _ => return Err(format!("the Mesh workspace has no tool {tool}")),
    })
}

fn vec3_arg(v: &Value) -> Result<DVec3, String> {
    match v.as_array().map(|a| a.iter().filter_map(Value::as_f64).collect::<Vec<_>>()).as_deref() {
        Some([x, y, z]) => Ok(DVec3::new(*x, *y, *z)),
        _ => Err("a point is [x, y, z] in mm".into()),
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
    // Numbered as mesh_select's `boundary` counts them.
    let mut loops: Vec<Value> = boundary_loops(m).iter().enumerate().map(|(k, l)| json!({"boundary": k, "edges": l.len(), "perimeter_mm": round(l.perimeter(m), 3), "centre_mm": vec3(l.centroid(m).to_array(), 3)})).collect();
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

#[cfg(test)]
mod command_tests {
    use super::*;
    use crate::app::command::tests::workspace;

    #[test]
    fn arguments_become_commands() {
        let (mut ws, ctx, vp) = workspace();
        ws.execute(MeshCommand::AddShape(Shape::Cube), &ctx, vp).unwrap();
        let p = |tool: &str, args: Value| parse(&ws, tool, &args);
        assert_eq!(p("mesh_select", json!({"action": "near", "at": [1, 2, 3], "radius": 2})).unwrap(), vec![MeshCommand::SelectNear { at: DVec3::new(1.0, 2.0, 3.0), radius: 2.0, add: false }]);
        assert_eq!(p("mesh_select", json!({"action": "grow"})).unwrap(), vec![MeshCommand::Select(SelectCmd::Expand)]);
        assert!(p("mesh_select", json!({"action": "near", "at": [1, 2]})).is_err());
        assert!(p("mesh_select", json!({"action": "sideways"})).unwrap_err().contains("unknown action"));
        assert_eq!(p("mesh_history", json!({"action": "undo", "steps": 3})).unwrap().len(), 3);
        assert_eq!(p("mesh_view", json!({"view": "top", "wireframe": true})).unwrap(), vec![MeshCommand::StandardView(4), MeshCommand::Wireframe(true)]);
        assert!(p("mesh_tool", json!({"action": "open", "tool": "teleport"})).unwrap_err().contains("No tool"));
        assert_eq!(p("mesh_tool", json!({"action": "open", "tool": "remesh", "object": "cube", "settings": {"density": 10}})).unwrap().len(), 3);
        assert!(p("mesh_file", json!({"action": "open", "paths": ["relative.stl"]})).unwrap_err().contains("absolute"));
        let existing = std::env::temp_dir();
        let e = p("mesh_file", json!({"action": "export", "path": existing.join("x.stl").display().to_string()}));
        assert!(e.is_ok() || e.unwrap_err().contains("exists"));
        assert!(p("mesh_file", json!({"action": "save_project", "path": existing.join("x.stl").display().to_string(), "overwrite": true})).unwrap_err().contains(".polymix"));
    }

    #[test]
    fn agent_changes_report_and_label() {
        let (mut ws, ctx, _) = workspace();
        let r = act(&mut ws, "mesh_edit", &json!({"action": "add_shape", "shape": "sphere"}), &ctx).unwrap();
        assert!(r["done"].as_str().unwrap().contains("sphere"), "{r}");
        assert_eq!(ws.history.undo_label(), Some("Agent: Add Sphere"));
        assert!(ws.status.starts_with("Agent: "));
        let r = act(&mut ws, "mesh_select", &json!({"action": "all"}), &ctx).unwrap();
        assert!(r["selected_triangles"].as_u64().unwrap() > 0);
        let r = act(&mut ws, "mesh_tool", &json!({"action": "open", "tool": "smooth", "settings": {"amount": 0.5}}), &ctx).unwrap();
        assert_eq!(r["tool"]["name"], "Smooth");
        assert_eq!(r["tool"]["settings"]["amount"], 0.5);
        assert!(act(&mut ws, "mesh_objects", &json!({"action": "rename", "object": 999999, "name": "x"}), &ctx).is_err());
    }
}
