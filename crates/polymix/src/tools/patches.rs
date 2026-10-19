//! Boundary patches for CFD: name face groups (wall / inlet / outlets) and export them as a
//! multi-solid ASCII STL (one `solid <name>` per patch, as read by OpenFOAM, Fluent, CFX,
//! SimVascular and others) or as one STL file per patch.

use super::*;
use crate::scene::SceneObject;
use crate::ui_util;
use std::collections::BTreeMap;
use std::io::Write;

pub struct PatchesTool {
    object: Option<ObjectId>,
    status: String,
}

impl PatchesTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, status: String::new() }
    }
}

/// Triangle count and area per face group.
pub fn group_stats(mesh: &Mesh) -> BTreeMap<i32, (usize, f64)> {
    let mut m = BTreeMap::new();
    for t in mesh.triangle_ids() {
        let e = m.entry(mesh.tri_group(t)).or_insert((0usize, 0.0f64));
        e.0 += 1;
        e.1 += mesh.tri_area(t);
    }
    m
}

pub fn patch_name(o: &SceneObject, g: i32) -> String {
    o.group_names.get(&g).cloned().unwrap_or_else(|| if g == 0 { "wall".into() } else { format!("patch_{g}") })
}

fn sanitize(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '_' || c == '-' { c } else { '_' }).collect();
    if s.is_empty() {
        "patch".into()
    } else {
        s
    }
}

/// Write a multi-solid ASCII STL with one solid per face group.
pub fn write_multisolid_stl(path: &std::path::Path, o: &SceneObject) -> anyhow::Result<usize> {
    let mesh = &o.mesh;
    let mut by_group: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for t in mesh.triangle_ids() {
        by_group.entry(mesh.tri_group(t)).or_default().push(t);
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (g, tris) in &by_group {
        let name = sanitize(&patch_name(o, *g));
        writeln!(w, "solid {name}")?;
        for &t in tris {
            let n = mesh.tri_normal(t);
            let [a, b, c] = mesh.tri_positions(t);
            writeln!(w, "  facet normal {:e} {:e} {:e}", n.x, n.y, n.z)?;
            writeln!(w, "    outer loop")?;
            for p in [a, b, c] {
                writeln!(w, "      vertex {:e} {:e} {:e}", p.x, p.y, p.z)?;
            }
            writeln!(w, "    endloop")?;
            writeln!(w, "  endfacet")?;
        }
        writeln!(w, "endsolid {name}")?;
    }
    w.flush()?;
    Ok(by_group.len())
}

/// Write each face group to `<dir>/<name>.stl` (binary).
pub fn write_patch_files(dir: &std::path::Path, o: &SceneObject) -> anyhow::Result<usize> {
    let mesh = &o.mesh;
    let mut by_group: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for t in mesh.triangle_ids() {
        by_group.entry(mesh.tri_group(t)).or_default().push(t);
    }
    for (g, tris) in &by_group {
        let (part, _) = mesh.submesh(tris);
        let name = sanitize(&patch_name(o, *g));
        mm_io::write_mesh(&dir.join(format!("{name}.stl")), &part, &mm_io::WriteOptions::default())?;
    }
    Ok(by_group.len())
}

impl PatchesTool {
    /// Group 0 = wall; among flat caps the largest = inlet, the rest = outlet_1..n by area.
    fn auto_name(o: &mut SceneObject) -> usize {
        let stats = group_stats(&o.mesh);
        let cap_groups: Vec<i32> = crate::tools::vessel::find_vessel_ends(&o.mesh)
            .iter()
            .filter_map(|e| if let crate::tools::vessel::EndKind::Cap(g) = e.kind { Some(g) } else { None })
            .collect();
        let mut caps: Vec<(i32, f64)> = stats.iter().filter(|(g, _)| **g != 0 && cap_groups.contains(g)).map(|(g, (_, a))| (*g, *a)).collect();
        caps.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        o.group_names.insert(0, "wall".into());
        for (k, (g, _)) in caps.iter().enumerate() {
            o.group_names.insert(*g, if k == 0 { "inlet".into() } else { format!("outlet_{k}") });
        }
        caps.len()
    }

    fn select_group(cx: &mut ToolCx, id: ObjectId, g: i32) {
        cx.history.push("Select patch", cx.scene);
        let o = cx.scene.get_mut(id).unwrap();
        let m = o.mesh.clone();
        o.selection.sync(&m);
        o.selection.clear();
        for t in m.triangle_ids().filter(|&t| m.tri_group(t) == g) {
            o.selection.insert(t);
        }
        o.touch_selection();
    }
}

/// An absolute path for an agent's export (existing files kept unless `overwrite`).
fn export_path(args: &serde_json::Value, key: &str) -> Result<std::path::PathBuf, String> {
    let a = Args(args);
    let p = std::path::PathBuf::from(a.str(key)?);
    if !p.is_absolute() {
        return Err(format!("{} is not an absolute path", p.display()));
    }
    if p.is_file() && !a.opt_bool("overwrite").unwrap_or(false) {
        return Err(format!("{} exists: pass overwrite: true to replace it", p.display()));
    }
    Ok(p)
}

impl Tool for PatchesTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        use serde_json::json;
        let Some(o) = self.object.and_then(|id| scene.get(id)) else { return json!({"object": null}) };
        let caps: Vec<i32> = crate::tools::vessel::find_vessel_ends(&o.mesh).iter().filter_map(|e| if let crate::tools::vessel::EndKind::Cap(g) = e.kind { Some(g) } else { None }).collect();
        json!({
            "object": o.id,
            "patches": group_stats(&o.mesh).iter().map(|(g, (n, area))| json!({
                "group": g,
                "name": patch_name(o, *g),
                "named": o.group_names.contains_key(g),
                "triangles": n,
                "area_mm2": polymix_workspace::agent::round(*area, 3),
                "flat_cap": caps.contains(g),
            })).collect::<Vec<_>>(),
            "message": (!self.status.is_empty()).then(|| self.status.clone()),
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new("rename", "Name a patch (face group)", json!({"group": arg::integer("Face group number"), "name": arg::text("New name (inlet, outlet_1, wall, …)")})).required(&["group", "name"]),
            Action::new("select", "Select a patch's triangles", json!({"group": arg::integer("Face group number")})).required(&["group"]),
            Action::plain("auto_name", "Group 0 becomes wall; among flat caps the largest becomes inlet, the rest outlet_1..n by decreasing area"),
            Action::new("export_multisolid", "Write a multi-solid ASCII STL, one solid per patch (OpenFOAM snappyHexMesh, Fluent, CFX, SimVascular)", json!({"path": arg::text("Absolute path ending in .stl"), "overwrite": arg::boolean("Replace an existing file")})).required(&["path"]),
            Action::new("export_per_patch", "Write one binary STL per patch, named after it, into a folder", json!({"folder": arg::text("Absolute path of an existing folder"), "overwrite": arg::boolean("Replace existing files")})).required(&["folder"]),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        let id = self.object.filter(|id| cx.scene.get(*id).is_some()).ok_or("no active object")?;
        let group = || -> Result<i32, String> {
            let g = a.0.get("group").and_then(serde_json::Value::as_i64).ok_or("`group` (a face group number) is needed")? as i32;
            if group_stats(&cx.scene.get(id).unwrap().mesh).contains_key(&g) { Ok(g) } else { Err(format!("the object has no face group {g}")) }
        };
        match action {
            "rename" => {
                let g = group()?;
                let name = a.str("name")?.trim().to_string();
                if name.is_empty() {
                    return Err("a patch name cannot be empty".into());
                }
                cx.scene.get_mut(id).unwrap().group_names.insert(g, name.clone());
                Ok(format!("Group {g} is now {name}"))
            }
            "select" => {
                let g = group()?;
                Self::select_group(cx, id, g);
                Ok(format!("Selected patch {}", patch_name(cx.scene.get(id).unwrap(), g)))
            }
            "auto_name" => {
                let n = Self::auto_name(cx.scene.get_mut(id).unwrap());
                Ok(format!("Named group 0 wall and {n} flat cap(s) inlet / outlet_N"))
            }
            "export_multisolid" => {
                let path = export_path(args, "path")?;
                let n = write_multisolid_stl(&path, cx.scene.get(id).unwrap()).map_err(|e| format!("Export failed: {e:#}"))?;
                self.status = format!("Wrote {n} patches to {}", path.display());
                Ok(self.status.clone())
            }
            "export_per_patch" => {
                let dir = std::path::PathBuf::from(a.str("folder")?);
                if !dir.is_absolute() || !dir.is_dir() {
                    return Err(format!("{} is not an absolute path to a folder", dir.display()));
                }
                let o = cx.scene.get(id).unwrap();
                if !a.opt_bool("overwrite").unwrap_or(false) {
                    if let Some(g) = group_stats(&o.mesh).keys().find(|g| dir.join(format!("{}.stl", sanitize(&patch_name(o, **g)))).exists()) {
                        return Err(format!("{}.stl exists there: pass overwrite: true to replace", sanitize(&patch_name(o, *g))));
                    }
                }
                let n = write_patch_files(&dir, o).map_err(|e| format!("Export failed: {e:#}"))?;
                self.status = format!("Wrote {n} files to {}", dir.display());
                Ok(self.status.clone())
            }
            _ => Err(format!("Boundary patches has no action {action}")),
        }
    }

    fn name(&self) -> &'static str {
        "Boundary patches"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Boundary patches (CFD)");
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.small("Each face group becomes a named boundary patch. Caps made by Trim / Cap / Flow extension get their own groups automatically. Turn on View > Face group colors to see them.");
        let stats = group_stats(&cx.scene.get(id).unwrap().mesh);
        let mut select_group = None;
        let mut renames: Vec<(i32, String)> = Vec::new();
        egui::Grid::new("patches").num_columns(4).striped(true).show(ui, |ui| {
            ui.strong("Name");
            ui.strong("Tris");
            ui.strong("Area");
            ui.label("");
            ui.end_row();
            for (g, (n, area)) in &stats {
                let o = cx.scene.get(id).unwrap();
                let mut name = patch_name(o, *g);
                if ui.add(egui::TextEdit::singleline(&mut name).desired_width(110.0)).changed() {
                    renames.push((*g, name));
                }
                ui.label(ui_util::format_count(*n));
                ui.label(format!("{area:.2}"));
                if ui.small_button("select").clicked() {
                    select_group = Some(*g);
                }
                ui.end_row();
            }
        });
        if !renames.is_empty() {
            let o = cx.scene.get_mut(id).unwrap();
            for (g, n) in renames {
                o.group_names.insert(g, n);
            }
        }
        if let Some(g) = select_group {
            Self::select_group(cx, id, g);
        }
        if ui.button("Auto-name caps (largest = inlet)").on_hover_text("Group 0 = wall; among flat caps the largest = inlet, the rest = outlet_1..n by decreasing area").clicked() {
            Self::auto_name(cx.scene.get_mut(id).unwrap());
        }
        ui_util::section(ui, "Export");
        if ui.button("Multi-solid ASCII STL…").clicked() {
            let o = cx.scene.get(id).unwrap();
            if let Some(path) = rfd::FileDialog::new().add_filter("STL", &["stl"]).set_file_name(format!("{}_patches.stl", o.name)).save_file() {
                self.status = match write_multisolid_stl(&path, o) {
                    Ok(n) => format!("Wrote {n} patches to {}", path.display()),
                    Err(e) => format!("Export failed: {e:#}"),
                };
            }
        }
        if ui.button("One STL per patch…").clicked() {
            let o = cx.scene.get(id).unwrap();
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                self.status = match write_patch_files(&dir, o) {
                    Ok(n) => format!("Wrote {n} files to {}", dir.display()),
                    Err(e) => format!("Export failed: {e:#}"),
                };
            }
        }
        if !self.status.is_empty() {
            ui.label(&self.status);
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }
}
