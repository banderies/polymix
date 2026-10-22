//! Boundary patches for CFD: name face groups (wall / inlet / outlets) and export them as a
//! multi-solid ASCII STL (one `solid <name>` per patch, as read by OpenFOAM, Fluent, CFX,
//! SimVascular and others) or as one STL file per patch.

use super::*;
use crate::jobs::Job;
use crate::scene::SceneObject;
use crate::ui_util;
use std::collections::{BTreeMap, HashMap};
use std::io::Write;
use std::sync::{Arc, Weak};

pub struct PatchesTool {
    object: Option<ObjectId>,
    status: String,
    stats: StatsCache,
    /// An export writing in the background; it ends with the message to show.
    export: Option<Job<String>>,
}

/// Triangle count and area per face group, as [`group_stats`] gives them.
type GroupStats = BTreeMap<i32, (usize, f64)>;

/// The group statistics of the mesh last shown: counting the groups of a large mesh takes
/// milliseconds, too long to repeat every frame the panel is drawn.
#[derive(Default)]
struct StatsCache {
    /// The mesh they were counted on, with its shape and group versions. Held weakly: while it
    /// is held, the allocation cannot be reused for another mesh (group versions are only
    /// counted per mesh, so a copy edited elsewhere can repeat them), yet an object that drops
    /// the mesh still frees it.
    mesh: Option<(Weak<Mesh>, u64, u64)>,
    stats: Arc<GroupStats>,
}

impl StatsCache {
    fn is_for(&self, mesh: &Arc<Mesh>) -> bool {
        self.mesh.as_ref().is_some_and(|(w, shape, groups)| std::ptr::eq(w.as_ptr(), Arc::as_ptr(mesh)) && *shape == mesh.shape_version() && *groups == mesh.group_version())
    }

    /// The statistics of `mesh`, counted again only when it is not the mesh of last time.
    fn get(&mut self, mesh: &Arc<Mesh>) -> Arc<GroupStats> {
        if !self.is_for(mesh) {
            self.stats = Arc::new(group_stats(mesh));
            self.mesh = Some((Arc::downgrade(mesh), mesh.shape_version(), mesh.group_version()));
        }
        self.stats.clone()
    }

    /// The statistics of `mesh` without keeping them (kept ones if they are for it).
    fn peek(&self, mesh: &Arc<Mesh>) -> Arc<GroupStats> {
        if self.is_for(mesh) {
            self.stats.clone()
        } else {
            Arc::new(group_stats(mesh))
        }
    }
}

/// More patches than this are listed in a scrolling box that lays out only the rows in view.
const LONG_LIST: usize = 200;

impl PatchesTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, status: String::new(), stats: StatsCache::default(), export: None }
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
    patch_name_in(&o.group_names, g)
}

/// [`patch_name`] from an object's group names.
fn patch_name_in(names: &HashMap<i32, String>, g: i32) -> String {
    names.get(&g).cloned().unwrap_or_else(|| if g == 0 { "wall".into() } else { format!("patch_{g}") })
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
    write_multisolid_stl_of(path, &o.mesh, &o.group_names)
}

/// [`write_multisolid_stl`] from a mesh and its group names (for a background export).
fn write_multisolid_stl_of(path: &std::path::Path, mesh: &Mesh, names: &HashMap<i32, String>) -> anyhow::Result<usize> {
    let mut by_group: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for t in mesh.triangle_ids() {
        by_group.entry(mesh.tri_group(t)).or_default().push(t);
    }
    let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
    for (g, tris) in &by_group {
        let name = sanitize(&patch_name_in(names, *g));
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
    write_patch_files_of(dir, &o.mesh, &o.group_names)
}

/// [`write_patch_files`] from a mesh and its group names (for a background export).
fn write_patch_files_of(dir: &std::path::Path, mesh: &Mesh, names: &HashMap<i32, String>) -> anyhow::Result<usize> {
    let mut by_group: BTreeMap<i32, Vec<u32>> = BTreeMap::new();
    for t in mesh.triangle_ids() {
        by_group.entry(mesh.tri_group(t)).or_default().push(t);
    }
    for (g, tris) in &by_group {
        let (part, _) = mesh.submesh(tris);
        let name = sanitize(&patch_name_in(names, *g));
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

    /// Write the object's patches in the background (the files can take seconds for a large
    /// mesh): `write` gets a copy of the mesh handle and of the group names, so later edits
    /// do not change what is written.
    fn start_export(&mut self, o: &SceneObject, cx: &ToolCx, write: impl FnOnce(&Mesh, &HashMap<i32, String>) -> String + Send + 'static) {
        let (mesh, names) = (o.mesh.clone(), o.group_names.clone());
        self.status.clear();
        self.export = Some(Job::spawn("Export patches", cx.egui.clone(), move |_| Ok(write(&mesh, &names))));
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
            "patches": self.stats.peek(&o.mesh).iter().map(|(g, (n, area))| json!({
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
        let stats = self.stats.get(&cx.scene.get(id).unwrap().mesh);
        let group = || -> Result<i32, String> {
            let g = a.0.get("group").and_then(serde_json::Value::as_i64).ok_or("`group` (a face group number) is needed")? as i32;
            if stats.contains_key(&g) { Ok(g) } else { Err(format!("the object has no face group {g}")) }
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
                    if let Some(g) = stats.keys().find(|g| dir.join(format!("{}.stl", sanitize(&patch_name(o, **g)))).exists()) {
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

    fn busy(&self) -> Option<(String, f32)> {
        self.export.as_ref().map(|j| ("Exporting patches".to_string(), j.progress()))
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Boundary patches (CFD)");
        // Collect a finished export first, so it ends even if the object went away meanwhile.
        if let Some(r) = self.export.as_ref().and_then(|j| j.poll()) {
            self.export = None;
            self.status = r.unwrap_or_else(|e| format!("Export failed: {e:#}"));
        }
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        ui.small("Each face group becomes a named boundary patch. Caps made by Trim / Cap / Flow extension, and flow extension walls (extension_N), get their own groups automatically. Turn on View > Face group colors to see them.");
        let stats = self.stats.get(&cx.scene.get(id).unwrap().mesh);
        let mut select_group = None;
        let mut renames: Vec<(i32, String)> = Vec::new();
        if stats.len() <= LONG_LIST {
            egui::Grid::new("patches").num_columns(4).striped(true).show(ui, |ui| {
                ui.strong("Name");
                ui.strong("Tris");
                ui.strong("Area");
                ui.label("");
                ui.end_row();
                for (g, (n, area)) in stats.iter() {
                    let o = cx.scene.get(id).unwrap();
                    let mut name = patch_name(o, *g);
                    if ui.add(egui::TextEdit::singleline(&mut name).id_salt(("patch_name", *g)).desired_width(110.0)).changed() {
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
        } else {
            // Thousands of groups (e.g. after Generate Face Groups on a noisy surface): lay out
            // only the rows in view. Each name field is tied to its group, not to its row, so
            // scrolling while typing keeps editing the same patch.
            let o = cx.scene.get(id).unwrap();
            let row_height = ui.spacing().interact_size.y;
            let cell = |ui: &mut egui::Ui, w: f32, text: egui::WidgetText| {
                ui.add_sized([w, row_height], egui::Label::new(text).truncate());
            };
            ui.horizontal(|ui| {
                cell(ui, 110.0, egui::RichText::new("Name").strong().into());
                cell(ui, 70.0, egui::RichText::new("Tris").strong().into());
                cell(ui, 70.0, egui::RichText::new("Area").strong().into());
            });
            egui::ScrollArea::vertical().id_salt("patch_rows").max_height(row_height * 16.0).auto_shrink([false, true]).show_rows(ui, row_height, stats.len(), |ui, rows| {
                for (g, (n, area)) in stats.iter().skip(rows.start).take(rows.len()) {
                    ui.horizontal(|ui| {
                        let mut name = patch_name(o, *g);
                        if ui.add(egui::TextEdit::singleline(&mut name).id_salt(("patch_name", *g)).desired_width(110.0)).changed() {
                            renames.push((*g, name));
                        }
                        cell(ui, 70.0, ui_util::format_count(*n).into());
                        cell(ui, 70.0, format!("{area:.2}").into());
                        if ui.small_button("select").clicked() {
                            select_group = Some(*g);
                        }
                    });
                }
            });
        }
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
        let idle = self.export.is_none();
        if ui.add_enabled(idle, egui::Button::new("Multi-solid ASCII STL…")).clicked() {
            let o = cx.scene.get(id).unwrap();
            if let Some(path) = rfd::FileDialog::new().add_filter("STL", &["stl"]).set_file_name(format!("{}_patches.stl", o.name)).save_file() {
                self.start_export(o, cx, move |mesh, names| match write_multisolid_stl_of(&path, mesh, names) {
                    Ok(n) => format!("Wrote {n} patches to {}", path.display()),
                    Err(e) => format!("Export failed: {e:#}"),
                });
            }
        }
        if ui.add_enabled(idle, egui::Button::new("One STL per patch…")).clicked() {
            let o = cx.scene.get(id).unwrap();
            if let Some(dir) = rfd::FileDialog::new().pick_folder() {
                self.start_export(o, cx, move |mesh, names| match write_patch_files_of(&dir, mesh, names) {
                    Ok(n) => format!("Wrote {n} files to {}", dir.display()),
                    Err(e) => format!("Export failed: {e:#}"),
                });
            }
        }
        if !idle {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Writing…");
            });
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

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    /// A sphere whose triangles are spread over `groups` face groups.
    fn grouped(groups: i32) -> Mesh {
        let mut m = primitives::icosphere(5.0, 4);
        let tris: Vec<u32> = m.triangle_ids().collect();
        for (k, t) in tris.into_iter().enumerate() {
            m.set_tri_group(t, k as i32 % groups);
        }
        m
    }

    #[test]
    fn group_statistics_are_kept_per_mesh_and_counted_again_after_edits() {
        let mut o = SceneObject::new(1, "patches", grouped(7));
        let mut cache = StatsCache::default();
        let first = cache.get(&o.mesh);
        assert_eq!(*first, group_stats(&o.mesh));
        assert!(Arc::ptr_eq(&first, &cache.get(&o.mesh)), "the same mesh is not counted again");
        // A group change in place (copy-on-write, as edits do) is seen.
        let t = o.mesh.triangle_ids().next().unwrap();
        o.mesh_mut().set_tri_group(t, 40);
        let after = cache.get(&o.mesh);
        assert_eq!(*after, group_stats(&o.mesh));
        assert!(after.contains_key(&40));
        // A copy with the same versions but other groups, in a new allocation, is not
        // mistaken for the cached mesh.
        let mut copy = (*o.mesh).clone();
        copy.set_tri_group(t, 41);
        let copy = Arc::new(copy);
        assert_eq!(*cache.get(&copy), group_stats(&copy));
        assert_eq!(*cache.peek(&o.mesh), group_stats(&o.mesh));
    }

    #[test]
    fn the_panel_lists_few_and_many_patches() {
        for groups in [7, 600] {
            let mut scene = Scene::default();
            scene.add("patches", grouped(groups));
            let (mut history, mut camera, mut status) = (crate::scene::History::default(), crate::camera::Camera::default(), String::new());
            let ctx = egui::Context::default();
            let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
            let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
            let mut tool = PatchesTool::new(&mut cx);
            for _ in 0..3 {
                let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 900.0))), ..Default::default() };
                let _ = ctx.run_ui(raw, |ui| {
                    let _ = tool.panel(ui, &mut cx);
                });
            }
            assert_eq!(tool.stats.stats.len(), groups as usize);
            assert!(tool.stats.is_for(&cx.scene.active().unwrap().mesh));
            assert_eq!(tool.state(cx.scene)["patches"].as_array().map(Vec::len), Some(groups as usize));
        }
    }

    #[test]
    fn background_exports_write_what_the_direct_ones_do() {
        let mut o = SceneObject::new(1, "vessel", grouped(4));
        o.group_names.insert(0, "wall".into());
        o.group_names.insert(2, "inlet".into());
        let dir = std::env::temp_dir().join(format!("polymix_patches_{}", std::process::id()));
        std::fs::create_dir_all(dir.join("a")).unwrap();
        std::fs::create_dir_all(dir.join("b")).unwrap();
        assert_eq!(write_multisolid_stl(&dir.join("a.stl"), &o).unwrap(), 4);
        assert_eq!(write_multisolid_stl_of(&dir.join("b.stl"), &o.mesh, &o.group_names).unwrap(), 4);
        assert_eq!(std::fs::read(dir.join("a.stl")).unwrap(), std::fs::read(dir.join("b.stl")).unwrap());
        assert_eq!(write_patch_files(&dir.join("a"), &o).unwrap(), 4);
        assert_eq!(write_patch_files_of(&dir.join("b"), &o.mesh, &o.group_names).unwrap(), 4);
        for name in ["wall", "patch_1", "inlet", "patch_3"] {
            let f = format!("{name}.stl");
            assert_eq!(std::fs::read(dir.join("a").join(&f)).unwrap(), std::fs::read(dir.join("b").join(&f)).unwrap(), "{f}");
        }
        std::fs::remove_dir_all(&dir).ok();
    }
}
