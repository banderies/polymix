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

impl Tool for PatchesTool {
    fn survives_scene_change(&self) -> bool {
        true
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
        if ui.button("Auto-name caps (largest = inlet)").on_hover_text("Group 0 = wall; among flat caps the largest = inlet, the rest = outlet_1..n by decreasing area").clicked() {
            let o = cx.scene.get_mut(id).unwrap();
            // Only flat cap groups are inlet/outlet candidates.
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
