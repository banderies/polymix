//! Stamp (Meshmixer's Stamp tool, hotkey 4): pick a stencil from the library, then drag on
//! the model from where the stamp should be centred; the drag sets its size. On release the
//! outline is cut into the surface and the inside becomes a new face group (ready for
//! Extrude, Offset…) or is removed, leaving a hole of that shape.
//!
//! The library, all Polymix's own designs ([`mm_edit::stencil`]): plain shapes, icons, a
//! text label in the built-in font (for labelling printed models), and the user's stamps —
//! made with Select > Convert To > Stamp, or text files of points put in the stamps folder —
//! shown as a grid of thumbnails. Stencils may have holes (a ring, the counters of letters).

use super::*;
use crate::ui_util;
use mm_edit::{StampFrame, StencilShape};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StampMode {
    FaceGroup,
    Hole,
}

/// Library sections.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shelf {
    Shapes,
    Icons,
    Text,
    Mine,
}

struct Entry {
    shape: StencilShape,
    shelf: Shelf,
    /// Filled drawing for its button, made when first shown.
    thumb: Option<egui::TextureHandle>,
}

/// Name of the library entry that holds the text label.
const TEXT: &str = "Text";

struct StampDrag {
    object: ObjectId,
    seed: u32,
    frame: StampFrame,
    radius: f64,
}

pub struct StampTool {
    /// Shapes, icons, the text label, then the user's stamps.
    entries: Vec<Entry>,
    choice: usize,
    /// The text field.
    text: String,
    /// The label the Text entry holds (the last text the font could draw).
    label_text: String,
    text_error: Option<String>,
    mode: StampMode,
    /// Stencil rotation in the view, degrees.
    rotation: f64,
    /// Whole-unit stamp sizes (Meshmixer "Snap Dimension"): diameters, or letter heights.
    snap: bool,
    drag: Option<StampDrag>,
}

/// The built-in library and the user's stamps (renamed where they share a built-in's name).
fn library(text: &str) -> Vec<Entry> {
    let entry = |shape, shelf| Entry { shape, shelf, thumb: None };
    let mut out: Vec<Entry> = mm_edit::shape_stencils().into_iter().map(|s| entry(s, Shelf::Shapes)).collect();
    out.extend(mm_edit::icon_stencils().into_iter().map(|s| entry(s, Shelf::Icons)));
    let label = mm_edit::text_stencil(text).unwrap_or_else(|_| mm_edit::text_stencil("A").unwrap());
    out.push(entry(StencilShape { name: TEXT.into(), ..label }, Shelf::Text));
    out.extend(user_entries(&out));
    out
}

fn user_entries(builtin: &[Entry]) -> Vec<Entry> {
    crate::parts::user_stencils()
        .into_iter()
        .map(|mut s| {
            if builtin.iter().any(|e| e.shape.name == s.name) {
                s.name += " (mine)";
            }
            Entry { shape: s, shelf: Shelf::Mine, thumb: None }
        })
        .collect()
}

impl StampTool {
    pub fn new(_cx: &mut ToolCx) -> Self {
        Self::with_text("A1")
    }

    fn with_text(text: &str) -> Self {
        Self { entries: library(text), choice: 0, text: text.into(), label_text: text.into(), text_error: None, mode: StampMode::FaceGroup, rotation: 0.0, snap: false, drag: None }
    }

    fn name_of(&self) -> &str {
        &self.entries[self.choice].shape.name
    }

    /// What the status line calls the chosen stencil.
    fn label(&self) -> String {
        if self.name_of() == TEXT {
            format!("text \"{}\"", self.label_text.trim().to_uppercase())
        } else {
            self.name_of().to_string()
        }
    }

    fn text_index(&self) -> usize {
        self.entries.iter().position(|e| e.shelf == Shelf::Text).unwrap_or(0)
    }

    /// Outline the text label anew after an edit.
    fn update_text(&mut self) {
        match mm_edit::text_stencil(&self.text) {
            Ok(s) => {
                let i = self.text_index();
                self.entries[i] = Entry { shape: StencilShape { name: TEXT.into(), ..s }, shelf: Shelf::Text, thumb: None };
                self.label_text = self.text.clone();
                self.text_error = None;
            }
            Err(e) => self.text_error = Some(e),
        }
    }

    /// Read the user's stamps again (after Convert To > Stamp or new files).
    fn reload_user(&mut self) {
        let chosen = self.name_of().to_string();
        self.entries.retain(|e| e.shelf != Shelf::Mine);
        let mine = user_entries(&self.entries);
        self.entries.extend(mine);
        self.choice = self.entries.iter().position(|e| e.shape.name == chosen).unwrap_or(0);
    }

    /// The stencil's loops in the drag's plane coordinates.
    fn outline(&self, radius: f64) -> Vec<Vec<DVec2>> {
        let (s, c) = self.rotation.to_radians().sin_cos();
        self.entries[self.choice].shape.loops.iter().map(|l| l.iter().map(|p| DVec2::new(p.x * c - p.y * s, p.x * s + p.y * c) * radius).collect()).collect()
    }

    fn snapped(&self, radius: f64) -> f64 {
        if !self.snap {
            return radius;
        }
        match self.entries[self.choice].shape.cap_height {
            // Whole-unit letter heights.
            Some(h) if h > 0.0 => (h * radius).round().max(1.0) / h,
            // Whole-unit diameters.
            _ => ((2.0 * radius).round().max(1.0)) * 0.5,
        }
    }

    /// The size readout: diameter, and letter height for text.
    fn size_text(&self, radius: f64) -> String {
        match self.entries[self.choice].shape.cap_height {
            Some(h) => format!("letters {:.3} tall", h * radius),
            None => format!("Ø {:.3}", 2.0 * radius),
        }
    }

    fn apply(&mut self, d: StampDrag, cx: &mut ToolCx) {
        let Some(o) = cx.scene.get(d.object) else { return };
        let mut mesh = (*o.mesh).clone();
        let loops = self.outline(d.radius);
        match mm_edit::stamp_loops(&mut mesh, d.seed, &d.frame, &loops) {
            Ok(r) => {
                let what = match self.mode {
                    StampMode::FaceGroup => {
                        let g = mesh.max_group_id() + 1;
                        for &t in &r.inside {
                            mesh.set_tri_group(t, g);
                        }
                        format!("new face group {g}")
                    }
                    StampMode::Hole => {
                        mesh.remove_triangles(r.inside.iter().copied());
                        "hole".to_string()
                    }
                };
                cx.history.push(format!("Stamp {}", self.label()), cx.scene);
                if let Some(o) = cx.scene.get_mut(d.object) {
                    o.set_mesh(mesh);
                }
                *cx.status = format!("Stamp {}: {}, {what} ({} triangles)", self.label(), self.size_text(d.radius), r.inside.len());
            }
            Err(e) => *cx.status = format!("Stamp: {e}"),
        }
    }
}

/// Coverage (0..1) of a stencil drawn into a `w` × `h` pixel image (rows downward), fitted
/// inside a `margin`-pixel border; even-odd fill, 4 samples per row.
fn coverage(loops: &[Vec<DVec2>], w: usize, h: usize, margin: f64) -> Vec<f32> {
    let mut out = vec![0.0f32; w * h];
    let (mut lo, mut hi) = (DVec2::splat(f64::INFINITY), DVec2::splat(f64::NEG_INFINITY));
    for p in loops.iter().flatten() {
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    if !lo.is_finite() || w == 0 || h == 0 {
        return out;
    }
    let size = (hi - lo).max(DVec2::splat(1e-12));
    let k = ((w as f64 - 2.0 * margin) / size.x).min((h as f64 - 2.0 * margin) / size.y).max(0.0);
    let mid = (lo + hi) * 0.5;
    let px = |p: DVec2| DVec2::new(w as f64 * 0.5 + (p.x - mid.x) * k, h as f64 * 0.5 - (p.y - mid.y) * k);
    let edges: Vec<(DVec2, DVec2)> = loops.iter().flat_map(|l| (0..l.len()).map(move |i| (px(l[i]), px(l[(i + 1) % l.len()])))).collect();
    const SUB: usize = 4;
    let mut xs = Vec::new();
    for row in 0..h {
        for s in 0..SUB {
            let y = row as f64 + (s as f64 + 0.5) / SUB as f64;
            xs.clear();
            xs.extend(edges.iter().filter(|(a, b)| (a.y > y) != (b.y > y)).map(|(a, b)| a.x + (y - a.y) / (b.y - a.y) * (b.x - a.x)));
            xs.sort_by(f64::total_cmp);
            for [a, b] in xs.as_chunks::<2>().0 {
                let (x0, x1) = (a.clamp(0.0, w as f64), b.clamp(0.0, w as f64));
                for i in (x0.floor() as usize)..(x1.ceil() as usize).min(w) {
                    let cover = (x1.min(i as f64 + 1.0) - x0.max(i as f64)).max(0.0);
                    out[row * w + i] += (cover / SUB as f64) as f32;
                }
            }
        }
    }
    out.iter_mut().for_each(|c| *c = c.min(1.0));
    out
}

/// A stencil button: its filled drawing on a tile.
fn stamp_button(ui: &mut egui::Ui, entry: &mut Entry, size: egui::Vec2, selected: bool) -> egui::Response {
    let (rect, resp) = ui.allocate_exact_size(size, egui::Sense::click());
    let v = ui.visuals();
    let (bg, fg) = if selected { (v.selection.bg_fill, v.selection.stroke.color) } else if resp.hovered() { (v.widgets.hovered.bg_fill, v.text_color()) } else { (v.widgets.inactive.bg_fill, v.text_color()) };
    ui.painter().rect_filled(rect, 4.0, bg);
    let inner = rect.shrink(4.0);
    let ppp = ui.ctx().pixels_per_point();
    let want = [(inner.width() * ppp).round().max(1.0) as usize, (inner.height() * ppp).round().max(1.0) as usize];
    if entry.thumb.as_ref().is_none_or(|t| t.size() != want) {
        let cov = coverage(&entry.shape.loops, want[0], want[1], 1.0);
        let img = egui::ColorImage::new(want, cov.iter().map(|&c| egui::Color32::from_white_alpha((c * 255.0).round() as u8)).collect());
        entry.thumb = Some(ui.ctx().load_texture(format!("stamp {}", entry.shape.name), img, egui::TextureOptions::LINEAR));
    }
    if let Some(t) = &entry.thumb {
        ui.painter().image(t.id(), inner, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), fg);
    }
    resp
}

/// Open a folder in the file manager.
fn show_folder(dir: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let opener = if cfg!(target_os = "macos") {
        "open"
    } else if cfg!(windows) {
        "explorer"
    } else {
        "xdg-open"
    };
    std::process::Command::new(opener).arg(dir).spawn().map(|_| ())
}

impl Tool for StampTool {
    fn name(&self) -> &'static str {
        "Stamp"
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"stencil": self.name_of(), "text": self.text, "mode": if self.mode == StampMode::Hole { "hole" } else { "face_group" }, "rotation_deg": self.rotation, "whole_unit_sizes": self.snap}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let names: Vec<&str> = self.entries.iter().map(|e| e.shape.name.as_str()).collect();
        Some(serde_json::json!({"type": "object", "properties": {
            "stencil": {"type": "string", "enum": names, "description": "The outline stamped: built-in shapes and icons, \"Text\" (the `text` label), then the user's own stamps"},
            "text": {"type": "string", "description": "The label the \"Text\" stencil stamps: capital letters (lower case is drawn as capitals), digits, spaces and - . + / :"},
            "mode": {"type": "string", "enum": ["face_group", "hole"], "description": "The inside becomes a new face group, or is removed"},
            "rotation_deg": {"type": "number", "description": "Turn the stencil in its plane, degrees"},
            "whole_unit_sizes": {"type": "boolean", "description": "Round diameters (text: letter heights) to whole millimetres"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, _cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let name = v["stencil"].as_str().unwrap_or_default().to_string();
        let choice = self.entries.iter().position(|e| e.shape.name == name).ok_or_else(|| format!("no stencil {name}"))?;
        let text = v["text"].as_str().ok_or("text must be a string")?.to_string();
        if text != self.text {
            mm_edit::text_stencil(&text)?;
            self.text = text;
            self.update_text();
        }
        self.choice = choice;
        self.mode = if v["mode"] == "hole" { StampMode::Hole } else { StampMode::FaceGroup };
        self.rotation = v["rotation_deg"].as_f64().ok_or("rotation_deg must be a number")?;
        self.snap = v["whole_unit_sizes"].as_bool().unwrap_or(false);
        Ok(())
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        vec![
            Action::new(
                "stamp_at",
                "Cut the stencil into the active object, centred at the surface point nearest `at` and facing out of the surface there (as pressing there and dragging to the size does)",
                json!({"at": arg::point("Centre, mm"), "diameter_mm": arg::number("The stamp's size (the diameter of the circle around it)"), "up": arg::point("Which way the stencil's top points (default the view's up)")}),
            )
            .required(&["at", "diameter_mm"]),
            Action::plain("reload_stamps", "Read the user's stamps again (after Select > Convert To > Stamp or new files in the stamps folder)"),
        ]
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        let a = Args(args);
        match action {
            "stamp_at" => {
                let id = cx.scene.active.ok_or("no active object")?;
                let o = cx.scene.get_mut(id).ok_or("no object")?;
                let h = o.bvh().nearest(&o.mesh, DVec3::from_array(a.point("at")?), f64::INFINITY).ok_or("no surface")?;
                let n = o.mesh.triangle(h.tid).iter().map(|&v| o.mesh.vertex_normal(v)).sum::<DVec3>().normalize_or(o.mesh.tri_normal(h.tid));
                let up = a.opt_point("up").map(DVec3::from_array).unwrap_or_else(|| cx.camera.up());
                let d = a.f64("diameter_mm")?;
                if d <= 0.0 {
                    return Err("diameter_mm must be positive".into());
                }
                let drag = StampDrag { object: id, seed: h.tid, frame: StampFrame::new(h.point, n, up), radius: self.snapped(0.5 * d) };
                let before = cx.history.generation();
                self.apply(drag, cx);
                if cx.history.generation() == before { Err(cx.status.clone()) } else { Ok(cx.status.clone()) }
            }
            "reload_stamps" => {
                self.reload_user();
                Ok(format!("{} stamps of the user's", self.entries.iter().filter(|e| e.shelf == Shelf::Mine).count()))
            }
            _ => Err(format!("Stamp has no action {action}")),
        }
    }

    fn shows_groups(&self) -> bool {
        true
    }

    fn survives_scene_change(&self) -> bool {
        true
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Stamp");
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.mode, StampMode::FaceGroup, "Face group").on_hover_text("The inside becomes a new face group");
            ui.selectable_value(&mut self.mode, StampMode::Hole, "Hole").on_hover_text("The inside is removed");
        });
        ui.add(egui::Slider::new(&mut self.rotation, -180.0..=180.0).text("rotation").suffix("°"));
        ui.checkbox(&mut self.snap, "Snap Dimension").on_hover_text("Whole-unit stamp diameters (text: whole-unit letter heights)");
        ui.small("Drag on the model from the centre of the stamp outward to size it. The outline is cut into the surface facing you; a new face group can then be extruded or offset (double-click it in Select).");

        let tile = egui::vec2(38.0, 38.0);
        for (shelf, title) in [(Shelf::Shapes, "Shapes"), (Shelf::Icons, "Icons")] {
            ui_util::section(ui, title);
            ui.horizontal_wrapped(|ui| {
                for i in 0..self.entries.len() {
                    if self.entries[i].shelf == shelf {
                        let selected = self.choice == i;
                        let name = self.entries[i].shape.name.clone();
                        if stamp_button(ui, &mut self.entries[i], tile, selected).on_hover_text(name).clicked() {
                            self.choice = i;
                        }
                    }
                }
            });
        }

        ui_util::section(ui, "Text");
        let ti = self.text_index();
        let edit = ui.add(egui::TextEdit::singleline(&mut self.text).hint_text("e.g. L-ICA 2").desired_width(f32::INFINITY));
        if edit.changed() {
            self.update_text();
        }
        if edit.changed() || edit.gained_focus() {
            self.choice = ti;
        }
        let wide = egui::vec2(ui.available_width(), 40.0);
        let selected = self.choice == ti;
        if stamp_button(ui, &mut self.entries[ti], wide, selected).on_hover_text("Stamp this text").clicked() {
            self.choice = ti;
        }
        if let Some(e) = &self.text_error {
            ui.colored_label(ui.visuals().warn_fg_color, e);
        }
        ui.small("Capital letters, digits and - . + / : for labelling parts. Stamp it as a face group, then Extrude the group to emboss or engrave it.");

        ui_util::section(ui, "My stamps");
        let mine: Vec<usize> = (0..self.entries.len()).filter(|&i| self.entries[i].shelf == Shelf::Mine).collect();
        if mine.is_empty() {
            ui.weak("None yet.");
        }
        ui.horizontal_wrapped(|ui| {
            for i in mine {
                let selected = self.choice == i;
                let name = self.entries[i].shape.name.clone();
                if stamp_button(ui, &mut self.entries[i], tile, selected).on_hover_text(name).clicked() {
                    self.choice = i;
                }
            }
        });
        let dir = crate::parts::stencils_dir();
        ui.horizontal(|ui| {
            if ui.button("Reload").on_hover_text("Read the stamps folder again").clicked() {
                self.reload_user();
            }
            if let Some(dir) = &dir {
                if ui.button("Show folder").on_hover_text(dir.display().to_string()).clicked() {
                    if let Err(e) = show_folder(dir) {
                        *cx.status = format!("Cannot open {}: {e}", dir.display());
                    }
                }
            }
        });
        ui.small("Select > Convert To > Stamp saves a selection's outline here. Your own outlines go in the stamps folder as text files of \"x y\" points, one per line, with a blank line between loops (holes are loops inside loops).");
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.drag = cx.pick(pos, cx.scene.active).map(|(id, h)| {
                let o = cx.scene.get(id).unwrap();
                // Smoothed normal under the click, so faceting does not tilt the stamp.
                let n = o.mesh.triangle(h.tid).iter().map(|&v| o.mesh.vertex_normal(v)).sum::<DVec3>().normalize_or(o.mesh.tri_normal(h.tid));
                StampDrag { object: id, seed: h.tid, frame: StampFrame::new(h.point, n, cx.camera.up()), radius: 0.0 }
            });
        }
        if let Some(d) = &mut self.drag {
            if input.down {
                // Size from where the pointer ray meets the stencil plane.
                let ray = cx.ray_at(pos);
                let denom = ray.dir.dot(d.frame.n);
                if denom.abs() > 1e-9 {
                    let t = (d.frame.origin - ray.origin).dot(d.frame.n) / denom;
                    d.radius = d.frame.project(ray.origin + ray.dir * t).length();
                }
            }
        }
        if input.released {
            if let Some(mut d) = self.drag.take() {
                // A click without a drag does nothing.
                if d.radius > cx.world_per_point(d.frame.origin) * 4.0 {
                    d.radius = self.snapped(d.radius);
                    self.apply(d, cx);
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        let Some(d) = &self.drag else { return };
        if d.radius <= 0.0 {
            return;
        }
        let Some(o) = cx.scene.get_mut(d.object) else { return };
        let bvh = o.bvh();
        let col = Vec4::new(1.0, 0.75, 0.2, 1.0);
        // Outline dropped onto the surface (edges subdivided so it follows the curvature).
        for poly in self.outline(self.snapped(d.radius)) {
            let mut pts = Vec::new();
            for i in 0..poly.len() {
                let (a, b) = (poly[i], poly[(i + 1) % poly.len()]);
                let n = ((b - a).length() / (d.radius * 0.08)).ceil().max(1.0) as usize;
                for k in 0..n {
                    let q = d.frame.lift(a.lerp(b, k as f64 / n as f64));
                    pts.push(bvh.nearest(&o.mesh, q, d.radius * 2.0).map_or(q, |h| h.point + o.mesh.tri_normal(h.tid) * (d.radius * 0.01)));
                }
            }
            for i in 0..pts.len() {
                ov.line(pts[i], pts[(i + 1) % pts.len()], col);
            }
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        // Size readout beside the pointer while dragging.
        let Some(d) = &self.drag else { return };
        if d.radius <= 0.0 {
            return;
        }
        let Some(at) = cx.egui.pointer_latest_pos() else { return };
        let text = self.size_text(self.snapped(d.radius));
        painter.text(at + egui::vec2(14.0, -14.0), egui::Align2::LEFT_BOTTOM, text, egui::FontId::proportional(13.0), egui::Color32::from_rgb(255, 190, 50));
    }

    fn script(&mut self, cmd: &str, _cx: &mut ToolCx) {
        // `stencil:Star`, `text:L-ICA 2`, `mode:hole`, `rotate:30`.
        match cmd.split_once(':') {
            Some(("stencil", name)) => self.choice = self.entries.iter().position(|e| e.shape.name.eq_ignore_ascii_case(name)).unwrap_or(self.choice),
            Some(("text", t)) => {
                self.text = t.to_string();
                self.update_text();
                self.choice = self.text_index();
            }
            Some(("mode", m)) => self.mode = if m == "hole" { StampMode::Hole } else { StampMode::FaceGroup },
            Some(("rotate", a)) => self.rotation = a.parse().unwrap_or(self.rotation),
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool() -> StampTool {
        StampTool::with_text("A1")
    }

    #[test]
    fn rotation_and_snapping() {
        let mut t = tool();
        t.choice = t.entries.iter().position(|e| e.shape.name == "Square").unwrap();
        t.rotation = 45.0;
        // A square rotated by 45° has its corners on the axes.
        let p = t.outline(2.0);
        assert!(p[0].iter().any(|q| (q.x - 2.0).abs() < 1e-9 && q.y.abs() < 1e-9));
        assert_eq!(t.snapped(1.3), 1.3);
        t.snap = true;
        assert_eq!(t.snapped(1.3), 1.5);
        assert_eq!(t.snapped(0.1), 0.5);
        // Text snaps its letter height.
        t.choice = t.text_index();
        let h = t.entries[t.choice].shape.cap_height.unwrap();
        assert!((h * t.snapped(7.3 / h) - 7.0).abs() < 1e-9);
    }

    #[test]
    fn library_has_its_shelves_and_the_text_follows_edits() {
        let mut t = tool();
        for shelf in [Shelf::Shapes, Shelf::Icons, Shelf::Text] {
            assert!(t.entries.iter().any(|e| e.shelf == shelf), "{shelf:?}");
        }
        let names: Vec<&str> = t.entries.iter().map(|e| e.shape.name.as_str()).collect();
        for n in ["Circle", "Ring", "Rounded rectangle", "Slot", "Double arrow", TEXT] {
            assert!(names.contains(&n), "{n}");
        }
        let before = t.entries[t.text_index()].shape.loops.len();
        t.text = "B8".into();
        t.update_text();
        // B and 8: two pieces, four counters.
        assert_eq!(t.entries[t.text_index()].shape.loops.len(), 6);
        assert_ne!(before, 6);
        t.text = "Ω".into();
        t.update_text();
        assert!(t.text_error.is_some());
        assert_eq!(t.entries[t.text_index()].shape.loops.len(), 6, "the last good label stays");
        assert_eq!(t.label(), "Circle");
        t.choice = t.text_index();
        assert_eq!(t.label(), "text \"B8\"");
    }

    #[test]
    fn coverage_fills_inside_and_leaves_holes() {
        let ring = mm_edit::shape_stencils().into_iter().find(|s| s.name == "Ring").unwrap();
        let (w, h) = (40, 40);
        let c = coverage(&ring.loops, w, h, 0.0);
        // Centre (the hole) empty, the band full, the corners empty.
        assert!(c[20 * w + 20] < 0.01);
        assert!(c[20 * w + 3] > 0.99);
        assert!(c[0] < 0.01);
        // Total ≈ the ring's area in pixels.
        let total: f32 = c.iter().sum();
        let want = std::f64::consts::PI * (1.0 - 0.62f64.powi(2)) * 400.0;
        assert!((total as f64 - want).abs() < 0.03 * want, "{total} vs {want}");
    }
}
