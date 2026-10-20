//! Vessel analysis of a segment, as SlicerVMTK's Extract centerline and Cross-section
//! analysis: the centerline tree (`mm_segment::centerlines`), the cross-sections along each
//! branch, its narrowest point, a plot of the diameter along the branch (a click there moves
//! the crosshair), and the centerline drawn in the views.

use crate::effects::Outcome;
use crate::SegmentationWorkspace;
use egui::{Color32, Pos2, Rect, RichText, Stroke};
use glam::DVec3;
use mm_segment::{CenterlineParams, Centerlines, CrossSection};
use polymix_imaging::ViewMap;
use polymix_workspace::jobs::Job;

/// A segment's centerline and its cross-sections.
pub(crate) struct VesselResult {
    pub label: u8,
    /// The label map's version it was made from.
    pub version: u64,
    pub tree: Centerlines,
    /// Per branch: the sections along it, and the narrowest (index, narrowing %).
    pub sections: Vec<Vec<CrossSection>>,
    pub narrowest: Vec<Option<(usize, f64)>>,
}

/// The settings and results of the vessel analysis.
pub(crate) struct Vessels {
    pub min_branch_mm: f64,
    pub step_mm: f64,
    /// Points the centerline must start from (the first) and reach (voxel coordinates), placed
    /// by clicking in the slice views while `placing`.
    pub ends: Vec<DVec3>,
    pub placing: bool,
    pub result: Option<VesselResult>,
    /// The branch shown in the plot.
    pub branch: usize,
    /// Going to a section also turns the planes: one across the vessel, two along it.
    pub align: bool,
}

impl Default for Vessels {
    fn default() -> Self {
        Self { min_branch_mm: 5.0, step_mm: 0.5, ends: Vec::new(), placing: false, result: None, branch: 0, align: true }
    }
}

const LINE: Color32 = Color32::from_rgb(255, 150, 60);

impl SegmentationWorkspace {
    /// Extract the selected segment's centerline and cross-sections (in the background).
    pub(crate) fn extract_centerline(&mut self, ctx: &egui::Context) {
        let (None, Some(vol), Some(seg)) = (&self.running, self.volume.clone(), &self.seg) else { return };
        let Some(label) = seg.selected_segment().map(|s| s.label) else {
            self.say("Select the vessel's segment first", true);
            return;
        };
        let mask = seg.mask_of(label);
        if mask.count() == 0 {
            self.say("The segment is empty", true);
            return;
        }
        let version = seg.version;
        let params = CenterlineParams { min_branch_mm: self.vessels.min_branch_mm, ends: self.vessels.ends.iter().map(|v| v.to_array()).collect(), ..Default::default() };
        let step = self.vessels.step_mm.max(0.05);
        let job = Job::spawn("Extracting the centerline", ctx.clone(), move |progress| {
            let tree = mm_segment::centerlines(&mask, &vol.geometry, &params, &|f| progress(f * 0.8))?;
            if tree.branches.is_empty() {
                anyhow::bail!("no centerline: the segment is shorter than the shortest branch kept");
            }
            let sections: Vec<Vec<CrossSection>> = tree.branches.iter().map(|b| mm_segment::cross_sections(&mask, &vol.geometry, b, step)).collect();
            let narrowest = sections.iter().map(|s| narrowest(s)).collect();
            Ok(Outcome::Centerline(Box::new(VesselResult { label, version, tree, sections, narrowest })))
        });
        self.running = Some(crate::Running { what: "Extracting the centerline".into(), job, bypass: false });
    }

    /// The Vessels section of the panel.
    pub(crate) fn vessels_section(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let idle = self.running.is_none();
        let mut extract = false;
        let mut go_to = None;
        let title = match &self.vessels.result {
            Some(r) => format!("Centerline ({} branch{})", r.tree.branches.len(), if r.tree.branches.len() == 1 { "" } else { "es" }),
            None => "Centerline".into(),
        };
        let stale = match (&self.vessels.result, &self.seg) {
            (Some(r), Some(s)) => r.version != s.version,
            _ => false,
        };
        egui::CollapsingHeader::new(RichText::new(title).strong()).id_salt("segmentation.vessels").default_open(false).show(ui, |ui| {
            let v = &mut self.vessels;
            ui.weak("The selected segment's centerline tree, and its diameter along each branch (as SlicerVMTK's Extract centerline and Cross-section analysis).");
            ui.horizontal(|ui| {
                ui.add(egui::DragValue::new(&mut v.min_branch_mm).speed(0.1).range(0.5..=200.0).suffix(" mm"));
                ui.label("shortest branch");
                ui.add(egui::DragValue::new(&mut v.step_mm).speed(0.05).range(0.1..=10.0).suffix(" mm"));
                ui.label("between sections");
            });
            ui.horizontal(|ui| {
                ui.toggle_value(&mut v.placing, "Place ends").on_hover_text("Click points in the slice views: the centerline starts at the first and reaches the others (none: found automatically)");
                ui.label(format!("{} end{}", v.ends.len(), if v.ends.len() == 1 { "" } else { "s" }));
                if ui.add_enabled(!v.ends.is_empty(), egui::Button::new("Clear")).clicked() {
                    v.ends.clear();
                }
            });
            extract = ui.add_enabled(idle, egui::Button::new("Extract centerline")).clicked();
            let Some(r) = &v.result else { return };
            if stale {
                ui.weak("(the segments changed since: extract again)");
            }
            egui::Grid::new("segmentation.branches").num_columns(4).striped(true).spacing([10.0, 2.0]).show(ui, |ui| {
                for h in ["Branch", "Length mm", "⌀ mm (min–max)", "Narrowest"] {
                    ui.label(RichText::new(h).small().strong());
                }
                ui.end_row();
                for (k, b) in r.tree.branches.iter().enumerate() {
                    if ui.selectable_label(v.branch == k, format!("{}{}", k + 1, b.parent.map_or(String::new(), |p| format!(" (of {})", p + 1)))).clicked() {
                        v.branch = k;
                    }
                    ui.label(format!("{:.1}", b.length_mm));
                    let d: Vec<f64> = r.sections[k].iter().map(|s| s.equivalent_diameter_mm).filter(|d| *d > 0.0).collect();
                    let (lo, hi) = d.iter().fold((f64::INFINITY, 0f64), |(a, b), &x| (a.min(x), b.max(x)));
                    ui.label(if d.is_empty() { "—".into() } else { format!("{lo:.2} – {hi:.2}") });
                    match r.narrowest[k] {
                        Some((i, pct)) => {
                            if ui.link(format!("{pct:.0}% at {:.1} mm", r.sections[k][i].distance_mm)).on_hover_text("Show it: the crosshair moves there").clicked() {
                                go_to = Some((DVec3::from(r.sections[k][i].ras), DVec3::from(r.sections[k][i].normal)));
                                v.branch = k;
                            }
                        }
                        None => {
                            ui.weak("—");
                        }
                    }
                    ui.end_row();
                }
            });
            let k = v.branch.min(r.sections.len().saturating_sub(1));
            if let Some(sections) = r.sections.get(k) {
                if let Some(i) = diameter_plot(ui, sections, r.narrowest[k].map(|n| n.0)) {
                    go_to = Some((DVec3::from(sections[i].ras), DVec3::from(sections[i].normal)));
                }
                ui.checkbox(&mut v.align, "Turn the planes with the vessel").on_hover_text("Going to a section (a click in the plot or on the narrowest) shows the vessel's cross-section in the view most nearly across it (axial for a vessel running up, sagittal for one running left to right), and the vessel along its length in the other two");
            }
            if ui.button("Copy sections").on_hover_text("Every cross-section of every branch, as CSV").clicked() {
                let mut csv = String::from("branch,distance_mm,r_mm,a_mm,s_mm,area_mm2,equivalent_diameter_mm,min_diameter_mm,max_diameter_mm,inscribed_radius_mm\n");
                for (b, sections) in r.sections.iter().enumerate() {
                    for s in sections {
                        csv += &format!("{},{:.3},{:.3},{:.3},{:.3},{:.4},{:.4},{:.4},{:.4},{:.4}\n", b + 1, s.distance_mm, s.ras[0], s.ras[1], s.ras[2], s.area_mm2, s.equivalent_diameter_mm, s.min_diameter_mm, s.max_diameter_mm, s.radius_mm);
                    }
                }
                ui.ctx().copy_text(csv);
            }
        });
        if extract {
            self.extract_centerline(&ctx);
        }
        if let Some((ras, tangent)) = go_to {
            self.go_to_section(ras, tangent);
        }
    }

    /// Move the crosshair to a cross-section (RAS), and turn the planes with the vessel there
    /// (when chosen).
    pub(crate) fn go_to_section(&mut self, ras: DVec3, tangent: DVec3) {
        let (Some(vol), Some(st)) = (&self.volume, &mut self.state) else { return };
        st.cursor = vol.geometry.matrix().inverse().transform_point3(ras);
        if self.vessels.align {
            st.frames = Some(polymix_imaging::oblique::along_vessel(tangent).0);
        }
    }

    /// The centerline in the slice views (the stretch within a slice of each one) and over the
    /// 3D view, and the ends being placed.
    pub(crate) fn paint_vessels(&self, ui: &egui::Ui, maps: &[ViewMap], view3d: Option<Rect>) {
        let (Some(vol), Some(r)) = (&self.volume, &self.vessels.result) else {
            self.paint_ends(ui, maps);
            return;
        };
        let to_voxel = vol.geometry.matrix().inverse();
        for (k, b) in r.tree.branches.iter().enumerate() {
            let width = if k == self.vessels.branch { 2.5 } else { 1.5 };
            let pts: Vec<DVec3> = b.points.iter().map(|p| DVec3::from(p.voxel)).collect();
            for m in maps {
                let painter = ui.painter_at(m.rect);
                for w in pts.windows(2) {
                    let ((a, da), (b, db)) = (m.project(w[0]), m.project(w[1]));
                    if da.abs() <= 1.0 && db.abs() <= 1.0 {
                        painter.line_segment([a, b], Stroke::new(width, LINE));
                    }
                }
            }
            if let Some(rect) = view3d {
                let painter = ui.painter_at(rect);
                let line: Vec<Pos2> = pts.iter().filter_map(|&p| self.view3d.project(vol, rect, p)).collect();
                painter.add(egui::Shape::line(line, Stroke::new(width, LINE)));
            }
        }
        // The narrowest section of the branch shown.
        if let Some((i, _)) = r.narrowest.get(self.vessels.branch).copied().flatten() {
            let s = &r.sections[self.vessels.branch][i];
            let v = to_voxel.transform_point3(DVec3::from(s.ras));
            for m in maps {
                let (at, d) = m.project(v);
                if d.abs() <= 1.0 {
                    ui.painter_at(m.rect).circle_stroke(at, 6.0, Stroke::new(2.0, Color32::from_rgb(255, 80, 80)));
                }
            }
        }
        self.paint_ends(ui, maps);
    }

    fn paint_ends(&self, ui: &egui::Ui, maps: &[ViewMap]) {
        for (k, e) in self.vessels.ends.iter().enumerate() {
            for m in maps {
                let (p, d) = m.project(*e);
                if d.abs() > 1.0 {
                    continue;
                }
                let painter = ui.painter_at(m.rect);
                painter.circle_filled(p, 4.5, LINE);
                painter.text(p + egui::vec2(6.0, -6.0), egui::Align2::LEFT_BOTTOM, if k == 0 { "start".to_string() } else { format!("end {k}") }, egui::FontId::proportional(11.0), LINE);
            }
        }
    }
}

/// The narrowest section of a branch and its narrowing (`mm_segment::stenosis`), leaving out the
/// sections within a radius of either end: where a vessel leaves the image, its last sections
/// are cut short, not narrowed.
fn narrowest(sections: &[CrossSection]) -> Option<(usize, f64)> {
    let length = sections.last()?.distance_mm;
    let inner: Vec<usize> = (0..sections.len()).filter(|&i| {
        let (d, r) = (sections[i].distance_mm, sections[i].radius_mm.max(0.5));
        d >= r && length - d >= r
    }).collect();
    let (first, last) = (*inner.first()?, *inner.last()?);
    mm_segment::stenosis(&sections[first..=last]).map(|(i, pct)| (first + i, pct))
}

/// The equivalent diameter along a branch (the narrowest section marked); the section
/// clicked, if any.
fn diameter_plot(ui: &mut egui::Ui, sections: &[CrossSection], narrowest: Option<usize>) -> Option<usize> {
    let (rect, response) = ui.allocate_exact_size(egui::vec2(ui.available_width().min(360.0), 110.0), egui::Sense::click());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let valid: Vec<&CrossSection> = sections.iter().filter(|s| s.equivalent_diameter_mm > 0.0).collect();
    if valid.len() < 2 {
        return None;
    }
    let (d0, d1) = (valid[0].distance_mm, valid[valid.len() - 1].distance_mm.max(valid[0].distance_mm + 1e-6));
    let top = valid.iter().map(|s| s.max_diameter_mm.max(s.equivalent_diameter_mm)).fold(0.0, f64::max) * 1.1;
    let inner = rect.shrink2(egui::vec2(28.0, 12.0));
    let at = |d: f64, y: f64| Pos2::new(inner.min.x + ((d - d0) / (d1 - d0)) as f32 * inner.width(), inner.max.y - (y / top.max(1e-6)) as f32 * inner.height());
    let text = ui.visuals().weak_text_color();
    painter.text(Pos2::new(rect.min.x + 4.0, inner.min.y), egui::Align2::LEFT_TOP, format!("{top:.1}"), egui::FontId::proportional(10.0), text);
    painter.text(Pos2::new(rect.min.x + 4.0, inner.max.y), egui::Align2::LEFT_BOTTOM, "0", egui::FontId::proportional(10.0), text);
    painter.text(Pos2::new(inner.max.x, rect.max.y - 1.0), egui::Align2::RIGHT_BOTTOM, format!("{:.0} mm along", d1), egui::FontId::proportional(10.0), text);
    // Min and max diameter as a band, the equivalent diameter as the line.
    let band: Vec<Pos2> = valid.iter().map(|s| at(s.distance_mm, s.max_diameter_mm)).chain(valid.iter().rev().map(|s| at(s.distance_mm, s.min_diameter_mm))).collect();
    painter.add(egui::Shape::closed_line(band, Stroke::new(1.0, LINE.gamma_multiply(0.35))));
    painter.add(egui::Shape::line(valid.iter().map(|s| at(s.distance_mm, s.equivalent_diameter_mm)).collect(), Stroke::new(1.5, LINE)));
    if let Some(s) = narrowest.and_then(|i| sections.get(i)) {
        painter.circle_stroke(at(s.distance_mm, s.equivalent_diameter_mm), 4.0, Stroke::new(1.5, Color32::from_rgb(255, 80, 80)));
    }
    // The section under the pointer, and a click there.
    let pointer = response.hover_pos()?;
    let d = d0 + ((pointer.x - inner.min.x) / inner.width()).clamp(0.0, 1.0) as f64 * (d1 - d0);
    let i = (0..sections.len()).filter(|&i| sections[i].equivalent_diameter_mm > 0.0).min_by(|&a, &b| (sections[a].distance_mm - d).abs().total_cmp(&(sections[b].distance_mm - d).abs()))?;
    let s = &sections[i];
    painter.vline(at(s.distance_mm, 0.0).x, inner.y_range(), Stroke::new(1.0, text));
    painter.text(at(s.distance_mm, top * 0.95), egui::Align2::LEFT_TOP, format!(" {:.1} mm: ⌀ {:.2} ({:.2}–{:.2}), {:.1} mm²", s.distance_mm, s.equivalent_diameter_mm, s.min_diameter_mm, s.max_diameter_mm, s.area_mm2), egui::FontId::proportional(10.0), ui.visuals().text_color());
    response.clicked().then_some(i)
}
