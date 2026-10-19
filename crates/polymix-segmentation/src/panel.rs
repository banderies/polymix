//! The editor panel on the left, as in 3D Slicer's Segment Editor: the segments, the effects
//! and their options, masking, and saving / meshing.

use crate::effects::{Effect, Histogram, IslandOp, LogicalOp, SmoothMethod};
use crate::segmentation::{EditableArea, Overwrite};
use crate::SegmentationWorkspace;
use egui::{Color32, RichText};
use mm_volume::load::LoadedVolume;
use polymix_workspace::style::section;
use std::sync::Arc;

impl SegmentationWorkspace {
    pub(crate) fn editor(&mut self, ui: &mut egui::Ui) {
        let Some(vol) = self.volume.clone().filter(|_| self.seg.is_some() && self.settings.is_some()) else {
            ui.add_space(8.0);
            ui.weak("Select a volume in the Volume tab (⌘1) to segment it.");
            return;
        };
        egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
            self.history_row(ui);
            self.segments_section(ui, &vol);
            self.effects_section(ui, &vol);
            self.masking_section(ui);
            self.output_section(ui);
        });
    }

    fn history_row(&mut self, ui: &mut egui::Ui) {
        let Some(seg) = &mut self.seg else { return };
        ui.horizontal(|ui| {
            let idle = self.running.is_none();
            if ui.add_enabled(idle && seg.can_undo(), egui::Button::new("⟲ Undo")).on_hover_text("⌘Z").clicked() {
                seg.undo();
            }
            if ui.add_enabled(idle && seg.can_redo(), egui::Button::new("⟳ Redo")).on_hover_text("⇧⌘Z").clicked() {
                seg.redo();
            }
        });
        // On a line of its own, so the panel never widens (the views must not move).
        let mut cancel = false;
        if let Some(r) = &self.running {
            ui.horizontal(|ui| {
                ui.spinner();
                ui.label(format!("{}…", r.what));
                cancel = ui.small_button("Cancel").clicked();
            });
        }
        if cancel {
            self.running = None;
        }
        if let Some((m, error)) = &self.message {
            if *error {
                ui.colored_label(crate::WARN, m);
            } else {
                ui.weak(m);
            }
        }
    }

    fn segments_section(&mut self, ui: &mut egui::Ui, vol: &LoadedVolume) {
        let Some(seg) = &mut self.seg else { return };
        section(ui, "Segments");
        ui.horizontal(|ui| {
            if ui.button("+ Add").on_hover_text("Add a segment").clicked() && seg.add_segment(None).is_none() {
                self.message = Some(("All 255 segment labels are in use".into(), true));
            }
            if ui.add_enabled(seg.selected.is_some(), egui::Button::new("− Remove")).on_hover_text("Remove the selected segment and its voxels").clicked() {
                if let Some(k) = seg.selected {
                    seg.remove_segment(k);
                }
            }
            ui.add(egui::Slider::new(&mut self.opacity, 0.0..=1.0).show_value(false).text("fill")).on_hover_text("Opacity of the segments' fill in the slice views");
        });
        let ml_per_voxel = vol.geometry.spacing.iter().product::<f64>() / 1000.0;
        let counts = seg.counts().to_vec();
        let mut redraw = false;
        for k in 0..seg.segments.len() {
            ui.horizontal(|ui| {
                let selected = seg.selected == Some(k);
                let s = &mut seg.segments[k];
                if ui.checkbox(&mut s.visible, "").on_hover_text("Show the segment").changed() {
                    redraw = true;
                }
                let mut c = [s.colour.r(), s.colour.g(), s.colour.b()];
                if ui.color_edit_button_srgb(&mut c).changed() {
                    s.colour = Color32::from_rgb(c[0], c[1], c[2]);
                    redraw = true;
                }
                match &mut self.renaming {
                    Some((r, name, focused)) if *r == k => {
                        let edit = ui.add(egui::TextEdit::singleline(name).desired_width(120.0));
                        // Focus once only: Enter, Esc (keeps the old name), Tab or a click
                        // elsewhere then ends the rename (egui's focus, not `has_focus`, which
                        // is also false while another app is in front).
                        if !*focused {
                            edit.request_focus();
                            *focused = true;
                        } else if !ui.memory(|m| m.has_focus(edit.id)) {
                            if !name.trim().is_empty() && !ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                s.name = name.trim().to_string();
                                redraw = true;
                            }
                            self.renaming = None;
                        }
                    }
                    _ => {
                        let r = ui.selectable_label(selected, &s.name).on_hover_text("Double-click to rename");
                        if r.clicked() {
                            seg.selected = Some(k);
                        }
                        if r.double_clicked() {
                            self.renaming = Some((k, seg.segments[k].name.clone(), false));
                        }
                    }
                }
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.weak(format!("{:.2} mL", counts[seg.segments[k].label as usize] as f64 * ml_per_voxel));
                });
            });
        }
        if seg.segments.is_empty() {
            ui.weak("No segments yet: Add one, or start with Threshold or Paint.");
        }
        if redraw {
            seg.changed(true);
        }
    }

    fn effects_section(&mut self, ui: &mut egui::Ui, vol: &Arc<LoadedVolume>) {
        section(ui, "Effects");
        egui::Grid::new("segmentation.effects").num_columns(4).spacing([4.0, 4.0]).show(ui, |ui| {
            for (i, e) in Effect::ALL.into_iter().enumerate() {
                if ui.selectable_label(self.effect == Some(e), e.label()).on_hover_text(e.hint()).clicked() {
                    self.effect = if self.effect == Some(e) { None } else { Some(e) };
                }
                if i % 4 == 3 {
                    ui.end_row();
                }
            }
        });
        let Some(effect) = self.effect else {
            ui.weak("Choose an effect. Esc puts it away.");
            return;
        };
        ui.add_space(4.0);
        ui.label(RichText::new(effect.hint()).italics());
        let idle = self.running.is_none();
        let segments: Vec<(u8, String)> = self.seg.as_ref().map(|s| s.segments.iter().map(|x| (x.label, x.name.clone())).collect()).unwrap_or_default();
        let colour = self.seg.as_ref().and_then(|s| s.selected_segment()).map_or(Color32::from_rgb(128, 174, 128), |s| s.colour);
        let (Some(st), Some(hist)) = (&mut self.settings, &self.hist) else { return };
        let mut apply = false;
        match effect {
            Effect::Threshold => {
                histogram(ui, hist, &mut st.threshold, colour);
                let speed = ((vol.range.1 - vol.range.0) / 500.0).max(1e-3);
                ui.horizontal(|ui| {
                    ui.label("Range");
                    ui.add(egui::DragValue::new(&mut st.threshold.0).speed(speed));
                    ui.label("–");
                    ui.add(egui::DragValue::new(&mut st.threshold.1).speed(speed));
                });
                ui.horizontal(|ui| {
                    if ui.button("Auto").on_hover_text("Otsu's method: the value that best splits the histogram in two").clicked() {
                        if let Some(t) = hist.otsu() {
                            st.threshold = (t, vol.range.1);
                        }
                    }
                    if ui.button("Use for masking").on_hover_text("Edits then only change voxels in this range").clicked() {
                        self.masking.intensity = Some(st.threshold);
                    }
                    apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
                });
                ui.weak("The preview shows the segment after Apply.");
            }
            Effect::Paint | Effect::Erase => {
                ui.add(egui::Slider::new(&mut st.brush_mm, 0.2..=50.0).logarithmic(true).text("radius (mm)"));
                ui.checkbox(&mut st.sphere, "Sphere brush").on_hover_text("Paint a ball across slices, not a disc in this slice");
                ui.weak("Left drag in a slice view. Shift + wheel: brush size.");
            }
            Effect::FloodFill => {
                ui.radio_value(&mut st.flood_use_threshold, false, "Similar to the clicked voxel, within");
                ui.horizontal(|ui| {
                    ui.add_space(20.0);
                    ui.add(egui::DragValue::new(&mut st.flood_tolerance).speed(((vol.range.1 - vol.range.0) / 500.0).max(1e-3)).range(0.0..=f32::MAX));
                });
                ui.radio_value(&mut st.flood_use_threshold, true, format!("In the threshold range ({:.0} – {:.0})", st.threshold.0, st.threshold.1));
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.flood_min_diameter_mm).speed(0.1).range(0.0..=50.0).suffix(" mm"));
                    ui.label("minimum diameter").on_hover_text("Stops the fill from leaking through gaps narrower than this (0: no limit)");
                });
                ui.weak("Click a structure in a slice view.");
            }
            Effect::Islands => {
                for op in IslandOp::ALL {
                    ui.radio_value(&mut st.islands, op, op.label());
                }
                if matches!(st.islands, IslandOp::RemoveSmall | IslandOp::Split) {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut st.min_island).speed(10.0).range(1..=usize::MAX));
                        ui.label("voxels minimum size");
                    });
                }
                if st.islands.by_click() {
                    ui.weak("Click an island in a slice view.");
                } else {
                    apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
                }
            }
            Effect::Margin => {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.margin_mm).speed(0.1).range(-100.0..=100.0).suffix(" mm"));
                    ui.label(if st.margin_mm >= 0.0 { "grow" } else { "shrink" });
                });
                apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
            }
            Effect::Smoothing => {
                egui::ComboBox::from_id_salt("segmentation.smoothing").selected_text(st.smoothing.label()).show_ui(ui, |ui| {
                    for m in SmoothMethod::ALL {
                        ui.selectable_value(&mut st.smoothing, m, m.label());
                    }
                });
                if st.smoothing != SmoothMethod::FillHoles {
                    ui.horizontal(|ui| {
                        ui.add(egui::DragValue::new(&mut st.smoothing_mm).speed(0.1).range(0.1..=50.0).suffix(" mm"));
                        ui.label("kernel size");
                    });
                }
                apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
            }
            Effect::Logical => {
                egui::ComboBox::from_id_salt("segmentation.logical").selected_text(st.logical.label()).show_ui(ui, |ui| {
                    for op in LogicalOp::ALL {
                        ui.selectable_value(&mut st.logical, op, op.label());
                    }
                });
                if st.logical.needs_other() {
                    let name = st.logical_other.and_then(|l| segments.iter().find(|s| s.0 == l)).map_or("(choose)", |s| s.1.as_str()).to_string();
                    egui::ComboBox::from_id_salt("segmentation.logical_other").selected_text(name).show_ui(ui, |ui| {
                        for (l, n) in &segments {
                            ui.selectable_value(&mut st.logical_other, Some(*l), n);
                        }
                    });
                }
                apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
            }
        }
        if apply {
            let ctx = ui.ctx().clone();
            self.start_op(effect, None, &ctx);
        }
    }

    fn masking_section(&mut self, ui: &mut egui::Ui) {
        let Some(seg) = &self.seg else { return };
        let active = self.masking != Default::default();
        let title = if active { "Masking (on)" } else { "Masking" };
        egui::CollapsingHeader::new(RichText::new(title).strong()).id_salt("segmentation.masking").default_open(false).show(ui, |ui| {
            let name = |l: u8| seg.segments.iter().find(|s| s.label == l).map_or("?".to_string(), |s| s.name.clone());
            let area_label = |a: EditableArea| match a {
                EditableArea::Everywhere => "Everywhere".to_string(),
                EditableArea::InsideAll => "Inside all segments".into(),
                EditableArea::InsideVisible => "Inside visible segments".into(),
                EditableArea::OutsideAll => "Outside all segments".into(),
                EditableArea::OutsideVisible => "Outside visible segments".into(),
                EditableArea::Inside(l) => format!("Inside {}", name(l)),
                EditableArea::Outside(l) => format!("Outside {}", name(l)),
            };
            ui.label("Editable area");
            egui::ComboBox::from_id_salt("segmentation.area").selected_text(area_label(self.masking.area)).width(220.0).show_ui(ui, |ui| {
                let mut areas = vec![EditableArea::Everywhere, EditableArea::InsideAll, EditableArea::InsideVisible, EditableArea::OutsideAll, EditableArea::OutsideVisible];
                areas.extend(seg.segments.iter().map(|s| EditableArea::Inside(s.label)));
                areas.extend(seg.segments.iter().map(|s| EditableArea::Outside(s.label)));
                for a in areas {
                    ui.selectable_value(&mut self.masking.area, a, area_label(a));
                }
            });
            let mut on = self.masking.intensity.is_some();
            if ui.checkbox(&mut on, "Editable intensity range").changed() {
                self.masking.intensity = on.then(|| self.settings.as_ref().map_or((0.0, 1.0), |s| s.threshold));
            }
            if let Some((lo, hi)) = &mut self.masking.intensity {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(lo));
                    ui.label("–");
                    ui.add(egui::DragValue::new(hi));
                });
            }
            ui.label("Modify other segments");
            let ow_label = |o: Overwrite| match o {
                Overwrite::All => "Overwrite all",
                Overwrite::Visible => "Overwrite visible",
                Overwrite::None => "Keep other segments",
            };
            egui::ComboBox::from_id_salt("segmentation.overwrite").selected_text(ow_label(self.masking.overwrite)).width(220.0).show_ui(ui, |ui| {
                for o in [Overwrite::All, Overwrite::Visible, Overwrite::None] {
                    ui.selectable_value(&mut self.masking.overwrite, o, ow_label(o));
                }
            });
            if active && ui.small_button("Reset").clicked() {
                self.masking = Default::default();
            }
        });
    }

    fn output_section(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let Some(st) = &mut self.settings else { return };
        section(ui, "Save and mesh");
        let idle = self.running.is_none();
        let has_segments = self.seg.as_ref().is_some_and(|s| !s.segments.is_empty());
        ui.horizontal(|ui| {
            ui.add(egui::DragValue::new(&mut st.mesh_smoothing_mm).speed(0.05).range(0.0..=5.0).suffix(" mm"));
            ui.label("surface smoothing");
        });
        ui.checkbox(&mut st.mesh_all_visible, "Every visible segment");
        let make = ui.add_enabled(idle && has_segments, egui::Button::new("Make mesh → Mesh tab")).on_hover_text("Closed surface of the segment(s), in LPS millimetres like 3D Slicer's models").clicked();
        let save = ui.add_enabled(idle && has_segments, egui::Button::new("Save segmentation…")).on_hover_text("3D Slicer .seg.nrrd (names and colours) or a NIfTI label map").clicked();
        if make {
            self.make_mesh(&ctx);
        }
        if save {
            self.save_dialog(&ctx);
        }
    }
}

/// The intensity histogram (log scale) with the range in the segment's colour; dragging
/// moves the nearer end of the range.
fn histogram(ui: &mut egui::Ui, h: &Histogram, range: &mut (f32, f32), colour: Color32) {
    let (rect, r) = ui.allocate_exact_size(egui::vec2(ui.available_width().max(100.0), 56.0), egui::Sense::click_and_drag());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, 2.0, ui.visuals().extreme_bg_color);
    let n = h.bins.len();
    let top = h.bins.iter().map(|&c| (c as f32).ln_1p()).fold(0.0, f32::max).max(1.0);
    let bar = rect.width() / n as f32;
    for (b, &c) in h.bins.iter().enumerate() {
        let v = h.value(b as f32 + 0.5);
        let inside = v >= range.0 && v <= range.1;
        let height = (c as f32).ln_1p() / top * (rect.height() - 2.0);
        let x = rect.left() + b as f32 * bar;
        let col = if inside { colour } else { ui.visuals().weak_text_color() };
        painter.rect_filled(egui::Rect::from_min_max(egui::pos2(x, rect.bottom() - height), egui::pos2(x + bar.max(1.0), rect.bottom())), 0.0, col);
    }
    if let Some(p) = r.interact_pointer_pos().filter(|_| r.dragged() || r.clicked()) {
        let v = h.value((p.x - rect.left()) / bar);
        if (v - range.0).abs() <= (v - range.1).abs() {
            range.0 = v.min(range.1);
        } else {
            range.1 = v.max(range.0);
        }
    }
}
