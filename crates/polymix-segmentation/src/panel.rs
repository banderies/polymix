//! The editor panel on the left, as in 3D Slicer's Segment Editor: the segments, the effects
//! and their options, masking, and saving / meshing.

use crate::effects::{local_range, AutoMode, Effect, Histogram, IslandOp, LocalPick, LocalShape, LogicalOp, MaskFill, Refine, ScissorsOp, ScissorsRange, ScissorsShape, SmoothMethod, HOLLOW_SIDES};
use crate::segmentation::{EditableArea, Overwrite, Status};
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
            self.statistics_section(ui);
            self.vessels_section(ui);
            let n = self.measure.items.len();
            egui::CollapsingHeader::new(RichText::new(if n > 0 { format!("Measurements ({n})") } else { "Measurements".into() }).strong()).id_salt("segmentation.measurements").default_open(false).show(ui, |ui| {
                self.measure.list(ui);
            });
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
                polymix_workspace::spinner(ui);
                ui.label(format!("{}…", r.what));
                cancel = ui.small_button("Cancel").clicked();
            });
        }
        if cancel {
            self.running = None;
            self.agent_op = false;
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
        // Showing or hiding a segment changes what edits may do and what the 3D view shows; a
        // colour, status or name changes only how the segments look.
        let (mut redraw, mut restyle) = (false, false);
        // Filter by name and status, when there are enough segments to need it.
        if seg.segments.len() > 4 || !self.filter.0.is_empty() || self.filter.1.is_some() {
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut self.filter.0).hint_text("Filter").desired_width(110.0));
                egui::ComboBox::from_id_salt("segmentation.status_filter").selected_text(self.filter.1.map_or("Any status", |s| s.label())).width(100.0).show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.filter.1, None, "Any status");
                    for st in Status::ALL {
                        ui.selectable_value(&mut self.filter.1, Some(st), st.label());
                    }
                });
            });
        }
        let needle = self.filter.0.to_lowercase();
        for k in 0..seg.segments.len() {
            if !seg.segments[k].name.to_lowercase().contains(&needle) || self.filter.1.is_some_and(|f| f != seg.segments[k].status) {
                continue;
            }
            ui.horizontal(|ui| {
                let selected = seg.selected == Some(k);
                let s = &mut seg.segments[k];
                if ui.checkbox(&mut s.visible, "").on_hover_text("Show the segment").changed() {
                    redraw = true;
                }
                let flag = ui.add(egui::Button::new(RichText::new("●").color(s.status.colour())).frame(false).small()).on_hover_text(format!("{} (click for the next state)", s.status.label()));
                if flag.clicked() {
                    s.status = s.status.next();
                    restyle = true;
                }
                let mut c = [s.colour.r(), s.colour.g(), s.colour.b()];
                if ui.color_edit_button_srgb(&mut c).changed() {
                    s.colour = Color32::from_rgb(c[0], c[1], c[2]);
                    restyle = true;
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
                                restyle = true;
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
        if restyle {
            seg.restyled();
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
        ui.weak("Keys: 1–0 / ⇧1–⇧0 effects · Space the one before · i intensity masking · q / w segments · ← → slices · − / + brush").on_hover_text("As in 3D Slicer's Segment Editor. Esc drops a shape being drawn, then the effect.");
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
                    egui::ComboBox::from_id_salt("segmentation.auto_method").selected_text(st.auto_method.label()).width(150.0).show_ui(ui, |ui| {
                        for m in mm_segment::AutoThreshold::ALL {
                            ui.selectable_value(&mut st.auto_method, m, m.label());
                        }
                    });
                    ui.selectable_value(&mut st.auto_mode, AutoMode::Lower, "↑").on_hover_text("From the threshold up (brighter structures)");
                    ui.selectable_value(&mut st.auto_mode, AutoMode::Upper, "↓").on_hover_text("Up to the threshold (darker structures)");
                    if ui.button("Auto").on_hover_text("Set the range from the automatic threshold of the method chosen").clicked() {
                        if let Some(r) = hist.auto_range(st.auto_method, st.auto_mode) {
                            st.threshold = r;
                        }
                    }
                });
                ui.checkbox(&mut st.local_histogram, "Local histogram").on_hover_text("Drag a box, circle or line over the structure in a slice view: the range is set from the values in it");
                if st.local_histogram {
                    ui.horizontal(|ui| {
                        for s in LocalShape::ALL {
                            ui.selectable_value(&mut st.local_shape, s, s.label());
                        }
                    });
                    let (lower, upper) = (st.local_lower, st.local_upper);
                    ui.horizontal(|ui| {
                        ui.label("Lower");
                        for p in LocalPick::ALL {
                            ui.selectable_value(&mut st.local_lower, p, p.label(false));
                        }
                    });
                    ui.horizontal(|ui| {
                        ui.label("Upper");
                        for p in [LocalPick::Average, LocalPick::Percentile, LocalPick::Extreme] {
                            ui.selectable_value(&mut st.local_upper, p, p.label(true));
                        }
                    });
                    match &self.local {
                        Some(l) => {
                            if (lower, upper) != (st.local_lower, st.local_upper) {
                                if let Some(r) = local_range(&l.values, st.local_lower, st.local_upper) {
                                    st.threshold = r;
                                }
                            }
                            let mut range = st.threshold;
                            histogram(ui, &l.hist, &mut range, colour);
                            st.threshold = range;
                            let n = l.values.len() as f64;
                            let mean = l.values.iter().map(|&v| v as f64).sum::<f64>() / n;
                            let sd = (l.values.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
                            ui.weak(format!("{} voxels: {:.4} ± {:.3}, {:.4} – {:.4}", l.values.len(), mean, sd, l.values[0], l.values[l.values.len() - 1]));
                        }
                        None => {
                            ui.weak("Drag over the structure in a slice view.");
                        }
                    }
                }
                ui.horizontal(|ui| {
                    if ui.button("Use for masking").on_hover_text("Edits then only change voxels in this range").clicked() {
                        self.masking.intensity = Some(st.threshold);
                    }
                    apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
                });
                ui.weak("The preview shows the segment after Apply.");
            }
            Effect::Paint | Effect::Erase => {
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut st.brush_relative, false, "mm");
                    ui.selectable_value(&mut st.brush_relative, true, "% of view").on_hover_text("The brush keeps its size on screen as you zoom: its diameter is a share of the view's smaller side");
                });
                if st.brush_relative {
                    ui.add(egui::Slider::new(&mut st.brush_percent, 0.5..=50.0).logarithmic(true).suffix(" %").text(format!("diameter ({:.1} mm radius)", st.brush_mm)));
                } else {
                    ui.add(egui::Slider::new(&mut st.brush_mm, 0.2..=50.0).logarithmic(true).text("radius (mm)"));
                }
                ui.checkbox(&mut st.sphere, "Sphere brush").on_hover_text("Paint a ball across slices, not a disc in this slice");
                ui.checkbox(&mut st.pixel_mode, "Pixel mode").on_hover_text("Single voxels: only the voxels the pointer passes over");
                ui.weak("Left drag in a slice view. Shift + wheel: brush size.");
                ui.weak("3D view: ⌘ / Ctrl + drag paints a ball on the surface under the pointer; a click adds one.");
            }
            Effect::Draw => {
                let points = self.sketch.as_ref().map_or(0, |s| s.points.len());
                ui.weak("In a slice view: click points, or drag a line, around the region.");
                ui.weak("Double-click or Enter fills it into the segment in that slice; x takes back the last point; Esc starts again.");
                ui.horizontal(|ui| {
                    ui.label(format!("{points} point{}", if points == 1 { "" } else { "s" }));
                    if ui.add_enabled(points >= 3, egui::Button::new("Fill")).clicked() {
                        let vol = vol.clone();
                        self.finish_draw(&vol);
                    }
                    if ui.add_enabled(points > 0, egui::Button::new("Clear")).clicked() {
                        self.sketch = None;
                    }
                });
            }
            Effect::LevelTracing => {
                ui.radio_value(&mut st.trace_darker, false, "Brighter: values at least the pointer's");
                ui.radio_value(&mut st.trace_darker, true, "Darker: values at most the pointer's");
                ui.weak("Move the pointer over a slice view: the connected region in that slice is outlined. Click to add it to the segment.");
            }
            Effect::Scissors => {
                ui.horizontal_wrapped(|ui| {
                    for op in ScissorsOp::ALL {
                        ui.selectable_value(&mut st.scissors_op, op, op.label());
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Shape");
                    for sh in ScissorsShape::ALL {
                        ui.selectable_value(&mut st.scissors_shape, sh, sh.label());
                    }
                });
                ui.horizontal(|ui| {
                    ui.label("Slices");
                    egui::ComboBox::from_id_salt("segmentation.scissors_range").selected_text(st.scissors_range.label()).show_ui(ui, |ui| {
                        for r in ScissorsRange::ALL {
                            ui.selectable_value(&mut st.scissors_range, r, r.label()).on_hover_text(match r {
                                ScissorsRange::Unlimited => "Every slice",
                                ScissorsRange::Positive => "This slice and those towards the patient's superior, anterior or right",
                                ScissorsRange::Negative => "This slice and those towards the patient's inferior, posterior or left",
                                ScissorsRange::Symmetric => "A slab of the thickness, centred on this slice",
                            });
                        }
                    });
                    if st.scissors_range == ScissorsRange::Symmetric {
                        ui.add(egui::DragValue::new(&mut st.scissors_mm).speed(0.2).range(0.1..=500.0).suffix(" mm"));
                    }
                });
                ui.weak("Drag the shape in a slice view; it cuts when you let go. In the 3D view: ⌘ / Ctrl + drag, and it cuts through the whole depth.");
            }
            Effect::GrowFromSeeds | Effect::Watershed | Effect::FillBetweenSlices => {
                if effect == Effect::FillBetweenSlices {
                    ui.horizontal(|ui| {
                        ui.label("Between slices of");
                        for p in crate::PLANES {
                            ui.selectable_value(&mut st.fill_plane, p, p.label());
                        }
                    });
                } else if effect == Effect::GrowFromSeeds {
                    ui.add(egui::Slider::new(&mut st.seed_locality, 0.0..=10.0).text("seed locality")).on_hover_text("0: only likeness counts; higher: nearer seeds win more easily");
                } else {
                    ui.add(egui::Slider::new(&mut st.watershed_mm, 0.0..=5.0).suffix(" mm").text("object scale")).on_hover_text("Smoothing before the edges are found: higher gives smoother boundaries but rounds small structures");
                }
                if effect != Effect::FillBetweenSlices {
                    ui.add(egui::Slider::new(&mut st.seed_margin_mm, 1.0..=100.0).logarithmic(true).suffix(" mm").text("margin")).on_hover_text("Works within this far of the seeds");
                }
                ui.checkbox(&mut st.auto_update, "Auto-update").on_hover_text("Update the result after every edit of the seeds");
                let previewing = self.seeded.as_ref().is_some_and(|s| s.effect == effect);
                ui.horizontal(|ui| {
                    let start = ui.add_enabled(idle, egui::Button::new(if previewing { "Update" } else { "Initialize" })).on_hover_text("Grow from the visible segments (at least two: the structure and its surroundings)").clicked();
                    let keep = ui.add_enabled(idle && previewing, egui::Button::new("Apply")).clicked();
                    let drop = ui.add_enabled(previewing, egui::Button::new("Cancel")).clicked();
                    let ctx = ui.ctx().clone();
                    if start {
                        self.start_seeded(effect, &ctx);
                    }
                    if keep {
                        self.apply_seeded();
                    }
                    if drop {
                        self.seeded = None;
                        self.preview = None;
                    }
                });
                ui.weak(if effect == Effect::FillBetweenSlices {
                    "Segment the structure on every few slices (Paint, Draw, Level tracing…); the empty slices between are filled by interpolating the shapes of the visible segments. The preview shows the result; Apply keeps it."
                } else {
                    "Paint seeds into two or more visible segments: the structure, and what surrounds it. The preview shows the result; Apply keeps it."
                });
            }
            Effect::MaskVolume => {
                egui::ComboBox::from_id_salt("segmentation.mask_fill").selected_text(st.mask_fill.label()).show_ui(ui, |ui| {
                    for f in MaskFill::ALL {
                        ui.selectable_value(&mut st.mask_fill, f, f.label());
                    }
                });
                let speed = ((vol.range.1 - vol.range.0) / 500.0).max(1e-3);
                ui.horizontal(|ui| {
                    if st.mask_fill != MaskFill::Outside {
                        ui.label("inside");
                        ui.add(egui::DragValue::new(&mut st.mask_inside).speed(speed));
                    }
                    if st.mask_fill != MaskFill::Inside {
                        ui.label("outside");
                        ui.add(egui::DragValue::new(&mut st.mask_outside).speed(speed));
                    }
                });
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.mask_soft_mm).speed(0.05).range(0.0..=20.0).suffix(" mm"));
                    ui.label("soft edge").on_hover_text("Blend across the segment's boundary over a Gaussian of this size (0: sharp)");
                });
                ui.weak("Saves a new image among the derived images: it appears in the Volume tab, where it can be selected.");
                apply = ui.add_enabled(idle, egui::Button::new("Save masked volume")).clicked();
            }
            Effect::Engrave => {
                ui.horizontal(|ui| {
                    ui.label("Text");
                    ui.add(egui::TextEdit::singleline(&mut st.engrave_text).desired_width(140.0));
                });
                ui.add(egui::Slider::new(&mut st.engrave_mm, 1.0..=50.0).logarithmic(true).suffix(" mm").text("letter height"));
                ui.add(egui::Slider::new(&mut st.engrave_depth_mm, 0.2..=20.0).logarithmic(true).suffix(" mm").text("depth"));
                ui.horizontal(|ui| {
                    ui.radio_value(&mut st.engrave_raise, false, "Engrave (cut in)");
                    ui.radio_value(&mut st.engrave_raise, true, "Emboss (raise)");
                });
                ui.weak("Click where the text goes in a slice view: it reads left to right there, through the depth around that slice. Letters A–Z, digits and - . + / :");
            }
            Effect::LevelSet => {
                ui.label(format!("Vessel values {:.0} – {:.0} (Threshold's range)", st.threshold.0, st.threshold.1));
                ui.add(egui::Slider::new(&mut st.ls_iterations, 10..=3000).logarithmic(true).text("iterations"));
                ui.add(egui::Slider::new(&mut st.ls_max_mm, 0.5..=100.0).logarithmic(true).suffix(" mm").text("moves at most"));
                ui.add(egui::Slider::new(&mut st.ls_curvature, 0.0..=2.0).text("smoothness"));
                ui.checkbox(&mut st.ls_overlay, "Follow the overlay (e.g. vesselness)").on_hover_text("The overlay's window decides: above its centre the front moves out, below it moves back. Show it with Overlay ▾ in the toolbar");
                ui.weak("Start from a rough segment inside the vessel (painted or thresholded).");
                apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
            }
            Effect::SplitVolume => {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.split_pad_mm).speed(0.1).range(0.0..=100.0).suffix(" mm"));
                    ui.label("margin around each segment");
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut st.split_blank, "Fill outside the segment with");
                    let speed = ((vol.range.1 - vol.range.0) / 500.0).max(1e-3);
                    ui.add_enabled(st.split_blank, egui::DragValue::new(&mut st.mask_outside).speed(speed));
                });
                ui.weak("One image per visible segment, saved among the derived images (Volume tab).");
                apply = ui.add_enabled(idle, egui::Button::new("Split")).clicked();
            }
            Effect::FastMarching => {
                ui.add(egui::Slider::new(&mut st.march_percent, 0.01..=20.0).logarithmic(true).suffix(" %").text("of the volume")).on_hover_text("The segment grows over the most similar voxels nearest to it until it is this big");
                ui.weak("Paint a few voxels of the structure into the segment first.");
                apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
            }
            Effect::DrawTube | Effect::SurfaceCut | Effect::GuidedVessel => {
                if effect == Effect::DrawTube {
                    ui.add(egui::Slider::new(&mut st.tube_mm, 0.1..=20.0).logarithmic(true).suffix(" mm").text("radius"));
                    ui.weak("Click points along the structure in the slice views (any view, any slice) or on the 3D view's surface: the tube runs smoothly through them in order.");
                } else if effect == Effect::GuidedVessel {
                    ui.add(egui::Slider::new(&mut st.tube_mm, 0.2..=20.0).logarithmic(true).suffix(" mm").text("about the vessel's radius"));
                    ui.label(format!("Vessel values {:.0} – {:.0} (Threshold's range)", st.threshold.0, st.threshold.1));
                    ui.weak("Click points along the vessel in order (its ends and a few between, in any view): the path between them follows the vessel, then a level set grows to its wall.");
                } else {
                    ui.weak("Click points on the structure's surface in the slice views (several slices) or in the 3D view: the closed shape through them is added.");
                }
                let n = self.markers.len();
                ui.horizontal(|ui| {
                    ui.label(format!("{n} point{}", if n == 1 { "" } else { "s" }));
                    if ui.add_enabled(n > 0, egui::Button::new("Remove last")).clicked() {
                        self.markers.pop();
                    }
                    if ui.add_enabled(n > 0, egui::Button::new("Clear")).clicked() {
                        self.markers.clear();
                    }
                    if ui.add_enabled(n > 0 && idle, egui::Button::new("Apply")).clicked() {
                        if effect == Effect::GuidedVessel {
                            let ctx = ui.ctx().clone();
                            self.start_guided(&ctx);
                        } else {
                            self.apply_markers(effect);
                        }
                    }
                });
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
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("segmentation.flood_refine").selected_text(st.flood_refine.label()).show_ui(ui, |ui| {
                        for r in Refine::ALL {
                            ui.selectable_value(&mut st.flood_refine, r, r.label());
                        }
                    }).response.on_hover_text("Edges: as the threshold leaves them, or decided again near them by Grow from seeds or Watershed (smoother on noisy images)");
                    if st.flood_refine != Refine::None {
                        ui.add(egui::DragValue::new(&mut st.flood_feature_mm).speed(0.05).range(0.1..=20.0).suffix(" mm")).on_hover_text("Feature size: how far from the thresholded edge the edge may move");
                    }
                });
                ui.weak("Click a structure in a slice view or in the 3D view.");
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
                    ui.weak("Click an island in a slice view or in the 3D view.");
                } else {
                    apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
                }
            }
            Effect::Margin => {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.margin_mm).speed(0.1).range(-100.0..=100.0).suffix(" mm"));
                    ui.label(if st.margin_mm >= 0.0 { "grow" } else { "shrink" });
                });
                ui.checkbox(&mut st.apply_visible, "Apply to visible segments");
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
                        ui.label(if st.smoothing.sigma() { "standard deviation" } else { "kernel size" });
                    });
                }
                if st.smoothing == SmoothMethod::Joint {
                    ui.weak("Every visible segment is smoothed together: no gaps or overlaps open where they touch.");
                } else {
                    ui.checkbox(&mut st.apply_visible, "Apply to visible segments");
                }
                apply = ui.add_enabled(idle, egui::Button::new("Apply")).clicked();
            }
            Effect::Hollow => {
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut st.hollow_mm).speed(0.05).range(0.05..=50.0).suffix(" mm"));
                    ui.label("shell thickness");
                });
                let side = HOLLOW_SIDES.iter().find(|h| h.0 == st.hollow_side).map_or("", |h| h.1);
                egui::ComboBox::from_id_salt("segmentation.hollow").selected_text(side).show_ui(ui, |ui| {
                    for (s, label, _) in HOLLOW_SIDES {
                        ui.selectable_value(&mut st.hollow_side, s, label);
                    }
                });
                ui.weak("For a printable wall, Thicken in the Mesh tab gives an exact thickness on the mesh.");
                ui.checkbox(&mut st.apply_visible, "Apply to visible segments");
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
                ui.checkbox(&mut st.bypass_masking, "Bypass masking").on_hover_text("Ignore the editable area, intensity range and overwrite settings for this operation");
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

    /// Segment statistics: volume, surface, intensities and shape of every segment.
    fn statistics_section(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        let Some(seg) = &self.seg else { return };
        let idle = self.running.is_none();
        let mut measure = false;
        egui::CollapsingHeader::new(RichText::new("Statistics").strong()).id_salt("segmentation.statistics").default_open(false).show(ui, |ui| {
            ui.horizontal(|ui| {
                measure = ui.add_enabled(idle && !seg.segments.is_empty(), egui::Button::new("Measure segments")).clicked();
                if let Some((version, rows)) = &self.stats {
                    if ui.button("Copy table").on_hover_text("Every measure, as CSV for a spreadsheet").clicked() {
                        ui.ctx().copy_text(crate::stats_csv(rows, seg));
                    }
                    if *version != seg.version {
                        ui.weak("(segments changed since)");
                    }
                }
            });
            let Some((_, rows)) = &self.stats else {
                ui.weak("Volume, surface, intensities and shape of each segment.");
                return;
            };
            egui::Grid::new("segmentation.stats").num_columns(5).striped(true).spacing([10.0, 2.0]).show(ui, |ui| {
                for h in ["Segment", "mL", "cm²", "Mean ± SD", "Max ⌀ mm"] {
                    ui.label(RichText::new(h).small().strong());
                }
                ui.end_row();
                for (label, s) in rows {
                    let Some(g) = seg.segments.iter().find(|g| g.label == *label) else { continue };
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("●").color(g.colour));
                        let name: String = if g.name.chars().count() > 16 { g.name.chars().take(15).chain(['…']).collect() } else { g.name.clone() };
                        ui.label(name).on_hover_text(format!(
                            "{}\n{} voxels\nmedian {:.4}, range {:.4} – {:.4}\nroundness {:.3} · elongation {:.3}\nprincipal axes {:.1} × {:.1} × {:.1} mm\ncentre (RAS) {:.1}, {:.1}, {:.1} mm",
                            g.name,
                            polymix_workspace::style::format_count(s.voxels),
                            s.median,
                            s.min,
                            s.max,
                            s.roundness,
                            s.elongation,
                            s.principal_mm[0],
                            s.principal_mm[1],
                            s.principal_mm[2],
                            s.centroid_ras[0],
                            s.centroid_ras[1],
                            s.centroid_ras[2],
                        ));
                    });
                    ui.label(format!("{:.3}", s.volume_mm3 / 1000.0));
                    ui.label(format!("{:.2}", s.surface_mm2 / 100.0));
                    ui.label(format!("{:.4} ± {:.3}", s.mean, s.sd));
                    ui.label(format!("{:.1}", s.feret_mm));
                    ui.end_row();
                }
            });
        });
        if measure {
            self.measure(&ctx);
        }
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
        let load = ui.add_enabled(idle, egui::Button::new("Load segmentation…")).on_hover_text("Add the segments of a 3D Slicer .seg.nrrd or a NIfTI label map (read onto this volume's grid)").clicked();
        ui.weak("Meshes come in from the Mesh tab: right-click an object → Send to Segmentation.");
        if make {
            self.make_mesh(&ctx);
        }
        if save {
            self.save_dialog(&ctx);
        }
        if load {
            self.load_dialog(&ctx);
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
