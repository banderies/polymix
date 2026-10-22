//! The editor panel on the left, as in 3D Slicer's Segment Editor: the segments, the effects
//! and their options, masking, and saving / meshing.

use crate::card::{self, Card, Ready};
use crate::drag;
use crate::effects::{local_range, AutoMode, Effect, Group, Histogram, IslandOp, LocalPick, LocalShape, LogicalOp, MaskFill, Refine, ScissorsOp, ScissorsRange, ScissorsShape, SmoothMethod, HOLLOW_SIDES};
use crate::segmentation::{EditableArea, Masking, Overwrite, Status};
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
        });
        ui.horizontal(|ui| {
            ui.label("Opacity").on_hover_text("How solid the segments' colours are in the slice views (0: hidden)");
            ui.add(egui::Slider::new(&mut self.opacity, 0.0..=1.0).custom_formatter(|v, _| format!("{:.0} %", v * 100.0)));
        });
        if !seg.segments.is_empty() {
            ui.weak("☑ shown · ● status (click: next) · colour · name (double-click: rename) · volume");
        }
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
        // The list, by group; folded to a line once an effect is chosen, so its card sits
        // right below.
        let list_id = egui::Id::NULL.with("segmentation.effects_list");
        let mut open = self.effect.is_none() || ui.data(|d| d.get_temp::<bool>(list_id).unwrap_or(false));
        if let (Some(e), false) = (self.effect, open) {
            ui.horizontal(|ui| {
                if ui.small_button("◀ All effects").on_hover_text("Show every effect, to choose another").clicked() {
                    open = true;
                }
                ui.label(RichText::new(e.group().label()).small().color(ui.visuals().weak_text_color()));
            });
        } else {
            for g in Group::ALL {
                ui.add_space(2.0);
                ui.label(RichText::new(g.label()).small().strong().color(ui.visuals().weak_text_color())).on_hover_text(g.hint());
                ui.horizontal_wrapped(|ui| {
                    ui.spacing_mut().item_spacing.x = 4.0;
                    for e in Effect::LISTED.into_iter().filter(|e| e.group() == g) {
                        if ui.selectable_label(self.effect == Some(e), e.label()).on_hover_text(e.summary()).clicked() {
                            self.effect = if self.effect == Some(e) { None } else { Some(e) };
                            open = false;
                        }
                    }
                });
            }
        }
        // Open on request only: the next effect chosen (by click or key) folds it again.
        ui.data_mut(|d| d.insert_temp(list_id, open && self.effect.is_some()));
        let Some(effect) = self.effect else {
            ui.add_space(4.0);
            ui.weak("Choose an effect: its card opens here, with its steps in order.");
            return;
        };
        ui.add_space(6.0);
        let ctx = ui.ctx().clone();
        let idle = self.running.is_none();
        let Some(seg) = &mut self.seg else { return };
        let counts = seg.counts().to_vec();
        let rows: Vec<SegRow> = seg.segments.iter().map(|s| SegRow { label: s.label, name: s.name.clone(), colour: s.colour, visible: s.visible, voxels: counts[s.label as usize] }).collect();
        let selected = seg.selected;
        let colour = seg.selected_segment().map_or(Color32::from_rgb(128, 174, 128), |s| s.colour);
        let visible_with_voxels = rows.iter().filter(|r| r.visible && r.voxels > 0).count();
        let markers = self.markers.len();
        let sketch_points = self.sketch.as_ref().map_or(0, |s| s.points.len());
        let previewing = self.seeded.as_ref().is_some_and(|s| s.effect == effect);
        let region = match effect {
            Effect::ThresholdGrow => self.grow.as_ref(),
            Effect::SelectPart => self.part.as_ref(),
            _ => None,
        }
        .map(|r| Region { mm: r.mm, stats: r.stats(vol), ignored: r.ignored, label: r.label(), dragging: r.dragging() });
        let (Some(st), Some(hist)) = (&mut self.settings, &self.hist) else { return };
        let local = &self.local;
        let masking = &mut self.masking;
        let ml = |mm3: f64| mm3 / 1000.0;
        let mut acts: Vec<Act> = Vec::new();
        let mut close = false;
        let help = format!(
            "{}.\n\nKeys: 1–0 and ⇧1–⇧0 choose effects · Space: the effect before · i: limit to the threshold range on / off · q / w: previous / next segment · ← →: slices · − / +: brush size · Esc: drop what is being drawn or the points, then the effect.",
            effect.hint()
        );
        card::show(ui, effect.key(), effect.label(), effect.summary(), &help, &mut close, |ui, card| {
            match effect {
                Effect::Threshold => {
                    segment_step(card, ui, "Segment to fill", &rows, selected, None, &mut acts);
                    card.step(ui, "Range of values", Ready::Either, |ui| {
                        range_ui(ui, hist, &mut st.threshold, colour, vol);
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("segmentation.auto_method").selected_text(st.auto_method.label()).width(130.0).show_ui(ui, |ui| {
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
                        ui.checkbox(&mut st.local_histogram, "From a region of the image").on_hover_text("Drag a box, circle or line over the structure in a slice view: the range is set from the values in it");
                        if st.local_histogram {
                            local_histogram_ui(ui, st, local, colour);
                        }
                    });
                    card.step(ui, "Check the preview", Ready::Either, |ui| {
                        ui.weak("The views show the segment as Apply will leave it: the voxels in the range in it, the rest out.");
                    });
                    card.step(ui, "Apply", Ready::Todo, |ui| {
                        ui.horizontal(|ui| {
                            if ui.add_enabled(idle, egui::Button::new("Apply")).clicked() {
                                acts.push(Act::Apply);
                            }
                            if ui.button("Use as a limit").on_hover_text("Other effects then only change voxels in this range (Limits)").clicked() {
                                masking.intensity = Some(st.threshold);
                            }
                        });
                    });
                }
                Effect::ThresholdGrow => {
                    segment_step(card, ui, "Segment to grow", &rows, selected, None, &mut acts);
                    card.step(ui, "Range of values", Ready::Either, |ui| {
                        range_ui(ui, hist, &mut st.threshold, colour, vol);
                        ui.weak("The same range as Threshold's. Growth only passes through voxels in it.");
                    });
                    card.step(ui, "Seeds", Ready::when(region.is_some()), |ui| {
                        ui.weak("Click in the structure in a slice view (or on the 3D surface): each click adds a seed. Seeds must be in the range.");
                        points_row(ui, markers, &mut acts);
                        if let Some(r) = region.as_ref().filter(|r| r.ignored > 0) {
                            ui.colored_label(crate::WARN, format!("{} seed(s) outside the range are left out", r.ignored));
                        } else if markers > 0 && region.is_none() {
                            ui.colored_label(crate::WARN, "No seed is in the range");
                        }
                    });
                    card.step(ui, "Grow", Ready::when(region.as_ref().is_some_and(|r| r.stats.voxels > 0)), |ui| {
                        ui.weak("⌘ / Ctrl + right drag anywhere in a slice view: right grows from the seeds, left shrinks. Each drag goes on from the last, so scroll to other slices to check and drag again.");
                        reach_row(ui, region.as_ref(), &mut acts);
                        if let Some(r) = &region {
                            ui.label(format!("{} voxels · {:.2} mL", r.stats.voxels, ml(r.stats.volume_mm3)));
                        }
                    });
                    card.step(ui, "Apply", Ready::Todo, |ui| {
                        let ready = idle && region.as_ref().is_some_and(|r| r.stats.voxels > 0 && !r.dragging);
                        ui.horizontal(|ui| {
                            if ui.add_enabled(ready, egui::Button::new("Apply")).on_hover_text("Keep the growth in the segment (one undo step)").clicked() {
                                acts.push(Act::ApplyGrow);
                            }
                            if ui.add_enabled(ready, egui::Button::new("Cancel")).on_hover_text("Drop the growth (the seeds stay)").clicked() {
                                acts.push(Act::SetReach(0.0));
                            }
                        });
                        ui.weak("Until then the segment is unchanged: the views show a preview.");
                    });
                }
                Effect::SelectPart => {
                    card.step(ui, "Seeds", Ready::when(region.is_some()), |ui| {
                        ui.weak("Click in the segment to take a part of (in a slice view, or on the 3D surface): each click adds a seed.");
                        points_row(ui, markers, &mut acts);
                        if let Some(r) = region.as_ref().filter(|r| r.ignored > 0) {
                            ui.colored_label(crate::WARN, format!("{} seed(s) in other segments are left out", r.ignored));
                        } else if markers > 0 && region.is_none() {
                            ui.colored_label(crate::WARN, "The first seed is not in a segment");
                        }
                    });
                    card.step(ui, "Select", Ready::when(region.as_ref().is_some_and(|r| r.stats.voxels > 0)), |ui| {
                        ui.weak("⌘ / Ctrl + right drag anywhere in a slice view: right takes more of the segment around the seeds, left less. The part shows in yellow.");
                        reach_row(ui, region.as_ref(), &mut acts);
                        if let Some(r) = region.as_ref().filter(|r| r.stats.voxels > 0) {
                            let s = &r.stats;
                            ui.label(format!("Part of {}: {} voxels · {:.2} mL", name_of(&rows, r.label), s.voxels, ml(s.volume_mm3)));
                            ui.weak(format!("Reaches {:.1} mm along the segment · box {:.1} × {:.1} × {:.1} mm", s.reach_mm, s.extent_mm[0], s.extent_mm[1], s.extent_mm[2]));
                        }
                    });
                    card.step(ui, "Use the part", Ready::Todo, |ui| {
                        let ready = region.as_ref().is_some_and(|r| r.stats.voxels > 0 && !r.dragging);
                        let label = region.as_ref().map_or(0, |r| r.label);
                        ui.horizontal(|ui| {
                            if ui.add_enabled(ready, egui::Button::new("Erase")).on_hover_text("Remove the part from the segment").clicked() {
                                acts.push(Act::UsePart(drag::PartAction::Erase));
                            }
                            if ui.add_enabled(ready, egui::Button::new("Keep only")).on_hover_text("Remove the rest of the segment").clicked() {
                                acts.push(Act::UsePart(drag::PartAction::KeepOnly));
                            }
                        });
                        ui.horizontal(|ui| {
                            if st.part_target.is_some_and(|l| l == label || !rows.iter().any(|x| x.label == l)) {
                                st.part_target = None;
                            }
                            if ui.add_enabled(ready, egui::Button::new("Move to")).on_hover_text("Split the part off into another segment").clicked() {
                                acts.push(Act::UsePart(drag::PartAction::MoveTo(st.part_target)));
                            }
                            let target = st.part_target.map_or("a new segment".to_string(), |l| name_of(&rows, l));
                            egui::ComboBox::from_id_salt("segmentation.part_target").selected_text(target).show_ui(ui, |ui| {
                                ui.selectable_value(&mut st.part_target, None, "a new segment");
                                for r in rows.iter().filter(|x| x.label != label) {
                                    ui.selectable_value(&mut st.part_target, Some(r.label), &r.name);
                                }
                            });
                        });
                    });
                }
                Effect::Paint | Effect::Erase => {
                    let erase = effect == Effect::Erase;
                    segment_step(card, ui, if erase { "Segment to erase from" } else { "Segment to paint" }, &rows, selected, None, &mut acts);
                    card.step(ui, "Brush", Ready::Either, |ui| {
                        ui.horizontal(|ui| {
                            ui.selectable_value(&mut st.brush_relative, false, "mm");
                            ui.selectable_value(&mut st.brush_relative, true, "% of view").on_hover_text("The brush keeps its size on screen as you zoom: its diameter is a share of the view's smaller side");
                        });
                        if st.brush_relative {
                            ui.add(egui::Slider::new(&mut st.brush_percent, 0.5..=50.0).logarithmic(true).suffix(" %").text(format!("diameter ({:.1} mm radius)", st.brush_mm)));
                        } else {
                            ui.add(egui::Slider::new(&mut st.brush_mm, 0.2..=50.0).logarithmic(true).text("radius (mm)"));
                        }
                        ui.checkbox(&mut st.sphere, "Sphere brush").on_hover_text("A ball across slices, not a disc in this slice");
                        ui.checkbox(&mut st.pixel_mode, "Pixel mode").on_hover_text("Single voxels: only the voxels the pointer passes over");
                    });
                    card.step(ui, if erase { "Erase" } else { "Paint" }, Ready::Todo, |ui| {
                        ui.weak("Left drag in a slice view (⇧ + wheel: brush size). In the 3D view, ⌘ / Ctrl + drag on the surface; a click adds one ball. Each stroke is one undo step.");
                    });
                }
                Effect::Draw => {
                    segment_step(card, ui, "Segment to fill", &rows, selected, None, &mut acts);
                    card.step(ui, "Outline", Ready::when(sketch_points >= 3), |ui| {
                        ui.weak("In a slice view, click points around the region, or drag a line. x takes back the last point.");
                        ui.horizontal(|ui| {
                            ui.label(format!("{sketch_points} point{}", if sketch_points == 1 { "" } else { "s" }));
                            if ui.add_enabled(sketch_points > 0, egui::Button::new("Clear")).clicked() {
                                acts.push(Act::ClearSketch);
                            }
                        });
                    });
                    card.step(ui, "Fill", Ready::Todo, |ui| {
                        if ui.add_enabled(sketch_points >= 3, egui::Button::new("Fill")).clicked() {
                            acts.push(Act::FinishDraw);
                        }
                        ui.weak("Or double-click, or press Enter: the outline is filled into the segment in that slice.");
                    });
                }
                Effect::LevelTracing => {
                    segment_step(card, ui, "Segment to fill", &rows, selected, None, &mut acts);
                    card.step(ui, "Brighter or darker", Ready::Either, |ui| {
                        ui.radio_value(&mut st.trace_darker, false, "Brighter: values at least the pointer's");
                        ui.radio_value(&mut st.trace_darker, true, "Darker: values at most the pointer's");
                    });
                    card.step(ui, "Trace", Ready::Todo, |ui| {
                        ui.weak("Point at a structure in a slice view: the connected region in that slice is outlined. Click to add it.");
                    });
                }
                Effect::FloodFill => {
                    segment_step(card, ui, "Segment to fill", &rows, selected, None, &mut acts);
                    card.step(ui, "Which values", Ready::Either, |ui| {
                        ui.radio_value(&mut st.flood_use_threshold, false, "Similar to the clicked voxel, within");
                        ui.horizontal(|ui| {
                            ui.add_space(20.0);
                            ui.add(egui::DragValue::new(&mut st.flood_tolerance).speed(((vol.range.1 - vol.range.0) / 500.0).max(1e-3)).range(0.0..=f32::MAX));
                        });
                        ui.radio_value(&mut st.flood_use_threshold, true, format!("In the threshold range ({:.0} – {:.0})", st.threshold.0, st.threshold.1));
                    });
                    card.step(ui, "Leaks and edges", Ready::Either, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut st.flood_min_diameter_mm).speed(0.1).range(0.0..=50.0).suffix(" mm"));
                            ui.label("narrowest gap it passes").on_hover_text("Stops the fill from leaking through gaps narrower than this (0: no limit)");
                        });
                        ui.horizontal(|ui| {
                            egui::ComboBox::from_id_salt("segmentation.flood_refine")
                                .selected_text(st.flood_refine.label())
                                .show_ui(ui, |ui| {
                                    for r in Refine::ALL {
                                        ui.selectable_value(&mut st.flood_refine, r, r.label());
                                    }
                                })
                                .response
                                .on_hover_text("Edges: as the values leave them, or decided again near them by Grow from seeds or Watershed (smoother on noisy images)");
                            if st.flood_refine != Refine::None {
                                ui.add(egui::DragValue::new(&mut st.flood_feature_mm).speed(0.05).range(0.1..=20.0).suffix(" mm")).on_hover_text("Feature size: how far from the first edge the edge may move");
                            }
                        });
                    });
                    card.step(ui, "Fill", Ready::Todo, |ui| {
                        ui.weak("Click the structure in a slice view or in the 3D view.");
                    });
                }
                Effect::Scissors => {
                    segment_step(card, ui, "Segment to cut or fill", &rows, selected, None, &mut acts);
                    card.step(ui, "What the shape does", Ready::Either, |ui| {
                        ui.horizontal_wrapped(|ui| {
                            for op in ScissorsOp::ALL {
                                ui.selectable_value(&mut st.scissors_op, op, op.label());
                            }
                        });
                    });
                    card.step(ui, "Shape and depth", Ready::Either, |ui| {
                        ui.horizontal(|ui| {
                            for sh in ScissorsShape::ALL {
                                ui.selectable_value(&mut st.scissors_shape, sh, sh.label());
                            }
                        });
                        ui.horizontal(|ui| {
                            ui.label("Through");
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
                    });
                    card.step(ui, "Cut", Ready::Todo, |ui| {
                        ui.weak("Drag the shape in a slice view; it cuts when you let go. In the 3D view, ⌘ / Ctrl + drag cuts through the whole depth.");
                    });
                }
                Effect::Islands => {
                    segment_step(card, ui, "Segment", &rows, selected, Some("It has no voxels yet: make it first."), &mut acts);
                    card.step(ui, "What to do with its pieces", Ready::Either, |ui| {
                        for op in IslandOp::ALL {
                            ui.radio_value(&mut st.islands, op, op.label());
                        }
                        if matches!(st.islands, IslandOp::RemoveSmall | IslandOp::Split) {
                            ui.horizontal(|ui| {
                                ui.add(egui::DragValue::new(&mut st.min_island).speed(10.0).range(1..=usize::MAX));
                                ui.label("voxels minimum size");
                            });
                        }
                    });
                    if st.islands.by_click() {
                        card.step(ui, "Pick the island", Ready::Todo, |ui| {
                            ui.weak("Click an island in a slice view or in the 3D view.");
                        });
                    } else {
                        apply_step(card, ui, "Apply", idle, &mut acts);
                    }
                }
                Effect::Margin => {
                    segment_step(card, ui, "Segment", &rows, selected, Some("It has no voxels yet: make it first."), &mut acts);
                    card.step(ui, "Distance", Ready::Either, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut st.margin_mm).speed(0.1).range(-100.0..=100.0).suffix(" mm"));
                            ui.label(if st.margin_mm >= 0.0 { "grow (positive)" } else { "shrink (negative)" });
                        });
                        ui.checkbox(&mut st.apply_visible, "Every visible segment, not just this one");
                    });
                    apply_step(card, ui, "Apply", idle, &mut acts);
                }
                Effect::Smoothing => {
                    segment_step(card, ui, "Segment", &rows, selected, Some("It has no voxels yet: make it first."), &mut acts);
                    card.step(ui, "Method and size", Ready::Either, |ui| {
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
                            ui.checkbox(&mut st.apply_visible, "Every visible segment, not just this one");
                        }
                    });
                    apply_step(card, ui, "Apply", idle, &mut acts);
                }
                Effect::Hollow => {
                    segment_step(card, ui, "Segment", &rows, selected, Some("It has no voxels yet: make it first."), &mut acts);
                    card.step(ui, "Shell", Ready::Either, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut st.hollow_mm).speed(0.05).range(0.05..=50.0).suffix(" mm"));
                            ui.label("thick");
                        });
                        let side = HOLLOW_SIDES.iter().find(|h| h.0 == st.hollow_side).map_or("", |h| h.1);
                        egui::ComboBox::from_id_salt("segmentation.hollow").selected_text(side).show_ui(ui, |ui| {
                            for (s, label, _) in HOLLOW_SIDES {
                                ui.selectable_value(&mut st.hollow_side, s, label);
                            }
                        });
                        ui.checkbox(&mut st.apply_visible, "Every visible segment, not just this one");
                        ui.weak("For a printable wall, Thicken in the Mesh tab gives an exact thickness on the mesh.");
                    });
                    apply_step(card, ui, "Apply", idle, &mut acts);
                }
                Effect::Logical => {
                    segment_step(card, ui, "Segment to change", &rows, selected, None, &mut acts);
                    card.step(ui, "Operation", Ready::Either, |ui| {
                        egui::ComboBox::from_id_salt("segmentation.logical").selected_text(st.logical.label()).show_ui(ui, |ui| {
                            for op in LogicalOp::ALL {
                                ui.selectable_value(&mut st.logical, op, op.label());
                            }
                        });
                    });
                    if st.logical.needs_other() {
                        card.step(ui, "With segment", Ready::when(st.logical_other.is_some()), |ui| {
                            let name = st.logical_other.map_or("(choose)".to_string(), |l| name_of(&rows, l));
                            egui::ComboBox::from_id_salt("segmentation.logical_other").selected_text(name).show_ui(ui, |ui| {
                                for r in &rows {
                                    ui.selectable_value(&mut st.logical_other, Some(r.label), &r.name);
                                }
                            });
                        });
                    }
                    card.step(ui, "Apply", Ready::Todo, |ui| {
                        if ui.add_enabled(idle, egui::Button::new("Apply")).clicked() {
                            acts.push(Act::Apply);
                        }
                        ui.checkbox(&mut st.bypass_masking, "Ignore the limits").on_hover_text("This operation changes voxels wherever it says, whatever Limits are set");
                    });
                }
                Effect::GrowFromSeeds | Effect::Watershed | Effect::FillBetweenSlices => {
                    if effect == Effect::FillBetweenSlices {
                        card.step(ui, "Segment some slices", Ready::when(visible_with_voxels >= 1), |ui| {
                            ui.weak("Segment the structure on every few slices (Paint, Draw, Level tracing…): the slices between are filled from the visible segments.");
                        });
                        card.step(ui, "Slices of", Ready::Either, |ui| {
                            ui.horizontal(|ui| {
                                for p in crate::PLANES {
                                    ui.selectable_value(&mut st.fill_plane, p, p.label());
                                }
                            });
                        });
                    } else {
                        card.step(ui, "Seeds", Ready::when(visible_with_voxels >= 2), |ui| {
                            ui.weak("With Paint, mark a few strokes in two or more visible segments: the structure in one, what surrounds it in another.");
                            ui.label(format!("{visible_with_voxels} visible segment{} with voxels", if visible_with_voxels == 1 { "" } else { "s" }));
                        });
                        card.step(ui, "Options", Ready::Either, |ui| {
                            if effect == Effect::GrowFromSeeds {
                                ui.add(egui::Slider::new(&mut st.seed_locality, 0.0..=10.0).text("seed locality")).on_hover_text("0: only likeness counts; higher: nearer seeds win more easily");
                            } else {
                                ui.add(egui::Slider::new(&mut st.watershed_mm, 0.0..=5.0).suffix(" mm").text("object scale")).on_hover_text("Smoothing before the edges are found: higher gives smoother boundaries but rounds small structures");
                            }
                            ui.add(egui::Slider::new(&mut st.seed_margin_mm, 1.0..=100.0).logarithmic(true).suffix(" mm").text("margin")).on_hover_text("Works within this far of the seeds");
                        });
                    }
                    card.step(ui, "Preview", Ready::when(previewing), |ui| {
                        ui.horizontal(|ui| {
                            if ui.add_enabled(idle, egui::Button::new(if previewing { "Update" } else { "Initialize" })).clicked() {
                                acts.push(Act::StartSeeded);
                            }
                            ui.checkbox(&mut st.auto_update, "Auto-update").on_hover_text("Update the preview after every change of the seeds");
                        });
                        ui.weak("The views show the result before it is kept.");
                    });
                    card.step(ui, "Apply", Ready::Todo, |ui| {
                        ui.horizontal(|ui| {
                            if ui.add_enabled(idle && previewing, egui::Button::new("Apply")).clicked() {
                                acts.push(Act::ApplySeeded);
                            }
                            if ui.add_enabled(previewing, egui::Button::new("Cancel")).clicked() {
                                acts.push(Act::CancelSeeded);
                            }
                        });
                    });
                }
                Effect::FastMarching => {
                    segment_step(card, ui, "Start", &rows, selected, Some("Paint a few voxels of the structure into it first."), &mut acts);
                    card.step(ui, "Size", Ready::Either, |ui| {
                        ui.add(egui::Slider::new(&mut st.march_percent, 0.01..=20.0).logarithmic(true).suffix(" %").text("of the volume")).on_hover_text("The segment grows over the most similar voxels nearest to it until it is this big");
                    });
                    apply_step(card, ui, "Apply", idle, &mut acts);
                }
                Effect::LevelSet => {
                    segment_step(card, ui, "Rough segment", &rows, selected, Some("Start from a rough segment inside the vessel (painted or thresholded)."), &mut acts);
                    card.step(ui, "Vessel values", Ready::Either, |ui| {
                        range_values(ui, &mut st.threshold, vol);
                        ui.weak("Threshold's range: inside it the front moves out, outside it moves back.");
                    });
                    card.step(ui, "Options", Ready::Either, |ui| {
                        ui.add(egui::Slider::new(&mut st.ls_iterations, 10..=3000).logarithmic(true).text("iterations"));
                        ui.add(egui::Slider::new(&mut st.ls_max_mm, 0.5..=100.0).logarithmic(true).suffix(" mm").text("moves at most"));
                        ui.add(egui::Slider::new(&mut st.ls_curvature, 0.0..=2.0).text("smoothness"));
                        ui.checkbox(&mut st.ls_overlay, "Follow the overlay (e.g. vesselness)").on_hover_text("The overlay's window decides: above its centre the front moves out, below it moves back. Show it with Overlay ▾ in the toolbar");
                    });
                    apply_step(card, ui, "Apply", idle, &mut acts);
                }
                Effect::DrawTube | Effect::SurfaceCut | Effect::GuidedVessel => {
                    segment_step(card, ui, "Segment to fill", &rows, selected, None, &mut acts);
                    if effect == Effect::DrawTube {
                        card.step(ui, "Radius", Ready::Either, |ui| {
                            ui.add(egui::Slider::new(&mut st.tube_mm, 0.1..=20.0).logarithmic(true).suffix(" mm").text("radius"));
                        });
                    } else if effect == Effect::GuidedVessel {
                        card.step(ui, "Vessel", Ready::Either, |ui| {
                            range_values(ui, &mut st.threshold, vol);
                            ui.add(egui::Slider::new(&mut st.tube_mm, 0.2..=20.0).logarithmic(true).suffix(" mm").text("about its radius"));
                        });
                    }
                    let (min, what) = match effect {
                        Effect::DrawTube => (1, "Click points along the structure in the slice views (any view, any slice) or on the 3D surface: the tube runs smoothly through them in order."),
                        Effect::GuidedVessel => (2, "Click points along the vessel in order: its two ends and a few between, in any view. The path between them follows the vessel."),
                        _ => (4, "Click points on the structure's surface in several slices, or on the 3D surface: the closed shape through them is filled."),
                    };
                    card.step(ui, "Points", Ready::when(markers >= min), |ui| {
                        ui.weak(what);
                        points_row(ui, markers, &mut acts);
                    });
                    card.step(ui, "Apply", Ready::Todo, |ui| {
                        if ui.add_enabled(idle && markers >= min, egui::Button::new("Apply")).clicked() {
                            acts.push(Act::ApplyMarkers);
                        }
                    });
                }
                Effect::Engrave => {
                    segment_step(card, ui, "Segment", &rows, selected, Some("It has no voxels yet: the text is cut into (or raised from) it."), &mut acts);
                    card.step(ui, "Text", Ready::when(!st.engrave_text.trim().is_empty()), |ui| {
                        ui.add(egui::TextEdit::singleline(&mut st.engrave_text).desired_width(160.0));
                        ui.weak("Letters A–Z, digits and - . + / :");
                    });
                    card.step(ui, "Size and depth", Ready::Either, |ui| {
                        ui.add(egui::Slider::new(&mut st.engrave_mm, 1.0..=50.0).logarithmic(true).suffix(" mm").text("letter height"));
                        ui.add(egui::Slider::new(&mut st.engrave_depth_mm, 0.2..=20.0).logarithmic(true).suffix(" mm").text("depth"));
                        ui.horizontal(|ui| {
                            ui.radio_value(&mut st.engrave_raise, false, "Cut in");
                            ui.radio_value(&mut st.engrave_raise, true, "Raise");
                        });
                    });
                    card.step(ui, "Place it", Ready::Todo, |ui| {
                        ui.weak("Click where the text goes in a slice view: it reads left to right there, through the depth around that slice.");
                    });
                }
                Effect::MaskVolume => {
                    segment_step(card, ui, "Segment to mask by", &rows, selected, Some("It has no voxels yet: make it first."), &mut acts);
                    card.step(ui, "Fill", Ready::Either, |ui| {
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
                    });
                    card.step(ui, "Save", Ready::Todo, |ui| {
                        if ui.add_enabled(idle, egui::Button::new("Save masked volume")).clicked() {
                            acts.push(Act::Apply);
                        }
                        ui.weak("A new image among the derived images: it appears in the Volume tab.");
                    });
                }
                Effect::SplitVolume => {
                    card.step(ui, "Segments", Ready::when(visible_with_voxels >= 1), |ui| {
                        ui.weak("One image per visible segment with voxels: show the ones to split out.");
                        ui.label(format!("{visible_with_voxels} visible segment{} with voxels", if visible_with_voxels == 1 { "" } else { "s" }));
                    });
                    card.step(ui, "Crop and fill", Ready::Either, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::DragValue::new(&mut st.split_pad_mm).speed(0.1).range(0.0..=100.0).suffix(" mm"));
                            ui.label("margin around each segment");
                        });
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut st.split_blank, "Fill outside the segment with");
                            let speed = ((vol.range.1 - vol.range.0) / 500.0).max(1e-3);
                            ui.add_enabled(st.split_blank, egui::DragValue::new(&mut st.mask_outside).speed(speed));
                        });
                    });
                    card.step(ui, "Save", Ready::Todo, |ui| {
                        if ui.add_enabled(idle, egui::Button::new("Split")).clicked() {
                            acts.push(Act::Apply);
                        }
                        ui.weak("The images are saved among the derived images (Volume tab).");
                    });
                }
            }
            if effect.uses_masking() {
                let on = *masking != Masking::default();
                let summary = limits_summary(masking, &rows);
                card.limits(ui, on, &summary, |ui| limits_ui(ui, masking, &rows, st.threshold));
            }
        });
        if close {
            self.effect = None;
        }
        for act in acts {
            match act {
                Act::Apply => self.start_op(effect, None, &ctx),
                Act::Select(k) => {
                    if let Some(seg) = &mut self.seg {
                        seg.selected = Some(k);
                    }
                }
                Act::NewSegment => {
                    if self.seg.as_mut().is_some_and(|s| s.add_segment(None).is_none()) {
                        self.message = Some(("All 255 segment labels are in use".into(), true));
                    }
                }
                Act::FinishDraw => self.finish_draw(vol),
                Act::ClearSketch => self.sketch = None,
                Act::StartSeeded => self.start_seeded(effect, &ctx),
                Act::ApplySeeded => self.apply_seeded(),
                Act::CancelSeeded => {
                    self.seeded = None;
                    self.preview = None;
                }
                Act::PopMarker => {
                    self.markers.pop();
                }
                Act::ClearMarkers => self.markers.clear(),
                Act::ApplyMarkers => {
                    if effect == Effect::GuidedVessel {
                        self.start_guided(&ctx);
                    } else {
                        self.apply_markers(effect);
                    }
                }
                Act::UsePart(a) => self.use_part(vol, a),
                Act::ApplyGrow => self.apply_grow(vol),
                Act::SetReach(mm) => {
                    if let Some(r) = if effect == Effect::ThresholdGrow { &mut self.grow } else { &mut self.part } {
                        r.mm = mm;
                    }
                }
            }
        }
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

/// A segment as the effect cards list it.
struct SegRow {
    label: u8,
    name: String,
    colour: Color32,
    visible: bool,
    voxels: usize,
}

fn name_of(rows: &[SegRow], label: u8) -> String {
    rows.iter().find(|r| r.label == label).map_or("?".to_string(), |r| r.name.clone())
}

/// Threshold grow's or Select part's region, as its card shows it.
struct Region {
    mm: f64,
    stats: drag::RegionStats,
    ignored: usize,
    label: u8,
    dragging: bool,
}

/// What an effect card asked for, done once it is drawn.
enum Act {
    Apply,
    Select(usize),
    NewSegment,
    FinishDraw,
    ClearSketch,
    StartSeeded,
    ApplySeeded,
    CancelSeeded,
    PopMarker,
    ClearMarkers,
    ApplyMarkers,
    UsePart(drag::PartAction),
    ApplyGrow,
    SetReach(f64),
}

/// The step that picks the segment an effect works on (`needs_voxels`: it must have some,
/// and this says how to get them).
fn segment_step(card: &mut Card, ui: &mut egui::Ui, title: &str, rows: &[SegRow], selected: Option<usize>, needs_voxels: Option<&str>, acts: &mut Vec<Act>) {
    let has = selected.is_some_and(|k| rows[k].voxels > 0);
    card.step(ui, title, Ready::when(selected.is_some() && (needs_voxels.is_none() || has)), |ui| {
        ui.horizontal(|ui| {
            let text = selected.map_or(RichText::new("(choose)"), |k| RichText::new(format!("● {}", rows[k].name)).color(rows[k].colour));
            egui::ComboBox::from_id_salt("segmentation.card_segment").selected_text(text).width(150.0).show_ui(ui, |ui| {
                for (k, r) in rows.iter().enumerate() {
                    if ui.selectable_label(selected == Some(k), RichText::new(format!("● {}", r.name)).color(r.colour)).clicked() {
                        acts.push(Act::Select(k));
                    }
                }
            });
            if ui.button("+ New").on_hover_text("Add a segment and use it").clicked() {
                acts.push(Act::NewSegment);
            }
        });
        if let (Some(why), true, false) = (needs_voxels, selected.is_some(), has) {
            ui.colored_label(crate::WARN, why);
        }
    });
}

/// A final step with an Apply button.
fn apply_step(card: &mut Card, ui: &mut egui::Ui, title: &str, idle: bool, acts: &mut Vec<Act>) {
    card.step(ui, title, Ready::Todo, |ui| {
        if ui.add_enabled(idle, egui::Button::new("Apply")).clicked() {
            acts.push(Act::Apply);
        }
    });
}

/// The intensity histogram with the range on it, and the range's ends.
fn range_ui(ui: &mut egui::Ui, hist: &Histogram, range: &mut (f32, f32), colour: Color32, vol: &LoadedVolume) {
    histogram(ui, hist, range, colour);
    range_values(ui, range, vol);
    ui.weak("Drag on the histogram to move the nearer end.");
}

fn range_values(ui: &mut egui::Ui, range: &mut (f32, f32), vol: &LoadedVolume) {
    let speed = ((vol.range.1 - vol.range.0) / 500.0).max(1e-3);
    ui.horizontal(|ui| {
        ui.label("From");
        ui.add(egui::DragValue::new(&mut range.0).speed(speed));
        ui.label("to");
        ui.add(egui::DragValue::new(&mut range.1).speed(speed));
    });
}

/// Threshold's range from a region of the image: how its ends are picked from the region's
/// values, and their histogram.
fn local_histogram_ui(ui: &mut egui::Ui, st: &mut crate::effects::Settings, local: &Option<crate::sketch::LocalRegion>, colour: Color32) {
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
    match local {
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

/// The points placed (seeds): how many, and taking them back.
fn points_row(ui: &mut egui::Ui, n: usize, acts: &mut Vec<Act>) {
    ui.horizontal(|ui| {
        ui.label(format!("{n} point{}", if n == 1 { "" } else { "s" }));
        if ui.add_enabled(n > 0, egui::Button::new("Remove last")).clicked() {
            acts.push(Act::PopMarker);
        }
        if ui.add_enabled(n > 0, egui::Button::new("Clear")).on_hover_text("Esc").clicked() {
            acts.push(Act::ClearMarkers);
        }
    });
}

/// How far the region reaches from its seeds: the drags set it, and so does this slider.
fn reach_row(ui: &mut egui::Ui, region: Option<&Region>, acts: &mut Vec<Act>) {
    let mut mm = region.map_or(0.0, |r| r.mm);
    let top = (mm * 1.5).max(50.0);
    let r = ui.add_enabled(region.is_some(), egui::Slider::new(&mut mm, 0.0..=top).suffix(" mm").text("reach").fixed_decimals(1));
    if r.changed() {
        acts.push(Act::SetReach(mm));
    }
}

fn limits_summary(m: &Masking, rows: &[SegRow]) -> String {
    let mut parts = Vec::new();
    if m.area != EditableArea::Everywhere {
        parts.push(area_label(m.area, rows).to_lowercase());
    }
    if let Some((lo, hi)) = m.intensity {
        parts.push(format!("values {lo:.0}–{hi:.0}"));
    }
    match m.overwrite {
        Overwrite::All => {}
        Overwrite::Visible => parts.push("takes over visible segments only".into()),
        Overwrite::None => parts.push("leaves other segments alone".into()),
    }
    parts.join(" · ")
}

fn area_label(a: EditableArea, rows: &[SegRow]) -> String {
    match a {
        EditableArea::Everywhere => "Everywhere".to_string(),
        EditableArea::InsideAll => "Inside any segment".into(),
        EditableArea::InsideVisible => "Inside visible segments".into(),
        EditableArea::OutsideAll => "Outside every segment".into(),
        EditableArea::OutsideVisible => "Outside visible segments".into(),
        EditableArea::Inside(l) => format!("Inside {}", name_of(rows, l)),
        EditableArea::Outside(l) => format!("Outside {}", name_of(rows, l)),
    }
}

/// The limits on where effects may change voxels (masking).
fn limits_ui(ui: &mut egui::Ui, m: &mut Masking, rows: &[SegRow], threshold: (f32, f32)) {
    ui.horizontal(|ui| {
        ui.label("Where");
        egui::ComboBox::from_id_salt("segmentation.area").selected_text(area_label(m.area, rows)).width(180.0).show_ui(ui, |ui| {
            let mut areas = vec![EditableArea::Everywhere, EditableArea::InsideAll, EditableArea::InsideVisible, EditableArea::OutsideAll, EditableArea::OutsideVisible];
            areas.extend(rows.iter().map(|r| EditableArea::Inside(r.label)));
            areas.extend(rows.iter().map(|r| EditableArea::Outside(r.label)));
            for a in areas {
                ui.selectable_value(&mut m.area, a, area_label(a, rows));
            }
        });
    });
    let mut on = m.intensity.is_some();
    if ui.checkbox(&mut on, "Only voxels with values from").on_hover_text("i turns this on and off, with the threshold range").changed() {
        m.intensity = on.then_some(threshold);
    }
    if let Some((lo, hi)) = &mut m.intensity {
        ui.horizontal(|ui| {
            ui.add_space(20.0);
            ui.add(egui::DragValue::new(lo));
            ui.label("to");
            ui.add(egui::DragValue::new(hi));
        });
    }
    ui.horizontal(|ui| {
        ui.label("Other segments' voxels");
        let label = |o: Overwrite| match o {
            Overwrite::All => "taken over",
            Overwrite::Visible => "taken over if visible",
            Overwrite::None => "left alone",
        };
        egui::ComboBox::from_id_salt("segmentation.overwrite").selected_text(label(m.overwrite)).show_ui(ui, |ui| {
            for o in [Overwrite::All, Overwrite::Visible, Overwrite::None] {
                ui.selectable_value(&mut m.overwrite, o, label(o));
            }
        });
    });
    if *m != Masking::default() && ui.small_button("No limits").clicked() {
        *m = Masking::default();
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
