//! Boolean (union / difference / intersection) of the active object with another object,
//! computed volumetrically so the result is always closed and manifold.

use super::*;
use crate::jobs::Job;
use crate::ui_util;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Op {
    Union,
    Difference,
    Intersection,
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Key {
    other: ObjectId,
    op: Op,
    accuracy: f64,
}

pub struct BooleanTool {
    object: Option<ObjectId>,
    other: Option<ObjectId>,
    op: Op,
    accuracy: f64,
    job: Option<(Key, Job<Mesh>)>,
    result: Option<(Key, Mesh)>,
    error: Option<String>,
}

impl BooleanTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let other = cx.scene.objects.iter().find(|o| Some(o.id) != object && o.visible).map(|o| o.id);
        Self { object, other, op: Op::Union, accuracy: 0.5, job: None, result: None, error: None }
    }

    fn key(&self) -> Option<Key> {
        Some(Key { other: self.other?, op: self.op, accuracy: self.accuracy })
    }

    fn start(&mut self, cx: &mut ToolCx) {
        let (Some(a), Some(key)) = (self.object, self.key()) else { return };
        let (Some(ma), Some(mb)) = (cx.scene.get(a).map(|o| o.mesh.clone()), cx.scene.get(key.other).map(|o| o.mesh.clone())) else {
            return;
        };
        if let Some((_, j)) = &self.job {
            j.cancel();
        }
        self.error = None;
        let job = Job::spawn("Boolean", cx.egui.clone(), move |progress| {
            let bounds = ma.bounds().union(&mb.bounds());
            let cells = (64.0 * 16f64.powf(key.accuracy)).round() as u32;
            let h = mm_solid::cell_size_for_resolution(&bounds, cells);
            let op = match key.op {
                Op::Union => mm_solid::BoolOp::Union,
                Op::Difference => mm_solid::BoolOp::Difference,
                Op::Intersection => mm_solid::BoolOp::Intersection,
            };
            mm_solid::sdf_boolean(&ma, &mb, op, h, Some(&|f| progress(f))).map_err(|e| anyhow::anyhow!("{e}"))
        });
        self.job = Some((key, job));
    }
}

impl Tool for BooleanTool {
    fn name(&self) -> &'static str {
        "Boolean"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Boolean");
        let Some(a) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let names: Vec<(ObjectId, String)> = cx.scene.objects.iter().filter(|o| o.id != a).map(|o| (o.id, o.name.clone())).collect();
        if names.is_empty() {
            ui.label("Import or create a second object to combine with.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        let before = self.key();
        ui.label(format!("Active: {}", cx.scene.get(a).map(|o| o.name.clone()).unwrap_or_default()));
        egui::ComboBox::from_label("with")
            .selected_text(self.other.and_then(|id| names.iter().find(|n| n.0 == id)).map(|n| n.1.clone()).unwrap_or_default())
            .show_ui(ui, |ui| {
                for (id, n) in &names {
                    ui.selectable_value(&mut self.other, Some(*id), n);
                }
            });
        ui.horizontal(|ui| {
            ui.selectable_value(&mut self.op, Op::Union, "Union");
            ui.selectable_value(&mut self.op, Op::Difference, "Difference");
            ui.selectable_value(&mut self.op, Op::Intersection, "Intersection");
        });
        ui.add(egui::Slider::new(&mut self.accuracy, 0.0..=1.0).text("accuracy"));
        ui.small("Volumetric boolean: robust to messy input, result is closed and manifold.");
        if self.key() != before || (self.result.is_none() && self.job.is_none() && self.error.is_none()) {
            self.start(cx);
        }
        if let Some((_, j)) = &self.job {
            if let Some(r) = j.poll() {
                let (k, _) = self.job.take().unwrap();
                match r {
                    Ok(m) => self.result = Some((k, m)),
                    Err(e) => self.error = Some(format!("{e:#}")),
                }
            } else {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.add(egui::ProgressBar::new(j.progress()));
                });
            }
        }
        if let Some((_, m)) = &self.result {
            ui.label(format!("Result: {} triangles", ui_util::format_count(m.triangle_count())));
        }
        if let Some(e) = &self.error {
            ui.colored_label(egui::Color32::from_rgb(240, 110, 90), e);
        }
        let ready = self.result.as_ref().map(|(k, _)| Some(*k) == self.key()).unwrap_or(false);
        let (accept, cancel) = ui_util::accept_cancel(ui, ready);
        if accept && self.accept(cx) {
            return ToolOutcome::Close;
        }
        if cancel {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn preview(&self) -> Option<(ObjectId, &Mesh)> {
        let (k, m) = self.result.as_ref()?;
        (Some(*k) == self.key()).then_some((self.object?, m))
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        // Only commit a finished result for the current operation and operands.
        if self.job.is_some() || self.result.as_ref().map(|(k, _)| Some(*k) != self.key()).unwrap_or(true) {
            return false;
        }
        let Some((k, m)) = self.result.take() else { return false };
        let Some(a) = self.object else { return true };
        if cx.scene.get(a).is_none() || cx.scene.get(k.other).is_none() {
            return true;
        }
        cx.history.push("Boolean", cx.scene);
        let Some(o) = cx.scene.get_mut(a) else { return true };
        o.set_mesh(m);
        cx.scene.remove(k.other);
        cx.scene.active = Some(a);
        *cx.status = "Boolean applied".into();
        true
    }

    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Boolean".to_string(), j.progress()))
    }
}
