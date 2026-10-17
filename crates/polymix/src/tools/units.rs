//! Units / Dimensions (Analysis > Units / Dimensions): read and set the object's size in
//! common units, convert between units, and set the scale from measurements: drag on the
//! model to measure, give each measurement its real length, and the scale that best fits
//! them all is applied (with the remaining average / maximum deviation shown).

use super::*;
use crate::ui_util;

/// Display units; model units are taken to be millimetres.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Unit {
    Mm,
    Cm,
    M,
    In,
}

impl Unit {
    const ALL: [Unit; 4] = [Unit::Mm, Unit::Cm, Unit::M, Unit::In];
    fn label(self) -> &'static str {
        match self {
            Unit::Mm => "mm",
            Unit::Cm => "cm",
            Unit::M => "m",
            Unit::In => "in",
        }
    }
    /// Millimetres per unit.
    fn mm(self) -> f64 {
        match self {
            Unit::Mm => 1.0,
            Unit::Cm => 10.0,
            Unit::M => 1000.0,
            Unit::In => 25.4,
        }
    }
}

struct Measure {
    a: DVec3,
    b: DVec3,
    /// The real length this measurement should have (model units).
    target: f64,
}

impl Measure {
    fn length(&self) -> f64 {
        self.a.distance(self.b)
    }
}

pub struct UnitsTool {
    object: Option<ObjectId>,
    unit: Unit,
    measures: Vec<Measure>,
    selected: Option<usize>,
    pressed: Option<(egui::Pos2, DVec3)>,
}

/// Least-squares scale `s` minimising Σ (s·Lᵢ − Tᵢ)², with the average and maximum
/// deviation |s·Lᵢ − Tᵢ| left over.
fn fit_scale(pairs: &[(f64, f64)]) -> Option<(f64, f64, f64)> {
    let num: f64 = pairs.iter().map(|(l, t)| l * t).sum();
    let den: f64 = pairs.iter().map(|(l, _)| l * l).sum();
    if den <= 0.0 || pairs.is_empty() {
        return None;
    }
    let s = num / den;
    let dev: Vec<f64> = pairs.iter().map(|(l, t)| (s * l - t).abs()).collect();
    Some((s, dev.iter().sum::<f64>() / dev.len() as f64, dev.iter().copied().fold(0.0, f64::max)))
}

impl UnitsTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self { object: cx.scene.active, unit: Unit::Mm, measures: Vec::new(), selected: None, pressed: None }
    }

    fn scale_by(&mut self, cx: &mut ToolCx, s: f64, label: &str) {
        let Some(id) = self.object.filter(|id| cx.scene.get(*id).is_some()) else { return };
        if !(s > 0.0 && s.is_finite()) || s == 1.0 {
            return;
        }
        cx.history.push(label, cx.scene);
        let Some(o) = cx.scene.get_mut(id) else { return };
        let c = o.bounds().center();
        o.mesh_mut().transform(|p| c + (p - c) * s);
        for m in &mut self.measures {
            m.a = c + (m.a - c) * s;
            m.b = c + (m.b - c) * s;
        }
        *cx.status = format!("{label}: scaled by {s}");
    }
}

impl Tool for UnitsTool {
    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Units / Dimensions"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Units / Dimensions");
        let Some(o) = self.object.and_then(|id| cx.scene.get(id)) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let e = o.bounds().extents();
        ui.horizontal(|ui| {
            ui.label("Dimensions in");
            egui::ComboBox::from_id_salt("unit").selected_text(self.unit.label()).show_ui(ui, |ui| {
                for u in Unit::ALL {
                    ui.selectable_value(&mut self.unit, u, u.label());
                }
            });
        });
        let k = self.unit.mm();
        let mut scale = None;
        egui::Grid::new("dims").num_columns(2).show(ui, |ui| {
            for (i, n) in ["X", "Y", "Z"].iter().enumerate() {
                ui.label(*n);
                let mut v = e[i] / k;
                if ui.add(egui::DragValue::new(&mut v).speed(e[i] / k * 0.005).max_decimals(4).suffix(format!(" {}", self.unit.label()))).changed() && e[i] > 0.0 && v > 0.0 {
                    scale = Some(v * k / e[i]);
                }
                ui.end_row();
            }
        });
        if let Some(s) = scale {
            self.scale_by(cx, s, "Set dimensions");
        }
        ui.small("Typing a dimension scales the object uniformly. Mesh files carry no units; medical exports are usually millimetres.");
        ui_util::section(ui, "Convert units");
        ui.horizontal_wrapped(|ui| {
            for (label, s) in [("m → mm", 1000.0), ("cm → mm", 10.0), ("in → mm", 25.4), ("mm → m", 0.001), ("mm → in", 1.0 / 25.4)] {
                if ui.button(label).clicked() {
                    self.scale_by(cx, s, label);
                }
            }
        });
        ui_util::section(ui, "Set from measurements");
        ui.small("Drag on the model to measure, then enter each measurement's real length.");
        let mut remove = None;
        for (i, m) in self.measures.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                if ui.radio(self.selected == Some(i), format!("{:.3}", m.length() / k)).on_hover_text("Measured length").clicked() {
                    self.selected = Some(i);
                }
                ui.label("→");
                let mut t = m.target / k;
                if ui.add(egui::DragValue::new(&mut t).speed(0.01).range(0.0..=1e9).max_decimals(4).suffix(format!(" {}", self.unit.label()))).on_hover_text("Real length").changed() {
                    m.target = t * k;
                }
                if ui.small_button("x").clicked() {
                    remove = Some(i);
                }
            });
        }
        if let Some(i) = remove {
            self.measures.remove(i);
            self.selected = None;
        }
        let pairs: Vec<(f64, f64)> = self.measures.iter().filter(|m| m.target > 0.0).map(|m| (m.length(), m.target)).collect();
        if let Some((s, avg, max)) = fit_scale(&pairs) {
            ui.label(format!("Scale {s:.6}"));
            if pairs.len() > 1 {
                ui.label(format!("Avg deviation {:.4} {u}, max {:.4} {u}", avg / k, max / k, u = self.unit.label()));
            }
            if ui.add_enabled((s - 1.0).abs() > 1e-12, egui::Button::new("Apply scale")).clicked() {
                self.scale_by(cx, s, "Scale from measurements");
                for m in &mut self.measures {
                    m.target = m.length();
                }
            }
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.pressed = cx.pick(pos, self.object).map(|(_, h)| (pos, h.point));
        }
        if input.released {
            if let (Some((p0, a)), Some((_, h))) = (self.pressed.take(), cx.pick(pos, self.object)) {
                if p0.distance(pos) > 5.0 {
                    let m = Measure { a, b: h.point, target: a.distance(h.point) };
                    self.measures.push(m);
                    self.selected = Some(self.measures.len() - 1);
                }
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        // Bounding box, edges coloured by axis.
        if let Some(o) = self.object.and_then(|id| cx.scene.get(id)) {
            let b = o.bounds();
            let c = |i: usize| DVec3::new(if i & 1 == 0 { b.min.x } else { b.max.x }, if i & 2 == 0 { b.min.y } else { b.max.y }, if i & 4 == 0 { b.min.z } else { b.max.z });
            let cols = [Vec4::new(0.9, 0.3, 0.3, 1.0), Vec4::new(0.3, 0.85, 0.35, 1.0), Vec4::new(0.35, 0.5, 0.95, 1.0)];
            for i in 0..8 {
                for (axis, bit) in [1usize, 2, 4].iter().enumerate() {
                    if i & bit == 0 {
                        ov.line(c(i), c(i | bit), cols[axis]);
                    }
                }
            }
        }
        for (i, m) in self.measures.iter().enumerate() {
            let col = if self.selected == Some(i) { Vec4::new(1.0, 0.6, 0.2, 1.0) } else { Vec4::new(1.0, 0.85, 0.2, 1.0) };
            ov.line(m.a, m.b, col);
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        let k = self.unit.mm();
        let font = egui::FontId::proportional(13.0);
        if let Some(o) = self.object.and_then(|id| cx.scene.get(id)) {
            let b = o.bounds();
            let e = b.extents();
            let mids = [
                (DVec3::new(b.center().x, b.min.y, b.min.z), e.x, egui::Color32::from_rgb(230, 90, 90)),
                (DVec3::new(b.max.x, b.center().y, b.min.z), e.y, egui::Color32::from_rgb(90, 210, 100)),
                (DVec3::new(b.max.x, b.min.y, b.center().z), e.z, egui::Color32::from_rgb(110, 140, 240)),
            ];
            for (p, len, col) in mids {
                if let Some(q) = cx.project(p) {
                    painter.text(q, egui::Align2::CENTER_TOP, format!("{:.3} {}", len / k, self.unit.label()), font.clone(), col);
                }
            }
        }
        for m in &self.measures {
            if let Some(q) = cx.project((m.a + m.b) * 0.5) {
                painter.text(q, egui::Align2::LEFT_BOTTOM, format!("{:.3}", m.length() / k), font.clone(), egui::Color32::from_rgb(255, 220, 80));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn least_squares_scale() {
        // Two measurements that both say "double it" fit exactly.
        let (s, avg, max) = fit_scale(&[(1.0, 2.0), (3.0, 6.0)]).unwrap();
        assert!((s - 2.0).abs() < 1e-12 && avg < 1e-12 && max < 1e-12);
        // Conflicting ones leave a deviation.
        let (s, avg, max) = fit_scale(&[(1.0, 2.0), (1.0, 3.0)]).unwrap();
        assert!((s - 2.5).abs() < 1e-12 && (avg - 0.5).abs() < 1e-12 && (max - 0.5).abs() < 1e-12);
        assert!(fit_scale(&[]).is_none());
    }
}
