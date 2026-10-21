//! Clearance (Analysis > Clearance): how close the active object comes to another one.
//! Both surfaces are coloured by distance (red below the minimum clearance), and the
//! closest points are joined by a line.

use super::*;
use crate::jobs::Job;
use crate::ui_util;
use mm_core::measure::Clearance;
use mm_core::Bvh;
use rayon::prelude::*;
use std::sync::Arc;

/// What a clearance result was measured on: both meshes' shapes (shape versions are never
/// reused), the other object, and one way or both (a one-way result has no distances back).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Key {
    a: u64,
    b: u64,
    other: ObjectId,
    both_ways: bool,
}

/// A clearance measured in the background, with the trees it had to build (for the objects).
type Measured = (Clearance, Option<Arc<Bvh>>, Option<Arc<Bvh>>);

pub struct ClearanceTool {
    object: Option<ObjectId>,
    other: Option<ObjectId>,
    min_clearance: f64,
    both_ways: bool,
    result: Option<(Key, Clearance)>,
    /// The panel's measurement, running.
    job: Option<(Key, Job<Measured>)>,
    /// A measurement that failed: not started again for the same objects and shapes.
    failed: Option<Key>,
    /// The result and minimum clearance the objects are coloured for, and how many vertices
    /// of the active object are closer than that minimum.
    shown: Option<(Key, f64, usize)>,
    /// Objects coloured by the tool (cleared on close).
    coloured: Vec<ObjectId>,
    dirty: bool,
}

/// Red below `limit`, then yellow → green → blue as the distance grows to 4 × `limit`.
fn shade(d: f64, limit: f64) -> u32 {
    let pack = |c: [f64; 3]| {
        let b = |x: f64| (x.clamp(0.0, 1.0) * 255.0).round() as u32;
        (b(c[0]) << 16) | (b(c[1]) << 8) | b(c[2])
    };
    if d < limit {
        return pack([0.9, 0.15, 0.12]);
    }
    let t = ((d - limit) / (3.0 * limit.max(1e-300))).clamp(0.0, 1.0);
    let stops = [[0.95, 0.85, 0.2], [0.3, 0.8, 0.35], [0.25, 0.45, 0.9]];
    let x = t * 2.0;
    let i = (x.floor() as usize).min(1);
    let f = x - i as f64;
    pack([0, 1, 2].map(|k| stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f))
}

/// Per-triangle colours from per-vertex distances: each triangle shows its closest vertex.
/// The triangle slots are independent of each other, so they are filled in parallel.
fn colours(m: &Mesh, d: &[f64], limit: f64) -> Vec<u32> {
    const CHUNK: usize = 1 << 14;
    let mut out = vec![0u32; m.max_tid()];
    out.par_chunks_mut(CHUNK).enumerate().for_each(|(k, chunk)| {
        for (i, c) in chunk.iter_mut().enumerate() {
            let t = (k * CHUNK + i) as u32;
            if m.is_triangle(t) {
                let dt = m.triangle(t).iter().map(|&v| d[v as usize]).fold(f64::INFINITY, f64::min);
                *c = shade(dt, limit);
            }
        }
    });
    out
}

impl ClearanceTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        let object = cx.scene.active;
        let other = cx.scene.objects.iter().find(|o| Some(o.id) != object && o.visible).map(|o| o.id);
        let edge = cx.scene.active().map(|o| o.mesh.average_edge_length()).unwrap_or(1.0);
        Self { object, other, min_clearance: edge * 5.0, both_ways: true, result: None, job: None, failed: None, shown: None, coloured: Vec::new(), dirty: true }
    }

    /// The key a result for the current settings and meshes has.
    fn key(&self, scene: &Scene) -> Option<Key> {
        let (a, other) = (scene.get(self.object?)?, scene.get(self.other?)?);
        Some(Key { a: a.mesh.shape_version(), b: other.mesh.shape_version(), other: other.id, both_ways: self.both_ways })
    }

    /// Measure now, on this thread (agents).
    fn compute(&mut self, cx: &mut ToolCx) {
        self.dirty = false;
        self.job = None;
        self.failed = None;
        self.clear(cx);
        self.result = None;
        let (Some(a), Some(b)) = (self.object, self.other) else { return };
        let (Some(oa), Some(ob)) = (cx.scene.get_mut(a).map(|o| (o.mesh.clone(), o.bvh())), cx.scene.get_mut(b).map(|o| (o.mesh.clone(), o.bvh()))) else {
            return;
        };
        let Some(key) = self.key(cx.scene) else { return };
        let r = mm_core::measure::clearance(&oa.0, &oa.1, &ob.0, &ob.1, self.both_ways);
        self.finish(cx, key, r);
    }

    /// Measure in the background (the panel). The objects' trees are used when they are
    /// current; otherwise the job builds them, and they go to the objects afterwards.
    fn start(&mut self, cx: &mut ToolCx, key: Key) {
        self.clear(cx);
        self.result = None;
        let (Some(a), Some(b)) = (self.object.and_then(|id| cx.scene.get(id)), self.other.and_then(|id| cx.scene.get(id))) else { return };
        let (ma, ta, mb, tb) = (a.mesh.clone(), a.cached_bvh(), b.mesh.clone(), b.cached_bvh());
        let both_ways = self.both_ways;
        let job = Job::spawn("Clearance", cx.egui.clone(), move |progress| {
            let built_a = ta.is_none().then(|| Arc::new(Bvh::build(&ma)));
            let built_b = tb.is_none().then(|| Arc::new(Bvh::build(&mb)));
            let (Some(ta), Some(tb)) = (ta.or_else(|| built_a.clone()), tb.or_else(|| built_b.clone())) else { anyhow::bail!("no tree to measure with") };
            progress(0.3);
            Ok((mm_core::measure::clearance(&ma, &ta, &mb, &tb, both_ways), built_a, built_b))
        });
        self.job = Some((key, job));
    }

    /// Take the background measurement once it is done, if it still measures the objects as
    /// they are (otherwise the panel starts another).
    fn poll(&mut self, cx: &mut ToolCx) {
        let Some((key, job)) = &self.job else { return };
        let Some(r) = job.poll() else { return };
        let key = *key;
        self.job = None;
        match r {
            Ok((r, built_a, built_b)) if self.key(cx.scene) == Some(key) => {
                for (id, t) in [(self.object, built_a), (self.other, built_b)] {
                    if let (Some(o), Some(t)) = (id.and_then(|id| cx.scene.get_mut(id)), t) {
                        o.install_bvh(t);
                    }
                }
                self.finish(cx, key, r);
            }
            Ok(_) => {}
            Err(e) => {
                *cx.status = format!("Clearance: {e:#}");
                self.failed = Some(key);
            }
        }
    }

    fn finish(&mut self, cx: &mut ToolCx, key: Key, r: Clearance) {
        *cx.status = if r.intersecting { "The objects intersect".into() } else { format!("Minimum distance {:.4}", r.min) };
        self.result = Some((key, r));
        self.shown = None;
        self.recolour(cx);
    }

    /// Colour the objects by the result for the current minimum clearance, and count the
    /// active object's vertices closer than it, unless that is what they show already.
    fn recolour(&mut self, cx: &mut ToolCx) {
        let Some((key, r)) = &self.result else { return };
        let limit = self.min_clearance;
        if self.shown.is_some_and(|(k, l, _)| k == *key && l == limit) {
            return;
        }
        let Some(a) = self.object else { return };
        let ca = cx.scene.get(a).map(|o| colours(&o.mesh, &r.dist_a, limit));
        let cb = r.dist_b.as_ref().and_then(|d| cx.scene.get(key.other).map(|o| colours(&o.mesh, d, limit)));
        let below = r.dist_a.par_iter().filter(|&&d| d < limit).count();
        for (id, c) in [(a, ca), (key.other, cb)] {
            if let (Some(c), Some(o)) = (c, cx.scene.get_mut(id)) {
                o.colors = Some((crate::scene::next_stamp(), Arc::new(c)));
                if !self.coloured.contains(&id) {
                    self.coloured.push(id);
                }
            }
        }
        self.shown = Some((*key, limit, below));
    }

    fn clear(&mut self, cx: &mut ToolCx) {
        self.shown = None;
        for id in self.coloured.drain(..) {
            if let Some(o) = cx.scene.get_mut(id) {
                o.colors = None;
            }
        }
    }
}

impl Tool for ClearanceTool {
    fn busy(&self) -> Option<(String, f32)> {
        self.job.as_ref().map(|(_, j)| ("Clearance".to_string(), j.progress()))
    }

    fn pending(&self) -> bool {
        self.dirty && self.other.is_some()
    }

    fn settings(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"other": self.other, "min_clearance_mm": self.min_clearance, "both_ways": self.both_ways}))
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        Some(serde_json::json!({"type": "object", "properties": {
            "other": {"type": "integer", "description": "Object id of the object to measure against"},
            "min_clearance_mm": {"type": "number", "description": "Closer than this shows red"},
            "both_ways": {"type": "boolean", "description": "Also colour (and measure from) the other object"},
        }}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        self.other = Some(v["other"].as_u64().filter(|id| cx.scene.get(*id).is_some() && Some(*id) != self.object).ok_or("other must be the id of another object")?);
        self.min_clearance = v["min_clearance_mm"].as_f64().filter(|d| *d >= 0.0).ok_or("min_clearance_mm must be a number ≥ 0")?;
        self.both_ways = v["both_ways"].as_bool().ok_or("both_ways must be true or false")?;
        self.compute(cx);
        Ok(())
    }

    fn state(&self, _scene: &Scene) -> serde_json::Value {
        use polymix_workspace::agent::{round, vec3};
        serde_json::json!({
            "object": self.object,
            "other": self.other,
            "result": self.result.as_ref().map(|(_, r)| serde_json::json!({
                "intersecting": r.intersecting,
                "min_distance_mm": round(r.min, 5),
                "closest_points_mm": [vec3(r.closest.0.to_array(), 4), vec3(r.closest.1.to_array(), 4)],
                "vertices_closer_than_min_clearance": r.dist_a.iter().filter(|&&d| d < self.min_clearance).count(),
            })),
        })
    }

    fn actions(&self) -> Vec<Action> {
        vec![Action::plain("compute", "Measure again (after the objects changed)")]
    }

    fn act(&mut self, action: &str, _args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        match action {
            "compute" => {
                self.compute(cx);
                Ok(cx.status.clone())
            }
            _ => Err(format!("Clearance has no action {action}")),
        }
    }

    fn name(&self) -> &'static str {
        "Clearance"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Clearance");
        let Some(a) = self.object.filter(|id| cx.scene.get(*id).is_some()) else {
            ui.label("No active object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        };
        let names: Vec<(ObjectId, String)> = cx.scene.objects.iter().filter(|o| o.id != a).map(|o| (o.id, o.name.clone())).collect();
        if names.is_empty() {
            ui.label("Needs a second object.");
            return if ui.button("Close").clicked() { ToolOutcome::Close } else { ToolOutcome::Continue };
        }
        egui::ComboBox::from_label("to")
            .selected_text(self.other.and_then(|id| names.iter().find(|n| n.0 == id)).map(|n| n.1.clone()).unwrap_or_default())
            .show_ui(ui, |ui| {
                for (id, n) in &names {
                    ui.selectable_value(&mut self.other, Some(*id), n);
                }
            });
        let max = cx.scene.get(a).map(|o| o.bounds().diagonal() * 0.2).unwrap_or(1.0).max(self.min_clearance);
        ui.add(egui::Slider::new(&mut self.min_clearance, 0.0..=max).text("min clearance")).on_hover_text("Closer than this shows red");
        ui.checkbox(&mut self.both_ways, "Bi-directional").on_hover_text("Also colour the other object");
        // Another object, direction or shape is measured again, in the background; another
        // minimum clearance only recolours the result.
        self.poll(cx);
        let want = self.key(cx.scene);
        let have = self.job.as_ref().map(|(k, _)| *k).or(self.result.as_ref().map(|(k, _)| *k)).or(self.failed);
        if let Some(key) = want.filter(|k| have != Some(*k)) {
            self.start(cx, key);
        }
        self.dirty = false;
        if self.result.as_ref().is_some_and(|(k, _)| Some(*k) == want) {
            self.recolour(cx);
        }
        if self.job.is_some() {
            ui.horizontal(|ui| {
                polymix_workspace::spinner(ui);
                ui.label("Measuring…");
            });
        } else if let Some((_, r)) = &self.result {
            if r.intersecting {
                ui.colored_label(egui::Color32::from_rgb(240, 110, 90), "The objects intersect.");
            } else {
                ui.label(format!("Min distance {:.4}", r.min));
            }
            if let Some((_, _, below)) = self.shown {
                ui.label(format!("{below} vertices closer than the minimum clearance"));
            }
        }
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            self.clear(cx);
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        if let Some((_, r)) = self.result.as_ref().filter(|(_, r)| !r.intersecting) {
            ov.line(r.closest.0, r.closest.1, Vec4::new(1.0, 0.85, 0.2, 1.0));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if let Some((_, r)) = self.result.as_ref().filter(|(_, r)| !r.intersecting) {
            if let Some(p) = cx.project((r.closest.0 + r.closest.1) * 0.5) {
                painter.text(p, egui::Align2::LEFT_BOTTOM, format!("{:.3}", r.min), egui::FontId::proportional(14.0), egui::Color32::from_rgb(255, 220, 80));
            }
        }
    }

    fn cancel(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn accept(&mut self, cx: &mut ToolCx) -> bool {
        self.clear(cx);
        true
    }

    fn on_close(&mut self, cx: &mut ToolCx) {
        self.clear(cx);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    #[test]
    fn shading_bands() {
        assert_eq!(shade(0.1, 1.0), shade(0.9, 1.0), "everything below the limit is red");
        assert_ne!(shade(1.0, 1.0), shade(4.0, 1.0));
        assert_eq!(shade(10.0, 1.0), shade(4.0, 1.0), "far is blue");
    }

    /// Two spheres of radius 10, 2 apart, the second with a hole (dead triangle slots).
    fn spheres() -> (Mesh, Mesh) {
        let a = primitives::icosphere(10.0, 4);
        let mut b = primitives::icosphere(10.0, 4);
        b.transform(|p| p + DVec3::new(22.0, 0.0, 0.0));
        let gone: Vec<u32> = b.triangle_ids().filter(|&t| b.tri_centroid(t).y > 9.0).collect();
        b.remove_triangles(gone);
        (a, b)
    }

    /// The colouring as it was: one triangle after another.
    fn colours_serial(m: &Mesh, d: &[f64], limit: f64) -> Vec<u32> {
        let mut out = vec![0u32; m.max_tid()];
        for t in m.triangle_ids() {
            let dt = m.triangle(t).iter().map(|&v| d[v as usize]).fold(f64::INFINITY, f64::min);
            out[t as usize] = shade(dt, limit);
        }
        out
    }

    #[test]
    fn colours_match_the_serial_colouring() {
        let (a, b) = spheres();
        let r = mm_core::measure::clearance(&a, &Bvh::build(&a), &b, &Bvh::build(&b), true);
        assert!((r.min - 2.0).abs() < 0.05, "{}", r.min);
        for limit in [0.0, 0.5, 2.0, 2.3, 5.0, 50.0] {
            assert_eq!(colours(&a, &r.dist_a, limit), colours_serial(&a, &r.dist_a, limit), "{limit}");
            let db = r.dist_b.as_ref().unwrap();
            assert_eq!(colours(&b, db, limit), colours_serial(&b, db, limit), "{limit}");
        }
    }

    #[test]
    fn the_panel_measures_in_the_background_and_recolours_on_slider_changes() {
        let (ma, mb) = spheres();
        let mut scene = Scene::default();
        let a = scene.add("a", ma.clone());
        let b = scene.add("b", mb.clone());
        scene.active = Some(a);
        let (mut history, mut camera, mut status, ctx) = (History::default(), crate::camera::Camera::default(), String::new(), egui::Context::default());
        let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
        let mut cx = ToolCx { scene: &mut scene, history: &mut history, camera: &mut camera, status: &mut status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
        let mut tool = ClearanceTool::new(&mut cx);
        let frame = |tool: &mut ClearanceTool, cx: &mut ToolCx| {
            let raw = egui::RawInput { screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 900.0))), ..Default::default() };
            let _ = ctx.run_ui(raw, |ui| {
                let _ = tool.panel(ui, cx);
            });
        };
        let settle = |tool: &mut ClearanceTool, cx: &mut ToolCx| {
            let start = std::time::Instant::now();
            frame(tool, cx);
            while tool.busy().is_some() {
                assert!(start.elapsed().as_secs() < 60, "the measurement finishes");
                std::thread::sleep(std::time::Duration::from_millis(2));
                frame(tool, cx);
            }
        };
        assert!(tool.pending());
        frame(&mut tool, &mut cx);
        assert!(tool.busy().is_some(), "opening the tool measures in the background");
        settle(&mut tool, &mut cx);
        assert!(!tool.pending());
        // The same result the agents' synchronous path gives, and the trees went to the objects.
        let want = mm_core::measure::clearance(&ma, &Bvh::build(&ma), &mb, &Bvh::build(&mb), true);
        let (_, r) = tool.result.as_ref().unwrap();
        assert_eq!((r.min, &r.dist_a, &r.dist_b), (want.min, &want.dist_a, &want.dist_b));
        assert!(cx.scene.get(a).unwrap().bvh_if_current().is_some() && cx.scene.get(b).unwrap().bvh_if_current().is_some());
        let shown = |cx: &ToolCx, id: ObjectId| cx.scene.get(id).unwrap().colors.clone();
        let limit = tool.min_clearance;
        assert_eq!(*shown(&cx, a).unwrap().1, colours_serial(&ma, &want.dist_a, limit));
        assert_eq!(*shown(&cx, b).unwrap().1, colours_serial(&mb, want.dist_b.as_ref().unwrap(), limit));

        // A slider change recolours at once, without measuring again.
        let stamp = shown(&cx, a).unwrap().0;
        tool.min_clearance = 2.5;
        frame(&mut tool, &mut cx);
        assert!(tool.busy().is_none());
        assert_ne!(shown(&cx, a).unwrap().0, stamp);
        assert_eq!(*shown(&cx, a).unwrap().1, colours_serial(&ma, &want.dist_a, 2.5));
        assert_eq!(*shown(&cx, b).unwrap().1, colours_serial(&mb, want.dist_b.as_ref().unwrap(), 2.5));
        assert_eq!(tool.shown.unwrap().2, want.dist_a.iter().filter(|&&d| d < 2.5).count());
        let stamp = shown(&cx, a).unwrap().0;
        frame(&mut tool, &mut cx);
        assert_eq!(shown(&cx, a).unwrap().0, stamp, "nothing changed: nothing is recoloured");

        // One way: measured again, and the other object is no longer coloured.
        tool.both_ways = false;
        frame(&mut tool, &mut cx);
        assert!(tool.busy().is_some());
        settle(&mut tool, &mut cx);
        let (key, r) = tool.result.as_ref().unwrap();
        assert!(!key.both_ways && r.dist_b.is_none());
        assert!(shown(&cx, b).is_none());
        assert_eq!(*shown(&cx, a).unwrap().1, colours_serial(&ma, &want.dist_a, 2.5));

        // Agents measure at once.
        tool.set_settings(&serde_json::json!({"both_ways": true, "min_clearance_mm": 1.0}), &mut cx).unwrap();
        assert!(tool.busy().is_none() && !tool.pending());
        assert_eq!(*shown(&cx, b).unwrap().1, colours_serial(&mb, want.dist_b.as_ref().unwrap(), 1.0));
        frame(&mut tool, &mut cx);
        assert!(tool.busy().is_none(), "the panel takes the agent's result");
    }
}
