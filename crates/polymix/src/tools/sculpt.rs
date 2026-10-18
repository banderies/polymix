//! Sculpt brushes (Meshmixer "Sculpt > Brushes"). Volume brushes refine the mesh under the
//! brush as they go and reach everything inside the brush sphere; surface brushes keep the
//! triangles and stay on the stroked surface. Shift switches to the secondary (smoothing) brush.

use super::*;
use crate::ui_util;
use rayon::prelude::*;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Brush {
    Draw,
    Draw2,
    Draw3,
    DrawMax,
    Inflate,
    Flatten,
    Move,
    Drag,
    Push,
    Pinch,
    Spikes,
    Nudge,
    ZipperEdge,
    Attract,
    BubbleSmooth,
    ShrinkSmooth,
    RobustSmooth,
}

impl Brush {
    pub const ALL: [Brush; 17] = [
        Brush::Draw,
        Brush::Draw2,
        Brush::Draw3,
        Brush::DrawMax,
        Brush::Inflate,
        Brush::Flatten,
        Brush::Move,
        Brush::Drag,
        Brush::Push,
        Brush::Pinch,
        Brush::Spikes,
        Brush::Nudge,
        Brush::ZipperEdge,
        Brush::Attract,
        Brush::BubbleSmooth,
        Brush::ShrinkSmooth,
        Brush::RobustSmooth,
    ];
    fn label(self) -> &'static str {
        match self {
            Brush::Draw => "Draw",
            Brush::Draw2 => "Draw2",
            Brush::Draw3 => "Draw3",
            Brush::DrawMax => "DrawMax",
            Brush::Inflate => "Inflate",
            Brush::Flatten => "Flatten",
            Brush::Move => "Move",
            Brush::Drag => "Drag",
            Brush::Push => "Push",
            Brush::Pinch => "Pinch",
            Brush::Spikes => "Spikes",
            Brush::Nudge => "Nudge",
            Brush::ZipperEdge => "ZipperEdge",
            Brush::Attract => "Attract",
            Brush::BubbleSmooth => "BubbleSmooth",
            Brush::ShrinkSmooth => "ShrinkSmooth",
            Brush::RobustSmooth => "RobustSmooth",
        }
    }
    fn hint(self) -> &'static str {
        match self {
            Brush::Draw => "Raise the surface along the brush's average normal",
            Brush::Draw2 => "Raise along each vertex normal, with a rounder ridge",
            Brush::Draw3 => "Build a layer of constant height over the stroke",
            Brush::DrawMax => "Lift the surface up to the highest point under the brush",
            Brush::Inflate => "Puff the surface out along its normals",
            Brush::Flatten => "Flatten onto the average plane under the brush",
            Brush::Move => "Grab the surface and move it with the pointer",
            Brush::Drag => "Smear the surface along the stroke",
            Brush::Push => "Press the surface in along the view direction",
            Brush::Pinch => "Pull the surface toward the brush centre (sharpens creases)",
            Brush::Spikes => "Sharp peaks along the normals",
            Brush::Nudge => "Slide the surface along the stroke without changing its height",
            Brush::ZipperEdge => "Pull the surface toward the stroke line, closing grooves",
            Brush::Attract => "Pull the surface onto the target surface",
            Brush::BubbleSmooth => "Smooth while keeping the volume",
            Brush::ShrinkSmooth => "Plain smoothing (shrinks the surface)",
            Brush::RobustSmooth => "Smooth bumps with little shrinkage",
        }
    }
    /// Brushes that grab once and follow the pointer instead of dabbing.
    fn grabs(self) -> bool {
        self == Brush::Move
    }
}

/// Dynamic refinement: the surface under the brush is re-triangulated after every dab so
/// detail can be added (Meshmixer's volume-brush Refinement).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Refinement {
    pub enabled: bool,
    /// 0..100: finer triangles relative to the brush size.
    pub refine: f64,
    /// 0..100: how eagerly short edges are collapsed.
    pub reduce: f64,
    /// 0..100: tangential relaxation of the new triangles.
    pub smooth: f64,
    /// Finer where the surface bends, coarser where it is flat.
    pub adaptive: bool,
    /// 0..100: strength of the adaptive variation.
    pub adaptivity: f64,
}

impl Default for Refinement {
    fn default() -> Self {
        Self { enabled: false, refine: 25.0, reduce: 5.0, smooth: 5.0, adaptive: false, adaptivity: 50.0 }
    }
}

impl Refinement {
    /// Target edge length for a brush of radius `r` over `tris`.
    fn target(&self, m: &Mesh, tris: &[u32], r: f64) -> f64 {
        let mut l = r * 0.3 * 2f64.powf(-self.refine / 25.0);
        if self.adaptive {
            // Normal spread of the region: 0 on a plane, toward 1 where it bends sharply.
            let (mut n, mut a) = (DVec3::ZERO, 0.0);
            for &t in tris {
                let ar = m.tri_area(t);
                n += m.tri_normal(t) * ar;
                a += ar;
            }
            let spread = if a > 0.0 { 1.0 - n.length() / a } else { 0.0 };
            let k = self.adaptivity / 100.0;
            l *= (1.0 + k) * (1.0 - (spread * 4.0).min(1.0) * k * 0.75);
        }
        l
    }
}

/// Re-triangulate the surface within the brush: split long edges, collapse short ones, relax.
/// Returns whether the topology changed.
fn refine_region(mesh: &mut Mesh, bvh: &mm_core::Bvh, c: DVec3, r: f64, p: &Refinement) -> bool {
    let mut tris = Vec::new();
    bvh.query_sphere(mesh, c, r * 1.15, |t| tris.push(t));
    tris.sort_unstable();
    tris.dedup();
    if tris.is_empty() {
        return false;
    }
    let target = p.target(mesh, &tris, r);
    let Ok(split) = mm_remesh::refine_uniform(mesh, target * 1.33, Some(&tris), None) else { return false };
    let tris = split.region_triangles.unwrap_or(tris);
    let min = target * (0.2 + 0.6 * p.reduce / 100.0);
    let Ok(col) = mm_remesh::collapse_short_edges(mesh, min, Some(&tris), None) else { return split.splits > 0 };
    let tris = col.region_triangles.unwrap_or(tris);
    if p.smooth > 0.0 {
        let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
        let mut verts: Vec<u32> = tris.iter().flat_map(|&t| mesh.triangle(t)).filter(|&v| !mesh.is_boundary_vertex(v) && mesh.vertex_triangles(v).iter().all(|t| set.contains(t))).collect();
        verts.sort_unstable();
        verts.dedup();
        let k = (p.smooth / 100.0).min(1.0);
        let moves: Vec<(u32, DVec3)> = verts
            .iter()
            .map(|&v| {
                let n = mesh.vertex_normal(v);
                let l = laplacian(mesh, v);
                (v, mesh.vertex(v) + (l - n * l.dot(n)) * k)
            })
            .collect();
        for (v, q) in moves {
            mesh.set_vertex(v, q);
        }
    }
    split.splits + col.collapses > 0
}

/// Meshmixer's two brush families.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SculptMode {
    /// Dynamic refinement on, the brush acts on everything inside its sphere.
    Volume,
    /// The mesh keeps its triangles and the brush stays on the stroked surface.
    Surface,
}

/// Brush falloff curves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Falloff {
    Smooth,
    Linear,
    Sharp,
    Dome,
    Flat,
}

impl Falloff {
    const ALL: [Falloff; 5] = [Falloff::Smooth, Falloff::Linear, Falloff::Sharp, Falloff::Dome, Falloff::Flat];
    fn label(self) -> &'static str {
        match self {
            Falloff::Smooth => "Smooth",
            Falloff::Linear => "Linear",
            Falloff::Sharp => "Sharp",
            Falloff::Dome => "Dome",
            Falloff::Flat => "Flat",
        }
    }
    /// Weight at `x` = distance / radius in 0..1.
    fn weight(self, x: f64) -> f64 {
        let x = x.clamp(0.0, 1.0);
        match self {
            Falloff::Smooth => (1.0 - x * x).powi(2),
            Falloff::Linear => 1.0 - x,
            Falloff::Sharp => (1.0 - x).powi(3),
            Falloff::Dome => (1.0 - x * x).sqrt(),
            Falloff::Flat => {
                let t = ((x - 0.7) / 0.3).clamp(0.0, 1.0);
                1.0 - t * t * (3.0 - 2.0 * t)
            }
        }
    }
}

/// Which vertices a dab reaches, and how strongly.
#[derive(Clone, Copy, Debug)]
struct Footprint {
    falloff: Falloff,
    /// Everything inside the sphere (otherwise only surface connected to the stroke).
    volumetric: bool,
    hold_boundary: bool,
    hold_groups: bool,
}

pub struct SculptTool {
    brush: Brush,
    mode: SculptMode,
    /// Used while Shift is held.
    secondary: Brush,
    /// -100..100: moves the brush centre into (+) or out of (−) the surface.
    depth: f64,
    /// 0..100: the brush trails the pointer for steadier strokes.
    lazyness: f64,
    /// Keep dabbing while the pointer rests (otherwise only while it moves).
    flow: bool,
    falloff: Falloff,
    volumetric: bool,
    /// Face-group border vertices stay put.
    hold_groups: bool,
    /// Pull the stroked surface toward the target surface as well.
    attract: bool,
    refinement: Refinement,
    /// Brush radius in percent of the model size.
    size_pct: f32,
    strength: f64,
    stroke: Option<SculptStroke>,
    armed: bool,
    hover: Option<(DVec3, f64)>,
    /// Keep open-boundary vertices fixed.
    hold_boundary: bool,
    /// Strength when a right-drag (strength adjust) started.
    strength_drag: Option<f64>,
}

struct SculptStroke {
    object: ObjectId,
    invert: bool,
    // Move brush state.
    drag_verts: Vec<(u32, f64)>,
    drag_origin: Option<(DVec3, egui::Pos2)>,
    drag_start_pos: Vec<DVec3>,
    /// Previous dab centre (for stroke-direction brushes).
    last_center: Option<DVec3>,
    /// Positions when the stroke first touched each vertex (Draw3's layer reference).
    start: std::collections::HashMap<u32, DVec3>,
    /// Lazy brush centre, and the pointer position of the last dab (Flow off).
    lazy: Option<DVec3>,
    last_pos: Option<egui::Pos2>,
}

impl SculptTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self::with_mode(cx, SculptMode::Surface, None)
    }

    /// Volume or surface brushes (Meshmixer hotkeys 3 and 2), optionally with a brush.
    pub fn with_mode(_cx: &mut ToolCx, mode: SculptMode, brush: Option<Brush>) -> Self {
        let volume = mode == SculptMode::Volume;
        Self {
            brush: brush.unwrap_or(if volume { Brush::Draw } else { Brush::RobustSmooth }),
            mode,
            secondary: if volume { Brush::BubbleSmooth } else { Brush::RobustSmooth },
            depth: 0.0,
            lazyness: 0.0,
            flow: true,
            falloff: Falloff::Smooth,
            volumetric: volume,
            hold_groups: false,
            attract: false,
            refinement: Refinement { enabled: volume, ..Refinement::default() },
            size_pct: 4.0,
            strength: 0.5,
            stroke: None,
            armed: false,
            hover: None,
            hold_boundary: true,
            strength_drag: None,
        }
    }
}

impl Default for Footprint {
    fn default() -> Self {
        Self { falloff: Falloff::Smooth, volumetric: true, hold_boundary: false, hold_groups: false }
    }
}

/// Vertices within `r` of `c`, with falloff weights. Without `volumetric`, only vertices
/// reachable along edges from `seed` (the stroked triangle) inside the sphere count, so a
/// nearby but separate sheet of surface is left alone.
fn brush_vertices(mesh: &Mesh, bvh: &mm_core::Bvh, c: DVec3, r: f64, seed: Option<u32>, fp: &Footprint) -> Vec<(u32, f64)> {
    let mut inside = std::collections::HashSet::new();
    bvh.query_sphere(mesh, c, r, |t| {
        for v in mesh.triangle(t) {
            if mesh.vertex(v).distance(c) < r {
                inside.insert(v);
            }
        }
    });
    let reached: std::collections::HashSet<u32> = match seed.filter(|&t| !fp.volumetric && mesh.is_triangle(t)) {
        Some(t) => {
            let mut seen: std::collections::HashSet<u32> = mesh.triangle(t).into_iter().filter(|v| inside.contains(v)).collect();
            let mut stack: Vec<u32> = seen.iter().copied().collect();
            while let Some(v) = stack.pop() {
                for u in mesh.vertex_neighbors(v) {
                    if inside.contains(&u) && seen.insert(u) {
                        stack.push(u);
                    }
                }
            }
            seen
        }
        None => inside,
    };
    let held = |v: u32| {
        (fp.hold_boundary && mesh.is_boundary_vertex(v)) || (fp.hold_groups && {
            let tris = mesh.vertex_triangles(v);
            tris.iter().any(|&t| mesh.tri_group(t) != mesh.tri_group(tris[0]))
        })
    };
    let mut out: Vec<(u32, f64)> = reached.into_iter().filter(|&v| !held(v)).map(|v| (v, fp.falloff.weight(mesh.vertex(v).distance(c) / r))).collect();
    out.sort_unstable_by_key(|x| x.0);
    out
}

/// What a dab needs besides the vertices.
struct Dab<'a> {
    center: DVec3,
    r: f64,
    strength: f64,
    invert: bool,
    /// Direction the camera looks.
    view: DVec3,
    /// World-space motion of the brush since the previous dab.
    motion: DVec3,
    /// Draw3: positions at the start of the stroke.
    start: &'a std::collections::HashMap<u32, DVec3>,
    target: Option<&'a crate::scene::TargetSurface>,
}

/// Uniform-weight Laplacian of `v` (average of neighbours minus the vertex).
fn laplacian(m: &Mesh, v: u32) -> DVec3 {
    let mut avg = DVec3::ZERO;
    let mut n = 0.0;
    for u in m.vertex_neighbors(v) {
        avg += m.vertex(u);
        n += 1.0;
    }
    if n == 0.0 {
        DVec3::ZERO
    } else {
        avg / n - m.vertex(v)
    }
}

fn apply_dab(mesh: &mut Mesh, verts: &[(u32, f64)], brush: Brush, d: &Dab) {
    if verts.is_empty() {
        return;
    }
    let m: &Mesh = mesh;
    let sign = if d.invert { -1.0 } else { 1.0 };
    let (s, r) = (d.strength, d.r);
    // Weighted average normal and centroid of the brush region.
    let (mut nsum, mut psum, mut wsum) = (DVec3::ZERO, DVec3::ZERO, 0.0);
    for &(v, w) in verts {
        nsum += m.vertex_normal(v) * w;
        psum += m.vertex(v) * w;
        wsum += w;
    }
    let navg = nsum.normalize_or_zero();
    let pavg = if wsum > 0.0 { psum / wsum } else { d.center };
    let tangential = |v: u32, x: DVec3| {
        let n = m.vertex_normal(v);
        x - n * x.dot(n)
    };
    let updates: Vec<(u32, DVec3)> = match brush {
        Brush::ShrinkSmooth | Brush::RobustSmooth => verts
            .par_iter()
            .map(|&(v, w)| {
                let mut delta = laplacian(m, v) * (s * w).min(1.0);
                if brush == Brush::RobustSmooth {
                    // Remove most of the normal component to limit shrinkage, keeping
                    // enough to flatten bumps.
                    let nrm = m.vertex_normal(v);
                    delta -= nrm * delta.dot(nrm) * 0.5;
                }
                (v, m.vertex(v) + delta)
            })
            .collect(),
        Brush::BubbleSmooth => {
            // Taubin λ|μ step: smooth, then push back out by a little more, keeping volume.
            let lambda = (0.6 * s).min(0.9);
            let mu = -lambda * 1.06;
            let first: Vec<(u32, DVec3)> = verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + laplacian(m, v) * lambda * w)).collect();
            let moved: std::collections::HashMap<u32, DVec3> = first.iter().copied().collect();
            let at = |v: u32| moved.get(&v).copied().unwrap_or_else(|| m.vertex(v));
            first
                .iter()
                .zip(verts)
                .map(|(&(v, p), &(_, w))| {
                    let (mut avg, mut n) = (DVec3::ZERO, 0.0);
                    for u in m.vertex_neighbors(v) {
                        avg += at(u);
                        n += 1.0;
                    }
                    (v, if n > 0.0 { p + (avg / n - p) * mu * w } else { p })
                })
                .collect()
        }
        Brush::Inflate => verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + m.vertex_normal(v) * (sign * s * w * r * 0.05))).collect(),
        Brush::Draw => verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + navg * (sign * s * w * r * 0.05))).collect(),
        Brush::Draw2 => verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + m.vertex_normal(v) * (sign * s * w.powf(1.5) * r * 0.06))).collect(),
        Brush::Spikes => verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + m.vertex_normal(v) * (sign * s * w.powi(6) * r * 0.15))).collect(),
        Brush::Draw3 => {
            // Rise toward a layer `height` above where the stroke found the surface.
            let height = sign * s * r * 0.3;
            verts
                .par_iter()
                .map(|&(v, w)| {
                    let p = m.vertex(v);
                    let base = d.start.get(&v).copied().unwrap_or(p);
                    let want = base + navg * height * w;
                    let step = (want - p).dot(navg);
                    let step = if height >= 0.0 { step.max(0.0) } else { step.min(0.0) };
                    (v, p + navg * step * 0.5)
                })
                .collect()
        }
        Brush::DrawMax => {
            let top = verts.iter().map(|&(v, _)| (m.vertex(v) - pavg).dot(navg)).fold(f64::NEG_INFINITY, f64::max);
            verts
                .par_iter()
                .map(|&(v, w)| {
                    let p = m.vertex(v);
                    let h = (p - pavg).dot(navg);
                    (v, p + navg * ((top - h).max(0.0) * (s * w * 0.5).min(1.0)))
                })
                .collect()
        }
        Brush::Flatten => verts
            .par_iter()
            .map(|&(v, w)| {
                let p = m.vertex(v);
                (v, p - navg * (p - pavg).dot(navg) * (s * w * 0.5).min(1.0))
            })
            .collect(),
        Brush::Push => verts.par_iter().map(|&(v, w)| (v, m.vertex(v) + d.view * (sign * s * w * r * 0.05))).collect(),
        Brush::Pinch => verts
            .par_iter()
            .map(|&(v, w)| {
                let p = m.vertex(v);
                (v, p + tangential(v, d.center - p) * (sign * s * w * 0.15))
            })
            .collect(),
        Brush::Drag | Brush::Nudge => verts
            .par_iter()
            .map(|&(v, w)| {
                let step = d.motion * (s * w).min(1.0);
                (v, m.vertex(v) + if brush == Brush::Nudge { tangential(v, step) } else { step })
            })
            .collect(),
        Brush::ZipperEdge => {
            // Pull toward the line the stroke runs along (in the surface).
            let dir = d.motion.normalize_or_zero();
            verts
                .par_iter()
                .map(|&(v, w)| {
                    let p = m.vertex(v);
                    let rel = p - d.center;
                    let off = rel - dir * rel.dot(dir);
                    (v, p - tangential(v, off) * (s * w * 0.2))
                })
                .collect()
        }
        Brush::Attract => {
            let Some(t) = d.target else { return };
            verts
                .par_iter()
                .filter_map(|&(v, w)| {
                    let p = m.vertex(v);
                    let h = t.bvh.nearest(&t.mesh, p, r * 4.0)?;
                    Some((v, p + (h.point - p) * (s * w * 0.5).min(1.0)))
                })
                .collect()
        }
        Brush::Move => Vec::new(),
    };
    let pos = mesh.positions_mut();
    for (v, p) in updates {
        pos[v as usize] = p;
    }
    mesh.bump_shape_version();
}

impl Tool for SculptTool {
    fn brush_size(&mut self) -> Option<&mut f32> {
        Some(&mut self.size_pct)
    }

    fn brush_strength(&mut self) -> Option<&mut f64> {
        Some(&mut self.strength)
    }

    fn survives_scene_change(&self) -> bool {
        true
    }

    fn name(&self) -> &'static str {
        "Sculpt"
    }

    fn panel(&mut self, ui: &mut egui::Ui, cx: &mut ToolCx) -> ToolOutcome {
        ui_util::section(ui, "Sculpt brushes");
        ui.horizontal(|ui| {
            let before = self.mode;
            ui.selectable_value(&mut self.mode, SculptMode::Volume, "Volume (3)").on_hover_text("Refines the mesh as you sculpt; acts on everything inside the brush");
            ui.selectable_value(&mut self.mode, SculptMode::Surface, "Surface (2)").on_hover_text("Keeps the triangles; stays on the stroked surface");
            if self.mode != before {
                let volume = self.mode == SculptMode::Volume;
                self.refinement.enabled = volume;
                self.volumetric = volume;
            }
        });
        ui.horizontal_wrapped(|ui| {
            for b in Brush::ALL {
                ui.selectable_value(&mut self.brush, b, b.label()).on_hover_text(b.hint());
            }
        });
        egui::ComboBox::from_label("Secondary (Shift)").selected_text(self.secondary.label()).show_ui(ui, |ui| {
            for b in [Brush::BubbleSmooth, Brush::ShrinkSmooth, Brush::RobustSmooth] {
                ui.selectable_value(&mut self.secondary, b, b.label());
            }
        });
        egui::ComboBox::from_label("Falloff").selected_text(self.falloff.label()).show_ui(ui, |ui| {
            for f in Falloff::ALL {
                ui.selectable_value(&mut self.falloff, f, f.label());
            }
        });
        if self.brush == Brush::Attract && cx.scene.target.is_none() {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Attract needs a target surface (Edit → Set as target).");
        }
        let r = cx.scene.active.map(|id| brush_radius(cx.scene, id, self.size_pct));
        brush_size_ui(ui, &mut self.size_pct, r);
        ui.horizontal(|ui| {
            ui.label("Strength");
            ui.add(egui::Slider::new(&mut self.strength, 0.01..=1.0));
        });
        ui.add(egui::Slider::new(&mut self.depth, -100.0..=100.0).text("depth")).on_hover_text("Moves the brush centre into (+) or out of (−) the surface");
        ui.add(egui::Slider::new(&mut self.lazyness, 0.0..=100.0).text("lazyness")).on_hover_text("The brush trails the pointer for steadier strokes");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.flow, "Flow").on_hover_text("Keep sculpting while the pointer rests");
            ui.checkbox(&mut self.volumetric, "Volumetric").on_hover_text("Affect everything inside the brush, not only the stroked surface");
        });
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.hold_boundary, "Hold boundary (B)");
            ui.checkbox(&mut self.hold_groups, "Preserve groups (G)");
            ui.checkbox(&mut self.attract, "Attract (A)").on_hover_text("Also pull the stroked surface toward the target surface");
        });
        ui_util::section(ui, "Refinement");
        let rf = &mut self.refinement;
        ui.checkbox(&mut rf.enabled, "Dynamic refinement (R)").on_hover_text("Re-triangulate under the brush so detail can be added");
        ui.add_enabled_ui(rf.enabled, |ui| {
            ui.add(egui::Slider::new(&mut rf.refine, 0.0..=100.0).text("refine"));
            ui.add(egui::Slider::new(&mut rf.reduce, 0.0..=100.0).text("reduce"));
            ui.add(egui::Slider::new(&mut rf.smooth, 0.0..=100.0).text("smooth"));
            ui.horizontal(|ui| {
                ui.checkbox(&mut rf.adaptive, "Adaptive (T)");
                ui.add_enabled(rf.adaptive, egui::Slider::new(&mut rf.adaptivity, 0.0..=100.0).text("adaptivity"));
            });
        });
        ui.small("Drag on the surface. Shift uses the secondary brush; Cmd/Ctrl inverts the draw brushes, Inflate, Push and Pinch. Size: [ ] or the wheel (Shift+wheel in Fusion 360 navigation). Right-drag: strength.");
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        let Some(pos) = input.pos else { return };
        let hover = cx.pick(pos, cx.scene.active);
        self.hover = hover.map(|(id, h)| (h.point, brush_radius(cx.scene, id, self.size_pct)));
        if input.pressed {
            self.armed = true;
        }
        if input.down && self.armed && self.stroke.is_none() {
            if let Some((id, _)) = hover {
                cx.scene.active = Some(id);
                cx.snapshot(&format!("Sculpt: {}", self.brush.label()));
                self.stroke = Some(SculptStroke {
                    object: id,
                    invert: input.command,
                    drag_verts: Vec::new(),
                    drag_origin: None,
                    drag_start_pos: Vec::new(),
                    last_center: None,
                    start: Default::default(),
                    lazy: None,
                    last_pos: None,
                });
            }
        }
        if input.released {
            self.armed = false;
            self.stroke = None;
            return;
        }
        if !input.down {
            return;
        }
        let Some(stroke) = &mut self.stroke else { return };
        let brush = if input.shift && !self.brush.grabs() { self.secondary } else { self.brush };
        let strength = self.strength;
        let fp = Footprint { falloff: self.falloff, volumetric: self.volumetric, hold_boundary: self.hold_boundary, hold_groups: self.hold_groups };
        if !self.flow && stroke.last_pos == Some(pos) {
            return;
        }
        stroke.last_pos = Some(pos);
        if brush.grabs() {
            if stroke.drag_origin.is_none() {
                let Some((_, h)) = hover else { return };
                let r = brush_radius(cx.scene, stroke.object, self.size_pct);
                let Some(o) = cx.scene.get_mut(stroke.object) else { return };
                let bvh = o.bvh();
                stroke.drag_verts = brush_vertices(&o.mesh, &bvh, h.point, r, Some(h.tid), &fp);
                stroke.drag_start_pos = stroke.drag_verts.iter().map(|&(v, _)| o.mesh.vertex(v)).collect();
                stroke.drag_origin = Some((h.point, pos));
                return;
            }
            let (p0, s0) = stroke.drag_origin.unwrap();
            let wpp = cx.world_per_point(p0);
            let d = pos - s0;
            let delta = (cx.camera.right() * d.x as f64 - cx.camera.up() * d.y as f64) * wpp;
            let Some(o) = cx.scene.get_mut(stroke.object) else { return };
            if o.moved_vertices.is_empty() {
                o.moved_from = o.mesh.shape_version();
            }
            let m = o.mesh_mut();
            let pos_arr = m.positions_mut();
            for (k, &(v, w)) in stroke.drag_verts.iter().enumerate() {
                pos_arr[v as usize] = stroke.drag_start_pos[k] + delta * w;
            }
            m.bump_shape_version();
            o.moved_vertices.extend(stroke.drag_verts.iter().map(|&(v, _)| v));
            return;
        }
        let Some((_, h)) = hover else { return };
        let r = brush_radius(cx.scene, stroke.object, self.size_pct);
        let view = cx.camera.forward();
        let target = cx.scene.target_surface(Some(stroke.object));
        let Some(o) = cx.scene.get_mut(stroke.object) else { return };
        // Depth moves the centre along the surface normal; lazyness lets it trail the pointer.
        let mut center = h.point - o.mesh.tri_normal(h.tid) * (self.depth / 100.0 * r);
        if let Some(prev) = stroke.lazy {
            center = prev + (center - prev) * (1.0 - 0.95 * self.lazyness / 100.0);
        }
        stroke.lazy = Some(center);
        let motion = stroke.last_center.map_or(DVec3::ZERO, |c| center - c);
        stroke.last_center = Some(center);
        let bvh = o.bvh();
        let verts = brush_vertices(&o.mesh, &bvh, center, r, Some(h.tid), &fp);
        for &(v, _) in &verts {
            stroke.start.entry(v).or_insert_with(|| o.mesh.vertex(v));
        }
        if o.moved_vertices.is_empty() {
            o.moved_from = o.mesh.shape_version();
        }
        let dab = Dab { center, r, strength, invert: stroke.invert, view, motion, start: &stroke.start, target: target.as_ref() };
        apply_dab(o.mesh_mut(), &verts, brush, &dab);
        if self.attract && target.is_some() && brush != Brush::Attract {
            apply_dab(o.mesh_mut(), &verts, Brush::Attract, &Dab { strength: strength * 0.5, ..dab });
        }
        o.moved_vertices.extend(verts.iter().map(|&(v, _)| v));
        if self.refinement.enabled {
            let bvh = o.bvh();
            if refine_region(o.mesh_mut(), &bvh, center, r, &self.refinement) {
                // New triangles: the GPU copy and the selection are rebuilt.
                o.moved_vertices.clear();
                let m = o.mesh.clone();
                o.selection.sync(&m);
                o.touch_selection();
            }
        }
    }

    fn overlay(&mut self, ov: &mut Overlay, _cx: &mut ToolCx) {
        if let Some((c, r)) = self.hover {
            ov.brush = Some((c.as_vec3(), r as f32));
        }
    }

    fn wheel(&mut self, delta: f32, _cx: &mut ToolCx) -> bool {
        self.size_pct = (self.size_pct * (delta * 0.004).exp()).clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1);
        true
    }

    fn claims_secondary_drag(&self, _scene: &Scene) -> bool {
        true
    }

    fn secondary_drag(&mut self, phase: DragPhase, _pos: egui::Pos2, total: egui::Vec2, cx: &mut ToolCx) {
        // Horizontal right-drag sets the strength, as in Meshmixer.
        let s0 = *self.strength_drag.get_or_insert(self.strength);
        self.strength = (s0 + total.x as f64 * 0.003).clamp(0.01, 1.0);
        *cx.status = format!("Strength {:.2}", self.strength);
        if phase == DragPhase::End {
            self.strength_drag = None;
        }
    }

    fn keys(&mut self, i: &egui::InputState, cx: &mut ToolCx) -> bool {
        let plain = |k| i.key_pressed(k) && !i.modifiers.command && !i.modifiers.shift && !i.modifiers.alt;
        if plain(egui::Key::R) {
            self.refinement.enabled = !self.refinement.enabled;
            *cx.status = format!("Refinement {}", if self.refinement.enabled { "on" } else { "off" });
            return true;
        }
        if plain(egui::Key::T) {
            self.refinement.adaptive = !self.refinement.adaptive;
            *cx.status = format!("Adaptive refinement {}", if self.refinement.adaptive { "on" } else { "off" });
            return true;
        }
        let toggle = |flag: &mut bool, name: &str, status: &mut String| {
            *flag = !*flag;
            *status = format!("{name} {}", if *flag { "on" } else { "off" });
            true
        };
        if plain(egui::Key::B) {
            return toggle(&mut self.hold_boundary, "Hold boundary", cx.status);
        }
        if plain(egui::Key::G) {
            return toggle(&mut self.hold_groups, "Preserve groups", cx.status);
        }
        if plain(egui::Key::A) {
            return toggle(&mut self.attract, "Attract", cx.status);
        }
        if i.key_pressed(egui::Key::OpenBracket) {
            step_brush_size(&mut self.size_pct, false);
            return true;
        }
        if i.key_pressed(egui::Key::CloseBracket) {
            step_brush_size(&mut self.size_pct, true);
            return true;
        }
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_core::primitives;

    fn dab_on_sphere(brush: Brush, motion: DVec3, target: Option<&crate::scene::TargetSurface>) -> (Mesh, Mesh, Vec<(u32, f64)>) {
        let m0 = primitives::icosphere(1.0, 4);
        let mut m = m0.clone();
        let bvh = mm_core::Bvh::build(&m);
        let c = DVec3::Z;
        let verts = brush_vertices(&m, &bvh, c, 0.4, None, &Footprint::default());
        let start = verts.iter().map(|&(v, _)| (v, m.vertex(v))).collect();
        let d = Dab { center: c, r: 0.4, strength: 0.8, invert: false, view: -DVec3::Z, motion, start: &start, target };
        for _ in 0..5 {
            apply_dab(&mut m, &verts, brush, &d);
        }
        (m0, m, verts)
    }

    fn mean_radius(m: &Mesh, verts: &[(u32, f64)]) -> f64 {
        verts.iter().map(|&(v, _)| m.vertex(v).length()).sum::<f64>() / verts.len() as f64
    }

    #[test]
    fn brushes_move_the_surface_their_way() {
        for b in [Brush::Draw, Brush::Draw2, Brush::Draw3, Brush::Inflate, Brush::Spikes, Brush::DrawMax] {
            let (m0, m, v) = dab_on_sphere(b, DVec3::ZERO, None);
            assert!(mean_radius(&m, &v) > mean_radius(&m0, &v) + 1e-4, "{b:?} raises the surface");
            m.check_validity().unwrap();
        }
        let (m0, m, v) = dab_on_sphere(Brush::Push, DVec3::ZERO, None);
        assert!(mean_radius(&m, &v) < mean_radius(&m0, &v), "Push presses in");
        // Smoothers: shrink smoothing loses more volume than bubble smoothing.
        let vol = |m: &Mesh| mm_core::measure::volume(m);
        let (m0, shrink, _) = dab_on_sphere(Brush::ShrinkSmooth, DVec3::ZERO, None);
        let (_, bubble, _) = dab_on_sphere(Brush::BubbleSmooth, DVec3::ZERO, None);
        assert!(vol(&shrink) < vol(&m0));
        assert!((vol(&bubble) - vol(&m0)).abs() < (vol(&shrink) - vol(&m0)).abs());
        // Pinch gathers vertices toward the centre.
        let spread = |m: &Mesh, v: &[(u32, f64)]| v.iter().map(|&(x, _)| m.vertex(x).truncate().length()).sum::<f64>();
        let (m0, m, v) = dab_on_sphere(Brush::Pinch, DVec3::ZERO, None);
        assert!(spread(&m, &v) < spread(&m0, &v));
        // Nudge slides along the stroke without lifting.
        let (m0, m, v) = dab_on_sphere(Brush::Nudge, DVec3::X * 0.05, None);
        let shift: f64 = v.iter().map(|&(x, _)| (m.vertex(x) - m0.vertex(x)).x).sum();
        assert!(shift > 0.0);
        assert!((mean_radius(&m, &v) - mean_radius(&m0, &v)).abs() < 0.01);
        // Attract pulls onto a target surface.
        let tm = primitives::icosphere(0.9, 4);
        let target = crate::scene::TargetSurface { id: 7, bvh: std::sync::Arc::new(mm_core::Bvh::build(&tm)), mesh: std::sync::Arc::new(tm) };
        let (m0, m, v) = dab_on_sphere(Brush::Attract, DVec3::ZERO, Some(&target));
        assert!(mean_radius(&m, &v) < mean_radius(&m0, &v) && mean_radius(&m, &v) > 0.89);
    }

    #[test]
    fn falloff_curves() {
        for f in Falloff::ALL {
            assert!((f.weight(0.0) - 1.0).abs() < 1e-12 && f.weight(1.0).abs() < 1e-12, "{f:?}");
            let w: Vec<f64> = (0..=20).map(|i| f.weight(i as f64 / 20.0)).collect();
            assert!(w.windows(2).all(|p| p[1] <= p[0] + 1e-12), "{f:?} decreases");
        }
        assert_eq!(Falloff::Flat.weight(0.6), 1.0);
        let at = |f: Falloff| f.weight(0.5);
        assert!(at(Falloff::Sharp) < at(Falloff::Linear) && at(Falloff::Linear) < at(Falloff::Smooth) && at(Falloff::Smooth) < at(Falloff::Dome));
    }

    #[test]
    fn footprint_volumetric_connected_and_held() {
        // Two parallel sheets 0.1 apart; the brush sphere reaches both.
        let mut m = primitives::plane_grid(2.0, 2.0, 20, 20);
        let mut top = m.clone();
        top.transform(|p| p + DVec3::Z * 0.1);
        let top_verts = m.append_mesh(&top, 1);
        let bvh = mm_core::Bvh::build(&m);
        let seed = m.triangle_ids().find(|&t| m.tri_group(t) == 0 && m.tri_centroid(t).truncate().length() < 0.1).unwrap();
        let ids = |v: &[(u32, f64)]| v.iter().map(|x| x.0).collect::<std::collections::HashSet<u32>>();
        let all = ids(&brush_vertices(&m, &bvh, DVec3::ZERO, 0.4, Some(seed), &Footprint::default()));
        let surface = ids(&brush_vertices(&m, &bvh, DVec3::ZERO, 0.4, Some(seed), &Footprint { volumetric: false, ..Default::default() }));
        assert!(all.iter().any(|v| top_verts.contains(v)));
        assert!(!surface.is_empty() && surface.iter().all(|v| !top_verts.contains(v)));
        assert!(surface.len() < all.len());
        // Preserve groups: split the bottom sheet in two along x = 0; its border stays out.
        for t in m.triangle_ids().collect::<Vec<_>>() {
            if m.tri_group(t) == 0 && m.tri_centroid(t).x > 0.0 {
                m.set_tri_group(t, 2);
            }
        }
        let held = brush_vertices(&m, &bvh, DVec3::ZERO, 0.4, Some(seed), &Footprint { volumetric: false, hold_groups: true, ..Default::default() });
        assert!(!held.is_empty() && held.iter().all(|&(v, _)| m.vertex(v).x.abs() > 1e-9));
        // Weights follow the falloff.
        let lin = brush_vertices(&m, &bvh, DVec3::ZERO, 0.4, None, &Footprint { falloff: Falloff::Linear, ..Default::default() });
        assert!(lin.iter().all(|&(v, w)| (w - (1.0 - m.vertex(v).length() / 0.4)).abs() < 1e-12));
    }

    #[test]
    fn dynamic_refinement_adds_detail_under_the_brush() {
        let mut m = primitives::icosphere(1.0, 2);
        let n0 = m.triangle_count();
        let p = Refinement { enabled: true, refine: 50.0, ..Default::default() };
        for _ in 0..3 {
            let bvh = mm_core::Bvh::build(&m);
            assert!(refine_region(&mut m, &bvh, DVec3::Z, 0.5, &p) || m.triangle_count() > n0);
        }
        m.check_validity().unwrap();
        assert!(m.is_closed());
        assert!(m.triangle_count() > n0 * 2, "{} vs {n0}", m.triangle_count());
        // Far from the brush nothing changed: the bottom keeps its coarse triangles.
        let bottom = m.triangle_ids().filter(|&t| m.tri_centroid(t).z < -0.5).map(|t| m.tri_area(t)).fold(0.0, f64::max);
        let top = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.9).map(|t| m.tri_area(t)).fold(0.0, f64::max);
        assert!(top < bottom * 0.5);
        // Adaptive: longer edges on a flat patch than on a curved one.
        let flat = primitives::plane_grid(2.0, 2.0, 10, 10);
        let ft: Vec<u32> = flat.triangle_ids().collect();
        let sphere = primitives::icosphere(0.3, 3);
        let st: Vec<u32> = sphere.triangle_ids().filter(|&t| sphere.tri_centroid(t).z > 0.0).collect();
        let a = Refinement { adaptive: true, ..p };
        assert!(a.target(&flat, &ft, 0.5) > a.target(&sphere, &st, 0.5));
        assert!(a.target(&flat, &ft, 0.5) > p.target(&flat, &ft, 0.5));
    }
}
