//! Sculpt brushes (Meshmixer "Sculpt > Brushes"). Volume brushes refine the mesh under the
//! brush as they go and reach everything inside the brush sphere; surface brushes keep the
//! triangles and stay on the stroked surface. Shift switches to the secondary (smoothing) brush.
//!
//! Dabs are placed a set distance apart along the stroke (Spacing). The primary brush's
//! footprint can be shaped by a brush alpha that moves and turns with the dab (AlignToStroke)
//! — on the draw brushes it embosses the image — and by a stencil fixed to the view
//! ([`stencil`]). Symmetry mirrors every dab across a freely placed plane ([`symmetry`]).

pub mod stencil;
pub mod symmetry;

use super::*;
use crate::ui_util;
use mm_edit::brush_alpha::{AlphaMap, DabFrame, ScreenMap, ScreenStencil, Spacer};
use rayon::prelude::*;
use stencil::Alpha;
pub use symmetry::{SymPlane, Symmetry};

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
    Refine,
    Reduce,
    AdaptiveReduce,
    PaintVertex,
    SmoothColor,
}

impl Brush {
    pub const ALL: [Brush; 22] = [
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
        Brush::Refine,
        Brush::Reduce,
        Brush::AdaptiveReduce,
        Brush::PaintVertex,
        Brush::SmoothColor,
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
            Brush::Refine => "Refine",
            Brush::Reduce => "Reduce",
            Brush::AdaptiveReduce => "AdaptiveReduce",
            Brush::PaintVertex => "PaintVertex",
            Brush::SmoothColor => "SmoothColor",
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
            Brush::Refine => "Make the triangles under the brush smaller",
            Brush::Reduce => "Make the triangles under the brush larger and even",
            Brush::AdaptiveReduce => "Remove triangles where the surface is flat, keeping curved detail",
            Brush::PaintVertex => "Paint vertex colours (Cmd/Ctrl paints the default grey back)",
            Brush::SmoothColor => "Blend vertex colours with their neighbours",
        }
    }
    /// Colour brushes change vertex colours instead of the shape.
    fn paints(self) -> bool {
        matches!(self, Brush::PaintVertex | Brush::SmoothColor)
    }
    /// Mesh density brushes change the triangles instead of moving the surface.
    fn density(self) -> bool {
        matches!(self, Brush::Refine | Brush::Reduce | Brush::AdaptiveReduce)
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
    retriangulate(mesh, tris, target, 0.2 + 0.6 * p.reduce / 100.0, p.smooth / 100.0)
}

/// Split edges of `tris` longer than 1.33 × `target`, collapse those shorter than
/// `min_frac` × `target`, then relax the region's interior vertices tangentially by `smooth`
/// (0..1). The region border is kept. Returns whether the topology changed.
fn retriangulate(mesh: &mut Mesh, tris: Vec<u32>, target: f64, min_frac: f64, smooth: f64) -> bool {
    let Ok(split) = mm_remesh::refine_uniform(mesh, target * 1.33, Some(&tris), None) else { return false };
    let tris = split.region_triangles.unwrap_or(tris);
    let Ok(col) = mm_remesh::collapse_short_edges(mesh, target * min_frac, Some(&tris), None) else { return split.splits > 0 };
    let tris = col.region_triangles.unwrap_or(tris);
    if smooth > 0.0 {
        relax_interior(mesh, &tris, smooth.min(1.0));
    }
    split.splits + col.collapses > 0
}

/// Tangential Laplacian relaxation of the vertices whose whole one-ring lies in `tris`.
fn relax_interior(mesh: &mut Mesh, tris: &[u32], k: f64) {
    let set: std::collections::HashSet<u32> = tris.iter().copied().collect();
    let mut verts: Vec<u32> = tris.iter().flat_map(|&t| mesh.triangle(t)).filter(|&v| !mesh.is_boundary_vertex(v) && mesh.vertex_triangles(v).iter().all(|t| set.contains(t))).collect();
    verts.sort_unstable();
    verts.dedup();
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

/// One dab of a colour brush: blend toward `color` (PaintVertex) or the neighbours' average
/// (SmoothColor) by strength × falloff.
fn paint_dab(mesh: &mut Mesh, verts: &[(u32, f64)], brush: Brush, color: [f32; 3], strength: f64) {
    if brush == Brush::SmoothColor && !mesh.has_vertex_colors() {
        return;
    }
    mesh.enable_vertex_colors();
    let m: &Mesh = mesh;
    let new: Vec<(u32, [f32; 3])> = verts
        .iter()
        .map(|&(v, w)| {
            let c = m.vertex_color(v).unwrap_or(mm_core::mesh::DEFAULT_VERTEX_COLOR);
            let to = if brush == Brush::PaintVertex {
                color
            } else {
                let (mut sum, mut n) = ([0.0f32; 3], 0.0f32);
                for u in m.vertex_neighbors(v) {
                    let cu = m.vertex_color(u).unwrap_or(c);
                    for i in 0..3 {
                        sum[i] += cu[i];
                    }
                    n += 1.0;
                }
                if n > 0.0 { sum.map(|x| x / n) } else { c }
            };
            let k = (strength * w).clamp(0.0, 1.0) as f32;
            (v, [0, 1, 2].map(|i| c[i] + (to[i] - c[i]) * k))
        })
        .collect();
    for (v, c) in new {
        mesh.set_vertex_color(v, c);
    }
}

/// One dab of a mesh density brush over the triangles whose corners are all in `verts`.
/// Each dab moves the edge length a step (scaled by `strength`) toward finer (Refine) or
/// coarser (Reduce) triangles; AdaptiveReduce removes triangles by quadric error, so flat
/// areas thin out while curved detail stays. Returns whether the topology changed.
fn density_dab(mesh: &mut Mesh, verts: &[(u32, f64)], brush: Brush, r: f64, strength: f64) -> bool {
    let inside: std::collections::HashSet<u32> = verts.iter().map(|x| x.0).collect();
    let mut tris: Vec<u32> = verts.iter().flat_map(|&(v, _)| mesh.vertex_triangles(v)).filter(|&t| mesh.triangle(t).iter().all(|v| inside.contains(v))).collect();
    tris.sort_unstable();
    tris.dedup();
    if tris.is_empty() {
        return false;
    }
    let s = strength.clamp(0.0, 1.0);
    let mean = mm_remesh::edge_length_stats_in(mesh, &tris).mean;
    match brush {
        // Bounded by the brush size: a smaller brush refines further.
        Brush::Refine => retriangulate(mesh, tris, (mean * (1.0 - 0.25 * s)).max(r * 0.04), 0.3, 0.25),
        Brush::Reduce => retriangulate(mesh, tris, (mean * (1.0 + 0.5 * s)).min(r * 0.5), 0.75, 0.25),
        Brush::AdaptiveReduce => {
            // Cheapest collapses first, and none that tilt a face by more than a few degrees,
            // so curved areas resist while flat ones thin out.
            let params = mm_remesh::ReduceParams { target: mm_remesh::ReduceTarget::Percentage(100.0 - 40.0 * s), region: Some(tris), max_normal_change_deg: 6.0, ..Default::default() };
            mm_remesh::reduce(mesh, &params, None).is_ok_and(|rep| rep.collapses > 0)
        }
        _ => false,
    }
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

/// SymSnap tolerance, as a fraction of the brush radius.
const SYM_SNAP_TOL: f64 = 0.03;

/// Put back onto the plane the vertices of `verts` whose position when the stroke first
/// reached them (`start`) was within `tol` of it.
fn sym_snap(mesh: &mut Mesh, verts: impl Iterator<Item = u32>, start: &std::collections::HashMap<u32, DVec3>, sym: SymPlane, tol: f64) {
    for v in verts {
        if mesh.is_vertex(v) && start.get(&v).is_some_and(|&p| sym.distance(p) < tol) {
            let p = sym.project(mesh.vertex(v));
            mesh.set_vertex(v, p);
        }
    }
}

/// Dab spacing range, in brush radii.
const SPACING_RANGE: (f64, f64) = (0.05, 3.0);
/// Most dabs one pointer move places (a longer jump spaces them wider).
const MAX_DABS_PER_MOVE: usize = 32;
/// Brush alpha size range (image side over the brush diameter) and stencil size range
/// (image side over the viewport height).
const ALPHA_SCALE_RANGE: (f64, f64) = (0.05, 8.0);
const STENCIL_SIZE_RANGE: (f64, f64) = (0.02, 10.0);

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
    /// Mirror every dab across a plane (Shift+S).
    symmetry: Symmetry,
    /// Vertices that sat on the symmetry plane stay on it, so the halves stay joined.
    sym_snap: bool,
    /// Distance between dabs along a stroke, in brush radii.
    spacing: f64,
    /// The brush alpha turns with the stroke (otherwise it stays upright in the view).
    align_to_stroke: bool,
    /// Footprint / texture of the primary brush, moving with the dab, and how it sits.
    alpha: Option<Alpha>,
    alpha_map: AlphaMap,
    /// Image fixed to the view masking the primary brush, and where it is.
    stencil: Option<Alpha>,
    screen_stencil: ScreenStencil,
    /// How strongly the stencil shows over the view (0..1).
    stencil_opacity: f32,
    /// Left-drags move the stencil (Shift scales, Cmd/Ctrl turns) instead of sculpting.
    placing_stencil: bool,
    stencil_drag: Option<(egui::Pos2, ScreenStencil)>,
    /// PaintVertex colour.
    paint: [f32; 3],
    refinement: Refinement,
    /// Brush radius in percent of the model size.
    size_pct: f32,
    strength: f64,
    stroke: Option<SculptStroke>,
    armed: bool,
    /// Surface point under the pointer, the brush radius there and the surface normal.
    hover: Option<(DVec3, f64)>,
    hover_normal: DVec3,
    /// Keep open-boundary vertices fixed.
    hold_boundary: bool,
    /// Strength when a right-drag (strength adjust) started.
    strength_drag: Option<f64>,
}

struct SculptStroke {
    object: ObjectId,
    invert: bool,
    // Move brush state (with the mirrored grab under symmetry).
    drag_verts: Vec<(u32, f64)>,
    drag_mirror: Vec<(u32, f64)>,
    drag_origin: Option<(DVec3, egui::Pos2)>,
    drag_start_pos: std::collections::HashMap<u32, DVec3>,
    drag_r: f64,
    /// Previous dab centre (for stroke-direction brushes).
    last_center: Option<DVec3>,
    /// Stroke direction, smoothed over the dabs (AlignToStroke).
    dir: Option<DVec3>,
    /// Positions when the stroke first touched each vertex (Draw3's layer reference).
    start: std::collections::HashMap<u32, DVec3>,
    /// Lazy brush point on the surface, and the pointer position last frame (resting test).
    lazy: Option<DVec3>,
    last_pos: Option<egui::Pos2>,
    /// Places the dabs along the stroke.
    spacer: Spacer,
}

impl SculptStroke {
    fn new(object: ObjectId, invert: bool, step: f64) -> Self {
        Self {
            object,
            invert,
            drag_verts: Vec::new(),
            drag_mirror: Vec::new(),
            drag_origin: None,
            drag_start_pos: Default::default(),
            drag_r: 0.0,
            last_center: None,
            dir: None,
            start: Default::default(),
            lazy: None,
            last_pos: None,
            spacer: Spacer::new(step),
        }
    }

    /// Note a dab at `center`: the motion since the previous one, and the smoothed stroke
    /// direction.
    fn advance(&mut self, center: DVec3) -> DVec3 {
        let motion = self.last_center.map_or(DVec3::ZERO, |c| center - c);
        self.last_center = Some(center);
        if let Some(m) = motion.try_normalize() {
            self.dir = Some(self.dir.map_or(m, |d| (d + m).normalize_or(m)));
        }
        motion
    }
}

impl SculptTool {
    pub fn new(cx: &mut ToolCx) -> Self {
        Self::with_mode(cx, SculptMode::Surface, None)
    }

    /// Volume or surface brushes (Meshmixer hotkeys 3 and 2), optionally with a brush.
    pub fn with_mode(cx: &mut ToolCx, mode: SculptMode, brush: Option<Brush>) -> Self {
        Self::build(mode, brush, Symmetry::for_scene(cx.scene))
    }

    fn build(mode: SculptMode, brush: Option<Brush>, symmetry: Symmetry) -> Self {
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
            symmetry,
            sym_snap: false,
            spacing: 0.25,
            align_to_stroke: false,
            alpha: None,
            alpha_map: AlphaMap::default(),
            stencil: None,
            screen_stencil: ScreenStencil::default(),
            stencil_opacity: 0.35,
            placing_stencil: false,
            stencil_drag: None,
            paint: [0.85, 0.25, 0.2],
            refinement: Refinement { enabled: volume, ..Refinement::default() },
            size_pct: 4.0,
            strength: 0.5,
            stroke: None,
            armed: false,
            hover: None,
            hover_normal: DVec3::Z,
            hold_boundary: true,
            strength_drag: None,
        }
    }
}

/// One dab of a stroke (before symmetry).
struct Stamp {
    brush: Brush,
    center: DVec3,
    /// Triangle under the brush centre.
    seed: u32,
    motion: DVec3,
    view: DVec3,
    r: f64,
    invert: bool,
    /// Where the brush alpha's x axis points (laid into the surface): the stroke direction
    /// with AlignToStroke, else the view's right.
    across: DVec3,
    /// How the view maps the surface to the screen (for the stencil).
    screen: Option<ScreenMap>,
}

impl SculptTool {
    fn footprint(&self) -> Footprint {
        Footprint { falloff: self.falloff, volumetric: self.volumetric, hold_boundary: self.hold_boundary, hold_groups: self.hold_groups }
    }

    /// The brush alpha and stencil shape `brush` (the primary brush, not the secondary
    /// smoothing one, the density brushes or Move).
    fn shapes(&self, brush: Brush) -> bool {
        (self.alpha.is_some() || self.stencil.is_some()) && brush == self.brush && !brush.density() && !brush.grabs()
    }

    /// Multiply the falloff weights by the brush alpha (in the dab's `frame`) and the stencil
    /// (where each vertex shows on screen), dropping vertices left at zero. `back` takes a
    /// mirrored dab's vertices to where the original dab sees them, so the mirror image of
    /// the footprint is applied.
    fn shape(&self, mesh: &Mesh, verts: &mut Vec<(u32, f64)>, frame: &DabFrame, screen: Option<&ScreenMap>, back: impl Fn(DVec3) -> DVec3) {
        let alpha = self.alpha.as_ref().map(|a| (&*a.image, self.alpha_map));
        let stencil = self.stencil.as_ref().zip(screen).map(|(a, s)| (&*a.image, self.screen_stencil, *s));
        for (v, w) in verts.iter_mut() {
            let p = back(mesh.vertex(*v));
            if let Some((img, map)) = alpha {
                *w *= map.factor(img, frame.local(p));
            }
            if let Some((img, st, screen)) = &stencil {
                *w *= st.factor(img, p, screen);
            }
        }
        verts.retain(|x| x.1 > 0.0);
    }

    /// Apply a dab and, with symmetry on, its mirror image (the same operation reflected;
    /// Meshmixer does not enforce strict mesh symmetry either). `start` collects where the
    /// stroke first found each vertex. Returns whether the triangles changed.
    fn stamp(&self, o: &mut crate::scene::SceneObject, s: &Stamp, start: &mut std::collections::HashMap<u32, DVec3>, target: Option<&crate::scene::TargetSurface>) -> bool {
        let (brush, r, fp) = (s.brush, s.r, self.footprint());
        let sym = self.symmetry.active();
        let mut dabs = vec![(s.center, s.motion, s.view, Some(s.seed), None)];
        if let Some(sym) = sym {
            dabs.push((sym.reflect(s.center), sym.reflect_dir(s.motion), sym.reflect_dir(s.view), None, Some(sym)));
        }
        let shaped = self.shapes(brush);
        // The footprint's frame comes from the original dab; its mirror image uses it too.
        let mut frame: Option<DabFrame> = None;
        let mut retriangulated = false;
        for (c, motion, view, seed, mirror) in dabs {
            let bvh = o.bvh();
            let Some(seed) = seed.or_else(|| bvh.nearest(&o.mesh, c, r).map(|n| n.tid)) else { continue };
            let mut verts = brush_vertices(&o.mesh, &bvh, c, r, Some(seed), &fp);
            for &(v, _) in &verts {
                start.entry(v).or_insert_with(|| o.mesh.vertex(v));
            }
            if shaped {
                let m: &Mesh = &o.mesh;
                let f = *frame.get_or_insert_with(|| {
                    let n = verts.iter().map(|&(v, w)| m.vertex_normal(v) * w).sum::<DVec3>().normalize_or(m.tri_normal(seed));
                    DabFrame::new(c, n, s.across, r)
                });
                self.shape(m, &mut verts, &f, s.screen.as_ref(), |p| mirror.map_or(p, |sym| sym.reflect(p)));
            }
            if brush.density() {
                retriangulated |= density_dab(o.mesh_mut(), &verts, brush, r, self.strength);
            } else if brush.paints() {
                let color = if s.invert { mm_core::mesh::DEFAULT_VERTEX_COLOR } else { self.paint };
                paint_dab(o.mesh_mut(), &verts, brush, color, self.strength);
                o.moved_vertices.extend(verts.iter().map(|&(v, _)| v));
            } else {
                let dab = Dab { center: c, r, strength: self.strength, invert: s.invert, view, motion, start, target };
                apply_dab(o.mesh_mut(), &verts, brush, &dab);
                if self.attract && target.is_some() && brush != Brush::Attract {
                    apply_dab(o.mesh_mut(), &verts, Brush::Attract, &Dab { strength: self.strength * 0.5, ..dab });
                }
                o.moved_vertices.extend(verts.iter().map(|&(v, _)| v));
                if self.refinement.enabled {
                    let bvh = o.bvh();
                    retriangulated |= refine_region(o.mesh_mut(), &bvh, c, r, &self.refinement);
                }
            }
        }
        if let (Some(sym), true) = (sym, self.sym_snap) {
            let bvh = o.bvh();
            let mut near = brush_vertices(&o.mesh, &bvh, s.center, r, None, &Footprint::default());
            near.extend(brush_vertices(&o.mesh, &bvh, sym.reflect(s.center), r, None, &Footprint::default()));
            sym_snap(o.mesh_mut(), near.iter().map(|x| x.0), start, sym, r * SYM_SNAP_TOL);
            o.moved_vertices.extend(near.iter().map(|x| x.0));
        }
        retriangulated
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
        Brush::Move | Brush::Refine | Brush::Reduce | Brush::AdaptiveReduce | Brush::PaintVertex | Brush::SmoothColor => Vec::new(),
    };
    let pos = mesh.positions_mut();
    for (v, p) in updates {
        pos[v as usize] = p;
    }
    mesh.bump_shape_version();
}

fn brush_key(b: Brush) -> String {
    b.label().to_lowercase().replace([' ', '-'], "_")
}

impl SculptTool {
    /// How the current view maps the surface to the screen.
    fn screen_map(cx: &ToolCx) -> ScreenMap {
        let size = cx.vp.rect.size();
        ScreenMap { view_proj: cx.camera.view_proj(cx.vp.aspect()), size: DVec2::new(size.x as f64, size.y as f64) }
    }

    /// One dab of `stroke` at surface point `p` (Depth applied): the motion since the last
    /// dab, the alpha's direction and symmetry as dragging does. Returns whether the
    /// triangles changed.
    fn dab_at(&self, cx: &mut ToolCx, stroke: &mut SculptStroke, brush: Brush, p: DVec3, r: f64, target: Option<&TargetSurface>) -> bool {
        let (view, right, screen) = (cx.camera.forward(), cx.camera.right(), Self::screen_map(cx));
        let Some(o) = cx.scene.get_mut(stroke.object) else { return false };
        let bvh = o.bvh();
        let Some(h) = bvh.nearest(&o.mesh, p, f64::INFINITY) else { return false };
        let center = h.point - o.mesh.tri_normal(h.tid) * (self.depth / 100.0 * r);
        let motion = stroke.advance(center);
        let across = if self.align_to_stroke { stroke.dir.unwrap_or(right) } else { right };
        if o.moved_vertices.is_empty() {
            o.moved_from = o.mesh.shape_version();
        }
        let stamp = Stamp { brush, center, seed: h.tid, motion, view, r, invert: stroke.invert, across, screen: Some(screen) };
        let retriangulated = self.stamp(o, &stamp, &mut stroke.start, target);
        if retriangulated {
            // New triangles: the GPU copy and the selection are rebuilt.
            o.moved_vertices.clear();
            let m = o.mesh.clone();
            o.selection.sync(&m);
            o.touch_selection();
        }
        retriangulated
    }

    /// Dabs along `path` (points in mm; the brush's spacing apart, from the first point)
    /// with the current brush, as one stroke: what dragging the brush along it does.
    fn stroke_along(&mut self, cx: &mut ToolCx, path: &[DVec3], invert: bool) -> Result<usize, String> {
        let id = cx.scene.active.ok_or("no active object")?;
        let r = brush_radius(cx.scene, id, self.size_pct);
        if r <= 0.0 {
            return Err("the brush has no size".into());
        }
        if let Some(b) = cx.scene.get(id).map(|o| o.bounds()) {
            self.symmetry.place(id, &b);
        }
        let mut stroke = SculptStroke::new(id, invert, self.spacing * r);
        let pts: Vec<DVec3> = path.iter().flat_map(|&p| stroke.spacer.advance(p, usize::MAX)).collect();
        cx.snapshot(&format!("Sculpt: {}", self.brush.label()));
        let target = cx.scene.target_surface(Some(id));
        for p in &pts {
            self.dab_at(cx, &mut stroke, self.brush, *p, r, target.as_ref());
        }
        Ok(pts.len())
    }

    /// The Move brush: carry the surface under the brush at `from` by `to − from` (and its
    /// mirror image with symmetry on), as dragging it does.
    fn drag_from_to(&mut self, cx: &mut ToolCx, from: DVec3, to: DVec3) -> Result<usize, String> {
        let id = cx.scene.active.ok_or("no active object")?;
        let r = brush_radius(cx.scene, id, self.size_pct);
        let fp = self.footprint();
        if let Some(b) = cx.scene.get(id).map(|o| o.bounds()) {
            self.symmetry.place(id, &b);
        }
        let sym = self.symmetry.active();
        cx.snapshot(&format!("Sculpt: {}", self.brush.label()));
        let o = cx.scene.get_mut(id).ok_or("no object")?;
        let bvh = o.bvh();
        let h = bvh.nearest(&o.mesh, from, f64::INFINITY).ok_or("no surface")?;
        let delta = to - h.point;
        let verts = brush_vertices(&o.mesh, &bvh, h.point, r, Some(h.tid), &fp);
        let mirror = sym.and_then(|s| bvh.nearest(&o.mesh, s.reflect(h.point), r).map(|n| (s, brush_vertices(&o.mesh, &bvh, s.reflect(h.point), r, Some(n.tid), &fp)))).unwrap_or((SymPlane::new(DVec3::ZERO, DVec3::X), Vec::new()));
        let mut moved: std::collections::HashMap<u32, DVec3> = verts.iter().chain(&mirror.1).map(|&(v, _)| (v, o.mesh.vertex(v))).collect();
        let start = moved.clone();
        for (vs, step) in [(&verts, delta), (&mirror.1, mirror.0.reflect_dir(delta))] {
            for &(v, w) in vs {
                if let Some(p) = moved.get_mut(&v) {
                    *p += step * w;
                }
            }
        }
        if o.moved_vertices.is_empty() {
            o.moved_from = o.mesh.shape_version();
        }
        let m = o.mesh_mut();
        for (&v, &p) in &moved {
            m.positions_mut()[v as usize] = p;
        }
        m.bump_shape_version();
        if let (Some(sym), true) = (sym, self.sym_snap) {
            sym_snap(m, moved.keys().copied(), &start, sym, r * SYM_SNAP_TOL);
        }
        o.moved_vertices.extend(moved.keys().copied());
        Ok(moved.len())
    }

    /// The stroke following the pointer at `pos`: Move grabs the surface once and carries
    /// it; the other brushes dab the spacing apart along the (lazy) path, and with Flow keep
    /// dabbing where the pointer rests.
    fn drag_stroke(&self, stroke: &mut SculptStroke, input: &PointerInput, pos: egui::Pos2, hover: Option<(ObjectId, mm_core::bvh::RayHit)>, cx: &mut ToolCx) {
        let fp = self.footprint();
        let brush = if input.shift && !self.brush.grabs() { self.secondary } else { self.brush };
        let resting = stroke.last_pos == Some(pos);
        stroke.last_pos = Some(pos);
        let sym = self.symmetry.active();
        if brush.grabs() {
            if stroke.drag_origin.is_none() {
                let Some((_, h)) = hover else { return };
                let r = brush_radius(cx.scene, stroke.object, self.size_pct);
                let Some(o) = cx.scene.get_mut(stroke.object) else { return };
                let bvh = o.bvh();
                stroke.drag_verts = brush_vertices(&o.mesh, &bvh, h.point, r, Some(h.tid), &fp);
                if let Some(sym) = sym {
                    let mc = sym.reflect(h.point);
                    if let Some(n) = bvh.nearest(&o.mesh, mc, r) {
                        stroke.drag_mirror = brush_vertices(&o.mesh, &bvh, mc, r, Some(n.tid), &fp);
                    }
                }
                stroke.drag_start_pos = stroke.drag_verts.iter().chain(&stroke.drag_mirror).map(|&(v, _)| (v, o.mesh.vertex(v))).collect();
                stroke.drag_origin = Some((h.point, pos));
                stroke.drag_r = r;
                return;
            }
            if resting {
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
            // Mirrored grabs move by the mirrored offset; where the two overlap both apply.
            let mut moved: std::collections::HashMap<u32, DVec3> = stroke.drag_start_pos.clone();
            let mirrored = sym.map_or(DVec3::ZERO, |s| s.reflect_dir(delta));
            for (verts, step) in [(&stroke.drag_verts, delta), (&stroke.drag_mirror, mirrored)] {
                for &(v, w) in verts {
                    if let Some(p) = moved.get_mut(&v) {
                        *p += step * w;
                    }
                }
            }
            let m = o.mesh_mut();
            let pos_arr = m.positions_mut();
            for (&v, &p) in &moved {
                pos_arr[v as usize] = p;
            }
            m.bump_shape_version();
            if let (Some(sym), true) = (sym, self.sym_snap) {
                sym_snap(m, moved.keys().copied(), &stroke.drag_start_pos, sym, stroke.drag_r * SYM_SNAP_TOL);
            }
            o.moved_vertices.extend(moved.keys().copied());
            return;
        }
        let Some((_, h)) = hover else { return };
        let r = brush_radius(cx.scene, stroke.object, self.size_pct);
        // Lazyness lets the brush trail the pointer.
        let mut p = h.point;
        if let Some(prev) = stroke.lazy {
            p = prev + (p - prev) * (1.0 - 0.95 * self.lazyness / 100.0);
        }
        stroke.lazy = Some(p);
        let mut dabs = stroke.spacer.advance(p, MAX_DABS_PER_MOVE);
        if self.flow {
            if dabs.is_empty() && resting {
                dabs.push(p);
            }
            // Frames keep coming while the button is held, so a resting brush keeps working.
            cx.egui.request_repaint();
        }
        let target = cx.scene.target_surface(Some(stroke.object));
        for q in dabs {
            self.dab_at(cx, stroke, brush, q, r, target.as_ref());
        }
    }

    /// Stencil placing: left-drag moves the stencil, Shift+drag sizes it (up: larger),
    /// Cmd/Ctrl+drag turns it.
    fn place_stencil(&mut self, input: &PointerInput, cx: &ToolCx) {
        let Some(pos) = input.pos else { return };
        if input.pressed {
            self.stencil_drag = Some((pos, self.screen_stencil));
        }
        if let (true, Some((p0, s0))) = (input.down, self.stencil_drag) {
            let d = pos - p0;
            let size = cx.vp.rect.size();
            let mut st = s0;
            if input.shift {
                st.map.scale = (s0.map.scale * (-d.y as f64 * 0.01).exp()).clamp(STENCIL_SIZE_RANGE.0, STENCIL_SIZE_RANGE.1);
            } else if input.command {
                st.map.rotation_deg = (s0.map.rotation_deg - d.x as f64 * 0.5 + 180.0).rem_euclid(360.0) - 180.0;
            } else {
                st.center = s0.center + DVec2::new((d.x / size.x.max(1.0)) as f64, (d.y / size.y.max(1.0)) as f64);
            }
            self.screen_stencil = st;
        }
        if input.released {
            self.stencil_drag = None;
        }
    }
}

impl Tool for SculptTool {
    fn brush_size(&mut self) -> Option<&mut f32> {
        Some(&mut self.size_pct)
    }

    fn settings(&self) -> Option<serde_json::Value> {
        use polymix_workspace::agent::round;
        let name = |a: &Option<Alpha>| a.as_ref().map_or_else(|| "none".to_string(), Alpha::key);
        let (m, st) = (self.alpha_map, self.screen_stencil);
        let mut v = serde_json::json!({
            "brush": brush_key(self.brush),
            "size_percent": self.size_pct,
            "strength": self.strength,
            "depth": self.depth,
            "hold_boundary": self.hold_boundary,
            "hold_groups": self.hold_groups,
            "sym_snap": self.sym_snap,
            "spacing_percent": round(self.spacing * 100.0, 3),
            "align_to_stroke": self.align_to_stroke,
            "alpha": {"image": name(&self.alpha), "scale": m.scale, "rotation_deg": m.rotation_deg, "strength": m.strength, "tile": m.tile, "invert": m.invert},
            "stencil": {
                "image": name(&self.stencil), "center": [round(st.center.x, 5), round(st.center.y, 5)], "size": st.map.scale, "rotation_deg": st.map.rotation_deg,
                "strength": st.map.strength, "tile": st.map.tile, "invert": st.map.invert,
            },
        });
        if let Some(o) = v.as_object_mut() {
            o.extend(self.symmetry.settings());
        }
        Some(v)
    }

    fn settings_schema(&self) -> Option<serde_json::Value> {
        let image = "none, a built-in pattern (".to_string() + &mm_edit::Pattern::ALL.map(|p| p.key()).join(", ") + "), an image in the stencils folder, or a path to a grey PNG / JPEG";
        let mut props = serde_json::json!({
            "brush": {"type": "string", "enum": Brush::ALL.map(brush_key), "description": "The brush (robust_smooth for smoothing bumps; draw / inflate / flatten / pinch to shape; move drags)"},
            "size_percent": {"type": "number", "description": "Brush radius as a percentage of the object's size (0.1..25); the state gives it in mm"},
            "strength": {"type": "number", "description": "0..1"},
            "depth": {"type": "number", "description": "-100..100: moves the brush centre into (+) or out of (−) the surface"},
            "hold_boundary": {"type": "boolean", "description": "Open-boundary vertices stay put"},
            "hold_groups": {"type": "boolean", "description": "Face-group border vertices stay put"},
            "sym_snap": {"type": "boolean", "description": "With symmetry: vertices on the symmetry plane stay on it"},
            "spacing_percent": {"type": "number", "description": "Distance between dabs along a stroke, as a percentage of the brush radius (5..300)"},
            "align_to_stroke": {"type": "boolean", "description": "The brush alpha turns with the stroke direction (otherwise it stays upright in the view)"},
            "alpha": {"type": "object", "description": "Brush alpha: an image shaping the brush's footprint, moving with each dab (the draw brushes emboss it)", "properties": {
                "image": {"type": "string", "description": image},
                "scale": {"type": "number", "description": "Image side over the brush diameter (0.05..8)"},
                "rotation_deg": {"type": "number", "description": "Turns the image counter-clockwise"},
                "strength": {"type": "number", "description": "0..1: how much the image shapes the brush"},
                "tile": {"type": "boolean", "description": "Repeat the image (otherwise the brush does nothing outside it)"},
                "invert": {"type": "boolean"},
            }},
            "stencil": {"type": "object", "description": "Stencil: an image fixed to the view; the brush acts where it is light", "properties": {
                "image": {"type": "string", "description": image},
                "center": {"type": "array", "items": {"type": "number"}, "minItems": 2, "maxItems": 2, "description": "Centre as fractions of the viewport's width and height from its top left"},
                "size": {"type": "number", "description": "Image side over the viewport height (0.02..10)"},
                "rotation_deg": {"type": "number"},
                "strength": {"type": "number", "description": "0..1"},
                "tile": {"type": "boolean", "description": "Repeat the image over the whole view"},
                "invert": {"type": "boolean"},
            }},
        });
        if let Some(o) = props.as_object_mut() {
            o.extend(Symmetry::schema());
        }
        Some(serde_json::json!({"type": "object", "properties": props}))
    }

    fn set_settings(&mut self, changes: &serde_json::Value, cx: &mut ToolCx) -> Result<(), String> {
        let mut v = self.settings().unwrap_or_default();
        merge_settings(&mut v, changes)?;
        let a = Args(&v);
        // Images first: a name that matches nothing changes nothing.
        let image = |k: &str| changes.get(k).and_then(|x| x.get("image")).map(|x| x.as_str().ok_or(format!("{k}.image is a name or path")).and_then(Alpha::find)).transpose();
        let (alpha, stencil) = (image("alpha")?, image("stencil")?);
        self.brush = Brush::ALL.into_iter().find(|b| a.opt_str("brush") == Some(brush_key(*b).as_str())).ok_or("unknown brush")?;
        self.size_pct = (a.f64("size_percent")? as f32).clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1);
        self.strength = a.f64("strength")?.clamp(0.0, 1.0);
        self.depth = a.f64("depth")?.clamp(-100.0, 100.0);
        self.hold_boundary = a.opt_bool("hold_boundary").unwrap_or(self.hold_boundary);
        self.hold_groups = a.opt_bool("hold_groups").unwrap_or(self.hold_groups);
        self.sym_snap = a.opt_bool("sym_snap").unwrap_or(self.sym_snap);
        self.spacing = (a.f64("spacing_percent")? / 100.0).clamp(SPACING_RANGE.0, SPACING_RANGE.1);
        self.align_to_stroke = a.opt_bool("align_to_stroke").unwrap_or(self.align_to_stroke);
        if let Some(x) = alpha {
            self.alpha = x;
        }
        if let Some(x) = stencil {
            self.stencil = x;
        }
        let map = |m: &serde_json::Value, old: AlphaMap, size_key: &str, range: (f64, f64)| -> Result<AlphaMap, String> {
            let m = Args(m);
            Ok(AlphaMap {
                scale: m.f64(size_key)?.clamp(range.0, range.1),
                rotation_deg: m.f64("rotation_deg")?,
                strength: m.f64("strength")?.clamp(0.0, 1.0),
                tile: m.opt_bool("tile").unwrap_or(old.tile),
                invert: m.opt_bool("invert").unwrap_or(old.invert),
            })
        };
        self.alpha_map = map(&v["alpha"], self.alpha_map, "scale", ALPHA_SCALE_RANGE)?;
        let st = map(&v["stencil"], self.screen_stencil.map, "size", STENCIL_SIZE_RANGE)?;
        let c = v["stencil"]["center"].as_array().and_then(|c| Some(DVec2::new(c.first()?.as_f64()?, c.get(1)?.as_f64()?))).ok_or("stencil.center is [x, y]")?;
        self.screen_stencil = ScreenStencil { center: c, map: st };
        self.symmetry.set_settings(&v, changes, cx)
    }

    fn state(&self, scene: &Scene) -> serde_json::Value {
        let radius = scene.active().map(|o| o.bounds().diagonal() * self.size_pct as f64 / 100.0);
        serde_json::json!({
            "object": scene.active,
            "brush": brush_key(self.brush),
            "brush_radius_mm": radius.map(|r| polymix_workspace::agent::round(r, 4)),
            "next": if self.brush.grabs() { "drag the surface from a point to another" } else { "stroke along points on the surface" },
        })
    }

    fn actions(&self) -> Vec<Action> {
        use serde_json::json;
        let mut out = vec![
            Action::new("stroke", "Brush along a path over the active object, as dragging the brush does: one undo step (not for the move brush)", json!({"path": {"type": "array", "items": {"type": "array", "items": {"type": "number"}, "minItems": 3, "maxItems": 3}, "description": "Points in mm on or near the surface, in order; dabs go the spacing apart from the first point (one point is a single dab)"}, "invert": arg::boolean("The brush's opposite (Ctrl held): draw in, deflate…")})).required(&["path"]),
            Action::new("drag", "The move brush: carry the surface under the brush at `from` to `to`", json!({"from": arg::point("A point on or near the surface, mm"), "to": arg::point("Where it goes, mm")})).required(&["from", "to"]),
        ];
        out.extend(Symmetry::actions());
        out
    }

    fn act(&mut self, action: &str, args: &serde_json::Value, cx: &mut ToolCx) -> Result<String, String> {
        if let Some(r) = self.symmetry.act(action, args, cx) {
            return r;
        }
        let a = Args(args);
        match action {
            "stroke" => {
                if self.brush.grabs() {
                    return Err("the move brush drags: use drag, or choose another brush".into());
                }
                let path: Vec<DVec3> = a.0.get("path").and_then(serde_json::Value::as_array).ok_or("`path` must be a list of points")?.iter().map(|p| Args(&serde_json::json!({"p": p})).point("p").map(DVec3::from_array)).collect::<Result<_, _>>()?;
                if path.is_empty() {
                    return Err("the path is empty".into());
                }
                let invert = a.opt_bool("invert").unwrap_or(false);
                let n = self.stroke_along(cx, &path, invert)?;
                Ok(format!("{}: {n} dabs", self.brush.label()))
            }
            "drag" => {
                if !self.brush.grabs() {
                    return Err("drag is for the move brush (setting brush: move)".into());
                }
                let n = self.drag_from_to(cx, DVec3::from_array(a.point("from")?), DVec3::from_array(a.point("to")?))?;
                Ok(format!("Moved {n} vertices"))
            }
            _ => Err(format!("{} has no action {action}", self.name())),
        }
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
        if self.brush.paints() {
            ui.horizontal(|ui| {
                ui.color_edit_button_rgb(&mut self.paint);
                ui.label("Paint colour");
            });
        }
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
        let mut spacing = self.spacing * 100.0;
        if ui
            .add(egui::Slider::new(&mut spacing, SPACING_RANGE.0 * 100.0..=SPACING_RANGE.1 * 100.0).logarithmic(true).suffix("%").text("spacing"))
            .on_hover_text("Distance between dabs along the stroke, as a percentage of the brush radius: strokes come out the same however fast they go. Large spacing stamps the brush alpha one dab at a time.")
            .changed()
        {
            self.spacing = spacing / 100.0;
        }
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.flow, "Flow").on_hover_text("Keep sculpting while the pointer rests");
            ui.checkbox(&mut self.volumetric, "Volumetric").on_hover_text("Affect everything inside the brush, not only the stroked surface");
        });
        self.symmetry.ui(ui, cx, "Symmetry (Shift+S)");
        ui.add_enabled(self.symmetry.enabled, egui::Checkbox::new(&mut self.sym_snap, "SymSnap")).on_hover_text("Vertices on the symmetry plane stay on it, so the two halves stay joined");
        ui.horizontal_wrapped(|ui| {
            ui.checkbox(&mut self.hold_boundary, "Hold boundary (B)");
            ui.checkbox(&mut self.hold_groups, "Preserve groups (G)");
            ui.checkbox(&mut self.attract, "Attract (A)").on_hover_text("Also pull the stroked surface toward the target surface");
        });
        egui::CollapsingHeader::new("Brush texture").default_open(self.alpha.is_some()).show(ui, |ui| {
            ui.small("An image shaping the primary brush's footprint; it moves with each dab, so the draw brushes emboss it.");
            stencil::picker(ui, "sculpt alpha", &mut self.alpha, self.alpha_map.invert, cx.status);
            ui.add_enabled_ui(self.alpha.is_some(), |ui| {
                let m = &mut self.alpha_map;
                ui.add(egui::Slider::new(&mut m.scale, ALPHA_SCALE_RANGE.0..=ALPHA_SCALE_RANGE.1).logarithmic(true).text("scale")).on_hover_text("Image size over the brush diameter");
                ui.add(egui::Slider::new(&mut m.rotation_deg, -180.0..=180.0).suffix("°").text("rotation"));
                ui.add(egui::Slider::new(&mut m.strength, 0.0..=1.0).text("strength")).on_hover_text("How much the image shapes the brush (0: a plain brush)");
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut m.tile, "Tile").on_hover_text("Repeat the image across the brush (otherwise nothing outside it)");
                    ui.checkbox(&mut m.invert, "Invert");
                    ui.checkbox(&mut self.align_to_stroke, "Align to stroke").on_hover_text("The image turns with the stroke direction (otherwise it stays upright in the view)");
                });
            });
        });
        egui::CollapsingHeader::new("Stencil").default_open(self.stencil.is_some()).show(ui, |ui| {
            ui.small("An image fixed to the view: the primary brush acts where it is light.");
            if stencil::picker(ui, "sculpt stencil", &mut self.stencil, self.screen_stencil.map.invert, cx.status) && self.stencil.is_none() {
                self.placing_stencil = false;
            }
            ui.add_enabled_ui(self.stencil.is_some(), |ui| {
                ui.horizontal(|ui| {
                    ui.toggle_value(&mut self.placing_stencil, "Place").on_hover_text("Left-drag moves the stencil, Shift+drag sizes it, Cmd/Ctrl+drag turns it (no sculpting until you turn this off or press Esc)");
                    if ui.button("Centre").on_hover_text("Back to the middle of the view, unturned").clicked() {
                        self.screen_stencil.center = DVec2::splat(0.5);
                        self.screen_stencil.map.rotation_deg = 0.0;
                    }
                });
                let st = &mut self.screen_stencil;
                ui.add(egui::Slider::new(&mut st.map.scale, STENCIL_SIZE_RANGE.0..=STENCIL_SIZE_RANGE.1).logarithmic(true).text("size")).on_hover_text("Image size over the view's height");
                ui.add(egui::Slider::new(&mut st.map.rotation_deg, -180.0..=180.0).suffix("°").text("rotation"));
                ui.add(egui::Slider::new(&mut st.map.strength, 0.0..=1.0).text("strength"));
                ui.horizontal_wrapped(|ui| {
                    ui.checkbox(&mut st.map.tile, "Tile").on_hover_text("Repeat the image over the whole view");
                    ui.checkbox(&mut st.map.invert, "Invert");
                });
                ui.add(egui::Slider::new(&mut self.stencil_opacity, 0.0..=1.0).text("show")).on_hover_text("How strongly the stencil shows over the view");
            });
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
        ui.small(format!("Drag on the surface. Shift uses the secondary brush; Cmd/Ctrl inverts the draw brushes, Inflate, Push and Pinch. Size: [ ], {} or Cmd/Ctrl+middle-drag. Cmd/Ctrl+right-drag: strength.", cx.brush_scroll.scroll_label()));
        ui.add_space(8.0);
        if ui.button("Done").clicked() {
            return ToolOutcome::Close;
        }
        ToolOutcome::Continue
    }

    fn pointer(&mut self, input: &PointerInput, cx: &mut ToolCx) {
        // Setting the symmetry plane and placing the stencil take the pointer: no sculpting.
        if self.symmetry.pointer(input, cx) {
            self.hover = None;
            return;
        }
        if self.placing_stencil && self.stencil.is_some() {
            self.hover = None;
            self.place_stencil(input, cx);
            return;
        }
        let Some(pos) = input.pos else { return };
        let hover = cx.pick(pos, cx.scene.active);
        self.hover = hover.map(|(id, h)| (h.point, brush_radius(cx.scene, id, self.size_pct)));
        if let Some(n) = hover.and_then(|(id, h)| cx.scene.get(id).map(|o| o.mesh.tri_normal(h.tid))) {
            self.hover_normal = n;
        }
        if input.pressed {
            self.armed = true;
        }
        if input.down && self.armed && self.stroke.is_none() {
            if let Some((id, _)) = hover {
                cx.scene.active = Some(id);
                cx.snapshot(&format!("Sculpt: {}", self.brush.label()));
                if let Some(b) = cx.scene.get(id).map(|o| o.bounds()) {
                    self.symmetry.place(id, &b);
                }
                let r = brush_radius(cx.scene, id, self.size_pct);
                self.stroke = Some(SculptStroke::new(id, input.command, self.spacing * r));
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
        let Some(mut stroke) = self.stroke.take() else { return };
        self.drag_stroke(&mut stroke, input, pos, hover, cx);
        self.stroke = Some(stroke);
    }

    fn overlay(&mut self, ov: &mut Overlay, cx: &mut ToolCx) {
        if let Some((c, r)) = self.hover {
            ov.brush = Some((c.as_vec3(), r as f32));
            if self.alpha.is_some() {
                // The brush alpha's square on the surface, turned as the next dab would be,
                // with a tick to its top edge.
                let right = cx.camera.right();
                let across = if self.align_to_stroke { self.stroke.as_ref().and_then(|s| s.dir).unwrap_or(right) } else { right };
                let f = DabFrame::new(c + self.hover_normal * (r * 0.02), self.hover_normal, across, r);
                let m = self.alpha_map;
                let q = [DVec2::ZERO, DVec2::X, DVec2::ONE, DVec2::Y].map(|uv| f.point(m.local(uv)));
                let col = Vec4::new(1.0, 0.55, 0.15, 0.9);
                for i in 0..4 {
                    ov.line(q[i], q[(i + 1) % 4], col);
                }
                ov.line(f.center, f.point(m.local(DVec2::new(0.5, 0.0))), col);
            }
        }
        self.symmetry.overlay(ov, cx);
        // Ring on the surface around the mirrored brush centre.
        let (Some(sym), Some((c, r))) = (self.symmetry.active(), self.hover) else { return };
        let Some(o) = cx.scene.active.and_then(|id| cx.scene.get_mut(id)) else { return };
        let bvh = o.bvh();
        let Some(h) = bvh.nearest(&o.mesh, sym.reflect(c), r) else { return };
        let n = o.mesh.tri_normal(h.tid);
        let (u, v) = mm_core::geom::orthonormal_basis(n);
        let ring: Vec<DVec3> = (0..=48)
            .map(|k| {
                let a = k as f64 / 48.0 * std::f64::consts::TAU;
                let p = h.point + (u * a.cos() + v * a.sin()) * r;
                bvh.nearest(&o.mesh, p, r).map_or(p, |q| q.point + o.mesh.tri_normal(q.tid) * (r * 0.02))
            })
            .collect();
        for w in ring.windows(2) {
            ov.line(w[0], w[1], Vec4::new(0.3, 0.6, 1.0, 0.9));
        }
    }

    fn paint(&mut self, painter: &egui::Painter, cx: &mut ToolCx) {
        if let Some(a) = &mut self.stencil {
            stencil::paint_stencil(painter, cx.vp.rect, a, &self.screen_stencil, self.stencil_opacity, cx.egui);
        }
        self.symmetry.paint(painter, cx);
    }

    fn cancel(&mut self, _cx: &mut ToolCx) -> bool {
        // Esc leaves Set (restoring the plane) or stencil placing before it closes the tool.
        if self.symmetry.cancel_edit() {
            return false;
        }
        if self.placing_stencil {
            self.placing_stencil = false;
            return false;
        }
        true
    }

    fn accept(&mut self, _cx: &mut ToolCx) -> bool {
        self.symmetry.accept_edit();
        self.placing_stencil = false;
        false
    }

    fn wheel(&mut self, delta: f32, _cx: &mut ToolCx) -> bool {
        self.size_pct = (self.size_pct * (delta * 0.004).exp()).clamp(BRUSH_SIZE_RANGE.0, BRUSH_SIZE_RANGE.1);
        true
    }

    fn script(&mut self, cmd: &str, cx: &mut ToolCx) {
        // `brush:Refine`, `size:8`, `sym:x` / `sym:view` / `sym:set` / `sym:off`, `spacing:50`,
        // `alpha:noise`, `stencil:rings`, `align:on`.
        match cmd.split_once(':') {
            Some(("brush", name)) => self.brush = Brush::ALL.into_iter().find(|b| b.label().eq_ignore_ascii_case(name)).unwrap_or(self.brush),
            Some(("size", pct)) => self.size_pct = pct.parse().unwrap_or(self.size_pct),
            Some(("color", hex)) => {
                if let Ok(x) = u32::from_str_radix(hex.trim_start_matches('#'), 16) {
                    self.paint = [(x >> 16) & 255, (x >> 8) & 255, x & 255].map(|c| c as f32 / 255.0);
                }
            }
            Some(("sym", axis)) => {
                self.symmetry.enabled = axis != "off";
                if axis == "set" {
                    self.symmetry.begin_edit(cx);
                } else if axis == "view" {
                    self.symmetry.set_plane(SymPlane::from_view(cx.camera));
                } else if let Some(i) = ["x", "y", "z"].iter().position(|a| *a == axis) {
                    self.symmetry.preset(cx, i);
                }
            }
            Some(("spacing", pct)) => self.spacing = pct.parse::<f64>().map_or(self.spacing, |p| (p / 100.0).clamp(SPACING_RANGE.0, SPACING_RANGE.1)),
            Some(("alpha", name)) => match Alpha::find(name) {
                Ok(a) => self.alpha = a,
                Err(e) => *cx.status = e,
            },
            Some(("stencil", name)) => match Alpha::find(name) {
                Ok(a) => self.stencil = a,
                Err(e) => *cx.status = e,
            },
            Some(("align", on)) => self.align_to_stroke = on == "on",
            _ => {}
        }
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
        if i.key_pressed(egui::Key::S) && i.modifiers.shift && !i.modifiers.command && !i.modifiers.alt {
            if self.symmetry.enabled {
                self.symmetry.cancel_edit();
            }
            return toggle(&mut self.symmetry.enabled, "Symmetry", cx.status);
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
    fn symmetric_strokes_stay_symmetric() {
        // An icosphere is symmetric across x = 0; a mirrored stroke must keep it so.
        let m0 = primitives::icosphere(1.0, 4);
        let mut o = crate::scene::SceneObject::new(1, "s", m0.clone());
        let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Draw), Symmetry::new(Some(SymPlane::axis(&m0.bounds(), 0)), None));
        tool.symmetry.enabled = true;
        tool.strength = 0.8;
        let mut start = Default::default();
        let mut c = DVec3::new(0.6, 0.0, 0.8).normalize();
        for _ in 0..6 {
            let bvh = o.bvh();
            let h = bvh.nearest(&o.mesh, c, 1.0).unwrap();
            let s = Stamp { brush: Brush::Draw, center: h.point, seed: h.tid, motion: DVec3::Y * 0.05, view: -c, r: 0.35, invert: false, across: DVec3::X, screen: None };
            tool.stamp(&mut o, &s, &mut start, None);
            c = (c + DVec3::Y * 0.05).normalize();
        }
        let m = &o.mesh;
        let bvh0 = mm_core::Bvh::build(&m0);
        let mut moved = 0;
        for v in m.vertex_ids() {
            let p = m.vertex(v);
            if p.distance(m0.vertex(v)) > 1e-6 {
                moved += 1;
            }
            // The mirror twin of every vertex moved the same way.
            let twin = bvh0.nearest(&m0, SymPlane::new(DVec3::ZERO, DVec3::X).reflect(m0.vertex(v)), 1e-6).map(|h| m0.triangle(h.tid).into_iter().min_by(|&a, &b| m0.vertex(a).distance(h.point).total_cmp(&m0.vertex(b).distance(h.point))).unwrap()).unwrap();
            let q = m.vertex(twin);
            assert!((p - DVec3::new(-q.x, q.y, q.z)).length() < 1e-9, "vertex {v}");
        }
        assert!(moved > 20);
        // Without symmetry the other side stays put.
        let mut o = crate::scene::SceneObject::new(2, "s", m0.clone());
        tool.symmetry.enabled = false;
        let bvh = o.bvh();
        let h = bvh.nearest(&o.mesh, DVec3::new(0.6, 0.0, 0.8), 1.0).unwrap();
        tool.stamp(&mut o, &Stamp { brush: Brush::Draw, center: h.point, seed: h.tid, motion: DVec3::ZERO, view: -DVec3::Z, r: 0.35, invert: false, across: DVec3::X, screen: None }, &mut Default::default(), None);
        assert!(o.mesh.vertex_ids().filter(|&v| m0.vertex(v).x < -0.1).all(|v| o.mesh.vertex(v) == m0.vertex(v)));
    }

    #[test]
    fn symsnap_keeps_the_seam_on_the_plane() {
        // Inflate across the seam of a sphere moves seam vertices off x = 0 only through
        // asymmetric normals; on a slightly perturbed sphere they drift unless snapped.
        let mut m0 = primitives::icosphere(1.0, 4);
        m0.transform(|p| p + DVec3::X * 0.02 * p.y * p.z);
        let seam: Vec<u32> = m0.vertex_ids().filter(|&v| m0.vertex(v).x.abs() < 1e-9).collect();
        assert!(!seam.is_empty());
        let run = |snap: bool| {
            let mut o = crate::scene::SceneObject::new(1, "s", m0.clone());
            let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Inflate), Symmetry::new(Some(SymPlane::new(DVec3::ZERO, DVec3::X)), None));
            tool.symmetry.enabled = true;
            tool.sym_snap = snap;
            let mut start = Default::default();
            for _ in 0..5 {
                let bvh = o.bvh();
                let h = bvh.nearest(&o.mesh, DVec3::new(0.05, 0.3, 0.95), 1.0).unwrap();
                tool.stamp(&mut o, &Stamp { brush: Brush::Inflate, center: h.point, seed: h.tid, motion: DVec3::ZERO, view: -DVec3::Z, r: 0.4, invert: false, across: DVec3::X, screen: None }, &mut start, None);
            }
            o.mesh.check_validity().unwrap();
            let drift = seam.iter().map(|&v| o.mesh.vertex(v).x.abs()).fold(0.0, f64::max);
            let lifted = seam.iter().filter(|&&v| o.mesh.vertex(v).length() > m0.vertex(v).length() + 1e-4).count();
            (drift, lifted)
        };
        let (free, lifted) = run(false);
        let (snapped, lifted_snapped) = run(true);
        assert!(free > 1e-6, "{free}");
        assert!(snapped < 1e-12, "{snapped}");
        assert!(lifted > 0 && lifted_snapped == lifted);
    }

    #[test]
    fn paint_and_smooth_vertex_colors() {
        let m0 = primitives::icosphere(1.0, 4);
        let mut o = crate::scene::SceneObject::new(1, "s", m0.clone());
        let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::PaintVertex), Symmetry::new(Some(SymPlane::new(DVec3::ZERO, DVec3::X)), None));
        tool.strength = 1.0;
        tool.paint = [1.0, 0.0, 0.0];
        let bvh = o.bvh();
        let h = bvh.nearest(&o.mesh, DVec3::Z, 1.0).unwrap();
        let dab = |brush, invert| Stamp { brush, center: h.point, seed: h.tid, motion: DVec3::ZERO, view: -DVec3::Z, r: 0.4, invert, across: DVec3::X, screen: None };
        assert!(!tool.stamp(&mut o, &dab(Brush::PaintVertex, false), &mut Default::default(), None));
        let m = &o.mesh;
        let top = m.vertex_ids().max_by(|&a, &b| m.vertex(a).z.total_cmp(&m.vertex(b).z)).unwrap();
        assert_eq!(m.vertex_color(top), Some([1.0, 0.0, 0.0]));
        let far = m.vertex_ids().find(|&v| m.vertex(v).z < 0.0).unwrap();
        assert_eq!(m.vertex_color(far), Some(mm_core::mesh::DEFAULT_VERTEX_COLOR));
        assert!(o.mesh.vertex_ids().all(|v| o.mesh.vertex(v) == m0.vertex(v)), "painting leaves the shape alone");
        // Smoothing blurs the painted edge: some rim vertex gets in-between red.
        let rim_before: Vec<f32> = o.mesh.vertex_ids().map(|v| o.mesh.vertex_color(v).unwrap()[1]).collect();
        tool.stamp(&mut o, &dab(Brush::SmoothColor, false), &mut Default::default(), None);
        let rim_after: Vec<f32> = o.mesh.vertex_ids().map(|v| o.mesh.vertex_color(v).unwrap()[1]).collect();
        assert!(rim_before.iter().zip(&rim_after).any(|(a, b)| (a - b).abs() > 0.01));
        // Inverted paint restores the default colour.
        tool.stamp(&mut o, &dab(Brush::PaintVertex, true), &mut Default::default(), None);
        assert_eq!(o.mesh.vertex_color(top), Some(mm_core::mesh::DEFAULT_VERTEX_COLOR));
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
    fn density_brushes_change_triangle_size_locally() {
        let dab = |m: &mut Mesh, brush: Brush, n: usize| {
            for _ in 0..n {
                let bvh = mm_core::Bvh::build(m);
                let verts = brush_vertices(m, &bvh, DVec3::Z, 0.5, None, &Footprint::default());
                density_dab(m, &verts, brush, 0.5, 0.8);
            }
            m.check_validity().unwrap();
            assert!(m.is_closed());
        };
        let near = |m: &Mesh| {
            let tris: Vec<u32> = m.triangle_ids().filter(|&t| m.tri_centroid(t).z > 0.95).collect();
            mm_remesh::edge_length_stats_in(m, &tris).mean
        };
        let far = |m: &Mesh| m.triangle_ids().filter(|&t| m.tri_centroid(t).z < -0.5).count();
        let m0 = primitives::icosphere(1.0, 4);
        let mut fine = m0.clone();
        dab(&mut fine, Brush::Refine, 3);
        assert!(near(&fine) < near(&m0) * 0.8, "{} vs {}", near(&fine), near(&m0));
        assert_eq!(far(&fine), far(&m0));
        let mut coarse = m0.clone();
        dab(&mut coarse, Brush::Reduce, 3);
        assert!(near(&coarse) > near(&m0) * 1.3, "{} vs {}", near(&coarse), near(&m0));
        assert_eq!(far(&coarse), far(&m0));
        // Adaptive reduce thins a flat patch far more than a curved one, and keeps its shape.
        let mut flat = primitives::plane_grid(2.0, 2.0, 30, 30);
        let mut bumpy = flat.clone();
        bumpy.transform(|p| p + DVec3::Z * (0.08 * (p.x * 25.0).sin() * (p.y * 25.0).cos()));
        let (f0, b0) = (flat.triangle_count(), bumpy.triangle_count());
        for m in [&mut flat, &mut bumpy] {
            for _ in 0..3 {
                let bvh = mm_core::Bvh::build(m);
                let verts = brush_vertices(m, &bvh, DVec3::ZERO, 0.6, None, &Footprint::default());
                density_dab(m, &verts, Brush::AdaptiveReduce, 0.6, 1.0);
            }
            m.check_validity().unwrap();
        }
        assert!(f0 - flat.triangle_count() > 2 * (b0 - bumpy.triangle_count()), "{} vs {}", f0 - flat.triangle_count(), b0 - bumpy.triangle_count());
        assert!(flat.vertex_ids().all(|v| flat.vertex(v).z.abs() < 1e-9));
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

    /// A scene holding `mesh`, seen from above (looking down −z) in an 800 × 600 view.
    struct Rig {
        scene: Scene,
        history: crate::scene::History,
        camera: crate::camera::Camera,
        status: String,
        ctx: egui::Context,
    }

    impl Rig {
        fn new(mesh: Mesh) -> Self {
            let mut scene = Scene::default();
            scene.add("m", mesh);
            let mut camera = crate::camera::Camera::default();
            camera.look_along(-DVec3::Z);
            camera.focus(&scene.bounds());
            Self { scene, history: Default::default(), camera, status: String::new(), ctx: egui::Context::default() }
        }
        fn cx(&mut self) -> ToolCx<'_> {
            let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
            ToolCx { scene: &mut self.scene, history: &mut self.history, camera: &mut self.camera, status: &mut self.status, egui: &self.ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift }
        }
    }

    #[test]
    fn spacing_sets_the_dabs_along_a_stroke() {
        // A path two radii long: at 25 % of the radius 9 dabs, at 100 % 3.
        for (spacing, want) in [(25, "Draw: 9 dabs"), (100, "Draw: 3 dabs")] {
            let mut rig = Rig::new(primitives::plane_grid(4.0, 4.0, 40, 40));
            let mut cx = rig.cx();
            let mut tool = SculptTool::new(&mut cx);
            tool.brush = Brush::Draw;
            tool.set_settings(&serde_json::json!({"spacing_percent": spacing}), &mut cx).unwrap();
            let r = brush_radius(cx.scene, cx.scene.active.unwrap(), tool.size_pct);
            let path = serde_json::json!({"path": [[-1.0, 0.0, 0.0], [-1.0 + r, 0.0, 0.0], [-1.0 + 2.0 * r, 0.0, 0.0]]});
            assert_eq!(tool.act("stroke", &path, &mut cx).unwrap(), want);
        }
    }

    #[test]
    fn a_fast_drag_dabs_like_a_slow_one() {
        // Flow off: the dabs follow the path, however many pointer moves cover it (before,
        // a fast drag dabbed once per frame and left gaps).
        let run = |moves: usize| {
            let mut rig = Rig::new(primitives::plane_grid(4.0, 4.0, 80, 80));
            let mut cx = rig.cx();
            let mut tool = SculptTool::new(&mut cx);
            tool.brush = Brush::Draw;
            tool.flow = false;
            let (a, b) = (cx.project(DVec3::new(-0.6, 0.1, 0.0)).unwrap(), cx.project(DVec3::new(0.6, -0.1, 0.0)).unwrap());
            tool.pointer(&PointerInput { pos: Some(a), pressed: true, down: true, ..Default::default() }, &mut cx);
            for k in 1..=moves {
                let p = a + (b - a) * (k as f32 / moves as f32);
                tool.pointer(&PointerInput { pos: Some(p), down: true, ..Default::default() }, &mut cx);
            }
            tool.pointer(&PointerInput { pos: Some(b), released: true, ..Default::default() }, &mut cx);
            cx.scene.active().unwrap().mesh.as_ref().clone()
        };
        let (fast, slow) = (run(1), run(40));
        let lift = fast.vertex_ids().map(|v| fast.vertex(v).z.abs()).fold(0.0, f64::max);
        let diff = fast.vertex_ids().map(|v| (fast.vertex(v) - slow.vertex(v)).length()).fold(0.0, f64::max);
        assert!(lift > 1e-3, "{lift}");
        assert!(diff < 0.05 * lift, "{diff} vs {lift}");
        // The ridge is continuous: every grid vertex on the stroke's middle line was raised.
        let along = fast.vertex_ids().filter(|&v| fast.vertex(v).x.abs() < 0.55 && (fast.vertex(v).y + fast.vertex(v).x / 6.0).abs() < 0.01).collect::<Vec<_>>();
        assert!(!along.is_empty() && along.iter().all(|&v| fast.vertex(v).z.abs() > 0.3 * lift));
    }

    #[test]
    fn brush_alpha_embosses_and_turns_with_the_stroke() {
        let m0 = primitives::plane_grid(2.0, 2.0, 80, 80);
        let raised = |across: DVec3, tile: bool| {
            let mut o = crate::scene::SceneObject::new(1, "g", m0.clone());
            let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Draw), Symmetry::new(None, None));
            tool.alpha = Some(Alpha::pattern(mm_edit::Pattern::Stripes));
            tool.alpha_map.tile = tile;
            tool.falloff = Falloff::Flat;
            let bvh = o.bvh();
            let h = bvh.nearest(&o.mesh, DVec3::ZERO, 1.0).unwrap();
            let s = Stamp { brush: Brush::Draw, center: h.point, seed: h.tid, motion: DVec3::ZERO, view: -DVec3::Z, r: 0.5, invert: false, across, screen: None };
            tool.stamp(&mut o, &s, &mut Default::default(), None);
            let m = o.mesh.as_ref().clone();
            move |p: DVec3| {
                let v = m.vertex_ids().min_by(|&a, &b| m.vertex(a).distance(p).total_cmp(&m.vertex(b).distance(p))).unwrap();
                m.vertex(v).z.abs()
            }
        };
        // The stripes run across the image's x: ridges 0.2 r either side of the middle, a
        // groove along it.
        let along_x = raised(DVec3::X, false);
        assert!(along_x(DVec3::new(0.1, 0.0, 0.0)) > 0.01 && along_x(DVec3::new(-0.1, 0.0, 0.0)) > 0.01);
        assert!(along_x(DVec3::ZERO) < 1e-4 && along_x(DVec3::new(0.0, 0.1, 0.0)) < 1e-4, "the groove runs along y");
        // AlignToStroke: the stroke now runs along y, and the stripes turn with it.
        let along_y = raised(DVec3::Y, false);
        assert!(along_y(DVec3::new(0.0, 0.1, 0.0)) > 0.01 && along_y(DVec3::new(0.1, 0.0, 0.0)) < 1e-4);
        // Not tiled the image ends at the brush's square; the secondary brush is never shaped.
        assert!(along_x(DVec3::new(0.45, 0.45, 0.0)) < 1e-12);
        let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Draw), Symmetry::new(None, None));
        assert!(!tool.shapes(Brush::Draw), "no alpha, no shaping");
        tool.alpha = Some(Alpha::pattern(mm_edit::Pattern::Noise));
        assert!(tool.shapes(Brush::Draw) && !tool.shapes(tool.secondary));
    }

    #[test]
    fn stencil_masks_the_brush_where_it_shows_on_screen() {
        let m0 = primitives::plane_grid(2.0, 2.0, 80, 80);
        let mut camera = crate::camera::Camera::default();
        camera.look_along(-DVec3::Z);
        camera.focus(&m0.bounds());
        let size = DVec2::new(800.0, 600.0);
        let screen = ScreenMap { view_proj: camera.view_proj(size.x / size.y), size };
        let mut o = crate::scene::SceneObject::new(1, "g", m0.clone());
        let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Draw), Symmetry::new(None, None));
        tool.falloff = Falloff::Flat;
        tool.stencil = Some(Alpha::pattern(mm_edit::Pattern::Disc));
        // A small disc right of the view's middle, inside the brush.
        tool.screen_stencil = ScreenStencil { center: DVec2::new(0.6, 0.45), map: AlphaMap { scale: 0.15, ..Default::default() } };
        let bvh = o.bvh();
        let h = bvh.nearest(&o.mesh, DVec3::ZERO, 1.0).unwrap();
        let s = Stamp { brush: Brush::Draw, center: h.point, seed: h.tid, motion: DVec3::ZERO, view: -DVec3::Z, r: 0.95, invert: false, across: DVec3::X, screen: Some(screen) };
        tool.stamp(&mut o, &s, &mut Default::default(), None);
        let (mut inside, mut moved) = (0, 0);
        for v in o.mesh.vertex_ids() {
            let p0 = m0.vertex(v);
            let uv = tool.screen_stencil.uv(screen.to_screen(p0).unwrap(), size);
            let d = (uv - DVec2::splat(0.5)).length();
            let lifted = (o.mesh.vertex(v) - p0).length() > 1e-9;
            moved += lifted as usize;
            if d < 0.35 && p0.length() < 0.9 {
                inside += 1;
                assert!(lifted, "vertex {v} under the stencil's disc");
            }
            if d > 0.55 {
                assert!(!lifted, "vertex {v} outside the stencil's disc");
            }
        }
        assert!(inside > 10 && moved >= inside);
    }

    #[test]
    fn a_tilted_symmetry_plane_mirrors_dabs_and_snaps_the_seam() {
        // The icosphere's x = 0 mirror plane, tilted by turning the sphere.
        let rot = glam::DQuat::from_euler(glam::EulerRot::XYZ, 0.3, 0.5, 0.2);
        let plane = SymPlane::new(DVec3::ZERO, rot * DVec3::X);
        assert_eq!(plane.axis_index(), None);
        let mut m0 = primitives::icosphere(1.0, 4);
        m0.transform(|p| rot * p);
        let mut o = crate::scene::SceneObject::new(1, "s", m0.clone());
        let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Draw), Symmetry::new(Some(plane), None));
        tool.symmetry.enabled = true;
        tool.strength = 0.8;
        let mut start = Default::default();
        let mut c = rot * DVec3::new(0.6, 0.0, 0.8).normalize();
        for _ in 0..6 {
            let bvh = o.bvh();
            let h = bvh.nearest(&o.mesh, c, 1.0).unwrap();
            let s = Stamp { brush: Brush::Draw, center: h.point, seed: h.tid, motion: rot * DVec3::Y * 0.05, view: -c, r: 0.35, invert: false, across: rot * DVec3::Y, screen: None };
            tool.stamp(&mut o, &s, &mut start, None);
            c = (c + rot * DVec3::Y * 0.05).normalize();
        }
        // Every vertex's mirror twin moved to the mirror image of where it went.
        let (m, bvh0) = (&o.mesh, mm_core::Bvh::build(&m0));
        let twin = |v: u32| {
            let h = bvh0.nearest(&m0, plane.reflect(m0.vertex(v)), 1e-6).unwrap();
            m0.triangle(h.tid).into_iter().min_by(|&a, &b| m0.vertex(a).distance(h.point).total_cmp(&m0.vertex(b).distance(h.point))).unwrap()
        };
        let mut moved = 0;
        for v in m.vertex_ids() {
            moved += (m.vertex(v).distance(m0.vertex(v)) > 1e-6) as usize;
            assert!((plane.reflect(m.vertex(twin(v))) - m.vertex(v)).length() < 1e-9, "vertex {v}");
        }
        assert!(moved > 20);
        // SymSnap on a slightly lopsided sphere: inflating across the seam lets seam vertices
        // drift off the tilted plane, unless snapped back onto it.
        let mut m1 = primitives::icosphere(1.0, 4);
        m1.transform(|p| p + DVec3::X * 0.02 * p.y * p.z);
        m1.transform(|p| rot * p);
        let seam: Vec<u32> = m1.vertex_ids().filter(|&v| plane.distance(m1.vertex(v)) < 1e-9).collect();
        assert!(!seam.is_empty());
        let run = |snap: bool| {
            let mut o = crate::scene::SceneObject::new(1, "s", m1.clone());
            let mut tool = SculptTool::build(SculptMode::Surface, Some(Brush::Inflate), Symmetry::new(Some(plane), None));
            tool.symmetry.enabled = true;
            tool.sym_snap = snap;
            let mut start = Default::default();
            for _ in 0..5 {
                let bvh = o.bvh();
                let h = bvh.nearest(&o.mesh, rot * DVec3::new(0.05, 0.3, 0.95), 1.0).unwrap();
                tool.stamp(&mut o, &Stamp { brush: Brush::Inflate, center: h.point, seed: h.tid, motion: DVec3::ZERO, view: -DVec3::Z, r: 0.4, invert: false, across: DVec3::X, screen: None }, &mut start, None);
            }
            seam.iter().map(|&v| plane.distance(o.mesh.vertex(v))).fold(0.0, f64::max)
        };
        assert!(run(false) > 1e-6);
        assert!(run(true) < 1e-12);
    }

    #[test]
    fn agents_set_spacing_alpha_stencil_and_a_free_plane() {
        use serde_json::json;
        let mut rig = Rig::new(primitives::icosphere(1.0, 3));
        let mut cx = rig.cx();
        let mut tool = SculptTool::new(&mut cx);
        let change = json!({
            "spacing_percent": 60, "align_to_stroke": true,
            "alpha": {"image": "dots", "scale": 0.5, "tile": true},
            "stencil": {"image": "rings", "center": [0.3, 0.4], "rotation_deg": 30},
            "symmetry": true, "sym_snap": true, "symmetry_plane": {"point_mm": [0.1, 0.0, 0.0], "normal": [1.0, 1.0, 0.0]},
        });
        tool.set_settings(&change, &mut cx).unwrap();
        let s = tool.settings().unwrap();
        assert_eq!(s["spacing_percent"], 60.0);
        assert_eq!((s["align_to_stroke"].as_bool(), s["sym_snap"].as_bool()), (Some(true), Some(true)));
        assert_eq!((s["alpha"]["image"].as_str(), s["alpha"]["scale"].as_f64(), s["alpha"]["tile"].as_bool()), (Some("dots"), Some(0.5), Some(true)));
        assert_eq!((s["stencil"]["image"].as_str(), s["stencil"]["center"][0].as_f64(), s["stencil"]["rotation_deg"].as_f64()), (Some("rings"), Some(0.3), Some(30.0)));
        assert_eq!(s["symmetry_axis"], "free");
        let p = tool.symmetry.active().unwrap();
        assert!((p.normal - DVec3::new(1.0, 1.0, 0.0).normalize()).length() < 1e-12 && p.distance(DVec3::new(0.1, 0.0, 0.0)) < 1e-12);
        // A name that matches nothing is an error and changes nothing.
        assert!(tool.set_settings(&json!({"alpha": {"image": "no such alpha"}, "spacing_percent": 10}), &mut cx).is_err());
        assert_eq!(tool.settings().unwrap()["spacing_percent"], 60.0);
        // Back to an axis, and the plane through three points.
        tool.set_settings(&json!({"symmetry_axis": "y"}), &mut cx).unwrap();
        assert_eq!(tool.symmetry.active().unwrap().axis_index(), Some(1));
        tool.act("symmetry_through_points", &json!({"points": [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]}), &mut cx).unwrap();
        assert_eq!(tool.symmetry.active().unwrap().axis_index(), Some(0));
        assert!(tool.act("symmetry_through_points", &json!({"points": [[0.0, 0.0, 1.0], [0.0, 0.0, 2.0], [0.0, 0.0, 3.0]]}), &mut cx).is_err());
        // Every setting is described.
        let schema = tool.settings_schema().unwrap();
        for k in tool.settings().unwrap().as_object().unwrap().keys() {
            assert!(schema["properties"].get(k).is_some(), "{k}");
        }
        assert!(tool.actions().iter().any(|a| a.name == "symmetry_from_selection"));
    }

    #[test]
    fn panel_and_overlays_draw_with_alpha_stencil_and_set_mode() {
        let mut rig = Rig::new(primitives::icosphere(1.0, 3));
        let mut tool = SculptTool::new(&mut rig.cx());
        tool.alpha = Some(Alpha::pattern(mm_edit::Pattern::Rings));
        tool.stencil = Some(Alpha::pattern(mm_edit::Pattern::Dots));
        tool.screen_stencil.map.tile = true;
        tool.symmetry.begin_edit(&rig.cx());
        assert!(tool.symmetry.editing() && tool.symmetry.enabled);
        let ctx = rig.ctx.clone();
        let mut lines = 0;
        let _ = ctx.run_ui(egui::RawInput::default(), |ui| {
            let ctx = ui.ctx().clone();
            let vp = Viewport { rect: egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(800.0, 600.0)) };
            let mut cx = ToolCx { scene: &mut rig.scene, history: &mut rig.history, camera: &mut rig.camera, status: &mut rig.status, egui: &ctx, vp, brush_scroll: crate::navigation::BrushScroll::Shift };
            tool.panel(ui, &mut cx);
            tool.hover = Some((DVec3::Z, 0.2));
            let mut ov = Overlay::default();
            tool.overlay(&mut ov, &mut cx);
            lines = ov.lines.len();
            tool.paint(ui.painter(), &mut cx);
        });
        assert!(lines >= 2 * (4 + 1 + 4), "alpha square and tick, symmetry plane: {lines}");
        // Esc leaves Set (keeping symmetry on) and stencil placing before it closes the tool.
        let mut cx = rig.cx();
        assert!(!tool.cancel(&mut cx) && !tool.symmetry.editing() && tool.symmetry.enabled);
        tool.placing_stencil = true;
        assert!(!tool.cancel(&mut cx));
        assert!(tool.cancel(&mut cx));
    }

    #[test]
    fn set_mode_gizmo_slides_and_tilts_the_plane() {
        let mut rig = Rig::new(primitives::icosphere(1.0, 3));
        let mut cx = rig.cx();
        let mut tool = SculptTool::new(&mut cx);
        tool.symmetry.begin_edit(&cx);
        let p0 = tool.symmetry.active().unwrap();
        assert_eq!(p0.axis_index(), Some(0));
        // The arrow along the normal (+x, to the right on screen): drag it right.
        let c = cx.project(p0.point).unwrap();
        let tip = cx.project(p0.point + p0.normal * cx.world_per_point(p0.point) * 80.0).unwrap();
        let grab = c + (tip - c) * 0.9;
        tool.pointer(&PointerInput { pos: Some(grab), pressed: true, down: true, ..Default::default() }, &mut cx);
        tool.pointer(&PointerInput { pos: Some(grab + egui::vec2(40.0, 0.0)), down: true, ..Default::default() }, &mut cx);
        tool.pointer(&PointerInput { pos: Some(grab + egui::vec2(40.0, 0.0)), released: true, ..Default::default() }, &mut cx);
        let p1 = tool.symmetry.active().unwrap();
        assert!(p1.offset() > p0.offset() + 0.05 && p1.normal == p0.normal, "slid along +x: {p1:?}");
        assert!(tool.hover.is_none(), "no brush while setting");
        // Picking three points sets the plane through them.
        tool.symmetry.cancel_edit();
        assert_eq!(tool.symmetry.active(), Some(p0), "Cancel restores the plane");
        tool.symmetry.begin_edit(&cx);
        let pts = [DVec3::new(0.3, 0.3, 0.9), DVec3::new(-0.3, 0.3, 0.9), DVec3::new(0.0, -0.4, 0.9)].map(|p| p.normalize());
        tool.symmetry.through_points(&pts).unwrap();
        assert!(tool.symmetry.accept_edit());
        let p2 = tool.symmetry.active().unwrap();
        assert!(pts.iter().all(|&p| p2.distance(p) < 1e-12));
    }
}
