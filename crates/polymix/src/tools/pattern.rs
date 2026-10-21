//! Make Pattern (Edit > Make Pattern): turn the object into a pattern of tubes and balls —
//! a frame along its edges, or perforations / lattices filling its volume — see
//! [`mm_solid::pattern`]. Click Update to preview, Accept to apply.

use super::ops::{OpInfo, OpOutput, OpSpec, OpTool};
use super::*;
use crate::jobs::ProgressFn;
use mm_solid::{Composition, Gradient, PatternParams, PatternType, Tiling};

const TYPES: [(PatternType, &str); 10] = [
    (PatternType::Edges, "Edges"),
    (PatternType::DualEdges, "Dual Edges"),
    (PatternType::DelaunayEdges, "Mesh + Delaunay Edges"),
    (PatternType::DelaunayDualEdges, "Mesh + Delaunay Dual Edges"),
    (PatternType::TiledTubes, "Tiled Tubes"),
    (PatternType::TiledSpheres, "Tiled Spheres"),
    (PatternType::Lattice, "Lattice"),
    (PatternType::RandomVolume, "Random Primitives (Volume)"),
    (PatternType::RandomSurface, "Random Primitives (Surface)"),
    (PatternType::GroupBorders, "FaceGroup Borders"),
];

#[derive(Clone, PartialEq, Debug, serde::Serialize, serde::Deserialize, schemars::JsonSchema)]
pub struct PatternUi {
    pub p: PatternParams,
    /// Grid Smoothing, 0..10.
    pub smoothing: f64,
    /// Grid Resolution multiplier.
    pub grid_resolution: f64,
    /// Reduce, 0..100: percentage of the result's triangles removed afterwards.
    pub reduce: f64,
}

/// A round-ish number near `x` (1, 2, 2.5, 5 × 10ⁿ).
fn nice(x: f64) -> f64 {
    let e = 10f64.powf(x.abs().max(1e-12).log10().floor());
    let m = x / e;
    let k = if m < 1.5 { 1.0 } else if m < 2.25 { 2.0 } else if m < 3.5 { 2.5 } else if m < 7.5 { 5.0 } else { 10.0 };
    k * e
}

pub struct MakePatternSpec;
impl OpSpec for MakePatternSpec {
    type Params = PatternUi;
    const NAME: &'static str = "Make Pattern";
    const USES_SELECTION: bool = false;
    const LIVE: bool = false;
    fn defaults(info: &OpInfo) -> Self::Params {
        let dimension = nice(info.bbox_diag * 0.03);
        PatternUi {
            p: PatternParams { dimension, spacing: dimension * 1.5, ..Default::default() },
            smoothing: 5.0,
            grid_resolution: 1.0,
            reduce: 50.0,
        }
    }
    fn ui(ui: &mut egui::Ui, u: &mut Self::Params, info: &OpInfo) {
        let p = &mut u.p;
        egui::ComboBox::from_label("Pattern Type").selected_text(TYPES.iter().find(|t| t.0 == p.kind).map_or("", |t| t.1)).show_ui(ui, |ui| {
            for (k, name) in TYPES {
                ui.selectable_value(&mut p.kind, k, name);
            }
        });
        if matches!(p.kind, PatternType::TiledTubes | PatternType::TiledSpheres) {
            ui.horizontal(|ui| {
                ui.label("Tiling Mode");
                ui.selectable_value(&mut p.tiling, Tiling::Regular, "Regular Grid");
                ui.selectable_value(&mut p.tiling, Tiling::Hex, "Hex Grid");
            });
        }
        let span = info.bbox_diag;
        ui.add(egui::Slider::new(&mut p.dimension, span * 0.002..=span * 0.2).logarithmic(true).text("Element Dimension").max_decimals(3));
        if p.kind.from_edges() {
            ui.checkbox(&mut p.clip_to_surface, "Clip To Surface").on_hover_text("Keep only the part of the pattern inside the object (closed objects)");
        } else {
            ui.add(egui::Slider::new(&mut p.spacing, span * 0.004..=span * 0.4).logarithmic(true).text("Element Spacing").max_decimals(3));
            ui.horizontal(|ui| {
                ui.label("Composition");
                ui.selectable_value(&mut p.composition, Composition::Subtract, "Subtract").on_hover_text("Cut the elements out of the object");
                ui.selectable_value(&mut p.composition, Composition::Intersect, "Intersect").on_hover_text("Keep the elements, inside the object");
            });
        }
        ui.add(egui::Slider::new(&mut u.smoothing, 0.0..=10.0).text("Grid Smoothing").integer()).on_hover_text("Fillets where elements meet");
        ui.add(egui::Slider::new(&mut u.grid_resolution, 0.5..=4.0).text("Grid Resolution").suffix("×"));
        ui.add(egui::Slider::new(&mut u.reduce, 0.0..=95.0).text("Reduce").integer()).on_hover_text("Percentage of the result's triangles removed afterwards");
        egui::ComboBox::from_label("Gradient")
            .selected_text(match p.gradient {
                Gradient::None => "No Gradient",
                Gradient::Line => "Line Gradient",
                Gradient::Sphere => "Sphere Gradient",
            })
            .show_ui(ui, |ui| {
                ui.selectable_value(&mut p.gradient, Gradient::None, "No Gradient");
                ui.selectable_value(&mut p.gradient, Gradient::Line, "Line Gradient").on_hover_text("Element size changes along the object's longest axis");
                ui.selectable_value(&mut p.gradient, Gradient::Sphere, "Sphere Gradient").on_hover_text("Element size changes from the centre outward");
            });
        if p.gradient != Gradient::None {
            ui.add(egui::Slider::new(&mut p.start_scale, 0.1..=4.0).text("Start Scale"));
            ui.add(egui::Slider::new(&mut p.end_scale, 0.1..=4.0).text("End Scale"));
        }
        let cells = info.mesh.bounds().extents().max_element() / (p.dimension / (4.0 * u.grid_resolution));
        if cells > 600.0 {
            ui.colored_label(egui::Color32::from_rgb(240, 170, 90), "Small elements on a large object: this can take a while and use a lot of memory.");
        }
    }
    fn run(mesh: &Mesh, _: Option<&[u32]>, u: &Self::Params, progress: ProgressFn) -> anyhow::Result<OpOutput> {
        let t0 = std::time::Instant::now();
        let p = PatternParams { smoothing: u.smoothing / 20.0, resolution: 4.0 * u.grid_resolution, ..u.p.clone() };
        let pattern_progress = |f: f32| progress(f * 0.85);
        let mut out = mm_solid::make_pattern(mesh, &p, Some(&pattern_progress)).map_err(|e| anyhow::anyhow!("{e}"))?;
        if u.reduce > 0.0 && out.triangle_count() > 0 {
            let rp = mm_remesh::ReduceParams { target: mm_remesh::ReduceTarget::Percentage(100.0 - u.reduce), ..Default::default() };
            let reduce_progress = |f: f32| progress(0.85 + 0.15 * f);
            mm_remesh::reduce(&mut out, &rp, Some(&reduce_progress))?;
        }
        let shells = mm_core::topology::connected_components(&out).len();
        Ok(OpOutput {
            message: format!("{} triangles, {shells} piece(s) ({:.1}s)", out.triangle_count(), t0.elapsed().as_secs_f32()),
            mesh: out,
            ..Default::default()
        })
    }
}
pub type MakePatternTool = OpTool<MakePatternSpec>;

/// Convex Decomposition (Edit menu): the object is kept; its convex pieces are added as a new
/// object with one face group per piece.
pub struct ConvexSpec;
impl OpSpec for ConvexSpec {
    type Params = mm_edit::ConvexParams;
    const NAME: &'static str = "Convex Decomposition";
    const USES_SELECTION: bool = false;
    const LIVE: bool = false;
    const SHOWS_GROUPS: bool = true;
    fn defaults(_: &OpInfo) -> Self::Params {
        mm_edit::ConvexParams::default()
    }
    fn ui(ui: &mut egui::Ui, p: &mut Self::Params, _: &OpInfo) {
        ui.add(egui::Slider::new(&mut p.max_pieces, 1..=64).text("max pieces"));
        let mut pct = p.concavity * 100.0;
        if ui.add(egui::Slider::new(&mut pct, 0.1..=20.0).logarithmic(true).text("concavity %")).on_hover_text("Stop splitting once no piece's hull adds more than this share of the total hull volume").changed() {
            p.concavity = pct / 100.0;
        }
    }
    fn run(mesh: &Mesh, _: Option<&[u32]>, p: &Self::Params, _: ProgressFn) -> anyhow::Result<OpOutput> {
        let hulls = mm_edit::convex_decomposition(mesh, p)?;
        let mut pieces = Mesh::new();
        for (i, h) in hulls.iter().enumerate() {
            pieces.append_mesh(h, i as i32);
        }
        let extra = hulls.iter().map(mm_core::measure::volume).sum::<f64>() / mm_core::measure::volume(mesh).max(1e-300) - 1.0;
        Ok(OpOutput {
            message: format!("{} convex pieces, {:.1}% more volume than the object", hulls.len(), extra * 100.0),
            keeps_source: true,
            preview: Some(pieces.clone()),
            new_objects: vec![("convex pieces".into(), pieces)],
            ..Default::default()
        })
    }
}
pub type ConvexTool = OpTool<ConvexSpec>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nice_numbers() {
        assert_eq!(nice(0.9), 1.0);
        assert_eq!(nice(3.1), 2.5);
        assert_eq!(nice(42.0), 50.0);
    }

    #[test]
    fn pattern_op_runs_with_reduction() {
        let cube = mm_core::primitives::cube(10.0, 2);
        let info = OpInfo::new(&cube, None);
        let mut u = MakePatternSpec::defaults(&info);
        u.p.kind = PatternType::Edges;
        u.p.clip_to_surface = false;
        u.p.dimension = 1.0;
        let full = MakePatternSpec::run(&cube, None, &PatternUi { reduce: 0.0, ..u.clone() }, &|_| true).unwrap().mesh;
        let half = MakePatternSpec::run(&cube, None, &u, &|_| true).unwrap().mesh;
        assert!(full.is_closed() && half.is_closed());
        assert!(half.triangle_count() < full.triangle_count() * 6 / 10);
    }

    #[test]
    fn convex_pieces_leave_the_object_as_it_is() {
        // An L of two boxes: the pieces are added, the object itself is kept (not copied).
        let mut l = mm_core::primitives::cube(2.0, 1);
        let mut arm = mm_core::primitives::cube(2.0, 1);
        arm.transform(|p| p + glam::DVec3::new(1.9, 0.0, 0.0));
        l.append_mesh(&arm, 0);
        let out = ConvexSpec::run(&l, None, &ConvexSpec::defaults(&OpInfo::new(&l, None)), &|_| true).unwrap();
        assert!(out.keeps_source && out.mesh.triangle_count() == 0);
        assert_eq!(out.new_objects.len(), 1);
        assert_eq!(out.preview.as_ref().map(|m| m.triangle_count()), Some(out.new_objects[0].1.triangle_count()));
    }
}
