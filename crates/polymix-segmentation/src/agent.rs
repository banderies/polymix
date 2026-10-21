//! The Segmentation workspace for agents: `segmentation_state` (what it shows, the segments
//! in it, the effects' settings) and the actions the shell offers as `segmentation_do` (what
//! the editor panel and the views do: segments, effects, masking, undo, meshes, saving).
//!
//! The volume is not named here (its name can carry a patient's): `volume_catalog` says which
//! volume is selected in the Volume workspace, which is the one shown here.

use crate::effects::{Effect, IslandOp, LocalPick, LocalShape, LogicalOp, MaskFill, ScissorsOp, ScissorsRange, ScissorsShape, SmoothMethod};
use crate::segmentation::{EditableArea, Overwrite, Segmentation};
use crate::{SegmentationWorkspace, PLANES};
use glam::DVec3;
use mm_volume::load::LoadedVolume;
use mm_volume::slice::Plane;
use polymix_workspace::agent::{arg, round, vec3, Action, AgentReply, AgentTool, Args};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

pub fn tools() -> Vec<AgentTool> {
    vec![
        AgentTool::new(
            "segmentation_state",
            "The Segmentation workspace: the volume shown (grid, spacing, orientation, value range, window), the crosshair (voxel, RAS mm, value), the segments (label, name, colour, visibility, voxel count and volume in mL), the selected segment and effect, masking, and any loading, running operation or message. The volume shown is the one selected in the Volume workspace (see volume_catalog).",
            json!({}),
        ),
        AgentTool::new(
            "segmentation_centerline",
            "The last centerline extracted (segmentation_do centerline): per branch its length, parent branch, narrowest section (narrowing % against the branch's median diameter outside it, where), and with `sections` every cross-section along it (distance, RAS mm, area, equivalent / min / max diameter, inscribed radius)",
            json!({"branch": {"type": "integer", "description": "Only this branch (1 = the first)"}, "sections": {"type": "boolean", "description": "Include every cross-section"}, "points": {"type": "boolean", "description": "Include the centerline's points (RAS mm, inscribed radius)"}}),
        ),
        AgentTool::new(
            "segmentation_statistics",
            "Measures of segments, as 3D Slicer's Segment Statistics: voxels, volume (mL), surface area (cm²), the volume's values under the segment (mean, SD, median, min, max), the largest diameter (Feret, mm), roundness, elongation, principal axis lengths (mm) and the centre (RAS mm)",
            json!({"segments": {"type": "array", "items": {}, "description": "Labels or names (default: every segment)"}}),
        ),
    ]
}

pub fn call(ws: &mut SegmentationWorkspace, tool: &str, args: &Value, _ctx: &egui::Context) -> AgentReply {
    match tool {
        "segmentation_state" => AgentReply::Now(Ok(state(ws))),
        "segmentation_statistics" => statistics(ws, args),
        "segmentation_centerline" => AgentReply::Now(centerline(ws, args)),
        _ => AgentReply::error(format!("the Segmentation workspace has no tool {tool}")),
    }
}

fn state(ws: &mut SegmentationWorkspace) -> Value {
    let volume = ws.volume.as_ref().map(|v| {
        let g = &v.geometry;
        json!({
            "dims": v.dims,
            "spacing_mm": g.spacing.map(|x| round(x, 4)),
            "orientation": g.orientation(),
            "value_range": [v.range.0, v.range.1],
            "auto_window": {"centre": round(v.auto_window.0, 3), "width": round(v.auto_window.1, 3)},
            "volumes_in_set": ws.source.as_ref().map_or(1, |s| s.volumes),
        })
    });
    let cursor = ws.volume.as_ref().zip(ws.state.as_ref()).map(|(v, st)| {
        let c = st.cursor;
        let (i, j, k) = (c.x.round().max(0.0) as usize, c.y.round().max(0.0) as usize, c.z.round().max(0.0) as usize);
        let inside = i < v.dims[0] && j < v.dims[1] && k < v.dims[2];
        let ras = v.geometry.world(c);
        json!({
            "voxel": [i, j, k],
            "ras_mm": vec3(ras.to_array(), 3),
            "value": inside.then(|| v.get(i, j, k)),
            "label": ws.seg.as_ref().filter(|_| inside).map(|s| s.labels[v.index(i, j, k)]),
        })
    });
    let voxel_ml = ws.volume.as_ref().map_or(0.0, |v| v.geometry.spacing.iter().product::<f64>() / 1000.0);
    let segments = ws.seg.as_mut().map(|s| {
        let counts = s.counts().to_vec();
        json!({
            "segments": s.segments.iter().map(|g| json!({
                "label": g.label,
                "name": g.name,
                "colour": format!("#{:02x}{:02x}{:02x}", g.colour.r(), g.colour.g(), g.colour.b()),
                "visible": g.visible,
                "status": g.status.key(),
                "voxels": counts[g.label as usize],
                "volume_ml": round(counts[g.label as usize] as f64 * voxel_ml, 3),
            })).collect::<Vec<_>>(),
            "selected_segment": s.selected_segment().map(|g| g.label),
            "modified": s.modified,
        })
    });
    let m = &ws.masking;
    json!({
        "volume": volume,
        "loading": ws.loading.as_ref().map(|l| round(l.progress() as f64, 2)),
        "error": ws.error.as_ref().map(|e| polymix_workspace::guard::scrub_paths(e)),
        "window": ws.state.as_ref().map(|st| json!({"centre": round(st.window.0, 3), "width": round(st.window.1, 3)})),
        "cursor": cursor,
        "segmentation": segments,
        "effect": ws.effect.map(|e| e.label()),
        "masking": {"editable_area": format!("{:?}", m.area), "intensity_range": m.intensity.map(|(a, b)| [a, b]), "overwrite": format!("{:?}", m.overwrite)},
        "effect_settings": ws.settings.as_ref().map(|st| json!({
            "threshold": [st.threshold.0, st.threshold.1],
            "threshold_suggested_otsu": ws.hist.as_ref().and_then(|h| h.otsu()),
            "brush_radius_mm": round(st.brush_mm as f64, 3),
            "brush_sphere": st.sphere,
            "islands": st.islands.key(),
            "min_island_voxels": st.min_island,
            "flood_tolerance": st.flood_tolerance,
            "flood_use_threshold": st.flood_use_threshold,
            "flood_min_diameter_mm": st.flood_min_diameter_mm,
            "margin_mm": st.margin_mm,
            "smoothing": st.smoothing.key(),
            "smoothing_mm": st.smoothing_mm,
            "hollow_mm": st.hollow_mm,
            "hollow_side": crate::effects::hollow_side_key(st.hollow_side),
            "logical": st.logical.key(),
            "logical_other": st.logical_other,
            "mesh_smoothing_mm": st.mesh_smoothing_mm,
            "mesh_all_visible": st.mesh_all_visible,
        })),
        "undo": ws.seg.as_ref().map(|s| json!({"can_undo": s.can_undo(), "can_redo": s.can_redo()})),
        "display": {
            "opacity": round(ws.opacity as f64, 2),
            "crosshair": ws.crosshair,
            "smooth_slices": ws.smooth,
            "colour_map": ws.state.as_ref().map(|s| s.colour_map.key()),
            "layout": ws.layout.key(),
            "oblique_planes": ws.state.as_ref().and_then(|s| s.frames).map(|f| f.iter().map(|fr| json!({"right": vec3(fr[0].to_array(), 3), "down": vec3(fr[1].to_array(), 3), "normal": vec3(fr[2].to_array(), 3)})).collect::<Vec<_>>()),
            "slab": ws.state.as_ref().map(|s| json!({"mode": s.slab.mode.key(), "mm": s.slab.mm})),
        },
        "overlay": ws.state.as_ref().and_then(|s| s.fusion.as_ref()).map(|f| json!({
            "name": polymix_workspace::guard::scrub_paths(&f.name),
            "compare": f.compare.key(),
            "opacity": round(f.opacity as f64, 2),
            "wipe": round(f.wipe as f64, 2),
            "showing_second": f.showing,
            "colour_map": f.colour_map.key(),
            "window": {"centre": round(f.window.0, 3), "width": round(f.window.1, 3)},
        })),
        "overlay_loading": ws.fusion_job.is_some(),
        "centerline": ws.vessels.result.as_ref().map(|r| json!({
            "branches": r.tree.branches.len(),
            "narrowest_percent": r.narrowest.iter().flatten().map(|n| round(n.1, 1)).reduce(f64::max),
            "segments_changed_since": ws.seg.as_ref().is_some_and(|s| s.version != r.version),
        })),
        "measurements": ws.measure.items.iter().map(|m| json!({"tool": m.tool.key(), "plane": m.plane.label(), "slice": m.slice, "result": m.text()})).collect::<Vec<_>>(),
        "overlay_choices": ws.overlay_choices.iter().map(|(_, l)| polymix_workspace::guard::scrub_paths(l)).collect::<Vec<_>>(),
        "volume_switch_prompt": ws.pending.as_ref().map(|_| "Another volume, on a different voxel grid, is selected in the Volume workspace. Opening it discards the segmentation: answer with segmentation_do answer_volume_switch."),
        "running": ws.running.as_ref().map(|r| r.what.clone()),
        "message": ws.message.as_ref().map(|(text, error)| json!({"text": polymix_workspace::guard::scrub_paths(text), "error": error})),
        "maximized_view": ws.maximized.map(|k| match k {
            1 => "3d".to_string(),
            k => format!("{:?}", PLANES[[0, 0, 1, 2][k.min(3)]]).to_lowercase(),
        }),
        "meshes_waiting_for_mesh_tab": ws.outbox.len(),
        "note": ws.volume.is_none().then_some("No volume loaded: the Segmentation tab loads the volume selected in the Volume workspace when the tab is shown."),
    })
}

fn centerline(ws: &mut SegmentationWorkspace, args: &Value) -> Result<Value, String> {
    let r = ws.vessels.result.as_ref().ok_or("No centerline yet: segmentation_do centerline")?;
    let name = ws.seg.as_ref().and_then(|s| s.index_of(r.label).map(|k| s.segments[k].name.clone()));
    let only = args.get("branch").and_then(Value::as_u64).map(|b| b as usize);
    let (with_sections, with_points) = (args.get("sections").and_then(Value::as_bool) == Some(true), args.get("points").and_then(Value::as_bool) == Some(true));
    let branches: Vec<Value> = r
        .tree
        .branches
        .iter()
        .enumerate()
        .filter(|(k, _)| only.is_none_or(|b| b == k + 1))
        .map(|(k, b)| {
            let sections = &r.sections[k];
            let d: Vec<f64> = sections.iter().map(|s| s.equivalent_diameter_mm).filter(|d| *d > 0.0).collect();
            let mut v = json!({
                "branch": k + 1,
                "parent": b.parent.map(|p| p + 1),
                "length_mm": round(b.length_mm, 2),
                "diameter_mm": {"min": d.iter().copied().reduce(f64::min).map(|x| round(x, 3)), "max": d.iter().copied().reduce(f64::max).map(|x| round(x, 3))},
                "narrowest": r.narrowest[k].map(|(i, pct)| json!({"narrowing_percent": round(pct, 1), "distance_mm": round(sections[i].distance_mm, 2), "ras_mm": vec3(sections[i].ras, 2), "equivalent_diameter_mm": round(sections[i].equivalent_diameter_mm, 3), "area_mm2": round(sections[i].area_mm2, 3)})),
            });
            if with_sections {
                v["sections"] = sections.iter().map(|s| json!({"distance_mm": round(s.distance_mm, 2), "ras_mm": vec3(s.ras, 2), "area_mm2": round(s.area_mm2, 3), "equivalent_diameter_mm": round(s.equivalent_diameter_mm, 3), "min_diameter_mm": round(s.min_diameter_mm, 3), "max_diameter_mm": round(s.max_diameter_mm, 3), "inscribed_radius_mm": round(s.radius_mm, 3)})).collect();
            }
            if with_points {
                v["points"] = b.points.iter().map(|p| json!({"ras_mm": vec3(p.ras, 2), "radius_mm": round(p.radius_mm, 3)})).collect();
            }
            v
        })
        .collect();
    let stale = ws.seg.as_ref().is_some_and(|s| s.version != r.version);
    Ok(json!({"segment": name, "segments_changed_since": stale, "branches": branches}))
}

/// Measured on a worker thread, from a copy of the label map (each segment's mask made in
/// its turn): the UI does not wait for it.
fn statistics(ws: &mut SegmentationWorkspace, args: &Value) -> AgentReply {
    let vol = match loaded(ws) {
        Ok(v) => v,
        Err(e) => return AgentReply::error(e),
    };
    let Some(seg) = ws.seg.as_ref() else { return AgentReply::error("No segmentation") };
    let wanted: Vec<usize> = match args.get("segments").and_then(Value::as_array) {
        Some(list) => match list.iter().map(|v| find(seg, v)).collect::<Result<_, _>>() {
            Ok(w) => w,
            Err(e) => return AgentReply::error(e),
        },
        None => (0..seg.segments.len()).collect(),
    };
    let segments: Vec<(u8, String)> = wanted.into_iter().map(|k| (seg.segments[k].label, seg.segments[k].name.clone())).collect();
    let (labels, dims) = (seg.labels.clone(), seg.dims);
    AgentReply::later(move || polymix_workspace::in_job_pool(|| Ok(statistics_of(&vol, &labels, dims, &segments))))
}

fn statistics_of(vol: &LoadedVolume, labels: &[u8], dims: [usize; 3], segments: &[(u8, String)]) -> Value {
    let rows = segments
        .iter()
        .map(|(label, name)| {
            let s = mm_segment::segment_stats(&mm_segment::Mask::from_labels(dims, labels, *label), vol);
            json!({
                "label": label,
                "name": name,
                "empty": s.is_none(),
                "stats": s.map(|s| json!({
                    "voxels": s.voxels,
                    "volume_ml": round(s.volume_mm3 / 1000.0, 4),
                    "surface_cm2": round(s.surface_mm2 / 100.0, 4),
                    "mean": round(s.mean, 4),
                    "sd": round(s.sd, 4),
                    "median": s.median,
                    "min": s.min,
                    "max": s.max,
                    "max_diameter_mm": round(s.feret_mm, 3),
                    "roundness": round(s.roundness, 4),
                    "elongation": round(s.elongation, 4),
                    "principal_axes_mm": vec3(s.principal_mm, 3),
                    "centre_ras_mm": vec3(s.centroid_ras, 3),
                })),
            })
        })
        .collect::<Vec<_>>();
    json!({"segments": rows})
}

/// The start of `segmentation_do`'s description.
pub const ABOUT: &str = "Work the Segmentation workspace as its editor panel and views do. Effects change the selected segment (give `segment` to select one first) where the masking lets them. Whole-volume effects, meshes and saving run in the background: app_wait, then segmentation_state for the result. Positions are RAS millimetres, or voxel indices (i, j, k) with in_voxels. Segments are named by label number or name (segmentation_state lists them). Edits are steps of the workspace's own undo (undo / redo).";

fn segment(about: &str) -> Value {
    json!({"type": ["integer", "string"], "description": about})
}

fn keys<T: Copy>(all: &[T], key: fn(T) -> &'static str) -> Vec<&'static str> {
    all.iter().map(|x| key(*x)).collect()
}

pub fn actions() -> Vec<Action> {
    let target = || segment("The segment to change (label number or name; default: the selected one)");
    let in_voxels = || arg::boolean("Positions are voxel indices (i, j, k), not RAS mm");
    let brush = || {
        json!({
            "points": arg::points("Where to dab the brush, joined into one stroke (RAS mm, or voxels with in_voxels)"),
            "in_voxels": in_voxels(),
            "radius_mm": arg::number("Brush radius in mm, 0.2–50 (kept for later strokes)"),
            "sphere": arg::boolean("A ball rather than a disc in the slice"),
            "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice the disc lies in (default axial)"),
            "segment": target(),
        })
    };
    vec![
        Action::new("add_segment", "Add a segment and select it (named Segment_n unless `name` is given)", json!({"name": arg::text("Its name")})),
        Action::new("select_segment", "Select the segment effects change", json!({"segment": segment("Label number or name")})).required(&["segment"]),
        Action::new("set_segment", "Rename, recolour, show or hide a segment, or set its status", json!({"segment": segment("Label number or name"), "name": arg::text("New name"), "colour": arg::text("New colour, #rrggbb"), "visible": arg::boolean("Shown in the views"), "status": arg::choice(&["notstarted", "inprogress", "completed", "flagged"], "How far its work has got")})).required(&["segment"]),
        Action::new("remove_segment", "Remove a segment and its voxels", json!({"segment": segment("Label number or name")})).required(&["segment"]).destructive(),
        Action::new(
            "threshold",
            "The segment becomes the voxels with values from min to max (default: the current range, which starts at Otsu's threshold). Makes a segment if there is none. With preview, the views only show the result until apply",
            json!({
                "min": arg::number("Lowest value"),
                "max": arg::number("Highest value"),
                "auto": arg::choice(&mm_segment::AutoThreshold::ALL.map(|m| m.key()), "Set the range from this automatic threshold method instead (Otsu, Huang, IsoData, Kittler–Illingworth, Li, maximum entropy, moments, Rényi, Shanbhag, triangle, Yen)"),
                "auto_mode": arg::choice(&["lower", "upper"], "With auto: lower (from the threshold up: brighter structures, default) or upper (up to it: darker ones)"),
                "preview": arg::boolean("Show the result without applying it"),
                "segment": target(),
            }),
        ),
        Action::new("paint", "Paint the segment with the brush along the points (a segment is made if none is selected)", brush()).required(&["points"]),
        Action::new("erase", "Erase the segment with the brush along the points", brush()).required(&["points"]),
        Action::new(
            "draw",
            "Draw: fill the outline through `points` (at least three, in one slice of `plane`: the slice through the first point) into the segment. Makes a segment if none is selected",
            json!({"points": arg::points("The outline's corners (RAS mm, or voxels with in_voxels)"), "in_voxels": in_voxels(), "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice the outline lies in (default axial)"), "segment": target()}),
        )
        .required(&["points"]),
        Action::new(
            "level_trace",
            "Level tracing: add the region of the slice through `at` (in `plane`) connected to it whose values are at least its value (darker: at most)",
            json!({"at": arg::point("The voxel traced from (RAS mm, or a voxel with in_voxels)"), "in_voxels": in_voxels(), "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice (default axial)"), "darker": arg::boolean("Trace the darker region"), "segment": target()}),
        )
        .required(&["at"]),
        Action::new(
            "local_histogram",
            "Threshold's range from the values in a shape on a slice of `plane` (3D Slicer's local histogram): box (corners points[0], points[1]), circle (centre points[0], radius_mm) or line (points[0] to points[1]). The range runs from `lower` (extreme = minimum, percentile = 5th, average) to `upper` (average, percentile = 95th, extreme = maximum). The threshold preview shows; apply applies it",
            json!({
                "shape": arg::choice(&keys(&LocalShape::ALL, LocalShape::key), "The shape (default box)"),
                "points": arg::points("The shape's points (RAS mm, or voxels with in_voxels)"),
                "in_voxels": in_voxels(),
                "radius_mm": arg::number("Circle radius, mm"),
                "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice the shape is drawn in (default axial)"),
                "lower": arg::choice(&keys(&LocalPick::ALL, LocalPick::key), "Where the range starts (default extreme: the minimum)"),
                "upper": arg::choice(&keys(&LocalPick::ALL, LocalPick::key), "Where the range ends (default extreme: the maximum)"),
            }),
        )
        .required(&["points"]),
        Action::new(
            "scissors",
            "Scissors: fill or erase inside or outside a shape drawn in a slice of `plane`, through a range of slices. Shapes: free (the outline through `points`), circle (centre `points[0]`, radius_mm), rectangle (corners `points[0]` and `points[1]`). The slice is the one through `points[0]`",
            json!({
                "op": arg::choice(&keys(&ScissorsOp::ALL, ScissorsOp::key), "What to do"),
                "shape": arg::choice(&keys(&ScissorsShape::ALL, ScissorsShape::key), "The shape (default free)"),
                "points": arg::points("The shape's points (RAS mm, or voxels with in_voxels)"),
                "in_voxels": in_voxels(),
                "radius_mm": arg::number("Circle radius, mm"),
                "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice the shape is drawn in (default axial)"),
                "range": arg::choice(&keys(&ScissorsRange::ALL, ScissorsRange::key), "Slices cut: unlimited (all), positive / negative (this one and those towards the patient's superior, anterior or right / the other way), symmetric (a slab of thickness_mm)"),
                "thickness_mm": arg::number("Symmetric slab thickness, mm"),
                "segment": target(),
            }),
        )
        .required(&["op", "points"]),
        Action::new(
            "grow_from_seeds",
            "Grow from seeds: every visible segment (at least two: the structure and its surroundings, painted as seeds) grows over the voxels most like its own, within margin_mm of the seeds. Shows the result as a preview; apply keeps it (or pass apply: true)",
            json!({"locality": arg::number("0–10: how much nearness counts besides likeness (default 0)"), "margin_mm": arg::number("Work within this far of the seeds, mm"), "apply": arg::boolean("Keep the result at once")}),
        ),
        Action::new(
            "watershed",
            "Watershed: the visible segments (seeds, at least two) flood the image until they meet at its edges, after smoothing by object_scale_mm. A preview; apply keeps it (or pass apply: true)",
            json!({"object_scale_mm": arg::number("Smoothing before the edges are found, mm"), "margin_mm": arg::number("Work within this far of the seeds, mm"), "apply": arg::boolean("Keep the result at once")}),
        ),
        Action::new(
            "fill_between_slices",
            "Fill between slices: the visible segments, segmented on some slices of `plane`, are interpolated into the empty slices between. A preview; apply keeps it (or pass apply: true)",
            json!({"plane": arg::choice(&["axial", "sagittal", "coronal"], "Whose slices to fill between (default axial)"), "apply": arg::boolean("Keep the result at once")}),
        ),
        Action::new(
            "mask_volume",
            "Mask volume: save a new image (listed in the Volume workspace among the derived images): the volume with `inside` filled in inside the segment, `outside` outside it, or both (a binary image), the boundary softened over soft_mm",
            json!({"fill": arg::choice(&keys(&MaskFill::ALL, MaskFill::key), "Where to fill in values"), "inside": arg::number("Value inside"), "outside": arg::number("Value outside"), "soft_mm": arg::number("Soft edge, mm (0: sharp)"), "segment": target()}),
        ),
        Action::new(
            "split_volume",
            "Split volume: save a new image per visible segment (listed in the Volume workspace among the derived images): the volume cropped to the segment's box grown by margin_mm, with fill_outside (when given) outside the segment",
            json!({"margin_mm": arg::number("Margin around each segment, mm (default 5)"), "fill_outside": arg::number("Value outside the segment (default: the image's own values)")}),
        ),
        Action::new("fast_marching", "Fast marching: grow the segment over the most similar voxels nearest to it until it is `percent` of the volume", json!({"percent": arg::number("Final size, % of the volume"), "segment": target()})).required(&["percent"]),
        Action::new(
            "draw_tube",
            "Draw tube: add a tube of radius_mm along a smooth curve through `points` (in order) to the segment. Makes a segment if none is selected",
            json!({"points": arg::points("Points along the structure (RAS mm, or voxels with in_voxels)"), "in_voxels": in_voxels(), "radius_mm": arg::number("Tube radius, mm"), "segment": target()}),
        )
        .required(&["points"]),
        Action::new(
            "surface_cut",
            "Surface cut: add the closed shape through `points` (at least four, on a structure's surface, not all in one plane) to the segment",
            json!({"points": arg::points("Points on the surface (RAS mm, or voxels with in_voxels)"), "in_voxels": in_voxels(), "segment": target()}),
        )
        .required(&["points"]),
        Action::new(
            "flood_fill",
            "Add the voxels connected to `at` whose values are within `tolerance` of its value (or in the threshold range with use_threshold), not passing gaps narrower than min_diameter_mm. Makes a segment if there is none",
            json!({
                "at": arg::point("The seed (RAS mm, or a voxel with in_voxels)"),
                "in_voxels": in_voxels(),
                "tolerance": arg::number("Value tolerance"),
                "use_threshold": arg::boolean("Fill voxels in the threshold range instead (3D Slicer's Local threshold)"),
                "min_diameter_mm": arg::number("Narrowest gap passed, mm (0: no limit)"),
                "refine": arg::choice(&["none", "grow", "watershed"], "Edges: as thresholded, or decided again within feature_mm by grow from seeds or watershed"),
                "feature_mm": arg::number("How far the refined edge may move, mm"),
                "segment": target(),
            }),
        )
        .required(&["at"]),
        Action::new(
            "islands",
            "The segment's separate pieces: largest (keep the largest), small (remove those under min_size voxels), split (each piece its own segment), keep / remove / add (the piece at `at`; add takes it from the unlabelled voxels)",
            json!({"op": arg::choice(&keys(&IslandOp::ALL, IslandOp::key), "What to do"), "at": arg::point("The piece (ops keep, remove, add)"), "in_voxels": in_voxels(), "min_size": arg::integer("Smallest piece kept, voxels (ops small, split)"), "segment": target()}),
        )
        .required(&["op"]),
        Action::new("margin", "Grow (mm > 0) or shrink (mm < 0) the segment (or every visible one)", json!({"mm": arg::number("Distance in mm"), "visible": arg::boolean("Every visible segment, not only the selected one"), "segment": target()})).required(&["mm"]),
        Action::new(
            "smooth",
            "Smooth the segment: median (removes bumps), open (removes extrusions), close (fills gaps), fill (fills holes), gaussian (mm = standard deviation), joint (every visible segment together, no gaps or overlaps between them; mm = standard deviation)",
            json!({"method": arg::choice(&keys(&SmoothMethod::ALL, SmoothMethod::key), "Method"), "mm": arg::number("Kernel size in mm"), "visible": arg::boolean("Every visible segment, not only the selected one"), "segment": target()}),
        ),
        Action::new(
            "level_set",
            "Level set: grow (or shrink) the segment to the vessel's wall, starting from it (a rough segment inside the vessel). The vessel's values are Threshold's range (give min / max to set it); with overlay, the overlay shown (e.g. vesselness, see `overlay`) drives it instead: above its window centre the front moves out",
            json!({"segment": target(), "min": arg::number("Vessel values from"), "max": arg::number("Vessel values to"), "iterations": arg::integer("Iterations (default 300)"), "max_mm": arg::number("How far the front may move, mm (default 10)"), "smoothness": arg::number("Curvature weight (default 0.3)"), "overlay": arg::boolean("Follow the overlay shown")}),
        ),
        Action::new(
            "guided_vessel",
            "Guided vessel: segment a vessel through `points` clicked along it in order (its ends and a few between): the path between them follows the vessel (values in Threshold's range: min / max), then a level set grows to its wall. Adds to the segment (makes one if none is selected)",
            json!({"points": arg::points("Points along the vessel (RAS mm, or voxels with in_voxels)"), "in_voxels": in_voxels(), "radius_mm": arg::number("About the vessel's radius, mm"), "min": arg::number("Vessel values from"), "max": arg::number("Vessel values to"), "segment": target()}),
        )
        .required(&["points"]),
        Action::new(
            "go_to_section",
            "Move the crosshair to a cross-section of the last centerline (branch number, distance along it in mm; default the narrowest), turning the planes with the vessel there unless align is false: the view most nearly across it (axial for a vessel running up, sagittal for one running left to right) shows its cross-section, the other two run along it",
            json!({"branch": arg::integer("Branch (1 = the first)"), "distance_mm": arg::number("Distance along the branch, mm"), "align": arg::boolean("Turn the planes with the vessel (default true)")}),
        ),
        Action::new(
            "engrave",
            "Engrave: `text` (A–Z, 0–9, - . + / :) centred on `at` in the slice of `plane` through it, reading left to right as the view shows it, `height_mm` tall and `depth_mm` deep through the slices; cut from the selected segment, or raised from it with emboss",
            json!({"text": arg::text("The text"), "at": arg::point("Its centre (RAS mm, or voxels with in_voxels)"), "in_voxels": in_voxels(), "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice it is drawn in (default axial)"), "height_mm": arg::number("Letter height, mm"), "depth_mm": arg::number("Depth through the slices, mm"), "emboss": arg::boolean("Raise the text (add) instead of cutting it"), "segment": target()}),
        )
        .required(&["text", "at"]),
        Action::new(
            "measure",
            "Measure in a slice of `plane` (through points[0]): ruler (distance between two points, mm), angle (at the second of three points, degrees), region (a circle: centre points[0], a point of its edge points[1]; area and the image's mean, SD, min and max in it). The measurement is listed and drawn on its slice; clear: remove them all",
            json!({"tool": arg::choice(&polymix_imaging::MeasureTool::ALL.map(|t| t.key()), "What to measure"), "points": arg::points("Points (RAS mm, or voxels with in_voxels)"), "in_voxels": in_voxels(), "plane": arg::choice(&["axial", "sagittal", "coronal"], "The slice measured in (default axial)"), "clear": arg::boolean("Remove every measurement")}),
        ),
        Action::new(
            "centerline",
            "Extract the segment's centerline tree and its cross-sections (in the background: app_wait, then segmentation_centerline). `ends`: points (RAS mm, or voxels with in_voxels) the centerline starts from (the first) and reaches; none: found automatically",
            json!({"segment": target(), "ends": arg::points("Start and end points (optional)"), "in_voxels": in_voxels(), "min_branch_mm": arg::number("Shortest branch kept, mm (default 5)"), "step_mm": arg::number("Distance between cross-sections, mm (default 0.5)")}),
        ),
        Action::new(
            "hollow",
            "Make the segment (or every visible one) a shell: `side` outside (the segment is the lumen; default), medial or inside its surface",
            json!({"mm": arg::number("Shell thickness in mm"), "side": arg::choice(&["outside", "medial", "inside"], "Where the shell lies"), "visible": arg::boolean("Every visible segment, not only the selected one"), "segment": target()}),
        )
        .required(&["mm"]),
        Action::new(
            "logical",
            "copy, add, subtract or intersect the `other` segment into this one; invert, clear or fill this one",
            json!({"op": arg::choice(&keys(&LogicalOp::ALL, LogicalOp::key), "What to do"), "other": segment("The other segment (copy, add, subtract, intersect)"), "bypass_masking": arg::boolean("Ignore the masking settings"), "segment": target()}),
        )
        .required(&["op"]),
        Action::plain("apply", "Apply the effect in use with its settings (e.g. a previewed threshold)"),
        Action::plain("cancel", "Stop the operation running"),
        Action::new(
            "masking",
            "Where effects may change voxels: the editable area, an intensity range, and which other segments' voxels they may take",
            json!({
                "editable_area": arg::choice(&["everywhere", "inside_all", "inside_visible", "outside_all", "outside_visible", "inside", "outside"], "Where (inside / outside: of area_segment)"),
                "area_segment": segment("The segment for editable_area inside / outside"),
                "intensity_range": {"type": "array", "items": {"type": "number"}, "maxItems": 2, "description": "[min, max]: only voxels with values in it; [] for any value"},
                "overwrite": arg::choice(&["all", "visible", "none"], "Other segments' voxels an effect may take: all, those of visible segments, or none"),
            }),
        ),
        Action::new("undo", "Undo edits", json!({"steps": arg::integer("How many (default 1)")})),
        Action::new("redo", "Redo edits", json!({"steps": arg::integer("How many (default 1)")})),
        Action::new(
            "make_mesh",
            "Make a surface mesh of the selected segment (or of every visible one) and send it to the Mesh workspace, which is then shown",
            json!({"all_visible": arg::boolean("Every visible segment, not only the selected one"), "smoothing_mm": arg::number("Surface smoothing in mm")}),
        ),
        Action::new(
            "save",
            "Save the segmentation: a 3D Slicer .seg.nrrd, or a .nii / .nii.gz label map, by the path's ending. An existing file is kept unless overwrite",
            json!({"path": arg::text("Absolute path"), "overwrite": arg::boolean("Replace an existing file")}),
        )
        .required(&["path"])
        .destructive(),
        Action::new(
            "load_segmentation",
            "Add the segments of a label map file onto this volume: a 3D Slicer .seg.nrrd (names, colours and status kept) or a NIfTI label map (names from Polymix's description; others as Label n). Another grid is resampled by position",
            json!({"path": arg::text("Absolute path")}),
        )
        .required(&["path"]),
        Action::new("go_to", "Move the crosshair, and so the slices shown, to a point", json!({"at": arg::point("RAS mm, or a voxel with in_voxels"), "in_voxels": in_voxels()})).required(&["at"]),
        Action::new(
            "display",
            "How the views show the volume: window / level, the segments' fill opacity, the crosshair, smooth slices, and one view filling the view area",
            json!({
                "window_centre": arg::number("Window centre (level)"),
                "window_width": arg::number("Window width"),
                "auto_window": arg::boolean("Back to the automatic window"),
                "opacity": arg::number("Segments' fill opacity in the slices, 0–1"),
                "crosshair": arg::boolean("Show the crosshair"),
                "smooth_slices": arg::boolean("Interpolate the slices"),
                "maximize": arg::choice(&["axial", "sagittal", "coronal", "3d", "none"], "The view filling the view area (none: all four)"),
                "colour_map": arg::choice(&polymix_imaging::ColourMap::ALL.map(|m| m.key()), "Colours of the image"),
                "window_preset": arg::choice(&polymix_imaging::layers::CT_PRESETS.map(|p| p.0), "A CT window (Hounsfield units)"),
                "slab": arg::choice(&polymix_imaging::SlabMode::ALL.map(|m| m.key()), "Thick slabs in the slice views: max (MIP), min (MinIP), mean, or off"),
                "slab_mm": arg::number("Slab thickness, mm"),
                "layout": arg::choice(&polymix_imaging::Layout::ALL.map(|l| l.key()), "four_up (axial and 3D above, sagittal and coronal below), conventional (3D above the three planes), row (the three planes side by side)"),
                "straighten": arg::boolean("Planes back along the voxel grid (after oblique planes)"),
                "turn": {"type": "object", "description": "Turn the planes: {\"view\": axial | sagittal | coronal, \"degrees\": n} turns the other two about that view's normal (as ⌥ + drag does)"},
            }),
        ),
        Action::new(
            "overlay",
            "Show another image over this one, or compare the two: `choice` is a number or name from segmentation_state's overlay_choices (images of the Volume catalog overlapping this one; resampled onto this grid in the background: app_wait). compare: blend (see-through by opacity), wipe (the first left of a divider at `wipe` 0–1, the second right), toggle (one or the other: showing_second). remove: stop showing it",
            json!({
                "choice": {"description": "Index or name in overlay_choices"},
                "compare": arg::choice(&polymix_imaging::Compare::ALL.map(|c| c.key()), "How the two are shown"),
                "opacity": arg::number("Blend: the second image's opacity, 0–1"),
                "wipe": arg::number("Wipe: the divider, 0 (left) – 1 (right)"),
                "showing_second": arg::boolean("Toggle: show the second image"),
                "colour_map": arg::choice(&polymix_imaging::ColourMap::ALL.map(|m| m.key()), "Colours of the second image"),
                "window_centre": arg::number("The second image's window centre"),
                "window_width": arg::number("The second image's window width"),
                "remove": arg::boolean("Stop showing the second image"),
            }),
        ),
        Action::new(
            "answer_volume_switch",
            "Answer the prompt shown when a volume on another voxel grid is selected in the Volume workspace (segmentation_state volume_switch_prompt): discard (open it, discarding the segmentation) or keep (keep segmenting this one)",
            json!({"choice": arg::choice(&["discard", "keep"], "The answer")}),
        )
        .required(&["choice"])
        .destructive(),
    ]
}

pub fn act(ws: &mut SegmentationWorkspace, action: &str, args: &Value, ctx: &egui::Context) -> AgentReply {
    let was_running = ws.running.is_some();
    let r = act_now(ws, action, &Args(args), ctx);
    if let Ok(done) = &r {
        ws.message = Some((format!("Agent: {done}"), false));
    }
    let running = ws.running.as_ref().map(|r| r.what.clone());
    // The outcome of what the agent started replaces the message, also labelled as the agent's.
    ws.agent_op = running.is_some() && (ws.agent_op || !was_running);
    AgentReply::Now(r.map(|done| match running {
        Some(what) => json!({"done": done, "running": what, "next": "app_wait, then segmentation_state"}),
        None => json!({"done": done}),
    }))
}

fn act_now(ws: &mut SegmentationWorkspace, action: &str, a: &Args, ctx: &egui::Context) -> Result<String, String> {
    if action == "cancel" {
        ws.agent_op = false;
        return match ws.running.take() {
            Some(r) => Ok(format!("Cancelled {}", r.what)),
            None => Err("Nothing is running".into()),
        };
    }
    if action == "answer_volume_switch" {
        let (p, v) = ws.pending.take().ok_or("No volume switch is waiting for an answer")?;
        return Ok(if a.str("choice")? == "discard" {
            ws.open(&v, &p.label, false, ctx);
            "Opening the other volume; the segmentation was discarded".into()
        } else {
            ws.declined = Some(p);
            "Kept segmenting this volume".into()
        });
    }
    let vol = loaded(ws)?;
    if let Some(r) = &ws.running {
        return Err(format!("{} is running: app_wait, then try again (or cancel)", r.what));
    }
    if action == "go_to" || action == "display" {
        return view(ws, action, a, &vol);
    }
    if action == "overlay" {
        return overlay(ws, a, vol, ctx);
    }
    let seg = ws.seg.as_mut().ok_or("No segmentation")?;
    if a.has("segment") && !matches!(action, "select_segment" | "set_segment" | "remove_segment") {
        seg.selected = Some(find(seg, a.0.get("segment").unwrap())?);
    }
    let named = |seg: &Segmentation, k: usize| format!("{} (label {})", seg.segments[k].name, seg.segments[k].label);
    match action {
        "add_segment" => {
            let k = seg.add_segment(a.opt_str("name").map(str::to_string)).ok_or("All 255 segment labels are in use")?;
            Ok(format!("Added {}", named(seg, k)))
        }
        "select_segment" => {
            let k = find(seg, a.0.get("segment").unwrap())?;
            seg.selected = Some(k);
            Ok(format!("Selected {}", named(seg, k)))
        }
        "set_segment" => {
            let k = find(seg, a.0.get("segment").unwrap())?;
            let colour = a.opt_str("colour").map(parse_colour).transpose()?;
            let s = &mut seg.segments[k];
            if let Some(name) = a.opt_str("name").map(str::trim).filter(|n| !n.is_empty()) {
                s.name = name.to_string();
            }
            if let Some(c) = colour {
                s.colour = c;
            }
            let shown = a.opt_bool("visible");
            if let Some(v) = shown {
                s.visible = v;
            }
            if let Some(st) = a.opt_str("status") {
                s.status = crate::segmentation::Status::ALL.into_iter().find(|x| x.key() == st).ok_or("status must be notstarted, inprogress, completed or flagged")?;
            }
            // Only showing or hiding changes what edits and the 3D view see; the rest is how
            // the segment looks.
            seg.restyled();
            if shown.is_some() {
                seg.changed(true);
            }
            Ok(format!("Changed {}", named(seg, k)))
        }
        "remove_segment" => {
            let k = find(seg, a.0.get("segment").unwrap())?;
            let what = named(seg, k);
            seg.remove_segment(k);
            Ok(format!("Removed {what}"))
        }
        "undo" | "redo" => {
            let steps = a.opt_usize("steps").unwrap_or(1).max(1);
            let done = (0..steps).take_while(|_| if action == "undo" { seg.undo() } else { seg.redo() }).count();
            if done == 0 {
                return Err(format!("Nothing to {action}"));
            }
            Ok(format!("{} {done} step(s)", if action == "undo" { "Undid" } else { "Redid" }))
        }
        "masking" => {
            let m = &mut ws.masking;
            if let Some(area) = a.opt_str("editable_area") {
                let label = || -> Result<u8, String> {
                    let v = a.0.get("area_segment").filter(|v| !v.is_null()).ok_or("editable_area inside / outside needs area_segment")?;
                    Ok(seg.segments[find(seg, v)?].label)
                };
                m.area = match area {
                    "inside_all" => EditableArea::InsideAll,
                    "outside_all" => EditableArea::OutsideAll,
                    "inside_visible" => EditableArea::InsideVisible,
                    "outside_visible" => EditableArea::OutsideVisible,
                    "inside" => EditableArea::Inside(label()?),
                    "outside" => EditableArea::Outside(label()?),
                    _ => EditableArea::Everywhere,
                };
            }
            if let Some(r) = a.0.get("intensity_range").and_then(Value::as_array) {
                m.intensity = match r.iter().map(Value::as_f64).collect::<Option<Vec<f64>>>().as_deref() {
                    Some([lo, hi]) => Some((lo.min(*hi) as f32, lo.max(*hi) as f32)),
                    Some([]) => None,
                    _ => return Err("intensity_range must be [min, max], or [] for any value".into()),
                };
            }
            if let Some(o) = a.opt_str("overwrite") {
                m.overwrite = match o {
                    "visible" => Overwrite::Visible,
                    "none" => Overwrite::None,
                    _ => Overwrite::All,
                };
            }
            Ok(format!("Masking: {:?}, intensity {:?}, overwrite {:?}", m.area, m.intensity, m.overwrite))
        }
        "save" => {
            let path = PathBuf::from(a.str("path")?);
            let name = path.file_name().map(|n| n.to_string_lossy().to_lowercase()).unwrap_or_default();
            if !path.is_absolute() {
                return Err("path must be absolute".into());
            }
            if ![".seg.nrrd", ".nrrd", ".nii", ".nii.gz"].iter().any(|e| name.ends_with(e)) {
                return Err("path must end in .seg.nrrd, .nii or .nii.gz".into());
            }
            if path.exists() && a.opt_bool("overwrite") != Some(true) {
                return Err("That file exists: pass overwrite to replace it".into());
            }
            if !path.parent().is_some_and(|d| d.is_dir()) {
                return Err("The folder does not exist".into());
            }
            ws.save_to(path, ctx);
            Ok("Saving".into())
        }
        "make_mesh" => {
            let st = ws.settings.as_mut().ok_or("No volume")?;
            if let Some(all) = a.opt_bool("all_visible") {
                st.mesh_all_visible = all;
            }
            if let Some(mm) = a.opt_f64("smoothing_mm") {
                st.mesh_smoothing_mm = mm.max(0.0) as f32;
            }
            ws.make_mesh(ctx);
            started(ws, "Making meshes")
        }
        "paint" | "erase" => paint(ws, &vol, a, action == "erase"),
        "draw" | "level_trace" | "scissors" | "local_histogram" => sketch(ws, &vol, action, a),
        "engrave" => {
            let st = ws.settings.as_mut().ok_or("No volume")?;
            st.engrave_text = a.str("text")?.to_string();
            if let Some(h) = a.opt_f64("height_mm") {
                st.engrave_mm = h.max(0.5) as f32;
            }
            if let Some(d) = a.opt_f64("depth_mm") {
                st.engrave_depth_mm = d.max(0.1) as f32;
            }
            st.engrave_raise = a.opt_bool("emboss").unwrap_or(false);
            let plane = match a.opt_str("plane") {
                Some("sagittal") => Plane::Sagittal,
                Some("coronal") => Plane::Coronal,
                _ => Plane::Axial,
            };
            let v = voxel(&vol, a.point("at")?, a.opt_bool("in_voxels") == Some(true))?;
            ws.effect = Some(Effect::Engrave);
            let (axes, _) = crate::sketch::slicing(&vol, plane);
            ws.engrave(&vol, axes, v);
            match &ws.message {
                Some((m, true)) => Err(m.clone()),
                Some((m, false)) => Ok(m.clone()),
                None => Ok(String::new()),
            }
        }
        "go_to_section" => {
            let r = ws.vessels.result.as_ref().ok_or("No centerline yet: centerline first")?;
            let b = a.opt_usize("branch").map_or(ws.vessels.branch, |b| b.saturating_sub(1));
            let sections = r.sections.get(b).ok_or(format!("there are {} branches", r.sections.len()))?;
            let i = match a.opt_f64("distance_mm") {
                Some(d) => (0..sections.len()).min_by(|&x, &y| (sections[x].distance_mm - d).abs().total_cmp(&(sections[y].distance_mm - d).abs())).ok_or("the branch has no sections")?,
                None => r.narrowest[b].map(|n| n.0).or((!sections.is_empty()).then_some(0)).ok_or("the branch has no sections")?,
            };
            let s = sections[i];
            ws.vessels.branch = b;
            ws.vessels.align = a.opt_bool("align").unwrap_or(true);
            ws.go_to_section(DVec3::from(s.ras), DVec3::from(s.normal));
            Ok(format!("At {:.1} mm along branch {}: ⌀ {:.2} mm ({:.2}–{:.2}), {:.1} mm²", s.distance_mm, b + 1, s.equivalent_diameter_mm, s.min_diameter_mm, s.max_diameter_mm, s.area_mm2))
        }
        "guided_vessel" => {
            let in_voxels = a.opt_bool("in_voxels") == Some(true);
            let points = a.points("points")?.into_iter().map(|p| voxel(&vol, p, in_voxels)).collect::<Result<Vec<_>, _>>()?;
            let st = ws.settings.as_mut().ok_or("No volume")?;
            if let Some(r) = a.opt_f64("radius_mm") {
                st.tube_mm = r.max(0.1) as f32;
            }
            let (lo, hi) = (a.opt_f64("min").map_or(st.threshold.0, |x| x as f32), a.opt_f64("max").map_or(st.threshold.1, |x| x as f32));
            st.threshold = (lo.min(hi), lo.max(hi));
            ws.effect = Some(Effect::GuidedVessel);
            ws.markers = points;
            ws.start_guided(ctx);
            if let Some((m, true)) = &ws.message {
                return Err(m.clone());
            }
            started(ws, "Guided vessel")
        }
        "measure" => {
            if a.opt_bool("clear") == Some(true) {
                ws.measure.items.clear();
                return Ok("Measurements removed".into());
            }
            let tool = polymix_imaging::MeasureTool::ALL.into_iter().find(|t| Some(t.key()) == a.opt_str("tool")).ok_or("tool is ruler, angle or region")?;
            let plane = match a.opt_str("plane") {
                Some("sagittal") => Plane::Sagittal,
                Some("coronal") => Plane::Coronal,
                _ => Plane::Axial,
            };
            let in_voxels = a.opt_bool("in_voxels") == Some(true);
            let mut points = a.points("points")?.into_iter().map(|p| voxel(&vol, p, in_voxels)).collect::<Result<Vec<_>, _>>()?;
            if points.len() != tool.points() {
                return Err(format!("{} takes {} points", tool.label(), tool.points()));
            }
            let axis = mm_volume::slice::plane_axes(&vol.geometry, plane).slice;
            let slice = seed(&vol, points[0])[axis];
            points.iter_mut().for_each(|p| p[axis] = slice as f64);
            let m = polymix_imaging::measure::measure(&vol, tool, plane, slice, points);
            let text = m.text();
            ws.measure.items.push(m);
            Ok(format!("{}: {text}", tool.label()))
        }
        "centerline" => {
            let in_voxels = a.opt_bool("in_voxels") == Some(true);
            if a.has("ends") {
                ws.vessels.ends = a.points("ends")?.into_iter().map(|p| voxel(&vol, p, in_voxels)).collect::<Result<_, _>>()?;
            }
            if let Some(m) = a.opt_f64("min_branch_mm") {
                ws.vessels.min_branch_mm = m.max(0.5);
            }
            if let Some(m) = a.opt_f64("step_mm") {
                ws.vessels.step_mm = m.max(0.05);
            }
            ws.extract_centerline(ctx);
            if let Some((m, true)) = &ws.message {
                return Err(m.clone());
            }
            started(ws, "Extracting the centerline")
        }
        "load_segmentation" => {
            let path = PathBuf::from(a.str("path")?);
            if !path.is_absolute() || !path.exists() {
                return Err(format!("{} is not an absolute path to an existing file", path.display()));
            }
            ws.load_segmentation(path, ctx);
            started(ws, "Loading segmentation")
        }
        "grow_from_seeds" | "watershed" | "fill_between_slices" => {
            let effect = match action {
                "watershed" => Effect::Watershed,
                "fill_between_slices" => Effect::FillBetweenSlices,
                _ => Effect::GrowFromSeeds,
            };
            let st = ws.settings.as_mut().ok_or("No volume")?;
            st.fill_plane = match a.opt_str("plane") {
                Some("sagittal") => Plane::Sagittal,
                Some("coronal") => Plane::Coronal,
                _ => Plane::Axial,
            };
            if let Some(x) = a.opt_f64("locality") {
                st.seed_locality = x.clamp(0.0, 10.0) as f32;
            }
            if let Some(x) = a.opt_f64("object_scale_mm") {
                st.watershed_mm = x.max(0.0) as f32;
            }
            if let Some(x) = a.opt_f64("margin_mm") {
                st.seed_margin_mm = x.max(0.5) as f32;
            }
            ws.effect = Some(effect);
            if a.opt_bool("apply") == Some(true) {
                // Computed here, and kept at once.
                let (labels, version) = ws.seed_labels().ok_or("No segmentation")?;
                let r = crate::effects::run_seeded(effect, ws.settings.as_ref().ok_or("No volume")?, &vol, labels, version);
                ws.finish_op(r, false);
                if ws.seeded.is_none() {
                    return Err(ws.message.as_ref().map_or("no result".into(), |m| m.0.clone()));
                }
                ws.apply_seeded();
                return Ok(ws.message.as_ref().map_or(String::new(), |m| m.0.clone()));
            }
            ws.start_seeded(effect, ctx);
            started(ws, effect.label())
        }
        "draw_tube" | "surface_cut" => {
            let in_voxels = a.opt_bool("in_voxels") == Some(true);
            let points = a.points("points")?.into_iter().map(|p| voxel(&vol, p, in_voxels)).collect::<Result<Vec<_>, _>>()?;
            let effect = if action == "draw_tube" { Effect::DrawTube } else { Effect::SurfaceCut };
            if let (Some(st), Some(r)) = (ws.settings.as_mut(), a.opt_f64("radius_mm")) {
                st.tube_mm = r.max(0.05) as f32;
            }
            ws.effect = Some(effect);
            ws.markers = points;
            ws.apply_markers(effect);
            match &ws.message {
                Some((m, true)) => Err(m.clone()),
                Some((m, false)) => Ok(m.clone()),
                None => Ok(String::new()),
            }
        }
        _ => effect(ws, &vol, action, a, ctx),
    }
}

/// The volume shown, or what the agent should do to get one.
fn loaded(ws: &SegmentationWorkspace) -> Result<Arc<LoadedVolume>, String> {
    if ws.loading.is_some() {
        return Err("The volume is still loading: app_wait, then try again".into());
    }
    if let Some(e) = &ws.error {
        return Err(format!("The volume could not be shown: {}", polymix_workspace::guard::scrub_paths(e)));
    }
    ws.volume.clone().ok_or_else(|| "No volume is loaded. The Segmentation tab loads the volume selected in the Volume workspace (volume_do select) while it is shown, and it is shown now: app_wait, then try again.".into())
}

/// A segment's index, from its label number or name.
fn find(seg: &Segmentation, v: &Value) -> Result<usize, String> {
    let k = match v {
        Value::Number(n) => n.as_u64().and_then(|l| u8::try_from(l).ok()).and_then(|l| seg.index_of(l)),
        Value::String(name) => seg.segments.iter().position(|s| s.name == *name).or_else(|| seg.segments.iter().position(|s| s.name.eq_ignore_ascii_case(name))),
        _ => None,
    };
    k.ok_or_else(|| {
        let list: Vec<String> = seg.segments.iter().map(|s| format!("{} {}", s.label, s.name)).collect();
        if list.is_empty() { format!("No segment {v}: there are none yet (add_segment)") } else { format!("No segment {v}; the segments are {}", list.join(", ")) }
    })
}

fn parse_colour(s: &str) -> Result<egui::Color32, String> {
    let h = s.trim().trim_start_matches('#');
    let c = (h.len() == 6).then(|| u32::from_str_radix(h, 16).ok()).flatten().ok_or_else(|| format!("colour {s:?} must be #rrggbb"))?;
    Ok(egui::Color32::from_rgb((c >> 16) as u8, (c >> 8) as u8, c as u8))
}

/// A position as continuous voxel coordinates, checked to be inside the volume.
fn voxel(vol: &LoadedVolume, p: [f64; 3], in_voxels: bool) -> Result<DVec3, String> {
    let v = if in_voxels { DVec3::from(p) } else { vol.geometry.matrix().inverse().transform_point3(DVec3::from(p)) };
    let inside = (0..3).all(|a| v[a] > -0.5 && v[a] < vol.dims[a] as f64 - 0.5);
    if !inside {
        return Err(format!("{p:?} is outside the volume (voxel {:.1}, {:.1}, {:.1}; the grid is {:?})", v.x, v.y, v.z, vol.dims));
    }
    Ok(v)
}

fn seed(vol: &LoadedVolume, v: DVec3) -> [usize; 3] {
    [0, 1, 2].map(|a| v[a].round().clamp(0.0, vol.dims[a] as f64 - 1.0) as usize)
}

/// After starting background work: what started, or why it did not.
fn started(ws: &SegmentationWorkspace, what: &str) -> Result<String, String> {
    match (&ws.running, &ws.message) {
        (Some(_), _) => Ok(format!("{what} started")),
        (None, Some((m, true))) => Err(m.clone()),
        (None, _) => Err(format!("{what} did not start")),
    }
}

fn paint(ws: &mut SegmentationWorkspace, vol: &LoadedVolume, a: &Args, erase: bool) -> Result<String, String> {
    let in_voxels = a.opt_bool("in_voxels") == Some(true);
    let points = a.points("points")?.into_iter().map(|p| voxel(vol, p, in_voxels)).collect::<Result<Vec<_>, _>>()?;
    let plane = match a.opt_str("plane") {
        Some("sagittal") => Plane::Sagittal,
        Some("coronal") => Plane::Coronal,
        _ => Plane::Axial,
    };
    let st = ws.settings.as_mut().ok_or("No volume")?;
    if let Some(mm) = a.opt_f64("radius_mm") {
        st.brush_mm = mm.clamp(0.2, 50.0) as f32;
    }
    if let Some(s) = a.opt_bool("sphere") {
        st.sphere = s;
    }
    let brush_mm = st.brush_mm;
    if erase && ws.seg.as_ref().is_some_and(|s| s.selected.is_none()) {
        return Err("Select a segment first".into());
    }
    ws.effect = Some(if erase { Effect::Erase } else { Effect::Paint });
    let count = |ws: &mut SegmentationWorkspace| ws.seg.as_mut().and_then(|s| s.selected_segment().map(|g| g.label)).map(|l| ws.seg.as_mut().unwrap().counts()[l as usize]).unwrap_or(0);
    let before = count(ws);
    ws.end_stroke();
    for p in points {
        ws.brush(vol, Some(plane), p, erase);
    }
    ws.end_stroke();
    let after = count(ws);
    let name = ws.seg.as_ref().and_then(|s| s.selected_segment()).map_or(String::new(), |s| s.name.clone());
    Ok(format!("{} {name} with a {brush_mm} mm radius brush: {} voxels {}", if erase { "Erased" } else { "Painted" }, before.abs_diff(after), if erase { "removed" } else { "added" }))
}

/// The effects drawn on a slice: draw, level_trace and scissors.
fn sketch(ws: &mut SegmentationWorkspace, vol: &LoadedVolume, action: &str, a: &Args) -> Result<String, String> {
    let in_voxels = a.opt_bool("in_voxels") == Some(true);
    let plane = match a.opt_str("plane") {
        Some("sagittal") => Plane::Sagittal,
        Some("coronal") => Plane::Coronal,
        _ => Plane::Axial,
    };
    let (axes, step) = crate::sketch::slicing(vol, plane);
    let points = if action == "level_trace" { vec![a.point("at")?] } else { a.points("points")? };
    let points = points.into_iter().map(|p| voxel(vol, p, in_voxels)).collect::<Result<Vec<_>, _>>()?;
    let first = *points.first().ok_or("points is empty")?;
    let slice = seed(vol, first)[axes.slice];
    let flat: Vec<(f64, f64)> = points.iter().map(|v| (v[axes.across], v[axes.down])).collect();
    if action == "local_histogram" {
        let st = ws.settings.as_mut().ok_or("No volume")?;
        let pick = |k: &str, now| a.opt_str(k).map_or(Ok(now), |x| LocalPick::ALL.into_iter().find(|p| p.key() == x).ok_or(format!("{k} is extreme, percentile or average")));
        st.local_lower = pick("lower", st.local_lower)?;
        st.local_upper = pick("upper", st.local_upper)?;
        st.local_shape = match a.opt_str("shape") {
            Some(x) => LocalShape::ALL.into_iter().find(|s| s.key() == x).ok_or("shape is box, circle or line")?,
            None => LocalShape::Box,
        };
        st.local_histogram = true;
        let s = vol.geometry.spacing;
        let (outline, line) = match st.local_shape {
            LocalShape::Circle => (mm_segment::shapes::circle(flat[0], a.f64("radius_mm")? / s[axes.across], s[axes.across] / s[axes.down], 72), false),
            shape => {
                let (p, q) = (flat[0], *flat.get(1).ok_or("a box or line needs two points")?);
                if shape == LocalShape::Line {
                    (vec![p, q], true)
                } else {
                    (vec![p, (q.0, p.1), q, (p.0, q.1)], false)
                }
            }
        };
        ws.effect = Some(Effect::Threshold);
        ws.set_local(vol, plane, axes, slice, outline, line);
        let l = ws.local.as_ref().ok_or("The shape holds no voxels")?;
        let t = ws.settings.as_ref().map_or((0.0, 0.0), |s| s.threshold);
        return Ok(format!("Local histogram of {} voxels ({} – {}): threshold range {} to {} shown; apply applies it", l.values.len(), l.values[0], l.values[l.values.len() - 1], t.0, t.1));
    }
    let count = |ws: &mut SegmentationWorkspace| ws.seg.as_mut().and_then(|s| s.selected_segment().map(|g| g.label)).map(|l| ws.seg.as_mut().unwrap().counts()[l as usize]).unwrap_or(0);
    let before = count(ws);
    match action {
        "draw" => {
            if flat.len() < 3 {
                return Err("draw needs at least three points".into());
            }
            ws.effect = Some(Effect::Draw);
            ws.fill_outline(vol, axes, slice, &flat);
        }
        "level_trace" => {
            let darker = a.opt_bool("darker").unwrap_or(false);
            if let Some(st) = &mut ws.settings {
                st.trace_darker = darker;
            }
            ws.effect = Some(Effect::LevelTracing);
            let voxels = mm_segment::shapes::trace_level(vol, [axes.across, axes.down, axes.slice], seed(vol, first), darker);
            let mut mask = mm_segment::Mask::new(vol.dims);
            for i in voxels {
                mask.data[i] = 1;
            }
            ws.edit_now(vol, mask, crate::segmentation::Mode::Add, "Level tracing", true);
        }
        _ => {
            let st = ws.settings.as_mut().ok_or("No volume")?;
            st.scissors_op = ScissorsOp::ALL.into_iter().find(|o| o.key() == a.str("op").unwrap_or("")).ok_or("op must be one of erase_inside, erase_outside, fill_inside, fill_outside")?;
            if let Some(sh) = a.opt_str("shape") {
                st.scissors_shape = ScissorsShape::ALL.into_iter().find(|x| x.key() == sh).ok_or("shape must be free, circle or rectangle")?;
            }
            if let Some(r) = a.opt_str("range") {
                st.scissors_range = ScissorsRange::ALL.into_iter().find(|x| x.key() == r).ok_or("range must be unlimited, positive, negative or symmetric")?;
            }
            if let Some(t) = a.opt_f64("thickness_mm") {
                st.scissors_mm = t.max(0.01) as f32;
            }
            let s = vol.geometry.spacing;
            let polygon = match st.scissors_shape {
                ScissorsShape::FreeForm => flat.clone(),
                ScissorsShape::Circle => {
                    let r = a.f64("radius_mm")?;
                    mm_segment::shapes::circle(flat[0], r / s[axes.across], s[axes.across] / s[axes.down], 72)
                }
                ScissorsShape::Rectangle => {
                    let (p, q) = (flat[0], *flat.get(1).ok_or("a rectangle needs two corners")?);
                    vec![p, (q.0, p.1), q, (p.0, q.1)]
                }
            };
            if polygon.len() < 3 {
                return Err("a free-form shape needs at least three points".into());
            }
            if matches!(st.scissors_op, ScissorsOp::EraseInside | ScissorsOp::EraseOutside) && ws.seg.as_ref().is_some_and(|x| x.selected.is_none()) {
                return Err("Select a segment first".into());
            }
            ws.effect = Some(Effect::Scissors);
            ws.cut(vol, axes, slice, step, &polygon);
        }
    }
    if let Some((m, true)) = &ws.message {
        return Err(m.clone());
    }
    let after = count(ws);
    let name = ws.seg.as_ref().and_then(|s| s.selected_segment()).map_or(String::new(), |s| s.name.clone());
    Ok(format!("{}: {name} {} by {} voxels", ws.effect.map_or("", |e| e.label()), if after >= before { "grew" } else { "shrank" }, before.abs_diff(after)))
}

/// Effects that run on the whole volume in the background.
fn effect(ws: &mut SegmentationWorkspace, vol: &LoadedVolume, action: &str, a: &Args, ctx: &egui::Context) -> Result<String, String> {
    let in_voxels = a.opt_bool("in_voxels") == Some(true);
    let at = a.opt_point("at").map(|p| voxel(vol, p, in_voxels)).transpose()?.map(|v| seed(vol, v));
    let other = match a.0.get("other").filter(|v| !v.is_null()) {
        Some(v) => {
            let seg = ws.seg.as_ref().ok_or("No segmentation")?;
            Some(seg.segments[find(seg, v)?].label)
        }
        None => None,
    };
    let st = ws.settings.as_mut().ok_or("No volume")?;
    let f32_of = |k: &str| a.opt_f64(k).map(|x| x as f32);
    let (effect, seed) = match action {
        "threshold" => {
            if let Some(m) = a.opt_str("auto") {
                st.auto_method = mm_segment::AutoThreshold::from_key(m).ok_or("unknown auto method")?;
                st.auto_mode = if a.opt_str("auto_mode") == Some("upper") { crate::effects::AutoMode::Upper } else { crate::effects::AutoMode::Lower };
                let hist = ws.hist.as_ref().ok_or("No volume")?;
                st.threshold = hist.auto_range(st.auto_method, st.auto_mode).ok_or("The histogram cannot be split (every voxel alike)")?;
            } else {
                let (lo, hi) = (f32_of("min").unwrap_or(st.threshold.0), f32_of("max").unwrap_or(st.threshold.1));
                st.threshold = (lo.min(hi), lo.max(hi));
            }
            if a.opt_bool("preview") == Some(true) {
                ws.effect = Some(Effect::Threshold);
                return Ok(format!("Threshold preview {} to {} shown; apply applies it", st.threshold.0, st.threshold.1));
            }
            (Effect::Threshold, None)
        }
        "flood_fill" => {
            if let Some(t) = f32_of("tolerance") {
                st.flood_tolerance = t.abs();
            }
            if let Some(u) = a.opt_bool("use_threshold") {
                st.flood_use_threshold = u;
            }
            if let Some(d) = f32_of("min_diameter_mm") {
                st.flood_min_diameter_mm = d.max(0.0);
            }
            if let Some(r) = a.opt_str("refine") {
                st.flood_refine = crate::effects::Refine::ALL.into_iter().find(|x| x.key() == r).ok_or("refine must be none, grow or watershed")?;
            }
            if let Some(f) = f32_of("feature_mm") {
                st.flood_feature_mm = f.max(0.05);
            }
            (Effect::FloodFill, at)
        }
        "islands" => {
            st.islands = IslandOp::ALL.into_iter().find(|o| o.key() == a.str("op").unwrap_or("")).unwrap_or(st.islands);
            if let Some(n) = a.opt_usize("min_size") {
                st.min_island = n;
            }
            if st.islands.by_click() && at.is_none() {
                return Err(format!("islands {} needs `at`: a point in the piece", st.islands.key()));
            }
            (Effect::Islands, at.filter(|_| st.islands.by_click()))
        }
        "margin" => {
            st.margin_mm = a.f64("mm")? as f32;
            st.apply_visible = a.opt_bool("visible").unwrap_or(false);
            (Effect::Margin, None)
        }
        "smooth" => {
            if let Some(m) = a.opt_str("method").and_then(|m| SmoothMethod::ALL.into_iter().find(|x| x.key() == m)) {
                st.smoothing = m;
            }
            if let Some(mm) = f32_of("mm") {
                st.smoothing_mm = mm.max(0.0);
            }
            st.apply_visible = a.opt_bool("visible").unwrap_or(false);
            (Effect::Smoothing, None)
        }
        "hollow" => {
            st.hollow_mm = (a.f64("mm")? as f32).max(0.05);
            if let Some(side) = a.opt_str("side") {
                st.hollow_side = crate::effects::HOLLOW_SIDES.iter().find(|h| h.2 == side).ok_or("side is outside, medial or inside")?.0;
            }
            st.apply_visible = a.opt_bool("visible").unwrap_or(false);
            (Effect::Hollow, None)
        }
        "logical" => {
            st.logical = LogicalOp::ALL.into_iter().find(|o| o.key() == a.str("op").unwrap_or("")).unwrap_or(st.logical);
            if other.is_some() {
                st.logical_other = other;
            }
            st.bypass_masking = a.opt_bool("bypass_masking").unwrap_or(false);
            if st.logical.needs_other() && st.logical_other.is_none() {
                return Err(format!("logical {} needs `other`: the segment to combine with", st.logical.key()));
            }
            (Effect::Logical, None)
        }
        "apply" if ws.effect.is_some_and(|e| e.seeded()) => {
            if ws.seeded.is_none() {
                return Err(format!("{} has no result to apply yet: run it first", ws.effect.map_or("", |e| e.label())));
            }
            ws.apply_seeded();
            return Ok(ws.message.as_ref().map_or(String::new(), |m| m.0.clone()));
        }
        "fast_marching" => {
            st.march_percent = a.f64("percent")?.clamp(0.0001, 100.0) as f32;
            (Effect::FastMarching, None)
        }
        "mask_volume" => {
            if let Some(f) = a.opt_str("fill") {
                st.mask_fill = MaskFill::ALL.into_iter().find(|x| x.key() == f).ok_or("fill must be inside, outside or both")?;
            }
            if let Some(x) = f32_of("inside") {
                st.mask_inside = x;
            }
            if let Some(x) = f32_of("outside") {
                st.mask_outside = x;
            }
            if let Some(x) = f32_of("soft_mm") {
                st.mask_soft_mm = x.max(0.0);
            }
            (Effect::MaskVolume, None)
        }
        "level_set" => {
            let (lo, hi) = (f32_of("min").unwrap_or(st.threshold.0), f32_of("max").unwrap_or(st.threshold.1));
            st.threshold = (lo.min(hi), lo.max(hi));
            if let Some(n) = a.opt_usize("iterations") {
                st.ls_iterations = n.clamp(1, 20_000);
            }
            if let Some(m) = f32_of("max_mm") {
                st.ls_max_mm = m.max(0.1);
            }
            if let Some(c) = f32_of("smoothness") {
                st.ls_curvature = c.max(0.0);
            }
            st.ls_overlay = a.opt_bool("overlay").unwrap_or(false);
            (Effect::LevelSet, None)
        }
        "split_volume" => {
            if let Some(m) = f32_of("margin_mm") {
                st.split_pad_mm = m.max(0.0);
            }
            st.split_blank = a.opt_f64("fill_outside").is_some();
            if let Some(v) = f32_of("fill_outside") {
                st.mask_outside = v;
            }
            (Effect::SplitVolume, None)
        }
        "apply" => match ws.effect {
            Some(e @ (Effect::Paint | Effect::Erase | Effect::FloodFill | Effect::Draw | Effect::LevelTracing | Effect::Scissors | Effect::DrawTube | Effect::SurfaceCut | Effect::GuidedVessel)) => return Err(format!("{} works at points: use its own action", e.label())),
            Some(e) if e == Effect::Islands && st.islands.by_click() => return Err("This islands operation works on a piece: use islands with `at`".into()),
            Some(e) => (e, None),
            None => return Err("No effect is in use".into()),
        },
        _ => return Err(format!("the Segmentation workspace has no action {action}")),
    };
    ws.effect = Some(effect);
    ws.start_op(effect, seed, ctx);
    started(ws, effect.label())
}

/// `go_to` and `display`.
fn view(ws: &mut SegmentationWorkspace, action: &str, a: &Args, vol: &LoadedVolume) -> Result<String, String> {
    let state = ws.state.as_mut().ok_or("No volume")?;
    if action == "go_to" {
        let v = voxel(vol, a.point("at")?, a.opt_bool("in_voxels") == Some(true))?;
        state.cursor = v;
        let ras = vol.geometry.world(v);
        return Ok(format!("Crosshair at voxel ({:.0}, {:.0}, {:.0}), RAS ({:.1}, {:.1}, {:.1}) mm", v.x, v.y, v.z, ras.x, ras.y, ras.z));
    }
    if a.opt_bool("auto_window") == Some(true) {
        state.window = vol.auto_window;
    }
    if let Some(c) = a.opt_f64("window_centre") {
        state.window.0 = c;
    }
    if let Some(w) = a.opt_f64("window_width") {
        state.window.1 = w.max(1e-6);
    }
    if let Some(o) = a.opt_f64("opacity") {
        ws.opacity = o.clamp(0.0, 1.0) as f32;
    }
    if let Some(c) = a.opt_bool("crosshair") {
        ws.crosshair = c;
    }
    if let Some(s) = a.opt_bool("smooth_slices") {
        ws.smooth = s;
    }
    if let Some(m) = a.opt_str("maximize") {
        ws.maximized = ["axial", "3d", "sagittal", "coronal"].iter().position(|n| *n == m);
    }
    if let Some(m) = a.opt_str("colour_map") {
        state.colour_map = polymix_imaging::ColourMap::ALL.into_iter().find(|x| x.key() == m).ok_or("unknown colour map")?;
    }
    if let Some(p) = a.opt_str("window_preset") {
        let (_, c, w) = polymix_imaging::layers::CT_PRESETS.into_iter().find(|x| x.0 == p).ok_or("unknown window preset")?;
        state.window = (c, w);
    }
    if let Some(m) = a.opt_str("slab") {
        state.slab.mode = polymix_imaging::SlabMode::ALL.into_iter().find(|x| x.key() == m).ok_or("slab is off, max, min or mean")?;
    }
    if let Some(mm) = a.opt_f64("slab_mm") {
        state.slab.mm = mm.clamp(0.1, 500.0);
    }
    if a.opt_bool("straighten") == Some(true) {
        state.frames = None;
    }
    if let Some(t) = a.0.get("turn").filter(|t| t.is_object()) {
        let i = ["axial", "sagittal", "coronal"].iter().position(|n| Some(*n) == t.get("view").and_then(Value::as_str)).ok_or("turn.view is axial, sagittal or coronal")?;
        let deg = t.get("degrees").and_then(Value::as_f64).ok_or("turn.degrees is a number")?;
        let mut f = state.frames.unwrap_or(polymix_imaging::PLANES.map(polymix_imaging::oblique::standard_frame));
        let axis = f[i][2];
        for (j, frame) in f.iter_mut().enumerate() {
            if j != i {
                *frame = polymix_imaging::oblique::rotated(*frame, axis, deg.to_radians());
            }
        }
        state.frames = Some(f);
    }
    if let Some(l) = a.opt_str("layout") {
        ws.layout = polymix_imaging::Layout::ALL.into_iter().find(|x| x.key() == l).ok_or("layout is four_up, conventional or row")?;
        ws.maximized = None;
    }
    Ok(format!("Window {:.1} / {:.1}, opacity {:.2}", state.window.0, state.window.1, ws.opacity))
}

/// `overlay`: another image over this one, or compared with it.
fn overlay(ws: &mut SegmentationWorkspace, a: &Args, vol: Arc<LoadedVolume>, ctx: &egui::Context) -> Result<String, String> {
    if a.opt_bool("remove") == Some(true) {
        ws.fusion_job = None;
        ws.state.as_mut().ok_or("No volume")?.fusion = None;
        return Ok("The second image is no longer shown".into());
    }
    let mut started = None;
    if let Some(c) = a.0.get("choice").filter(|c| !c.is_null()) {
        let k = match (c.as_u64(), c.as_str()) {
            (Some(k), _) => k as usize,
            (_, Some(name)) => ws.overlay_choices.iter().position(|(_, l)| l.to_lowercase().contains(&name.to_lowercase())).ok_or(format!("no overlay choice matches {name}"))?,
            _ => return Err("choice is a number or a name".into()),
        };
        if k >= ws.overlay_choices.len() {
            return Err(format!("there are {} overlay choices", ws.overlay_choices.len()));
        }
        started = Some(polymix_workspace::guard::scrub_paths(&ws.overlay_choices[k].1));
        ws.start_overlay(vol, k, ctx);
    }
    let f = ws.state.as_mut().and_then(|s| s.fusion.as_mut());
    match f {
        Some(f) => {
            if let Some(c) = a.opt_str("compare") {
                f.compare = polymix_imaging::Compare::ALL.into_iter().find(|x| x.key() == c).ok_or("compare is blend, wipe or toggle")?;
            }
            if let Some(o) = a.opt_f64("opacity") {
                f.opacity = o.clamp(0.0, 1.0) as f32;
            }
            if let Some(w) = a.opt_f64("wipe") {
                f.wipe = w.clamp(0.0, 1.0) as f32;
            }
            if let Some(s) = a.opt_bool("showing_second") {
                f.showing = s;
            }
            if let Some(m) = a.opt_str("colour_map") {
                f.colour_map = polymix_imaging::ColourMap::ALL.into_iter().find(|x| x.key() == m).ok_or("unknown colour map")?;
            }
            if let Some(c) = a.opt_f64("window_centre") {
                f.window.0 = c;
            }
            if let Some(w) = a.opt_f64("window_width") {
                f.window.1 = w.max(1e-6);
            }
        }
        None if started.is_none() => return Err("No second image is shown: give `choice`".into()),
        None => {}
    }
    Ok(match started {
        Some(name) => format!("Loading {name} to show over the image (app_wait); its settings apply once it is shown"),
        None => "Overlay updated".into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::segmentation::Segmentation;
    use glam::DVec3;
    use mm_volume::geometry::Geometry;
    use mm_volume::load::LoadedVolume;
    use polymix_imaging::ViewState;
    use std::sync::Arc;

    #[test]
    fn segments_with_their_volumes() {
        let mut ws = SegmentationWorkspace::new(&egui::Context::default(), None);
        assert_eq!(state(&mut ws)["volume"], Value::Null);
        // 4 × 4 × 4 voxels of 0.5 × 0.5 × 2 mm (0.5 mm³ each), values 0..64.
        let g = Geometry::from_axes([4, 4, 4, 1], [0.5, 0.5, 2.0], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let vol = LoadedVolume::from_values("t", [4, 4, 4], (0..64).map(|v| v as f32).collect(), g);
        ws.state = Some(ViewState::new(&vol));
        let mut seg = Segmentation::new([4, 4, 4]);
        seg.add_segment(Some("aneurysm".into()));
        for v in &mut seg.labels[..10] {
            *v = 1;
        }
        seg.version += 1;
        ws.seg = Some(seg);
        ws.volume = Some(Arc::new(vol));
        let s = state(&mut ws);
        assert_eq!(s["volume"]["dims"], json!([4, 4, 4]));
        let a = &s["segmentation"]["segments"][0];
        assert_eq!(a["name"], "aneurysm");
        assert_eq!(a["voxels"], 10);
        assert_eq!(a["volume_ml"], 0.005);
        // The cursor starts in the middle voxel (1, 1, 1): value 1 + 4 + 16.
        assert_eq!(s["cursor"]["voxel"], json!([1, 1, 1]));
        assert_eq!(s["cursor"]["value"], 21.0);
        assert_eq!(s["cursor"]["ras_mm"], json!([0.5, 0.5, 2.0]));
    }

    /// A workspace showing an 8 × 8 × 8 volume of 1 mm voxels whose values are 10 · i.
    fn loaded_ws() -> (SegmentationWorkspace, egui::Context) {
        let ctx = egui::Context::default();
        let mut ws = SegmentationWorkspace::new(&ctx, None);
        let g = Geometry::from_axes([8, 8, 8, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::new(100.0, 0.0, 0.0));
        let values = (0..512).map(|n| (n % 8) as f32 * 10.0).collect();
        ws.finish_loading(Ok(LoadedVolume::from_values("t", [8, 8, 8], values, g)), false);
        (ws, ctx)
    }

    fn run(ws: &mut SegmentationWorkspace, ctx: &egui::Context, action: &str, args: Value) -> Result<Value, String> {
        let a = actions().into_iter().find(|a| a.name == action).expect("listed");
        a.check(&args)?;
        let AgentReply::Now(r) = act(ws, action, &args, ctx) else { panic!("answered now") };
        for _ in 0..1000 {
            polymix_workspace::Workspace::tick(ws, ctx, true);
            if ws.running.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        r
    }

    fn voxels(ws: &mut SegmentationWorkspace, label: u8) -> usize {
        ws.seg.as_mut().unwrap().counts()[label as usize]
    }

    #[test]
    fn agents_edit_the_segmentation_through_actions() {
        let (mut ws, ctx) = loaded_ws();
        let r = run(&mut ws, &ctx, "add_segment", json!({"name": "vessel"})).unwrap();
        assert_eq!(r["done"], "Added vessel (label 1)");
        assert_eq!(ws.message.as_ref().unwrap().0, "Agent: Added vessel (label 1)");

        // Threshold i ≥ 3: 5 of 8 columns. The preview applies nothing.
        run(&mut ws, &ctx, "threshold", json!({"min": 30, "max": 70, "preview": true})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 0);
        let r = run(&mut ws, &ctx, "apply", json!({})).unwrap();
        assert_eq!(r["running"], "Threshold");
        assert_eq!(voxels(&mut ws, 1), 5 * 64);
        assert!(ws.message.as_ref().unwrap().0.starts_with("Agent: Threshold: 320 voxels changed"), "{:?}", ws.message);

        // Erase a stroke along x = 5 (RAS mm: the origin is at x = 100) with a ball smaller
        // than a voxel: one voxel per slice.
        let r = run(&mut ws, &ctx, "erase", json!({"points": [[105, 0, 0], [105, 7, 0]], "radius_mm": 0.4, "sphere": true})).unwrap();
        assert_eq!(r["done"], "Erased vessel with a 0.4 mm radius brush: 8 voxels removed");
        assert_eq!(voxels(&mut ws, 1), 312);
        // Paint in voxels into a second segment, picked by name.
        run(&mut ws, &ctx, "add_segment", json!({})).unwrap();
        run(&mut ws, &ctx, "paint", json!({"points": [[0, 0, 0]], "in_voxels": true, "segment": "Segment_1"})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 1);

        // Logical, masking, undo.
        run(&mut ws, &ctx, "logical", json!({"op": "add", "other": "vessel", "segment": 2})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 313);
        assert!(run(&mut ws, &ctx, "logical", json!({"op": "add", "other": "artery"})).unwrap_err().contains("1 vessel, 2 Segment_1"));
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 1);
        run(&mut ws, &ctx, "masking", json!({"editable_area": "inside", "area_segment": "vessel", "intensity_range": [60, 100]})).unwrap();
        assert_eq!(ws.masking.area, EditableArea::Inside(1));
        assert_eq!(ws.masking.intensity, Some((60.0, 100.0)));
        run(&mut ws, &ctx, "masking", json!({"intensity_range": []})).unwrap();
        assert_eq!(ws.masking.intensity, None);

        // Islands needs a point for the click operations; flood fill seeds at one.
        assert!(run(&mut ws, &ctx, "islands", json!({"op": "keep"})).unwrap_err().contains("needs `at`"));
        assert!(run(&mut ws, &ctx, "flood_fill", json!({"at": [0, 0, 0]})).unwrap_err().contains("outside the volume"));

        // Segments: rename, recolour, hide, remove.
        run(&mut ws, &ctx, "set_segment", json!({"segment": 2, "name": "aneurysm", "colour": "#ff8000", "visible": false})).unwrap();
        let s = &ws.seg.as_ref().unwrap().segments[1];
        assert_eq!((s.name.as_str(), s.colour, s.visible), ("aneurysm", egui::Color32::from_rgb(255, 128, 0), false));
        assert!(run(&mut ws, &ctx, "set_segment", json!({"segment": 2, "colour": "orange"})).unwrap_err().contains("#rrggbb"));
        run(&mut ws, &ctx, "remove_segment", json!({"segment": "aneurysm"})).unwrap();
        assert_eq!(ws.seg.as_ref().unwrap().segments.len(), 1);

        // Views.
        let r = run(&mut ws, &ctx, "go_to", json!({"at": [102, 3, 4]})).unwrap();
        assert!(r["done"].as_str().unwrap().starts_with("Crosshair at voxel (2, 3, 4)"), "{r}");
        run(&mut ws, &ctx, "display", json!({"window_centre": 40, "window_width": 20, "maximize": "coronal"})).unwrap();
        assert_eq!((ws.state.as_ref().unwrap().window, ws.maximized), ((40.0, 20.0), Some(3)));

        // Saving: absolute paths, known endings, existing files kept.
        let dir = std::env::temp_dir().join(format!("polymix-seg-agent-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("s.nii.gz");
        assert!(run(&mut ws, &ctx, "save", json!({"path": "s.nii.gz"})).unwrap_err().contains("absolute"));
        assert!(run(&mut ws, &ctx, "save", json!({"path": dir.join("s.stl")})).unwrap_err().contains(".seg.nrrd"));
        run(&mut ws, &ctx, "save", json!({"path": path})).unwrap();
        assert!(path.exists());
        assert!(run(&mut ws, &ctx, "save", json!({"path": path})).unwrap_err().contains("overwrite"));
        std::fs::remove_dir_all(&dir).ok();

        // State reports the settings the actions left.
        let st = state(&mut ws);
        assert_eq!(st["effect_settings"]["threshold"], json!([30.0, 70.0]));
        assert_eq!(st["effect_settings"]["brush_radius_mm"], 0.4);
        assert_eq!(st["effect"], "Logical");
    }

    #[test]
    fn agents_draw_trace_and_cut_in_slices() {
        // Values rise with i (0, 10, … 70); voxels in RAS are (100 + i, j, k).
        let (mut ws, ctx) = loaded_ws();
        // Draw: a 3 × 3 square of voxel centres in axial slice k = 2 (a segment is made).
        run(&mut ws, &ctx, "draw", json!({"points": [[0.5, 0.5, 2], [3.5, 0.5, 2], [3.5, 3.5, 2], [0.5, 3.5, 2]], "in_voxels": true})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 9);
        assert!(run(&mut ws, &ctx, "draw", json!({"points": [[0, 0, 2], [1, 1, 2]], "in_voxels": true})).is_err());
        // Level tracing from i = 5 (value 50) in slice k = 4: the voxels with i ≥ 5 there.
        run(&mut ws, &ctx, "add_segment", json!({})).unwrap();
        run(&mut ws, &ctx, "level_trace", json!({"at": [105, 3, 4]})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 3 * 8);
        run(&mut ws, &ctx, "level_trace", json!({"at": [1, 3, 4], "in_voxels": true, "darker": true})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 3 * 8 + 2 * 8);
        // Scissors: erase the circle of radius 1.5 about (2, 2) in every slice (Unlimited) from
        // segment 1, which loses its 5-voxel cross; then fill a rectangle in a 3-slice slab.
        run(&mut ws, &ctx, "select_segment", json!({"segment": 1})).unwrap();
        run(&mut ws, &ctx, "scissors", json!({"op": "erase_inside", "shape": "circle", "points": [[2, 2, 2]], "in_voxels": true, "radius_mm": 1.2})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 9 - 5);
        run(&mut ws, &ctx, "scissors", json!({"op": "fill_inside", "shape": "rectangle", "points": [[5.5, 5.5, 4], [7.4, 7.4, 4]], "in_voxels": true, "range": "symmetric", "thickness_mm": 2})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 4 + 4 * 3);
        // Erase outside the rectangle in this slice only (positive and negative overlap in it):
        // the slab of slice 4 keeps its rectangle; segment 1 elsewhere is untouched.
        run(&mut ws, &ctx, "scissors", json!({"op": "erase_outside", "shape": "rectangle", "points": [[5.5, 5.5, 4], [7.4, 7.4, 4]], "in_voxels": true, "range": "symmetric", "thickness_mm": 0.5})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 4 + 4 * 3);
        assert_eq!(ws.effect, Some(Effect::Scissors));
        assert!(run(&mut ws, &ctx, "scissors", json!({"op": "bogus", "points": [[1, 1, 1]], "in_voxels": true})).is_err());
    }

    #[test]
    fn visible_segments_together_and_bypassing_the_mask() {
        let (mut ws, ctx) = loaded_ws();
        // Two single voxels in separate segments, far apart.
        run(&mut ws, &ctx, "draw", json!({"points": [[0.5, 0.5, 1], [1.5, 0.5, 1], [1.5, 1.5, 1], [0.5, 1.5, 1]], "in_voxels": true})).unwrap();
        run(&mut ws, &ctx, "add_segment", json!({})).unwrap();
        run(&mut ws, &ctx, "draw", json!({"points": [[5.5, 5.5, 5], [6.5, 5.5, 5], [6.5, 6.5, 5], [5.5, 6.5, 5]], "in_voxels": true})).unwrap();
        assert_eq!((voxels(&mut ws, 1), voxels(&mut ws, 2)), (1, 1));
        // Margin 1 mm on both: each becomes a 7-voxel cross; one undo step takes both back.
        run(&mut ws, &ctx, "margin", json!({"mm": 1.0, "visible": true})).unwrap();
        assert_eq!((voxels(&mut ws, 1), voxels(&mut ws, 2)), (7, 7));
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        assert_eq!((voxels(&mut ws, 1), voxels(&mut ws, 2)), (1, 1));
        // Inside segment 1 only, adding segment 1 to segment 2 does nothing; bypassing the
        // masking it works.
        run(&mut ws, &ctx, "masking", json!({"editable_area": "inside", "area_segment": 1, "overwrite": "none"})).unwrap();
        run(&mut ws, &ctx, "logical", json!({"op": "add", "other": 1, "segment": 2})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 1);
        run(&mut ws, &ctx, "logical", json!({"op": "add", "other": 1, "segment": 2, "bypass_masking": true})).unwrap();
        assert_eq!(voxels(&mut ws, 2), 2);
    }

    #[test]
    fn keys_choose_effects_masking_segments_and_slices() {
        let (mut ws, ctx) = loaded_ws();
        let press = |ws: &mut SegmentationWorkspace, key: egui::Key, shift: bool| {
            let modifiers = egui::Modifiers { shift, ..Default::default() };
            let input = egui::RawInput { events: vec![egui::Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers }], ..Default::default() };
            let _ = ctx.run_ui(input, |ui| ws.keys(ui.ctx()));
        };
        press(&mut ws, egui::Key::Num3, false);
        assert_eq!(ws.effect, Some(Effect::ALL[2]));
        press(&mut ws, egui::Key::Num1, false);
        assert_eq!(ws.effect, Some(Effect::ALL[0]));
        press(&mut ws, egui::Key::Num1, true);
        assert_eq!(ws.effect, Some(Effect::ALL[10]));
        press(&mut ws, egui::Key::Space, false);
        assert_eq!(ws.effect, Some(Effect::ALL[0]), "back to the one before");
        press(&mut ws, egui::Key::I, false);
        assert!(ws.masking.intensity.is_some());
        press(&mut ws, egui::Key::I, false);
        assert!(ws.masking.intensity.is_none());
        run(&mut ws, &ctx, "add_segment", json!({})).unwrap();
        run(&mut ws, &ctx, "add_segment", json!({})).unwrap();
        run(&mut ws, &ctx, "set_segment", json!({"segment": 2, "status": "flagged"})).unwrap();
        assert_eq!(state(&mut ws)["segmentation"]["segments"][1]["status"], "flagged");
        assert!(run(&mut ws, &ctx, "set_segment", json!({"segment": 2, "status": "done"})).is_err());
        run(&mut ws, &ctx, "select_segment", json!({"segment": 1})).unwrap();
        press(&mut ws, egui::Key::W, false);
        assert_eq!(ws.seg.as_ref().unwrap().selected, Some(1));
        press(&mut ws, egui::Key::W, false);
        assert_eq!(ws.seg.as_ref().unwrap().selected, Some(1), "stays on the last");
        press(&mut ws, egui::Key::Q, false);
        assert_eq!(ws.seg.as_ref().unwrap().selected, Some(0));
        // The slice under the pointer: axial here, towards superior (k + 1).
        ws.hover = Some((Plane::Axial, DVec3::new(1.0, 1.0, 3.0)));
        let k0 = ws.state.as_ref().unwrap().cursor.z;
        press(&mut ws, egui::Key::ArrowUp, false);
        assert_eq!(ws.state.as_ref().unwrap().cursor.z, k0 + 1.0);
        press(&mut ws, egui::Key::Escape, false);
        assert_eq!(ws.effect, None);
    }

    #[test]
    fn seeds_grow_march_and_tubes() {
        // Values rise with i (0 … 70): seeds at i = 0 and i = 7 split the grid where the values
        // meet, both growing over their halves.
        let (mut ws, ctx) = loaded_ws();
        run(&mut ws, &ctx, "add_segment", json!({"name": "dark"})).unwrap();
        run(&mut ws, &ctx, "paint", json!({"points": [[0, 4, 4]], "in_voxels": true, "radius_mm": 0.5})).unwrap();
        run(&mut ws, &ctx, "add_segment", json!({"name": "bright"})).unwrap();
        run(&mut ws, &ctx, "paint", json!({"points": [[7, 4, 4]], "in_voxels": true, "radius_mm": 0.5})).unwrap();
        let r = run(&mut ws, &ctx, "grow_from_seeds", json!({"margin_mm": 20})).unwrap();
        assert_eq!(r["running"], "Grow from seeds");
        assert!(ws.seeded.is_some() && voxels(&mut ws, 1) == 1, "a preview only");
        run(&mut ws, &ctx, "apply", json!({})).unwrap();
        let (d, b) = (voxels(&mut ws, 1), voxels(&mut ws, 2));
        assert_eq!(d + b, 512, "every voxel of the region is given to a seed");
        assert!(d >= 4 * 64 - 64 && b >= 3 * 64, "{d} / {b}");
        // Watershed applied at once.
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        run(&mut ws, &ctx, "watershed", json!({"object_scale_mm": 0, "apply": true})).unwrap();
        assert_eq!(voxels(&mut ws, 1) + voxels(&mut ws, 2), 512);
        // One seed segment is not enough.
        run(&mut ws, &ctx, "set_segment", json!({"segment": 2, "visible": false})).unwrap();
        run(&mut ws, &ctx, "grow_from_seeds", json!({})).unwrap();
        assert!(ws.message.as_ref().unwrap().1, "{:?}", ws.message);
        // Pixel mode: a stroke along i paints just the voxels it passes, whatever the radius.
        run(&mut ws, &ctx, "add_segment", json!({"name": "pixels"})).unwrap();
        ws.settings.as_mut().unwrap().pixel_mode = true;
        let px = ws.seg.as_ref().unwrap().selected_segment().unwrap().label;
        run(&mut ws, &ctx, "paint", json!({"points": [[1, 6, 6], [5, 6, 6]], "in_voxels": true, "radius_mm": 10})).unwrap();
        assert_eq!(voxels(&mut ws, px), 5);
        ws.settings.as_mut().unwrap().pixel_mode = false;
        // Fast marching grows a seed to 10 % of the volume.
        run(&mut ws, &ctx, "add_segment", json!({"name": "march"})).unwrap();
        run(&mut ws, &ctx, "paint", json!({"points": [[3, 3, 3]], "in_voxels": true, "radius_mm": 0.5})).unwrap();
        run(&mut ws, &ctx, "masking", json!({"overwrite": "all"})).unwrap();
        run(&mut ws, &ctx, "fast_marching", json!({"percent": 10})).unwrap();
        let label = |ws: &SegmentationWorkspace, name: &str| ws.seg.as_ref().unwrap().segments.iter().find(|s| s.name == name).unwrap().label;
        let march = label(&ws, "march");
        assert!((voxels(&mut ws, march) as i64 - 51).abs() <= 1, "{}", voxels(&mut ws, march));
        // A tube along j at i = 4, k = 4, radius 1.2: 5 voxels per slice across it.
        run(&mut ws, &ctx, "add_segment", json!({"name": "tube"})).unwrap();
        run(&mut ws, &ctx, "draw_tube", json!({"points": [[4, 1, 4], [4, 6, 4]], "in_voxels": true, "radius_mm": 1.2})).unwrap();
        let tube = label(&ws, "tube");
        assert_eq!(voxels(&mut ws, tube), 6 * 5 + 2);
        // Surface cut: a cube's corners give the cube.
        run(&mut ws, &ctx, "add_segment", json!({"name": "cut"})).unwrap();
        let corners: Vec<[f64; 3]> = (0..8).map(|c| [if c & 1 == 0 { 1.0 } else { 3.0 }, if c & 2 == 0 { 1.0 } else { 3.0 }, if c & 4 == 0 { 5.0 } else { 7.0 }]).collect();
        run(&mut ws, &ctx, "surface_cut", json!({"points": corners, "in_voxels": true})).unwrap();
        let cut = label(&ws, "cut");
        assert_eq!(voxels(&mut ws, cut), 27);
        assert!(run(&mut ws, &ctx, "surface_cut", json!({"points": [[1, 1, 1], [2, 1, 1], [1, 2, 1]], "in_voxels": true})).is_err());
    }

    #[test]
    fn automatic_thresholds_and_filling_between_slices() {
        let (mut ws, ctx) = loaded_ws();
        // Values 0 … 70 in eight even steps: every method splits them somewhere inside.
        for m in mm_segment::AutoThreshold::ALL {
            run(&mut ws, &ctx, "threshold", json!({"auto": m.key(), "preview": true})).unwrap();
            let (lo, hi) = ws.settings.as_ref().unwrap().threshold;
            assert!(lo > 0.0 && lo <= 70.0 && hi == 70.0, "{}: {lo}..{hi}", m.key());
        }
        run(&mut ws, &ctx, "threshold", json!({"auto": "otsu", "auto_mode": "upper", "preview": true})).unwrap();
        let (lo, hi) = ws.settings.as_ref().unwrap().threshold;
        assert!(lo == 0.0 && hi < 70.0);
        assert!(run(&mut ws, &ctx, "threshold", json!({"auto": "magic"})).is_err());
        // Squares drawn on axial slices 1 and 5 are filled into slices 2–4.
        run(&mut ws, &ctx, "draw", json!({"points": [[1.5, 1.5, 1], [4.5, 1.5, 1], [4.5, 4.5, 1], [1.5, 4.5, 1]], "in_voxels": true})).unwrap();
        run(&mut ws, &ctx, "draw", json!({"points": [[1.5, 1.5, 5], [4.5, 1.5, 5], [4.5, 4.5, 5], [1.5, 4.5, 5]], "in_voxels": true})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 18);
        run(&mut ws, &ctx, "fill_between_slices", json!({"apply": true})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 5 * 9);
        // Between sagittal slices nothing is empty in between (each sagittal slice through
        // the squares holds voxels): nothing more.
        run(&mut ws, &ctx, "fill_between_slices", json!({"plane": "sagittal", "apply": true})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 5 * 9);
    }

    #[test]
    fn masked_volumes_are_saved_for_the_volume_workspace() {
        // Saved into a temporary data folder, never the user's.
        let dir = std::env::temp_dir().join(format!("polymix-mask-test-{}", std::process::id()));
        std::env::set_var("POLYMIX_DATA_DIR", &dir);
        let (mut ws, ctx) = loaded_ws();
        run(&mut ws, &ctx, "draw", json!({"points": [[0.5, 0.5, 2], [3.5, 0.5, 2], [3.5, 3.5, 2], [0.5, 3.5, 2]], "in_voxels": true})).unwrap();
        run(&mut ws, &ctx, "mask_volume", json!({"fill": "both", "inside": 1, "outside": 0})).unwrap();
        let paths = ws.take_volumes();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].starts_with(&dir) && paths[0].to_string_lossy().ends_with(" masked.nii.gz"), "{:?}", paths[0]);
        assert!(ws.message.as_ref().unwrap().0.contains("derived images"));
        // Split volume: the drawn square (voxels 1…3 of slice 2) with a 1 mm margin, blank
        // outside it.
        run(&mut ws, &ctx, "split_volume", json!({"margin_mm": 1, "fill_outside": -5})).unwrap();
        let paths = ws.take_volumes();
        assert_eq!(paths.len(), 1, "{:?}", ws.message);
        let size = std::fs::metadata(&paths[0]).unwrap().len();
        let mm_volume::scan::Found::Volume(v) = mm_volume::scan::classify(&paths[0], size) else { panic!("not an image") };
        let part = mm_volume::load::load_volume(&v, 1 << 30, &|_| true).unwrap();
        assert_eq!(part.dims, [5, 5, 3]);
        assert_eq!(part.values.iter().filter(|&&v| v != -5.0).count(), 3 * 3);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn without_a_volume_agents_hear_how_to_get_one() {
        let ctx = egui::Context::default();
        let mut ws = SegmentationWorkspace::new(&ctx, None);
        let AgentReply::Now(Err(e)) = act(&mut ws, "add_segment", &json!({}), &ctx) else { panic!() };
        assert!(e.contains("volume_do select"), "{e}");
    }

    /// A 32³ volume of 1 mm voxels, 100 inside a ball of radius 8 about the middle and 0
    /// elsewhere, with segment 1 (“ball”) the ball.
    fn ball_ws() -> (SegmentationWorkspace, egui::Context) {
        let ctx = egui::Context::default();
        let mut ws = SegmentationWorkspace::new(&ctx, None);
        let g = Geometry::from_axes([32, 32, 32, 1], [1.0; 3], [DVec3::X, DVec3::Y, DVec3::Z], DVec3::ZERO);
        let inside = |n: usize| DVec3::new((n % 32) as f64, ((n / 32) % 32) as f64, (n / 1024) as f64).distance(DVec3::splat(16.0)) <= 8.0;
        let values = (0..32 * 32 * 32).map(|n| if inside(n) { 100.0 } else { 0.0 }).collect();
        ws.finish_loading(Ok(LoadedVolume::from_values("t", [32, 32, 32], values, g)), false);
        run(&mut ws, &ctx, "add_segment", json!({"name": "ball"})).unwrap();
        let seg = ws.seg.as_mut().unwrap();
        for (n, l) in seg.labels.iter_mut().enumerate() {
            *l = inside(n) as u8;
        }
        seg.changed(true);
        (ws, ctx)
    }

    #[test]
    fn statistics_hollow_and_smoothing() {
        let (mut ws, ctx) = ball_ws();
        let ball = voxels(&mut ws, 1);
        let AgentReply::Later(measure) = call(&mut ws, "segmentation_statistics", &json!({}), &ctx) else { panic!("measured on a worker thread") };
        let st = measure().unwrap();
        let s = &st["segments"][0]["stats"];
        let exact = 4.0 / 3.0 * std::f64::consts::PI * 512.0 / 1000.0;
        assert!((s["volume_ml"].as_f64().unwrap() / exact - 1.0).abs() < 0.03, "{s}");
        assert_eq!((s["mean"].as_f64(), s["sd"].as_f64()), (Some(100.0), Some(0.0)));
        assert!((s["max_diameter_mm"].as_f64().unwrap() - 16.0).abs() <= 1.0, "{s}");
        assert!(s["roundness"].as_f64().unwrap() > 0.9, "{s}");
        // The table in the panel, measured in the background.
        ws.measure(&ctx);
        for _ in 0..1000 {
            polymix_workspace::Workspace::tick(&mut ws, &ctx, true);
            if ws.running.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let csv = crate::stats_csv(&ws.stats.as_ref().unwrap().1, ws.seg.as_ref().unwrap());
        assert!(csv.lines().nth(1).unwrap().starts_with(&format!("ball,1,{ball},")), "{csv}");

        // Hollow 2 mm outside: a shell around the ball, none of its inside left.
        run(&mut ws, &ctx, "hollow", json!({"mm": 2.0, "side": "outside"})).unwrap();
        let shell = voxels(&mut ws, 1);
        let expected = 4.0 / 3.0 * std::f64::consts::PI * (10f64.powi(3) - 8f64.powi(3));
        assert!((shell as f64 / expected - 1.0).abs() < 0.15, "{shell} vs {expected:.0}");
        let seg = ws.seg.as_ref().unwrap();
        assert_eq!(seg.labels[(16 * 32 + 16) * 32 + 16], 0, "the middle is hollow");
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        assert_eq!(voxels(&mut ws, 1), ball);

        // Gaussian smoothing keeps a ball a ball.
        run(&mut ws, &ctx, "smooth", json!({"method": "gaussian", "mm": 1.0})).unwrap();
        let smoothed = voxels(&mut ws, 1) as f64;
        assert!((smoothed / ball as f64 - 1.0).abs() < 0.05, "{smoothed} vs {ball}");

        // Joint smoothing: a box overlapping the ball's side; neither is lost, no gap opens
        // between them, and a hidden segment is left as it was.
        run(&mut ws, &ctx, "add_segment", json!({"name": "box"})).unwrap();
        run(&mut ws, &ctx, "add_segment", json!({"name": "hidden"})).unwrap();
        let seg = ws.seg.as_mut().unwrap();
        for (n, l) in seg.labels.iter_mut().enumerate() {
            let [i, j, k] = [n % 32, (n / 32) % 32, n / 1024];
            if (20..28).contains(&i) && (12..20).contains(&j) && (12..20).contains(&k) {
                *l = 2;
            } else if i < 3 {
                *l = 3;
            }
        }
        seg.changed(true);
        seg.segments[2].visible = false;
        let before: Vec<u8> = seg.labels.clone();
        run(&mut ws, &ctx, "smooth", json!({"method": "joint", "mm": 1.5})).unwrap();
        let seg = ws.seg.as_ref().unwrap();
        assert!(before.iter().zip(&seg.labels).all(|(&a, &b)| (a == 3) == (b == 3)), "the hidden segment changed");
        // Through the ball into the box nothing is empty; smoothing at σ = 1.5 mm shrinks such
        // small shapes (by about σ² / radius), not more.
        assert!((10..27).all(|i| seg.labels[(16 * 32 + 16) * 32 + i] != 0), "a gap opened");
        assert!(seg.labels[(16 * 32 + 16) * 32 + 12] == 1 && seg.labels[(16 * 32 + 16) * 32 + 24] == 2);
        let filled = |l: &[u8]| l.iter().filter(|&&x| x == 1 || x == 2).count() as f64;
        assert!((filled(&seg.labels) / filled(&before) - 1.0).abs() < 0.2, "{} vs {}", filled(&seg.labels), filled(&before));
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        assert_eq!(ws.seg.as_ref().unwrap().labels, before, "one undo step");
    }

    #[test]
    fn segments_from_files_and_meshes() {
        let (mut ws, ctx) = ball_ws();
        let dir = std::env::temp_dir().join(format!("polymix-seg-load-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        // Save, then load the file back: a second segment of the same voxels, its name, colour
        // and status kept.
        ws.seg.as_mut().unwrap().segments[0].status = crate::segmentation::Status::Completed;
        let path = dir.join("ball.seg.nrrd");
        run(&mut ws, &ctx, "save", json!({"path": path.to_string_lossy()})).unwrap();
        let r = run(&mut ws, &ctx, "load_segmentation", json!({"path": path.to_string_lossy()})).unwrap();
        assert_eq!(r["running"], "Loading segmentation");
        let seg = ws.seg.as_ref().unwrap();
        assert_eq!(seg.segments.len(), 2, "{:?}", ws.message);
        assert_eq!((seg.segments[1].name.as_str(), seg.segments[1].status), ("ball", crate::segmentation::Status::Completed));
        assert_eq!(seg.segments[1].colour, seg.segments[0].colour);
        let ball = ws.seg.as_ref().unwrap().labels.iter().filter(|&&l| l == 2).count();
        assert!(ball > 2000, "the loaded segment took the ball's voxels ({ball})");
        assert!(run(&mut ws, &ctx, "load_segmentation", json!({"path": "relative.nii"})).is_err());

        // A mesh from the Mesh tab (LPS millimetres, as Slicer's models): a cube at RAS 2–10.
        let mut cube = mm_core::primitives::cube(8.0, 4);
        for v in cube.vertex_ids().collect::<Vec<_>>() {
            cube.set_vertex(v, cube.vertex(v) + DVec3::new(-6.0, -6.0, 6.0));
        }
        ws.import_mesh("Cube".into(), cube, &ctx);
        for _ in 0..1000 {
            polymix_workspace::Workspace::tick(&mut ws, &ctx, true);
            if ws.running.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let seg = ws.seg.as_mut().unwrap();
        assert_eq!(seg.segments.last().unwrap().name, "Cube", "{:?}", ws.message);
        let label = seg.segments.last().unwrap().label;
        let n = seg.counts()[label as usize];
        assert!((400..=800).contains(&n), "8 mm cube: {n} voxels");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn threshold_range_from_a_local_histogram() {
        let (mut ws, ctx) = loaded_ws();
        // Values are 10 · i: a box over i = 2…5 holds 20 to 50.
        let r = run(&mut ws, &ctx, "local_histogram", json!({"points": [[1.6, 1.6, 4], [5.4, 5.4, 4]], "in_voxels": true})).unwrap();
        assert!(r["done"].as_str().unwrap().starts_with("Local histogram of 16 voxels"), "{r}");
        assert_eq!(ws.settings.as_ref().unwrap().threshold, (20.0, 50.0));
        assert_eq!(ws.effect, Some(Effect::Threshold));
        run(&mut ws, &ctx, "local_histogram", json!({"points": [[1.6, 1.6, 4], [5.4, 5.4, 4]], "in_voxels": true, "lower": "average"})).unwrap();
        assert_eq!(ws.settings.as_ref().unwrap().threshold, (35.0, 50.0));
        // The choices stay; a line across i = 1…6, and a circle of radius 1.2 about i = 3 (i = 2…4).
        run(&mut ws, &ctx, "local_histogram", json!({"shape": "line", "points": [[1, 3, 4], [6, 3, 4]], "in_voxels": true})).unwrap();
        assert_eq!(ws.settings.as_ref().unwrap().threshold, (35.0, 60.0));
        run(&mut ws, &ctx, "local_histogram", json!({"shape": "line", "points": [[1, 3, 4], [6, 3, 4]], "in_voxels": true, "lower": "extreme"})).unwrap();
        assert_eq!(ws.settings.as_ref().unwrap().threshold, (10.0, 60.0));
        assert!(ws.local.as_ref().unwrap().line);
        run(&mut ws, &ctx, "local_histogram", json!({"shape": "circle", "points": [[3, 3, 4]], "radius_mm": 1.2, "in_voxels": true})).unwrap();
        assert_eq!(ws.settings.as_ref().unwrap().threshold, (20.0, 40.0));
        // Then Apply thresholds with it; leaving Threshold drops the region.
        run(&mut ws, &ctx, "add_segment", json!({"name": "mid"})).unwrap();
        run(&mut ws, &ctx, "apply", json!({})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 3 * 64);
        ws.effect = Some(Effect::Margin);
        ws.view_input(&ws.volume.clone().unwrap(), &[], &ctx);
        assert!(ws.local.is_none());
        assert!(run(&mut ws, &ctx, "local_histogram", json!({"shape": "box", "points": [[3, 3, 4]], "in_voxels": true})).is_err());
    }

    #[test]
    fn effects_at_points_picked_in_the_3d_view() {
        let (mut ws, ctx) = loaded_ws();
        let vol = ws.volume.clone().unwrap();
        run(&mut ws, &ctx, "add_segment", json!({"name": "a"})).unwrap();
        ws.settings.as_mut().unwrap().brush_mm = 1.0;
        ws.effect = Some(Effect::Paint);
        // A click paints one ball; a ⌘-drag over three frames is one stroke, one undo step.
        ws.input_3d(&vol, Some(Effect::Paint), Some(DVec3::splat(2.0)), false, true, &ctx);
        assert_eq!(voxels(&mut ws, 1), 7, "a ball of radius 1 mm");
        for (x, stroking) in [(5.0, true), (6.0, true), (6.0, false)] {
            let picked = stroking.then(|| DVec3::new(x, 5.0, 5.0));
            ws.view_input(&vol, &[], &ctx);
            ws.input_3d(&vol, Some(Effect::Paint), picked, stroking, false, &ctx);
        }
        assert_eq!(voxels(&mut ws, 1), 7 + 12);
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        assert_eq!(voxels(&mut ws, 1), 7);
        // Islands: a click on the ball keeps only it; Draw tube takes the point.
        ws.settings.as_mut().unwrap().islands = IslandOp::RemoveSelected;
        ws.input_3d(&vol, Some(Effect::Islands), Some(DVec3::splat(2.0)), false, true, &ctx);
        for _ in 0..1000 {
            polymix_workspace::Workspace::tick(&mut ws, &ctx, true);
            if ws.running.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert_eq!(voxels(&mut ws, 1), 0);
        ws.input_3d(&vol, Some(Effect::DrawTube), Some(DVec3::splat(3.0)), false, true, &ctx);
        assert_eq!(ws.markers.len(), 1);
    }

    #[test]
    fn another_image_over_this_one_and_display_settings() {
        let (mut ws, ctx) = loaded_ws();
        let vol = ws.volume.clone().unwrap();
        // A second image on the same grid (twice the values), saved and found as the catalog would.
        let dir = std::env::temp_dir().join(format!("polymix-overlay-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("double.nii");
        let doubled: Vec<f32> = vol.values.iter().map(|v| v * 2.0).collect();
        mm_volume::write::nifti(&path, mm_volume::write::Data::F32(&doubled), vol.dims, &vol.geometry, "", mm_volume::write::Intent::None).unwrap();
        let size = std::fs::metadata(&path).unwrap().len();
        let mm_volume::scan::Found::Volume(v) = mm_volume::scan::classify(&path, size) else { panic!() };
        ws.set_overlay_choices(vec![(*v, "double.nii".into())]);
        assert_eq!(state(&mut ws)["overlay_choices"], json!(["double.nii"]));
        assert!(run(&mut ws, &ctx, "overlay", json!({"opacity": 0.3})).is_err(), "nothing shown yet");
        run(&mut ws, &ctx, "overlay", json!({"choice": "DOUBLE"})).unwrap();
        for _ in 0..1000 {
            polymix_workspace::Workspace::tick(&mut ws, &ctx, true);
            if ws.fusion_job.is_none() {
                break;
            }
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        let f = ws.state.as_ref().unwrap().fusion.as_ref().expect("shown");
        assert_eq!(*f.values, doubled);
        run(&mut ws, &ctx, "overlay", json!({"compare": "wipe", "wipe": 0.3, "colour_map": "green"})).unwrap();
        let s = state(&mut ws);
        assert_eq!((s["overlay"]["compare"].as_str(), s["overlay"]["wipe"].as_f64(), s["overlay"]["colour_map"].as_str()), (Some("wipe"), Some(0.3), Some("green")));
        // Display: colour map, a CT window, a MIP slab.
        run(&mut ws, &ctx, "display", json!({"colour_map": "hot", "window_preset": "CT bone", "slab": "max", "slab_mm": 3})).unwrap();
        let st = ws.state.as_ref().unwrap();
        assert_eq!((st.colour_map, st.window, st.slab.mode, st.slab.mm), (polymix_imaging::ColourMap::Hot, (400.0, 1800.0), polymix_imaging::SlabMode::Max, 3.0));
        run(&mut ws, &ctx, "overlay", json!({"remove": true})).unwrap();
        assert!(ws.state.as_ref().unwrap().fusion.is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn centerlines_and_measurements() {
        let (mut ws, ctx) = ball_ws();
        // A tube along x (radius 4) narrowed to radius 2 in the middle, as segment 2.
        run(&mut ws, &ctx, "add_segment", json!({"name": "sinus"})).unwrap();
        let seg = ws.seg.as_mut().unwrap();
        for (n, l) in seg.labels.iter_mut().enumerate() {
            let [i, j, k] = [n % 32, (n / 32) % 32, n / 1024];
            let r = if (13..19).contains(&i) { 2.0 } else { 4.0 };
            *l = if ((j as f64 - 16.0).powi(2) + (k as f64 - 16.0).powi(2)).sqrt() <= r && (2..30).contains(&i) { 2 } else { 0 };
        }
        seg.changed(true);
        run(&mut ws, &ctx, "centerline", json!({"segment": "sinus"})).unwrap();
        let AgentReply::Now(Ok(c)) = call(&mut ws, "segmentation_centerline", &json!({"sections": true}), &ctx) else { panic!("{:?}", ws.message) };
        assert_eq!(c["segment"], "sinus");
        let b = &c["branches"][0];
        assert!(b["length_mm"].as_f64().unwrap() > 15.0, "{b}");
        let n = &b["narrowest"];
        assert!(n["narrowing_percent"].as_f64().unwrap() > 30.0, "{n}");
        let x = n["ras_mm"][0].as_f64().unwrap();
        assert!((12.0..=20.0).contains(&x), "the narrowing is in the middle: {x}");
        assert!(b["sections"].as_array().unwrap().len() > 20);
        // To the narrowest, the planes turned with the tube (left to right): the sagittal view
        // across it.
        let r = run(&mut ws, &ctx, "go_to_section", json!({})).unwrap();
        assert!(r["done"].as_str().unwrap().contains("along branch 1"), "{r}");
        let f = ws.state.as_ref().unwrap().frames.expect("oblique");
        assert!(f[1][2].x.abs() > 0.95, "across the tube: {:?}", f[1][2]);
        assert!((12.0..=20.0).contains(&ws.state.as_ref().unwrap().cursor.x));
        run(&mut ws, &ctx, "display", json!({"straighten": true})).unwrap();
        assert!(ws.state.as_ref().unwrap().frames.is_none());
        run(&mut ws, &ctx, "display", json!({"turn": {"view": "axial", "degrees": 30}})).unwrap();
        let f = ws.state.as_ref().unwrap().frames.unwrap();
        assert_eq!(f[0], polymix_imaging::oblique::standard_frame(Plane::Axial), "the view turned about stays");
        assert!((f[1][2].dot(DVec3::X) - 30f64.to_radians().cos()).abs() < 1e-9, "the sagittal plane turned 30°");
        // Measurements: a ruler across the tube, a region in it.
        let r = run(&mut ws, &ctx, "measure", json!({"tool": "ruler", "points": [[5, 12, 16], [5, 20, 16]], "in_voxels": true})).unwrap();
        assert_eq!(r["done"], "Ruler: 8.00 mm");
        run(&mut ws, &ctx, "measure", json!({"tool": "region", "points": [[16, 16, 16], [16, 18, 16]], "in_voxels": true, "plane": "sagittal"})).unwrap();
        let s = state(&mut ws);
        assert_eq!(s["measurements"].as_array().unwrap().len(), 2);
        assert!(run(&mut ws, &ctx, "measure", json!({"tool": "angle", "points": [[1, 1, 1]], "in_voxels": true})).is_err());
        run(&mut ws, &ctx, "measure", json!({"clear": true})).unwrap();
        assert!(ws.measure.items.is_empty());
    }

    #[test]
    fn level_sets_and_guided_vessels() {
        // The ball (value 100, radius 8) from a seed of 7 voxels at its middle.
        let (mut ws, ctx) = ball_ws();
        let ball = voxels(&mut ws, 1);
        let seg = ws.seg.as_mut().unwrap();
        for (n, l) in seg.labels.iter_mut().enumerate() {
            let p = DVec3::new((n % 32) as f64, ((n / 32) % 32) as f64, (n / 1024) as f64);
            *l = (p.distance(DVec3::splat(16.0)) <= 1.0) as u8;
        }
        seg.changed(true);
        run(&mut ws, &ctx, "level_set", json!({"min": 50, "max": 150, "iterations": 600, "max_mm": 12})).unwrap();
        let grown = voxels(&mut ws, 1) as f64;
        assert!((grown / ball as f64 - 1.0).abs() < 0.1, "{grown} vs {ball}: {:?}", ws.message);
        // Guided: points across the ball into a second segment.
        run(&mut ws, &ctx, "add_segment", json!({"name": "guided"})).unwrap();
        run(&mut ws, &ctx, "guided_vessel", json!({"points": [[10, 16, 16], [22, 16, 16]], "in_voxels": true, "radius_mm": 4, "min": 50, "max": 150})).unwrap();
        let g = voxels(&mut ws, 2);
        assert!(g > 500, "guided: {g} voxels, {:?}", ws.message);
        assert!(run(&mut ws, &ctx, "apply", json!({})).is_err(), "points effects have their own action");
    }

    #[test]
    fn scissors_through_the_3d_view() {
        let (mut ws, ctx) = ball_ws();
        let vol = ws.volume.clone().unwrap();
        let ball = voxels(&mut ws, 1);
        ws.settings.as_mut().unwrap().scissors_op = ScissorsOp::EraseInside;
        let rect = egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(400.0, 300.0));
        // A small square at the middle of the view: a tunnel through the ball, front to back.
        let c = rect.center();
        let square = [c + egui::vec2(-20.0, -20.0), c + egui::vec2(20.0, -20.0), c + egui::vec2(20.0, 20.0), c + egui::vec2(-20.0, 20.0)];
        let project = ws.view3d.projector(&vol, rect);
        ws.cut_3d(&vol, &square, project);
        let left = voxels(&mut ws, 1);
        assert!(left < ball && left > ball / 2, "{left} of {ball}");
        let seg = ws.seg.as_ref().unwrap();
        assert_eq!(seg.labels[(16 * 32 + 16) * 32 + 16], 0, "the middle is cut out");
        // Erase outside a square covering the whole view: everything shown stays.
        run(&mut ws, &ctx, "undo", json!({})).unwrap();
        ws.settings.as_mut().unwrap().scissors_op = ScissorsOp::EraseOutside;
        let all = [rect.left_top(), rect.right_top(), rect.right_bottom(), rect.left_bottom()];
        let project = ws.view3d.projector(&vol, rect);
        ws.cut_3d(&vol, &all, project);
        assert_eq!(voxels(&mut ws, 1), ball);
    }

    #[test]
    fn engraving_text() {
        let (mut ws, ctx) = ball_ws();
        let ball = voxels(&mut ws, 1);
        // "8" cut 2 mm deep into the ball's middle slice: its loops' insides stay.
        let r = run(&mut ws, &ctx, "engrave", json!({"text": "8", "at": [16, 16, 16], "in_voxels": true, "height_mm": 10, "depth_mm": 1})).unwrap();
        assert!(r["done"].as_str().unwrap().contains("Engrave"), "{r}");
        let cut = ball - voxels(&mut ws, 1);
        assert!(cut > 30 && cut < 400, "{cut} (three slices of a 10 mm 8)");
        // "L" raised in empty space on another segment: on screen (radiological axial) its
        // upright stroke is on the left, the patient's right (+x), and its foot at the bottom,
        // posterior (−y).
        run(&mut ws, &ctx, "add_segment", json!({"name": "label"})).unwrap();
        run(&mut ws, &ctx, "engrave", json!({"text": "L", "at": [16, 16, 2], "in_voxels": true, "height_mm": 24, "depth_mm": 1, "emboss": true})).unwrap();
        let seg = ws.seg.as_ref().unwrap();
        let pts: Vec<(f64, f64)> = (0..32 * 32).filter(|&n| seg.labels[2 * 1024 + n] == 2).map(|n| ((n % 32) as f64, (n / 32) as f64)).collect();
        assert!(pts.len() > 20, "{}", pts.len());
        let (ylo, yhi) = pts.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(a, b), p| (a.min(p.1), b.max(p.1)));
        let mean_x = |y: f64| {
            let row: Vec<f64> = pts.iter().filter(|p| (p.1 - y).abs() < 0.5).map(|p| p.0).collect();
            row.iter().sum::<f64>() / row.len().max(1) as f64
        };
        assert!(mean_x(yhi) > 17.0, "the upright at the top is on the patient's right: {}", mean_x(yhi));
        let foot: Vec<f64> = pts.iter().filter(|p| (p.1 - ylo).abs() < 0.5).map(|p| p.0).collect();
        assert!(foot.len() > 3, "the foot runs across the bottom (posterior)");
        assert!(run(&mut ws, &ctx, "engrave", json!({"text": "é", "at": [16, 16, 2], "in_voxels": true})).is_err());
    }
}
