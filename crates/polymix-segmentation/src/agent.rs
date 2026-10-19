//! The Segmentation workspace for agents: `segmentation_state` (what it shows, the segments
//! in it, the effects' settings) and the actions the shell offers as `segmentation_do` (what
//! the editor panel and the views do: segments, effects, masking, undo, meshes, saving).
//!
//! The volume is not named here (its name can carry a patient's): `volume_catalog` says which
//! volume is selected in the Volume workspace, which is the one shown here.

use crate::effects::{Effect, IslandOp, LogicalOp, SmoothMethod};
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
    vec![AgentTool::new(
        "segmentation_state",
        "The Segmentation workspace: the volume shown (grid, spacing, orientation, value range, window), the crosshair (voxel, RAS mm, value), the segments (label, name, colour, visibility, voxel count and volume in mL), the selected segment and effect, masking, and any loading, running operation or message. The volume shown is the one selected in the Volume workspace (see volume_catalog).",
        json!({}),
    )]
}

pub fn call(ws: &mut SegmentationWorkspace, tool: &str, _args: &Value, _ctx: &egui::Context) -> AgentReply {
    match tool {
        "segmentation_state" => AgentReply::Now(Ok(state(ws))),
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
            "logical": st.logical.key(),
            "logical_other": st.logical_other,
            "mesh_smoothing_mm": st.mesh_smoothing_mm,
            "mesh_all_visible": st.mesh_all_visible,
        })),
        "undo": ws.seg.as_ref().map(|s| json!({"can_undo": s.can_undo(), "can_redo": s.can_redo()})),
        "display": {"opacity": round(ws.opacity as f64, 2), "crosshair": ws.crosshair, "smooth_slices": ws.smooth},
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
        Action::new("set_segment", "Rename, recolour, show or hide a segment", json!({"segment": segment("Label number or name"), "name": arg::text("New name"), "colour": arg::text("New colour, #rrggbb"), "visible": arg::boolean("Shown in the views")})).required(&["segment"]),
        Action::new("remove_segment", "Remove a segment and its voxels", json!({"segment": segment("Label number or name")})).required(&["segment"]).destructive(),
        Action::new(
            "threshold",
            "The segment becomes the voxels with values from min to max (default: the current range, which starts at Otsu's threshold). Makes a segment if there is none. With preview, the views only show the result until apply",
            json!({"min": arg::number("Lowest value"), "max": arg::number("Highest value"), "preview": arg::boolean("Show the result without applying it"), "segment": target()}),
        ),
        Action::new("paint", "Paint the segment with the brush along the points (a segment is made if none is selected)", brush()).required(&["points"]),
        Action::new("erase", "Erase the segment with the brush along the points", brush()).required(&["points"]),
        Action::new(
            "flood_fill",
            "Add the voxels connected to `at` whose values are within `tolerance` of its value (or in the threshold range with use_threshold), not passing gaps narrower than min_diameter_mm. Makes a segment if there is none",
            json!({"at": arg::point("The seed (RAS mm, or a voxel with in_voxels)"), "in_voxels": in_voxels(), "tolerance": arg::number("Value tolerance"), "use_threshold": arg::boolean("Fill voxels in the threshold range instead"), "min_diameter_mm": arg::number("Narrowest gap passed, mm (0: no limit)"), "segment": target()}),
        )
        .required(&["at"]),
        Action::new(
            "islands",
            "The segment's separate pieces: largest (keep the largest), small (remove those under min_size voxels), split (each piece its own segment), keep / remove / add (the piece at `at`; add takes it from the unlabelled voxels)",
            json!({"op": arg::choice(&keys(&IslandOp::ALL, IslandOp::key), "What to do"), "at": arg::point("The piece (ops keep, remove, add)"), "in_voxels": in_voxels(), "min_size": arg::integer("Smallest piece kept, voxels (ops small, split)"), "segment": target()}),
        )
        .required(&["op"]),
        Action::new("margin", "Grow (mm > 0) or shrink (mm < 0) the segment", json!({"mm": arg::number("Distance in mm"), "segment": target()})).required(&["mm"]),
        Action::new(
            "smooth",
            "Smooth the segment: median (removes bumps), open (removes extrusions), close (fills gaps), fill (fills holes)",
            json!({"method": arg::choice(&keys(&SmoothMethod::ALL, SmoothMethod::key), "Method"), "mm": arg::number("Kernel size in mm"), "segment": target()}),
        ),
        Action::new(
            "logical",
            "copy, add, subtract or intersect the `other` segment into this one; invert, clear or fill this one",
            json!({"op": arg::choice(&keys(&LogicalOp::ALL, LogicalOp::key), "What to do"), "other": segment("The other segment (copy, add, subtract, intersect)"), "segment": target()}),
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
            if let Some(v) = a.opt_bool("visible") {
                s.visible = v;
            }
            seg.changed(true);
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
        ws.brush(vol, plane, p, erase);
    }
    ws.end_stroke();
    let after = count(ws);
    let name = ws.seg.as_ref().and_then(|s| s.selected_segment()).map_or(String::new(), |s| s.name.clone());
    Ok(format!("{} {name} with a {brush_mm} mm radius brush: {} voxels {}", if erase { "Erased" } else { "Painted" }, before.abs_diff(after), if erase { "removed" } else { "added" }))
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
            let (lo, hi) = (f32_of("min").unwrap_or(st.threshold.0), f32_of("max").unwrap_or(st.threshold.1));
            st.threshold = (lo.min(hi), lo.max(hi));
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
            (Effect::Margin, None)
        }
        "smooth" => {
            if let Some(m) = a.opt_str("method").and_then(|m| SmoothMethod::ALL.into_iter().find(|x| x.key() == m)) {
                st.smoothing = m;
            }
            if let Some(mm) = f32_of("mm") {
                st.smoothing_mm = mm.max(0.0);
            }
            (Effect::Smoothing, None)
        }
        "logical" => {
            st.logical = LogicalOp::ALL.into_iter().find(|o| o.key() == a.str("op").unwrap_or("")).unwrap_or(st.logical);
            if other.is_some() {
                st.logical_other = other;
            }
            if st.logical.needs_other() && st.logical_other.is_none() {
                return Err(format!("logical {} needs `other`: the segment to combine with", st.logical.key()));
            }
            (Effect::Logical, None)
        }
        "apply" => match ws.effect {
            Some(e @ (Effect::Paint | Effect::Erase | Effect::FloodFill)) => return Err(format!("{} works at points: use the {} action", e.label(), e.key())),
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
    Ok(format!("Window {:.1} / {:.1}, opacity {:.2}", state.window.0, state.window.1, ws.opacity))
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
    fn without_a_volume_agents_hear_how_to_get_one() {
        let ctx = egui::Context::default();
        let mut ws = SegmentationWorkspace::new(&ctx, None);
        let AgentReply::Now(Err(e)) = act(&mut ws, "add_segment", &json!({}), &ctx) else { panic!() };
        assert!(e.contains("volume_do select"), "{e}");
    }
}
