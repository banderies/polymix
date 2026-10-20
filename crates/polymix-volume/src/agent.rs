//! The Volume workspace's agent tools: the catalog and one volume's details.
//!
//! Agents always get identifiers masked, whatever *Hide identifiers* is set to in the window:
//! patient names, IDs, birth dates, sex, age and accession numbers become `•••`, patients' name
//! parts are masked inside file names, and dates are cut to the year. Patients are numbered
//! instead of named.

use crate::prepare::{self, Op};
use crate::privacy::Privacy;
use crate::{relative, Selection, VolumeWorkspace};
use mm_volume::model::{Catalog, Field, Volume};
use mm_volume::scan::Progress;
use polymix_workspace::agent::{arg, arg_str, arg_usize, round, year_only, Action, AgentReply, AgentResult, AgentTool, Args};
use serde_json::{json, Map, Value};
use std::path::PathBuf;
use std::sync::Arc;

pub fn tools() -> Vec<AgentTool> {
    let info = AgentTool::new(
        "volume_info",
        "One volume of the Volume workspace's catalog: kind, format, voxel grid (dimensions, spacing in mm, orientation code, voxel → RAS affine), data type, warnings, files and the header fields by group. Identifiers are masked and dates cut to the year.",
        json!({"volume": {"type": "integer", "description": "Volume number from volume_catalog"}}),
    )
    .required(&["volume"]);
    vec![
        AgentTool::new(
            "volume_catalog",
            "The medical images found in the Volume workspace's folders: scan state, the DICOM tree (patients numbered, never named → studies → series with modality, description and the volume numbers they hold), every volume with its kind, format, dimensions and spacing, and what is selected and previewed. Volume numbers stay valid until the folders are scanned again. Identifiers are masked and dates cut to the year.",
            json!({
                "search": {"type": "string", "description": "Only list volumes whose name, kind or format contains this (case-insensitive)"},
                "limit": {"type": "integer", "description": "Most volumes to list (default 200, up to 2000)"}
            }),
        ),
        info,
    ]
}

/// The start of `volume_do`'s description.
pub const ABOUT: &str = "Work the Volume workspace: choose the image the Segmentation workspace shows, browse its slices, manage the folders scanned, and prepare images (bias correction, resampling, registration, …: each result is saved as a new derived image, which is selected when done). Volumes are numbered as volume_catalog lists them. Scans and image operations run in the background: app_wait, then volume_catalog.";

pub fn actions() -> Vec<Action> {
    let volume = |about| arg::integer(about);
    vec![
        Action::new("select", "Select and preview a volume; the Segmentation workspace then shows it", json!({"volume": volume("Volume number")})).required(&["volume"]),
        Action::new("slice", "Show a slice of the previewed volume", json!({"slice": arg::integer("Slice number from 0")})).required(&["slice"]),
        Action::new(
            "window",
            "The preview's window / level",
            json!({"centre": arg::number("Window centre (level)"), "width": arg::number("Window width"), "auto": arg::boolean("Back to the automatic window (1st–99th percentile of the slice)"), "file": arg::boolean("The window stored in the file")}),
        ),
        Action::new("add_folder", "Find the images in a folder and everything under it (or in one file)", json!({"path": arg::text("Absolute path")})).required(&["path"]),
        Action::new("remove_folder", "Stop looking in a folder", json!({"folder": arg::integer("Its number in volume_catalog's sources, from 0")})).required(&["folder"]),
        Action::plain("rescan", "Scan the folders again (volume numbers may change)"),
        Action::new(
            "viewer",
            "The viewer under the browser: the selected image in three planes. open: show or hide it. compare: an image to show over it or beside it, by volume number (images overlapping it; resampled onto its grid in the background: app_wait), with `mode` blend (see-through by opacity), wipe (divider at `wipe` 0–1) or toggle (showing_second). remove_compare: stop comparing. colour_map, slab (off / max / min / mean) and slab_mm, window_preset: how it is shown",
            json!({
                "open": arg::boolean("Show the viewer"),
                "compare": volume("Volume number to compare with"),
                "mode": arg::choice(&polymix_imaging::Compare::ALL.map(|c| c.key()), "How the two are shown"),
                "opacity": arg::number("Blend: the second image's opacity, 0–1"),
                "wipe": arg::number("Wipe: the divider, 0–1"),
                "showing_second": arg::boolean("Toggle: show the second image"),
                "remove_compare": arg::boolean("Stop comparing"),
                "colour_map": arg::choice(&polymix_imaging::ColourMap::ALL.map(|m| m.key()), "Colours of the image"),
                "slab": arg::choice(&polymix_imaging::SlabMode::ALL.map(|m| m.key()), "Thick slabs"),
                "slab_mm": arg::number("Slab thickness, mm"),
                "window_preset": arg::choice(&polymix_imaging::layers::CT_PRESETS.map(|p| p.0), "A CT window"),
            }),
        ),
        Action::plain("cancel_scan", "Stop the scan running"),
        Action::new(
            "prepare",
            "Run an image operation on a volume (default: the previewed one): n4 (MR bias correction), denoise, resample, crop (crop / pad), reorient (reorient / fix the header), register (to `reference`, onto its grid), subtract (`reference` from it), normalise, vessels (vesselness filter), slab (MIP / MinIP), time (4D time points), pc (phase contrast → velocities), export (a de-identified copy into options.out)",
            json!({
                "op": arg::choice(&Op::ALL.map(Op::key), "The operation"),
                "volume": volume("Volume number (default: the previewed one)"),
                "options": {"type": "object", "description": prepare::options_about()},
                "reference": volume("The other image (register, subtract, normalise with norm match)"),
                "phases": {"type": "array", "items": {"type": "integer"}, "minItems": 3, "maxItems": 3, "description": "Volume numbers of the x, y and z phase images (pc)"},
                "magnitude": volume("Volume number of the magnitude image (pc, optional)"),
            }),
        )
        .required(&["op"])
        .destructive(),
        Action::plain("cancel_prepare", "Stop the image operation running"),
    ]
}

pub fn act(ws: &mut VolumeWorkspace, action: &str, args: &Value, ctx: &egui::Context) -> AgentReply {
    AgentReply::Now(act_now(ws, action, &Args(args), ctx).map(|done| json!({"done": done})))
}

fn act_now(ws: &mut VolumeWorkspace, action: &str, a: &Args, ctx: &egui::Context) -> Result<String, String> {
    let catalog = |ws: &VolumeWorkspace| ws.catalog.clone().ok_or_else(|| "No catalog yet: add a folder (add_folder) or wait for the scan (app_wait).".to_string());
    let volume = |c: &Catalog, k: usize| if k < c.volumes.len() { Ok(k) } else { Err(format!("No volume {k}: the catalog has {}", c.volumes.len())) };
    match action {
        "viewer" => {
            if let Some(o) = a.opt_bool("open") {
                ws.viewer.open = o;
            }
            if let Some(k) = a.opt_usize("compare") {
                let c = catalog(ws)?;
                let k = volume(&c, k)?;
                if ws.viewer.volume.is_none() {
                    return Err("The viewer shows no image yet: open it (open: true) with an image selected, then app_wait".into());
                }
                let label = ws.agent_privacy.path(&c.volumes[k].name);
                ws.viewer.compare_with(c.volumes[k].clone(), label, ctx);
            }
            let st = ws.viewer.state.as_mut();
            if let Some(st) = st {
                if a.opt_bool("remove_compare") == Some(true) {
                    st.fusion = None;
                }
                if let Some(m) = a.opt_str("colour_map") {
                    st.colour_map = polymix_imaging::ColourMap::ALL.into_iter().find(|x| x.key() == m).ok_or("unknown colour map")?;
                }
                if let Some(m) = a.opt_str("slab") {
                    st.slab.mode = polymix_imaging::SlabMode::ALL.into_iter().find(|x| x.key() == m).ok_or("slab is off, max, min or mean")?;
                }
                if let Some(mm) = a.opt_f64("slab_mm") {
                    st.slab.mm = mm.clamp(0.1, 500.0);
                }
                if let Some(p) = a.opt_str("window_preset") {
                    let (_, c, w) = polymix_imaging::layers::CT_PRESETS.into_iter().find(|x| x.0 == p).ok_or("unknown window preset")?;
                    st.window = (c, w);
                }
                if let Some(f) = &mut st.fusion {
                    if let Some(m) = a.opt_str("mode") {
                        f.compare = polymix_imaging::Compare::ALL.into_iter().find(|x| x.key() == m).ok_or("mode is blend, wipe or toggle")?;
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
                }
            }
            Ok(format!("Viewer {}", if ws.viewer.open { "shown" } else { "hidden" }))
        }
        "select" => {
            let c = catalog(ws)?;
            let k = volume(&c, a.usize("volume")?)?;
            ws.selection = Selection::Volume(k);
            ws.select_volume(k, ctx);
            Ok(format!("Selected volume {k}, {}", ws.agent_privacy.path(&c.volumes[k].name)))
        }
        "slice" => {
            let c = catalog(ws)?;
            let n = a.usize("slice")?;
            let p = ws.preview.as_mut().ok_or("No volume is previewed: select one first")?;
            if n >= p.slices {
                return Err(format!("No slice {n}: the volume has {}", p.slices));
            }
            p.go_to(n, &c.volumes[p.volume], ctx);
            Ok(format!("Slice {n} of {}", p.slices))
        }
        "window" => {
            let p = ws.preview.as_mut().ok_or("No volume is previewed: select one first")?;
            if a.opt_bool("auto") == Some(true) {
                p.set_window(None);
            } else if a.opt_bool("file") == Some(true) {
                p.set_window(Some(p.file_window().ok_or("This file stores no window")?));
            } else {
                let (c, w) = p.shown_window();
                p.set_window(Some((a.opt_f64("centre").unwrap_or(c), a.opt_f64("width").unwrap_or(w).max(1e-6))));
            }
            let (c, w) = p.shown_window();
            Ok(format!("Window {w:.1} / level {c:.1}"))
        }
        "add_folder" => {
            let path = PathBuf::from(a.str("path")?);
            if !path.is_absolute() || !path.exists() {
                return Err(format!("{} is not an absolute path to a folder or file", ws.agent_privacy.path(&path.display().to_string())));
            }
            ws.add_roots(vec![path], ctx);
            Ok("Scanning: app_wait, then volume_catalog".into())
        }
        "remove_folder" => {
            let k = a.usize("folder")?;
            if k >= ws.roots.len() {
                return Err(format!("No folder {k}: there are {}", ws.roots.len()));
            }
            let gone = ws.roots.remove(k);
            ws.start_scan(ctx);
            Ok(format!("No longer looking in {}; scanning again (volume numbers may change)", ws.agent_privacy.path(&gone.display().to_string())))
        }
        "rescan" => {
            ws.start_scan(ctx);
            Ok("Scanning again: app_wait, then volume_catalog (volume numbers may change)".into())
        }
        "cancel_scan" => match &ws.scan {
            Some(task) => {
                task.cancel();
                Ok("Scan cancelled".into())
            }
            None => Err("No scan is running".into()),
        },
        "cancel_prepare" => if ws.prepare.cancel() { Ok("Image operation cancelled".into()) } else { Err("No image operation is running".into()) },
        "prepare" => prepare(ws, a, ctx),
        _ => Err(format!("the Volume workspace has no action {action}")),
    }
}

fn prepare(ws: &mut VolumeWorkspace, a: &Args, ctx: &egui::Context) -> Result<String, String> {
    let c = ws.catalog.clone().ok_or("No catalog yet")?;
    if ws.prepare.busy() {
        return Err("An image operation is running: app_wait, then try again (or cancel_prepare)".into());
    }
    let pick = |k: usize| c.volumes.get(k).ok_or_else(|| format!("No volume {k}: the catalog has {}", c.volumes.len()));
    let k = match a.opt_usize("volume") {
        Some(k) => k,
        None => ws.preview.as_ref().map(|p| p.volume).ok_or("No volume is previewed: give `volume`")?,
    };
    let v = pick(k)?;
    if !prepare::applies_to(v) {
        return Err(format!("Volume {k} is not an intensity image Polymix can read"));
    }
    // The options as the panel's settings (checked before anything changes), over the defaults
    // for this image.
    let mut options = vec![format!("op={}", a.str("op")?)];
    let mut crop = (None, None);
    for (key, value) in a.0.get("options").and_then(Value::as_object).into_iter().flatten() {
        let point = || -> Result<[usize; 3], String> {
            match value.as_array().map(|x| x.iter().filter_map(Value::as_u64).map(|n| n as usize).collect::<Vec<_>>()).as_deref() {
                Some(&[i, j, k]) => Ok([i, j, k]),
                _ => Err(format!("options.{key} must be voxel indices [i, j, k]")),
            }
        };
        match key.as_str() {
            "crop_from" => crop.0 = Some(point()?),
            "crop_to" => crop.1 = Some(point()?),
            k if prepare::OPTION_KEYS.contains(&k) => {
                let text = match value {
                    Value::Bool(b) => if *b { "on" } else { "off" }.to_string(),
                    Value::Number(n) => n.to_string(),
                    Value::String(t) if !t.contains([',', '=']) => t.clone(),
                    _ => return Err(format!("options.{key} must be a number, true / false or text without , or =")),
                };
                if k == "out" && !std::path::Path::new(&text).is_absolute() {
                    return Err("options.out must be an absolute folder".into());
                }
                options.push(format!("{k}={text}"));
            }
            k => return Err(format!("prepare has no option {k}; the options are {}, crop_from, crop_to", prepare::OPTION_KEYS.join(", "))),
        }
    }
    ws.prepare.fit_to(v);
    if let Some(r) = a.opt_usize("reference") {
        ws.prepare.choose_reference(pick(r)?);
    }
    if let Some(ph) = a.0.get("phases").and_then(Value::as_array) {
        let ks: Vec<usize> = ph.iter().filter_map(Value::as_u64).map(|n| n as usize).collect();
        let [x, y, z] = ks[..] else { return Err("phases must be three volume numbers".into()) };
        let magnitude = a.opt_usize("magnitude").map(pick).transpose()?;
        ws.prepare.choose_phases([pick(x)?, pick(y)?, pick(z)?], magnitude);
    }
    if crop != (None, None) {
        let dims = v.geometry.as_ref().map(|g| g.dims).unwrap_or_default();
        ws.prepare.set_crop(v, crop.0.unwrap_or([0; 3]), crop.1.unwrap_or([0, 1, 2].map(|a| dims[a].saturating_sub(1))))?;
    }
    ws.prepare.set_options(&options.join(","));
    // The panel shows the volume being prepared.
    ws.selection = Selection::Volume(k);
    ws.select_volume(k, ctx);
    let label = ws.agent_privacy.path(&v.name);
    ws.prepare.start(&c, k, &label, ctx);
    if !ws.prepare.busy() {
        return Err(ws.prepare.last_message().trim_start_matches("error: ").to_string());
    }
    Ok(format!("{} of volume {k} started: app_wait, then volume_catalog (prepare.last_run)", a.str("op")?))
}

/// What the tools need, taken on the UI thread (the catalog is shared, not copied).
struct Snapshot {
    catalog: Option<Arc<Catalog>>,
    privacy: Arc<Privacy>,
    roots: Vec<String>,
    scanning: Option<String>,
    scan_error: Option<String>,
    selection: Value,
    previewed: Value,
    prepare: Value,
    viewer: Value,
    hide_ids: bool,
}

pub fn call(ws: &mut VolumeWorkspace, tool: &str, args: &Value) -> AgentReply {
    let privacy = ws.agent_privacy.clone();
    let scanning = ws.scan.as_ref().map(|s| match s.progress() {
        Some(Progress::Listing { files }) => format!("listing files ({files} so far)"),
        Some(Progress::Reading { done, total }) => format!("reading headers ({done} of {total})"),
        Some(Progress::Organising) => "organising".into(),
        None => "starting".into(),
    });
    let snap = Snapshot {
        catalog: ws.catalog.clone(),
        roots: ws.roots.iter().map(|r| privacy.path(&r.display().to_string())).collect(),
        scanning: scanning.or_else(|| ws.scan_soon.then(|| "starting".into())),
        scan_error: ws.scan_error.as_ref().map(|e| privacy.path(&polymix_workspace::guard::scrub_paths(e))),
        selection: match ws.selection {
            Selection::None => Value::Null,
            Selection::Patient(p) => json!({"patient": p}),
            Selection::Study(p, s) => json!({"patient": p, "study": s}),
            Selection::Series(p, s, se) => json!({"patient": p, "study": s, "series": se}),
            Selection::Volume(k) => json!({"volume": k}),
        },
        previewed: ws.preview.as_ref().map_or(Value::Null, |p| {
            let (c, w) = p.shown_window();
            json!({"volume": p.volume, "slice": p.slice, "slices": p.slices, "window": {"centre": round(c, 3), "width": round(w, 3)}})
        }),
        prepare: ws.prepare.agent_state(),
        viewer: json!({
            "open": ws.viewer.open,
            "volume": ws.viewer.volume.as_ref().map(|(k, _)| k),
            "busy": ws.viewer.busy(),
            "colour_map": ws.viewer.state.as_ref().map(|s| s.colour_map.key()),
            "slab": ws.viewer.state.as_ref().map(|s| json!({"mode": s.slab.mode.key(), "mm": s.slab.mm})),
            "compared_with": ws.viewer.state.as_ref().and_then(|s| s.fusion.as_ref()).map(|f| json!({"name": f.name, "mode": f.compare.key(), "opacity": round(f.opacity as f64, 2), "wipe": round(f.wipe as f64, 2), "showing_second": f.showing})),
            "measurements": ws.viewer.measure.items.iter().map(|m| m.text()).collect::<Vec<_>>(),
            "error": ws.viewer.error.as_ref().map(|e| polymix_workspace::guard::scrub_paths(e)),
        }),
        hide_ids: ws.hide_ids,
        privacy,
    };
    let args = args.clone();
    match tool {
        "volume_catalog" => AgentReply::later(move || Ok(catalog(&snap, &args))),
        "volume_info" => AgentReply::later(move || info(&snap, &args)),
        _ => AgentReply::error(format!("the Volume workspace has no tool {tool}")),
    }
}

fn catalog(s: &Snapshot, args: &Value) -> Value {
    let mut out = json!({
        "sources": s.roots,
        "scanning": s.scanning,
        "scan_error": s.scan_error,
        "selected": s.selection,
        "previewed": s.previewed,
        "prepare": s.prepare,
        "viewer": s.viewer,
        "identifiers_hidden_in_window": s.hide_ids,
    });
    let Some(c) = &s.catalog else {
        out["note"] = json!(if s.roots.is_empty() { "No folders added yet (Volume tab → Add folder…)." } else { "Not scanned yet." });
        return out;
    };
    let p = &s.privacy;
    let search = arg_str(args, "search").map(str::to_lowercase);
    let limit = arg_usize(args, "limit", 200, 1, 2000);
    let matches = |v: &Volume| search.as_ref().is_none_or(|q| [p.path(&v.name), v.kind.label().to_string(), v.format.label().to_string()].iter().any(|t| t.to_lowercase().contains(q)));
    let all: Vec<usize> = (0..c.volumes.len()).filter(|&k| matches(&c.volumes[k])).collect();
    let volumes: Vec<Value> = all.iter().take(limit).map(|&k| volume_row(c, k, p)).collect();
    let patients: Vec<Value> = c
        .patients
        .iter()
        .enumerate()
        .map(|(pi, pa)| {
            let studies: Vec<Value> = pa
                .studies
                .iter()
                .enumerate()
                .map(|(si, st)| {
                    let series: Vec<Value> = st
                        .series
                        .iter()
                        .enumerate()
                        .map(|(sei, se)| json!({"series": sei, "number": se.number, "modality": se.modality, "description": p.path(&se.description), "year": year_only(&se.date), "instances": se.instances, "volumes": se.volumes, "other_objects": se.objects.len()}))
                        .collect();
                    json!({"study": si, "year": year_only(&st.date), "description": p.path(&st.description), "series": series})
                })
                .collect();
            json!({"patient": pi, "studies": studies, "other_volumes": pa.other_volumes})
        })
        .collect();
    let series: usize = c.patients.iter().flat_map(|p| &p.studies).map(|s| s.series.len()).sum();
    let extra = json!({
        "counts": {"files_seen": c.files_seen, "patients": c.patients.len(), "series": series, "volumes": c.volumes.len(), "unmatched_files": c.files.len(), "recognised_unreadable": c.recognised.len(), "problems": c.problems.len(), "skipped_files": c.skipped.file_count()},
        "patients": patients,
        "unmatched_files": c.files,
        "volumes": volumes,
        "volumes_matching": all.len(),
        "truncated": all.len() > limit,
    });
    merge(&mut out, extra);
    out
}

fn volume_row(c: &Catalog, k: usize, p: &Privacy) -> Value {
    let v = &c.volumes[k];
    let g = v.geometry.as_ref();
    json!({
        "volume": k,
        "name": p.path(&v.name),
        "kind": v.kind.label(),
        "format": v.format.label(),
        "dims": g.map(|g| g.dims),
        "spacing_mm": g.map(|g| g.spacing.map(|x| round(x, 4))),
        "data_type": v.data_type.map(|d| d.label()),
        "warnings": v.warnings.len(),
    })
}

fn info(s: &Snapshot, args: &Value) -> AgentResult {
    let c = s.catalog.as_ref().ok_or("No catalog yet: add a folder in the Volume tab, or wait for the scan (see volume_catalog).")?;
    let k = args.get("volume").and_then(Value::as_u64).ok_or("Give the volume number (from volume_catalog).")? as usize;
    let v = c.volumes.get(k).ok_or_else(|| format!("No volume {k}: the catalog has {} (numbers change when the folders are scanned again).", c.volumes.len()))?;
    let p = &s.privacy;
    let mut header = Map::new();
    for f in &v.fields {
        let group = header.entry(f.group.clone()).or_insert_with(|| json!({}));
        group[&f.name] = json!(agent_field(p, f));
    }
    let series = c.patients.iter().enumerate().find_map(|(pi, pa)| {
        pa.studies.iter().enumerate().find_map(|(si, st)| st.series.iter().position(|se| se.volumes.contains(&k)).map(|sei| json!({"patient": pi, "study": si, "series": sei})))
    });
    let mut out = volume_row(c, k, p);
    merge(
        &mut out,
        json!({
            "bytes": v.bytes,
            "geometry": v.geometry.as_ref().map(|g| json!({
                "dims": g.dims,
                "spacing_mm": g.spacing,
                "orientation": g.orientation(),
                "extent_mm": g.extent_mm().map(|x| round(x, 3)),
                "obliquity_deg": round(g.obliquity_deg(), 3),
                "voxel_to_ras": g.affine,
            })),
            "window_in_file": v.window.map(|(c, w)| json!({"centre": c, "width": w})),
            "photometric": v.photometric,
            "warnings": v.warnings.iter().map(|w| p.path(w)).collect::<Vec<_>>(),
            "files": {"count": v.files.len(), "first": v.files.iter().take(10).map(|f| p.path(&relative(f, &c.roots))).collect::<Vec<_>>()},
            "series": series,
            "header": header,
        }),
    );
    Ok(out)
}

/// A header field as agents see it: masked as when hiding identifiers, dates cut to the year.
fn agent_field(p: &Privacy, f: &Field) -> String {
    let v = p.field(f);
    if f.name.to_lowercase().contains("date") && v != crate::privacy::MASK { year_only(&v) } else { v }
}

fn merge(into: &mut Value, from: Value) {
    if let (Value::Object(a), Value::Object(b)) = (into, from) {
        a.extend(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mm_volume::model::{Format, Kind, Patient, PatientKey, Series, Study, Voxels};

    fn sample() -> Catalog {
        let mut c = Catalog { roots: vec!["/data/DOE_JANE scans".into()], ..Default::default() };
        let fields = vec![
            Field::new("Patient", "Name", "DOE JANE"),
            Field::new("Patient", "Birth date", "1980-01-01"),
            Field::new("Study", "Date", "2024-03-05 10:20"),
            Field::new("Study", "Accession", "ACC777"),
            Field::new("Series", "Description", "TOF 3D"),
        ];
        c.volumes.push(Volume {
            format: Format::Dicom,
            kind: Kind::Image,
            name: "Series 3 · TOF 3D".into(),
            files: vec!["/data/DOE_JANE scans/doe_jane_0001.dcm".into()],
            bytes: 10,
            geometry: None,
            data_type: None,
            fields,
            warnings: vec![],
            voxels: Voxels::Unsupported(String::new()),
            window: None,
            photometric: None,
            patient: None,
        });
        let series = Series { uid: "1.2".into(), number: Some(3), modality: "MR".into(), description: "TOF 3D".into(), date: "20240305".into(), time: String::new(), instances: 1, volumes: vec![0], objects: vec![] };
        let study = Study { uid: "1".into(), date: "20240305".into(), time: String::new(), description: "MRA HEAD".into(), accession: "ACC777".into(), series: vec![series] };
        c.patients.push(Patient { key: PatientKey { id: "MRN123".into(), name: "DOE^JANE".into() }, name: "DOE JANE".into(), id: "MRN123".into(), birth_date: "19800101".into(), sex: "F".into(), studies: vec![study], other_volumes: vec![] });
        c
    }

    fn snapshot(c: Catalog) -> Snapshot {
        let privacy = Arc::new(Privacy::new(true, Some(&c)));
        Snapshot { roots: c.roots.iter().map(|r| privacy.path(&r.display().to_string())).collect(), catalog: Some(Arc::new(c)), privacy, scanning: None, scan_error: None, selection: Value::Null, previewed: Value::Null, prepare: Value::Null, viewer: Value::Null, hide_ids: false }
    }

    #[test]
    fn agents_never_see_identifiers() {
        let s = snapshot(sample());
        let text = format!("{}{}", super::catalog(&s, &json!({})), info(&s, &json!({"volume": 0})).unwrap());
        for secret in ["DOE", "JANE", "Jane", "jane", "MRN123", "ACC777", "1980", "03-05", "0305"] {
            assert!(!text.contains(secret), "{secret} leaked: {text}");
        }
        assert!(text.contains("TOF 3D") && text.contains("2024"), "{text}");
        let i = info(&s, &json!({"volume": 0})).unwrap();
        assert_eq!(i["header"]["Study"]["Date"], "2024");
        assert_eq!(i["series"], json!({"patient": 0, "study": 0, "series": 0}));
    }

    #[test]
    fn search_limit_and_bad_numbers() {
        let s = snapshot(sample());
        assert_eq!(super::catalog(&s, &json!({"search": "tof"}))["volumes_matching"], 1);
        assert_eq!(super::catalog(&s, &json!({"search": "nifti"}))["volumes_matching"], 0);
        assert!(info(&s, &json!({"volume": 5})).unwrap_err().contains("No volume 5"));
        assert!(info(&s, &json!({})).is_err());
    }

    fn run(ws: &mut VolumeWorkspace, action: &str, args: Value) -> Result<Value, String> {
        let a = actions().into_iter().find(|a| a.name == action).expect("listed");
        a.check(&args)?;
        let AgentReply::Now(r) = act(ws, action, &args, &egui::Context::default()) else { panic!("answered now") };
        r
    }

    #[test]
    fn actions_check_what_they_are_given() {
        let mut ws = VolumeWorkspace::new(&egui::Context::default(), None);
        assert!(run(&mut ws, "select", json!({"volume": 0})).unwrap_err().contains("No catalog yet"));
        let mut c = sample();
        // A readable image, so preparing it gets as far as its options.
        let g = mm_volume::geometry::Geometry::from_affine([4, 5, 6, 1], Default::default());
        c.volumes[0].geometry = Some(g);
        c.volumes[0].voxels = Voxels::Raw { file: "/nowhere/x.nii".into(), file_offset: 352, gzip: false, skip: 0, interleave: 1, dtype: mm_volume::model::DataType::F32, big_endian: false, slope: 1.0, intercept: 0.0 };
        ws.finish_scan(Ok(c));
        assert!(run(&mut ws, "select", json!({"volume": 3})).unwrap_err().contains("No volume 3: the catalog has 1"));
        assert!(run(&mut ws, "remove_folder", json!({"folder": 2})).unwrap_err().contains("No folder 2"));
        assert!(run(&mut ws, "cancel_scan", json!({})).unwrap_err().contains("No scan"));
        assert!(run(&mut ws, "prepare", json!({"op": "n4"})).unwrap_err().contains("give `volume`"));
        assert!(run(&mut ws, "prepare", json!({"op": "blur", "volume": 0})).is_err());
        let e = run(&mut ws, "prepare", json!({"op": "n4", "volume": 0, "options": {"strength": 2}})).unwrap_err();
        assert!(e.contains("no option strength") && e.contains("levels"), "{e}");
        assert!(run(&mut ws, "prepare", json!({"op": "crop", "volume": 0, "options": {"crop_to": [9, 9, 9]}})).unwrap_err().contains("4 × 5 × 6"));
        assert!(run(&mut ws, "prepare", json!({"op": "export", "volume": 0, "options": {"out": "relative"}})).unwrap_err().contains("absolute"));
        assert!(run(&mut ws, "prepare", json!({"op": "pc", "volume": 0, "phases": [0, 0]})).unwrap_err().contains("must be"));
        // The catalog says how preparing stands.
        let AgentReply::Later(f) = call(&mut ws, "volume_catalog", &json!({})) else { panic!() };
        assert_eq!(f().unwrap()["prepare"]["running"], false);
    }

    #[test]
    fn the_options_list_names_every_key() {
        let about = prepare::options_about();
        for k in prepare::OPTION_KEYS {
            assert!(about.contains(&format!("{k} (")) || about.contains(&format!("{k},")) || about.contains(&format!(" {k} ")) || about.contains(&format!("{k}:")), "{k} is not described: {about}");
        }
        assert!(about.contains("windowed sinc") && about.contains("curvature diffusion") && about.contains("frangi"), "{about}");
    }
}
