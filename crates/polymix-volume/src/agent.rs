//! The Volume workspace's agent tools: the catalog and one volume's details.
//!
//! Agents always get identifiers masked, whatever *Hide identifiers* is set to in the window:
//! patient names, IDs, birth dates, sex, age and accession numbers become `•••`, patients' name
//! parts are masked inside file names, and dates are cut to the year. Patients are numbered
//! instead of named.

use crate::privacy::Privacy;
use crate::{relative, Selection, VolumeWorkspace};
use mm_volume::model::{Catalog, Field, Volume};
use mm_volume::scan::Progress;
use polymix_workspace::agent::{arg_str, arg_usize, round, year_only, AgentReply, AgentResult, AgentTool};
use serde_json::{json, Map, Value};
use std::sync::Arc;

pub fn tools() -> Vec<AgentTool> {
    let mut info = AgentTool::new(
        "volume_info",
        "One volume of the Volume workspace's catalog: kind, format, voxel grid (dimensions, spacing in mm, orientation code, voxel → RAS affine), data type, warnings, files and the header fields by group. Identifiers are masked and dates cut to the year.",
        json!({"volume": {"type": "integer", "description": "Volume number from volume_catalog"}}),
    );
    info.input_schema["required"] = json!(["volume"]);
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

/// What the tools need, taken on the UI thread (the catalog is shared, not copied).
struct Snapshot {
    catalog: Option<Arc<Catalog>>,
    privacy: Arc<Privacy>,
    roots: Vec<String>,
    scanning: Option<String>,
    scan_error: Option<String>,
    selection: Value,
    previewed: Value,
    hide_ids: bool,
}

pub fn call(ws: &VolumeWorkspace, tool: &str, args: &Value) -> AgentReply {
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
        previewed: ws.preview.as_ref().map_or(Value::Null, |p| json!({"volume": p.volume, "slice": p.slice, "slices": p.slices})),
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
        let mut c = Catalog::default();
        c.roots = vec!["/data/DOE_JANE scans".into()];
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
        Snapshot { roots: c.roots.iter().map(|r| privacy.path(&r.display().to_string())).collect(), catalog: Some(Arc::new(c)), privacy, scanning: None, scan_error: None, selection: Value::Null, previewed: Value::Null, hide_ids: false }
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
}
