// SPDX-License-Identifier: AGPL-3.0-only
//! The script steps that drive the surface features' boxes headless (tests, `POLYMIX_SCRIPT`):
//! each opens its kind's box as its button does, sets what the step says, and creates it.
//!
//! - `sheet:<from>:<to>` — a sheet of the last sketch.
//! - `split:<scan|largest|smallest|between>[:<tolerance>[:<overlap %>[:<within mm>]]]` — the
//!   last solid body split by every sheet.
//! - `trimsheet:<scan|largest|smallest>` — the last sheet split by every solid body.
//! - `thicken:<mm>[:in|out|both[:new|join|cut]]` — the last sheet thickened.
//! - `surface:<x>:<y>:<z>[:freeform|plane|cylinder[:<flood °>]]` — a surface through the flood
//!   from the scan's vertex nearest a point of the part.
//! - `surfacebox:<x0>:<y0>:<z0>:<x1>:<y1>:<z1>[:freeform|plane|cylinder]` — a surface through
//!   the scan's vertices in a box, read again when a parameter moves it.
//! - `trim[:<tolerance>]` — every surface body closed into a solid.
//! - `extract:<element id>[:section]` — an element, or a section's shadow, as a body.
//! - `import:<path>` — a STEP file's bodies.
//! - `surfreport` — what the surface features said of themselves, to the log.
//! - `surfbox:<kind>` — a kind's box opened as its button opens it, left open (for a picture).

use glam::DVec3;
use mm_reverse::model::{BodyId, BodyOp, BoxBounds, FeatureKind, RegionBox, SplitKeep, SurfaceShape, ThickenSide};

use super::surf_common::{build_scan, flood_from, region_in_box, scan_offsets, sheets};
use crate::reverse::{Mode, ReverseWorkspace};

fn keep_of(s: Option<&&str>) -> SplitKeep {
    match s.copied() {
        Some("largest") => SplitKeep::Largest,
        Some("smallest") => SplitKeep::Smallest,
        Some("between") => SplitKeep::Between,
        _ => SplitKeep::Scan,
    }
}

fn shape_of(s: Option<&&str>) -> SurfaceShape {
    match s.copied() {
        Some("plane") => SurfaceShape::Plane,
        Some("cylinder") => SurfaceShape::Cylinder,
        _ => SurfaceShape::Freeform,
    }
}

/// Open a kind's box as its button does; `false` (said on the status line) where it cannot.
fn open(ws: &mut ReverseWorkspace, kind: &str) -> bool {
    match super::new_feature(kind, ws) {
        Ok(k) => {
            ws.open_feature(k, None);
            true
        }
        Err(e) => {
            ws.status = e;
            log::warn!("reverse script {kind}: {}", ws.status);
            false
        }
    }
}

/// The open box's feature, to set.
fn feature(ws: &mut ReverseWorkspace) -> Option<&mut FeatureKind> {
    match &mut ws.mode {
        Mode::Feature(e) => Some(&mut e.kind),
        _ => None,
    }
}

/// A script step of the surface features; `false` for a step that is none of theirs.
pub(crate) fn run(ws: &mut ReverseWorkspace, cmd: &str, args: &[&str]) -> bool {
    let num = |i: usize, default: f64| args.get(i).and_then(|v| v.parse::<f64>().ok()).unwrap_or(default);
    match cmd {
        "sheet" => {
            if open(ws, "sheet") {
                if let Some(FeatureKind::Sheet(s)) = feature(ws) {
                    s.from = num(0, -10.0);
                    s.to = num(1, 10.0);
                }
                ws.commit_feature();
            }
        }
        "split" => {
            if open(ws, "split") {
                let tools = sheets(ws);
                if let Some(FeatureKind::Split(s)) = feature(ws) {
                    s.tools = tools.into_iter().filter(|t| *t != s.body).collect();
                    s.keep = keep_of(args.first());
                    s.tolerance = num(1, s.tolerance);
                    s.overlap = num(2, s.overlap * 100.0) / 100.0;
                    s.within = args.get(3).and_then(|v| v.parse::<f64>().ok());
                }
                ws.commit_feature();
            }
        }
        "trimsheet" => {
            let solids: Vec<BodyId> = ws.built.bodies.iter().filter(|b| !b.surface).map(|b| b.id).collect();
            let sheet = sheets(ws).last().copied();
            if let (Some(sheet), true) = (sheet, open(ws, "split")) {
                if let Some(FeatureKind::Split(s)) = feature(ws) {
                    s.body = sheet;
                    s.tools = solids;
                    s.keep = keep_of(args.first());
                }
                ws.commit_feature();
            }
        }
        "thicken" => {
            if open(ws, "thicken") {
                if let Some(FeatureKind::Thicken(t)) = feature(ws) {
                    t.thickness = num(0, 2.0);
                    t.side = match args.get(1).copied() {
                        Some("out") => ThickenSide::Out,
                        Some("both") => ThickenSide::Both,
                        _ => ThickenSide::In,
                    };
                    t.op = match args.get(2).copied() {
                        Some("join") => BodyOp::Join,
                        Some("cut") => BodyOp::Cut,
                        _ => BodyOp::New,
                    };
                }
                ws.commit_feature();
                if let Some(id) = ws.part.features.last().map(|f| f.id) {
                    super::surf_common::after_commit(ws, id);
                }
            }
        }
        "surface" => {
            let at = DVec3::new(num(0, 0.0), num(1, 0.0), num(2, 0.0));
            let Some(scan) = build_scan(ws) else {
                ws.status = "A surface is fitted through the scan, and none is open.".into();
                return true;
            };
            let nearest = scan.positions().iter().enumerate().min_by(|a, b| (*a.1 - at).length_squared().total_cmp(&(*b.1 - at).length_squared())).map(|(i, _)| i as u32);
            let Some(n) = nearest else { return true };
            let offsets = scan_offsets(ws);
            let Some((object, start, _)) = offsets.iter().copied().find(|(_, s, len)| n >= *s && n < s + len) else { return true };
            let region = flood_from(ws, object, n - start, num(4, 1.0));
            if open(ws, "surface") {
                if let Some(FeatureKind::Surface(s)) = feature(ws) {
                    s.region = region;
                    s.shape = Some(shape_of(args.get(3)));
                }
                ws.commit_feature();
            }
        }
        "surfacebox" => {
            let bounds = BoxBounds { min: DVec3::new(num(0, 0.0), num(1, 0.0), num(2, 0.0)), max: DVec3::new(num(3, 0.0), num(4, 0.0), num(5, 0.0)) };
            match region_in_box(ws, &bounds, None, &[], None) {
                Ok(region) => {
                    if open(ws, "surface") {
                        if let Some(FeatureKind::Surface(s)) = feature(ws) {
                            s.region = region;
                            s.region_box = Some(RegionBox { bounds, normal: None, exclude: None, limit: None });
                            s.shape = Some(shape_of(args.get(6)));
                        }
                        ws.commit_feature();
                    }
                }
                Err(e) => ws.status = e,
            }
        }
        "trim" => {
            if open(ws, "trim") {
                if let Some(FeatureKind::Trim(t)) = feature(ws) {
                    if let Some(tol) = args.first().and_then(|v| v.parse::<f64>().ok()) {
                        t.tolerance = tol;
                    }
                }
                ws.commit_feature();
            }
        }
        "extract" => {
            if open(ws, "extract") {
                let id = args.first().and_then(|v| v.parse::<u32>().ok());
                let section = args.get(1) == Some(&"section");
                if let Some(FeatureKind::Extract(x)) = feature(ws) {
                    if section {
                        x.section = id;
                    } else {
                        x.element = id;
                    }
                }
                ws.commit_feature();
            }
        }
        "import" => {
            let path = std::path::PathBuf::from(args.join(":"));
            if open(ws, "import") {
                match super::import::read_step(ws, &path) {
                    Ok(file) => {
                        if let Some(FeatureKind::Import(x)) = feature(ws) {
                            x.file = file;
                        }
                        ws.commit_feature();
                    }
                    Err(e) => {
                        ws.status = e;
                        ws.close_mode();
                    }
                }
            }
        }
        "surfbox" => {
            if let Some(kind) = args.first() {
                open(ws, kind);
            }
        }
        "surfreport" => {
            let mut ids: Vec<&u32> = ws.built.surf.keys().collect();
            ids.sort();
            for id in ids {
                let r = &ws.built.surf[id];
                let name = ws.part.feature(*id).map_or(String::new(), |f| f.name.clone());
                let surface = r.surface.as_ref().map(|s| format!("σ {:.4} max {:.4} net {}×{} points {} shape {:?} radius {:?} relations {}", s.rms, s.max, s.net.0, s.net.1, s.points, s.shape, s.radius, s.relations.len()));
                let trim = r.trim.as_ref().map(|t| format!("faces {} kept {} closed {} open {}", t.faces, t.kept, t.closed, t.open_edges.len()));
                log::info!("reverse surf {id} {name}: sheet {:?} surface {:?} trim {:?} note {:?} offscan {} strays {} where {} error {:?}", r.sheet_faces, surface, trim, r.note, r.off_scan.len(), r.strays.len(), r.where_lines.len(), ws.built.errors.get(id));
            }
        }
        _ => return false,
    }
    true
}
