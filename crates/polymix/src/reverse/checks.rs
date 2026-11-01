// SPDX-License-Identifier: AGPL-3.0-only
//! The bodies compared with the scan and checked, for the panel, the script steps and the agent
//! commands. Ported from ScanRuler (`plugins/reverse/app/reBodyCommands.ts`,
//! `app/reCompareFeature.ts`, `app/reSectionLayer.ts` and the Compare in Deviation of
//! `app/useReversePlugin.tsx`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only; the reading
//! itself is `mm-reverse`'s (`compare_detail.rs`, `body_check.rs`, `audit.rs`,
//! `section_layer.rs`).
//!
//! Each answer is JSON in the shape ScanRuler's `reverse.body.*` commands answer in, so the
//! agent commands can hand it on as it is.

use glam::DVec3;
use mm_core::Mesh;
use mm_reverse::audit::AUDIT_DEFAULTS;
use mm_reverse::body_check::{check_answer, clearance, clearance_text, AuditMode, Clearance, TINY_FACE_MM2};
use mm_reverse::compare::vertex_normals;
use mm_reverse::compare_detail::{compare_answer, compare_feature, score_of, CompareBody, CompareInput};
use mm_reverse::edge_lines::TangentEdges;
use mm_reverse::frame::{SectionFrame, WorldAxis};
use mm_reverse::model::{BodyId, FeatureId};
use mm_reverse::section_layer::{section_layer, LayerBody, CUT_CHORD_MM};
use mm_reverse::volume_compare::{volume_deviation, volume_out, VolumeKind, VolumeOptions, VOLUME_REGIONS};
use serde_json::{json, Value};

use super::ReverseWorkspace;

/// The name the reference carries in the Mesh workspace's Deviation tool.
pub const COMPARE_REFERENCE_NAME: &str = "Bodies from Reverse engineer.stl";
/// The chord the bodies are meshed to for the comparison, millimetres.
pub const COMPARE_CHORD_MM: f64 = 0.01;

impl ReverseWorkspace {
    /// The scan's vertices in the part's frame, every visible object's live ones, with their
    /// normals.
    pub(crate) fn scan_points(&mut self) -> (Vec<DVec3>, Vec<DVec3>) {
        let parts = self.scan_parts();
        let mut points = Vec::new();
        let mut normals = Vec::new();
        for p in parts.iter() {
            let n = vertex_normals(&p.positions, &p.triangles);
            for v in &p.live_vertices {
                points.push(p.positions[*v as usize]);
                normals.push(n[*v as usize]);
            }
        }
        (points, normals)
    }

    /// The bodies a comparison reads: those named, else the visible solid ones.
    fn compared(&self, ids: Option<&[BodyId]>) -> Vec<CompareBody<'_>> {
        self.bodies
            .iter()
            .filter(|b| match ids {
                Some(ids) => ids.contains(&b.id),
                None => !self.hidden_bodies.contains(&b.id) && !self.built.body(b.id).is_some_and(|x| x.surface),
            })
            .map(|b| CompareBody { id: b.id, name: &b.name, mesh: &b.part_mesh, sheet: self.built.body(b.id).is_some_and(|x| x.surface), topology: Some(&b.topology) })
            .collect()
    }

    /// `reverse.body.compare`: the bodies against the scan; "score" detail with `score`.
    pub(crate) fn compare_json(&mut self, input: &CompareInput, score: bool, ids: Option<&[BodyId]>) -> Result<Value, String> {
        if self.scan.is_empty() {
            return Err("There is no scan to compare the bodies with.".into());
        }
        let (points, normals) = self.scan_points();
        let bodies = self.compared(ids);
        let out = compare_answer(&bodies, &self.part.features, &points, &normals, input, None)?;
        Ok(if score { score_of(&out) } else { out })
    }

    /// `reverse.body.compare { feature }`: one feature against the scan.
    pub(crate) fn compare_feature_json(&mut self, feature: FeatureId, input: &CompareInput) -> Result<Value, String> {
        if self.scan.is_empty() {
            return Err("There is no scan to compare the bodies with.".into());
        }
        let (points, normals) = self.scan_points();
        let bodies = self.compared(None);
        compare_feature(&bodies, &self.part.features, feature, &points, &normals, input)
    }

    /// `reverse.body.check`: a body, or every solid one, looked over and audited.
    pub(crate) fn check_json(&self, body: Option<BodyId>, audit: AuditMode, tiny_area: f64) -> Result<Value, String> {
        let picked: Vec<&super::BodyView> = match body {
            Some(id) => vec![self.bodies.iter().find(|b| b.id == id).ok_or_else(|| format!("{id} is not among the bodies built."))?],
            None => self.bodies.iter().filter(|b| !self.built.body(b.id).is_some_and(|x| x.surface)).collect(),
        };
        if picked.is_empty() {
            return Err("There is no body to check — build one first.".into());
        }
        let rows: Vec<Value> = picked
            .iter()
            .filter_map(|b| {
                let built = self.built.body(b.id)?;
                Some(check_answer(b.id.as_number(), &b.name, &built.body, &b.topology, &b.part_mesh, &self.part.features, tiny_area, audit, AUDIT_DEFAULTS, None))
            })
            .collect();
        Ok(json!({"bodies": rows}))
    }

    /// How two bodies stand to each other, and the line the status bar says it in.
    pub(crate) fn clearance_of(&self, a: BodyId, b: BodyId) -> Result<(Clearance, String), String> {
        let (Some(ba), Some(bb)) = (self.built.body(a), self.built.body(b)) else { return Err("The bodies are not built.".into()) };
        if ba.surface || bb.surface {
            let s = if ba.surface { &ba.name } else { &bb.name };
            return Err(format!("{s} is a surface — a clearance is between solids."));
        }
        let c = clearance(&ba.body, &bb.body)?;
        Ok((c, clearance_text(&ba.name, &bb.name, &c)))
    }

    /// `reverse.body.clearance`: a body against others — every other solid one by default —
    /// nearest first.
    pub(crate) fn clearance_json(&self, body: BodyId, against: Option<&[BodyId]>) -> Result<Value, String> {
        let b = self.built.body(body).ok_or_else(|| format!("{body} is not among the bodies built."))?;
        if b.surface {
            return Err(format!("{} is a surface — a clearance is between solids.", b.name));
        }
        let others: Vec<&mm_reverse::build::BuiltBody> = match against {
            Some(ids) => ids.iter().filter_map(|id| self.built.body(*id)).collect(),
            None => self.built.bodies.iter().filter(|x| x.id != body && !x.surface).collect(),
        };
        if others.iter().any(|x| x.id == body) {
            return Err("A body is not measured against itself.".into());
        }
        if others.is_empty() {
            return Err("There is no other solid body to measure it against.".into());
        }
        let mut rows = Vec::new();
        for o in others {
            let c = clearance(&b.body, &o.body)?;
            let mut row = json!({"id": o.id.as_number(), "name": o.name, "gap": c.gap});
            if c.overlap > 0.0 {
                row["overlap"] = json!(c.overlap);
            }
            if c.gap == 0.0 {
                row["touching"] = json!(true);
            }
            rows.push(row);
        }
        rows.sort_by(|x, y| x["gap"].as_f64().unwrap_or(0.0).total_cmp(&y["gap"].as_f64().unwrap_or(0.0)).then(y["overlap"].as_f64().unwrap_or(0.0).total_cmp(&x["overlap"].as_f64().unwrap_or(0.0))));
        Ok(json!({"body": {"id": body.as_number(), "name": b.name}, "against": rows}))
    }

    /// `reverse.body.volume_compare`: the visible solid bodies against the scan as solids —
    /// meshed to the map's chord for the tolerance — each region named by the body face nearest
    /// its deepest point.
    pub(crate) fn volume_compare_json(&mut self, tolerance: f64, cell: Option<f64>, regions: usize, within: Option<(DVec3, DVec3)>, kinds: &[VolumeKind]) -> Result<Value, String> {
        if self.scan.is_empty() {
            return Err("There is no scan to compare the bodies with.".into());
        }
        let parts = self.scan_parts();
        let picked: Vec<&mm_reverse::build::BuiltBody> = self.built.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id) && !b.surface).collect();
        if picked.is_empty() {
            return Err("There is no solid body to compare — build one first.".into());
        }
        // The chord the bodies are meshed to for a tolerance (ScanRuler's mapChord).
        let chord = COMPARE_CHORD_MM.max(((2.0 * tolerance).clamp(0.25, 2.0) / 4.0).min(tolerance * 0.2));
        let mut positions = Vec::new();
        let mut triangles = Vec::new();
        let mut owner: Vec<(usize, u32)> = Vec::new();
        for (k, b) in picked.iter().enumerate() {
            let m = b.body.tessellate(chord).map_err(|e| format!("{} could not be meshed: {e}", b.name))?;
            let base = positions.len() as u32;
            positions.extend(m.positions.iter().map(|p| mm_reverse::build::dv3(*p)));
            for (t, tri) in m.triangles.iter().enumerate() {
                triangles.push([tri[0] + base, tri[1] + base, tri[2] + base]);
                owner.push((k, m.tri_face.get(t).copied().unwrap_or(0)));
            }
        }
        let mut sp = Vec::new();
        let mut st = Vec::new();
        for p in parts.iter() {
            let base = sp.len() as u32;
            sp.extend_from_slice(&p.positions);
            st.extend(p.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        }
        let v = volume_deviation((&positions, &triangles), (&sp, &st), &VolumeOptions { tolerance, cell, within })?;
        let name_face = |tri: Option<usize>| -> (Value, Value) {
            let Some((k, face)) = tri.and_then(|t| owner.get(t)) else { return (Value::Null, Value::Null) };
            let b = picked[*k];
            let topology = self.bodies.iter().find(|x| x.id == b.id).map(|x| x.topology.clone());
            (json!({"id": b.id.as_number(), "name": b.name}), mm_reverse::compare_detail::face_out(topology.as_deref(), *face as i64, &self.part.features))
        };
        let mut out = volume_out(&v, tolerance, regions, kinds, &name_face);
        out["bodies"] = Value::Array(picked.iter().map(|b| json!({"id": b.id.as_number(), "name": b.name})).collect());
        Ok(out)
    }

    /// What the Reverse Engineer draws into a picture of a section on `sheet`.
    pub(crate) fn section_layer_json(&self, sheet: &SectionFrame) -> Option<Value> {
        let meshes: Vec<(String, mm_reverse::build::BodyMesh, bool)> = self
            .built
            .bodies
            .iter()
            .filter(|b| !self.hidden_bodies.contains(&b.id))
            .filter_map(|b| b.body.tessellate(CUT_CHORD_MM).ok().map(|m| (b.name.clone(), mm_reverse::build::BodyMesh { positions: m.positions.iter().map(|p| mm_reverse::build::dv3(*p)).collect(), triangles: m.triangles.clone(), ..Default::default() }, b.surface)))
            .collect();
        let bodies: Vec<LayerBody> = meshes.iter().map(|(n, m, s)| LayerBody { name: n, positions: &m.positions, triangles: &m.triangles, surface: *s }).collect();
        section_layer(sheet, &bodies, &self.part.features, self.model_radius() * 2.0)
    }

    /// Compare in Deviation: the visible bodies, meshed to a hundredth of a millimetre, go to
    /// the Mesh workspace as the reference its Deviation tool measures the scan against — in
    /// the scan's own frame, no best fit run.
    pub(crate) fn compare_in_deviation(&mut self) {
        if self.scan.is_empty() {
            self.status = "There is no scan to compare the bodies with.".into();
            return;
        }
        let picked: Vec<&mm_reverse::build::BuiltBody> = self.built.bodies.iter().filter(|b| !self.hidden_bodies.contains(&b.id) && !b.surface).collect();
        if picked.is_empty() {
            self.status = "There is no solid body to compare — build one first.".into();
            return;
        }
        let mut positions: Vec<DVec3> = Vec::new();
        let mut triangles: Vec<[u32; 3]> = Vec::new();
        for b in picked {
            let Ok(m) = b.body.tessellate(COMPARE_CHORD_MM) else {
                self.status = format!("{} could not be meshed for the comparison.", b.name);
                return;
            };
            let base = positions.len() as u32;
            positions.extend(m.positions.iter().map(|p| self.to_scene(mm_reverse::build::dv3(*p))));
            triangles.extend(m.triangles.iter().map(|t| [t[0] + base, t[1] + base, t[2] + base]));
        }
        let (positions, triangles) = mm_reverse::build::welded(&positions, &triangles, 1e-7);
        let (mesh, _) = Mesh::from_triangles(&positions, &triangles, None);
        self.vb.to_compare = Some((COMPARE_REFERENCE_NAME.to_string(), mesh));
        self.status = "The scan measured against the bodies from Reverse engineer, in the scan’s own frame — no best fit was run.".into();
    }

    /// The reference for the Mesh workspace's Deviation tool, taken by the shell.
    pub fn take_compare(&mut self) -> Option<(String, Mesh)> {
        self.vb.to_compare.take()
    }

    /// The view bar's, the maps' and the checks' script steps; false for a step that is none of
    /// them.
    pub(crate) fn viewcmp_step(&mut self, cmd: &str, args: &[&str]) -> bool {
        let num = |i: usize| args.get(i).and_then(|v| v.parse::<f64>().ok());
        let body_at = |ws: &ReverseWorkspace, i: usize| args.get(i).and_then(|v| v.parse::<usize>().ok()).and_then(|k| ws.bodies.get(k)).map(|b| b.id);
        let log = |what: &str, v: &Result<Value, String>| match v {
            Ok(v) => log::info!("reverse {what}: {v}"),
            Err(e) => log::info!("reverse {what} refused: {e}"),
        };
        match cmd {
            // viewbar:<switch>[:<value>]
            "viewbar" => {
                let on = |v: bool| args.get(1).map_or(!v, |a| *a == "on" || *a == "1");
                match args.first().copied().unwrap_or("") {
                    "scan" => self.show_scan = on(self.show_scan),
                    "bodies" => self.show_bodies = on(self.show_bodies),
                    "scantransparent" => self.vb.scan_see_through = Some(on(self.scan_see_through)),
                    "scanmesh" => self.vb.scan_mesh = on(self.vb.scan_mesh),
                    "bodiestransparent" => self.vb.bodies_see_through = on(self.vb.bodies_see_through),
                    // The part's Overlaps switch (PICKLIST's).
                    "overlaps" => {
                        let shown = on(self.part.overlaps_shown);
                        self.set_overlaps_shown(shown);
                    }
                    "elements" => self.vb.elements_shown = on(self.vb.elements_shown),
                    "curvature" => {
                        self.vb.curvature_map = on(self.vb.curvature_map);
                        if self.vb.curvature_map {
                            self.deviation.on = false;
                        }
                    }
                    "deviation" => {
                        self.deviation.on = on(self.deviation.on);
                        if self.deviation.on {
                            self.vb.curvature_map = false;
                        }
                    }
                    "tangent" => {
                        self.vb.tangent_edges = match args.get(1).copied() {
                            Some("dimmed") => TangentEdges::Dimmed,
                            Some("off") => TangentEdges::Removed,
                            _ => TangentEdges::Visible,
                        }
                    }
                    "sketchview" => match args.get(1).copied() {
                        Some("none") => self.vb.sketch_model_shown = false,
                        Some("full") => {
                            self.vb.sketch_model_shown = true;
                            self.vb.sketch_sliced = false;
                        }
                        _ => {
                            self.vb.sketch_model_shown = true;
                            self.vb.sketch_sliced = true;
                        }
                    },
                    "plane" => {
                        let axis = match args.get(1).copied() {
                            Some("yz") | Some("x") => WorldAxis::X,
                            Some("xz") | Some("y") => WorldAxis::Y,
                            _ => WorldAxis::Z,
                        };
                        // The plane's eye (PICKLIST's).
                        self.set_plane_shown(axis, args.get(2).is_none_or(|a| *a == "on"));
                    }
                    other => log::warn!("viewbar step: {other} is no switch of the view bar"),
                }
            }
            // section:on|off|flip|offset:<mm>|across:<x|y|z>|turn:<u|v>:<deg>
            "section" => match args.first().copied().unwrap_or("on") {
                "on" => {
                    let view = self.to_part(self.view.camera.target - self.view.camera.eye()).normalize_or_zero();
                    let centre = self.scan_centre().unwrap_or(DVec3::ZERO);
                    self.vb.section.open(view, centre);
                }
                "off" => self.vb.section.close(),
                "flip" => self.vb.section.flip(),
                "offset" => self.vb.section.set_offset(num(1).unwrap_or(0.0)),
                "across" => {
                    let a = match args.get(1).copied() {
                        Some("x") => WorldAxis::X,
                        Some("y") => WorldAxis::Y,
                        _ => WorldAxis::Z,
                    };
                    let elements = self.part.elements.clone();
                    self.vb.section.set_ref(mm_reverse::section_view::SectionRef::World(a), None, &elements);
                }
                "turn" => {
                    if let Some((_, g)) = self.vb.section.frames() {
                        let about = if args.get(1) == Some(&"v") { g.v } else { g.u };
                        self.vb.section.turn(about, num(2).unwrap_or(10.0).to_radians());
                    }
                }
                other => log::warn!("section step: {other} is no section view action"),
            },
            "livemap" => self.vb.live_map = args.first() != Some(&"off"),
            // ghost:<to>[:<from>]: an extrude box opened on the last sketch and left open, its
            // ghost on the part (the live map's test).
            "ghost" => {
                self.new_extrude(None, Vec::new());
                if let super::Mode::Feature(edit) = &mut self.mode {
                    if let mm_reverse::model::FeatureKind::Extrude(e) = &mut edit.kind {
                        e.extent = mm_reverse::model::ExtrudeExtent::Distance { from: num(1).unwrap_or(0.0), to: num(0).unwrap_or(10.0) };
                    }
                }
                self.request_build();
            }
            "compare" | "comparescore" => {
                let range = num(0).unwrap_or(self.deviation.range);
                let input = if cmd == "compare" { CompareInput::new(range) } else { CompareInput::score(range) };
                let r = self.compare_json(&input, cmd == "comparescore", None);
                self.status = match &r {
                    Ok(v) => format!("Compared: {} of {} scan points read lie within ±{} mm.", v["within"], v["read"], mm_reverse::js_fmt::num(range)),
                    Err(e) => e.clone(),
                };
                log(cmd, &r);
            }
            "comparefeature" => {
                let id = num(0).unwrap_or(0.0) as FeatureId;
                let r = self.compare_feature_json(id, &CompareInput::new(num(1).unwrap_or(self.deviation.range)));
                log(cmd, &r);
            }
            "check" => {
                let body = body_at(self, 0);
                let mode = match args.get(1).copied() {
                    Some("full") => AuditMode::Full,
                    Some("lists") => AuditMode::Lists,
                    Some("off") => AuditMode::Off,
                    _ => AuditMode::Brief,
                };
                let r = self.check_json(body, mode, TINY_FACE_MM2);
                log(cmd, &r);
            }
            "clearance" => {
                let r = match (body_at(self, 0), body_at(self, 1)) {
                    (Some(a), Some(b)) => self.clearance_of(a, b).map(|(c, text)| {
                        self.status = text.clone();
                        json!({"gap": c.gap, "overlap": c.overlap, "text": text})
                    }),
                    (Some(a), None) => self.clearance_json(a, None),
                    _ => Err("clearance takes the bodies by their place in the list".into()),
                };
                log(cmd, &r);
            }
            "sectionlayer" => {
                let axis = match args.first().copied() {
                    Some("x") => WorldAxis::X,
                    Some("y") => WorldAxis::Y,
                    _ => WorldAxis::Z,
                };
                let frame = SectionFrame::along(mm_reverse::frame::world_cut_axis(axis, DVec3::ZERO), num(1).unwrap_or(0.0), None);
                let r = frame.and_then(|f| self.section_layer_json(&f)).ok_or_else(|| "nothing to draw on that sheet".to_string());
                log(cmd, &r);
            }
            "compareindeviation" => self.compare_in_deviation(),
            // volumecompare[:<tolerance>[:<cell>]]
            "volumecompare" => {
                let r = self.volume_compare_json(num(0).unwrap_or(self.deviation.range), num(1), VOLUME_REGIONS, None, &[VolumeKind::Excess, VolumeKind::Missing, VolumeKind::Unsure]);
                log(cmd, &r);
            }
            "viewreport" => {
                let r = self.vb.map.report;
                log::info!(
                    "reverse view: scan {} (veil {}), bodies {} (see-through {}), tangent {:?}, map {:?}, curvature {}, section {:?}, sketch part {}/{}, planes {:?}",
                    self.show_scan,
                    self.scan_see_through,
                    self.show_bodies,
                    self.vb.bodies_see_through,
                    self.vb.tangent_edges,
                    r,
                    self.vb.curvature_map,
                    self.vb.section.frames().map(|f| (f.0.origin, f.0.normal)),
                    self.vb.sketch_model_shown,
                    self.vb.sketch_sliced,
                    self.planes_drawn()
                );
            }
            _ => return false,
        }
        true
    }
}
