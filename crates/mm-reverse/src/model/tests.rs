// SPDX-License-Identifier: AGPL-3.0-only
//! The feature model's tests: every kind written and read back in ScanRuler's shape, the body
//! ids, the STEP key, and what a feature builds on — the cases of ScanRuler's own tests
//! (`reRound19Answers.test.ts`, `rePatternStep.test.ts`, `reStore.test.ts`,
//! `reImportMove.test.ts`), © Stefan Hermann (CNC Kitchen), AGPL-3.0-only.

use super::*;
use crate::pick_query::{StoredPick, StoredWhere};
use crate::topo::{EdgeKind, EdgeSide};

fn face(kind: FaceKind, centroid: [f64; 3], dir: [f64; 3], origin: i64) -> FaceRef {
    FaceFingerprint { kind, centroid: DVec3::from(centroid), dir: DVec3::from(dir), area: 100.0, radius: if kind == FaceKind::Cylinder { Some(3.0) } else { None }, center: None, origin }
}

fn edge(origin: i64) -> EdgeRef {
    EdgeFingerprint {
        kind: EdgeKind::Line,
        mid: DVec3::new(20.0, 0.0, 10.0),
        dir: DVec3::X,
        length: 40.0,
        radius: None,
        faces: vec![EdgeSide { kind: FaceKind::Plane, dir: DVec3::Z }, EdgeSide { kind: FaceKind::Plane, dir: DVec3::NEG_Y }],
        origin,
    }
}

fn frame() -> SectionFrame {
    SectionFrame { origin: DVec3::new(0.0, 0.0, 5.0), normal: DVec3::Z, u: DVec3::X, v: DVec3::Y }
}

fn exprs() -> Exprs {
    [("distance", "wall*2")].into_iter().collect()
}

fn feature(id: FeatureId, name: &str, kind: FeatureKind) -> Feature {
    Feature { id, name: name.into(), flags: FeatureFlags::default(), kind }
}

fn sketch_kind() -> FeatureKind {
    let mut s = Sketch::default();
    s.add_rectangle(DVec2::new(0.0, 0.0), DVec2::new(40.0, 30.0));
    FeatureKind::Sketch(SketchFeature { plane: PlaneRef::World { axis: WorldAxis::Z, offset: 0.0, up: None }, frame: frame(), sketch: s, exprs: Exprs::new(), image: None, inset: None })
}

fn pad(sketch: FeatureId, op: BodyOp, from: f64, to: f64) -> FeatureKind {
    FeatureKind::Extrude(ExtrudeFeature {
        sketch_id: sketch,
        regions: vec!["1,2,3,4".into()],
        region_pick: None,
        regions_by_place: vec![],
        face: None,
        extent: ExtrudeExtent::Distance { from, to },
        draft_deg: 0.0,
        direction: None,
        op,
        target: None,
        exprs: Exprs::new(),
    })
}

/// One feature of every kind, each with the fields it can carry filled.
fn every_kind() -> Vec<FeatureKind> {
    let body = BodyId::new(2);
    let island = island_id(2, 1);
    let picks = vec![StoredPick { query: StoredWhere { made_by: Some(4), kind: Some("line".into()), ..Default::default() }, found: vec![edge(4)] }];
    let face_picks = vec![StoredPick { query: StoredWhere { made_by: Some(2), ..Default::default() }, found: vec![face(FaceKind::Plane, [0.0, 0.0, 10.0], [0.0, 0.0, 1.0], 2)] }];
    let top = face(FaceKind::Plane, [20.0, 15.0, 10.0], [0.0, 0.0, 1.0], 2);
    let bore = face(FaceKind::Cylinder, [10.0, 10.0, 5.0], [0.0, 0.0, 1.0], 4);
    vec![
        FeatureKind::Sketch(SketchFeature {
            plane: PlaneRef::BodyFace { body, face: top, frame: frame(), offset: Some(1.5) },
            frame: frame(),
            sketch: Sketch::default(),
            exprs: exprs(),
            image: Some(SketchImage { src: "data:image/png;base64,AAAA".into(), width: 4, height: 3, origin: DVec2::new(-1.0, 2.0), mm_per_px: 0.5, angle_deg: 10.0, opacity: 0.6, name: Some("photo.png".into()) }),
            inset: Some(SketchInset { of: 1, by: 2.0, keep: vec![3, 4] }),
        }),
        FeatureKind::Extrude(ExtrudeFeature {
            sketch_id: 1,
            regions: vec!["1,2,3,4".into()],
            region_pick: Some(RegionPick { given: RegionGiven::List(vec![RegionAsk::Key("1,2,3,4".into()), RegionAsk::At { at: DVec2::new(3.0, 4.0) }]), keys: vec!["1,2,3,4".into()] }),
            regions_by_place: vec!["1,2,3,4".into()],
            face: Some(FaceProfile { body, face: top }),
            extent: ExtrudeExtent::ToFace { from: 1.0, body: island, face: top },
            draft_deg: 3.0,
            direction: Some(DVec3::new(0.0, 0.1, 1.0)),
            op: BodyOp::Cut,
            target: Some(island),
            exprs: exprs(),
        }),
        FeatureKind::Revolve(RevolveFeature {
            sketch_id: 1,
            regions: vec!["5".into()],
            region_pick: Some(RegionPick { given: RegionGiven::Word(RegionWord::Largest), keys: vec!["5".into()] }),
            regions_by_place: vec![],
            axis: AxisRef::SketchLine { entity: 3 },
            angle_deg: 720.0,
            pitch: Some(2.5),
            left_handed: true,
            op: BodyOp::Join,
            target: None,
            exprs: exprs(),
        }),
        FeatureKind::Extract(ExtractFeature { element: Some(7), section: None, thickness: 2.0, op: BodyOp::New, target: None, exprs: exprs() }),
        FeatureKind::Sheet(SheetFeature { sketch_id: 1, from: -5.0, to: 5.0, exprs: exprs() }),
        FeatureKind::Loft(LoftFeature { sketches: vec![1, 3], ruled: true, op: BodyOp::Join, target: Some(body) }),
        FeatureKind::Sweep(SweepFeature {
            profile: 1,
            path: None,
            route: Some(SweepRoute { points: vec![DVec3::ZERO, DVec3::new(0.0, 0.0, 10.0), DVec3::new(10.0, 0.0, 10.0)], bend_radius: 3.0, radii: Some(vec![None]) }),
            op: BodyOp::New,
            target: None,
            exprs: exprs(),
        }),
        FeatureKind::Thicken(ThickenFeature { sheet: island, thickness: 1.2, side: ThickenSide::Both, op: BodyOp::Join, target: Some(body), exprs: exprs() }),
        FeatureKind::Surface(SurfaceFeature {
            region: vec![1, 2, 3, 40, 41],
            fit: SurfaceFitParams { net: Some(NetSize { u: 8, v: 6 }), degree: 3, expand_pct: 10.0, smoothness: 0.01, robust: Some(true), normal: Some(DVec3::Z) },
            shape: Some(SurfaceShape::Cylinder),
            constrain: Some(ConstrainMode::Force { relation: Relation::Along, reference: "surface:5".into() }),
            region_box: Some(RegionBox { bounds: BoxBounds { min: DVec3::splat(-1.0), max: DVec3::splat(1.0) }, normal: Some(Facing { dir: DVec3::Z, within_deg: Some(20.0) }), exclude: Some(vec![BoxBounds { min: DVec3::ZERO, max: DVec3::ONE }]), limit: Some(5000) }),
            region_stale: Some(RegionStale::Why("the scan is gone".into())),
            exprs: exprs(),
        }),
        FeatureKind::Network(NetworkFeature {
            nodes: vec![
                NetworkNode { id: 1, at: DVec3::ZERO, on: Some(NodeOn::Edge { body, edge: edge(2), t: 0.25 }), corner: Some(true) },
                NetworkNode { id: 2, at: DVec3::X, on: Some(NodeOn::Vertex { body: island, vertex: edge(2), end: 1 }), corner: None },
            ],
            curves: vec![NetworkCurve { id: 1, from: 1, to: 2, via: vec![DVec3::new(0.5, 0.1, 0.0)], crease: Some(true), along: Some(CurveAlong { body, edge: edge(2) }), side: Some(CurveSide { body, face: top, mode: SideMode::Tangent }) }],
            cells: vec![NetworkCell { seed: DVec3::new(0.3, 0.3, 0.0), centre: Some(DVec3::new(0.5, 0.5, 0.0)), corners: Some(vec![1, 2]) }],
            spans: BTreeMap::from([("1".to_string(), 4)]),
            fit: NetworkFitParams { smoothness: 0.02, robust: Some(false), tolerance: Some(0.1) },
        }),
        FeatureKind::Trim(TrimFeature { tools: vec![TrimTool::Body { id: island }, TrimTool::Element { id: 7 }], tolerance: 0.3, overlap: 0.6, exprs: exprs() }),
        FeatureKind::Import(ImportFeature { file: StepFileRef { key: step_file_key("ISO-10303-21; A"), name: "a.step".into(), size: 15 } }),
        FeatureKind::MoveBody(MoveBodyFeature { bodies: vec![body, island], pivot: DVec3::new(1.0, 2.0, 3.0), translate: DVec3::new(0.0, 0.0, 10.0), rotate: DVec3::new(0.0, 0.0, 90.0), copy: true, exprs: exprs() }),
        FeatureKind::Shell(ShellFeature { body, faces: vec![top], faces_where: face_picks.clone(), thickness: 2.0, exprs: exprs() }),
        FeatureKind::Fillet(FilletFeature { body, edges: vec![edge(4)], edges_where: picks.clone(), radius: 1.0, chain: true, skip_failing: true, variable: vec![VariableEdge { edge: edge(2), radius: [1.0, 3.0] }], exprs: exprs() }),
        FeatureKind::Chamfer(ChamferFeature { body, edges: vec![edge(2)], edges_where: picks, distance: 1.0, chain: false, skip_failing: false, mode: Some(ChamferMode::DistanceAngle), distance2: Some(2.0), angle_deg: Some(30.0), flip: true, exprs: exprs() }),
        FeatureKind::Hole(HoleFeature { body, axis: HoleAxis::Point { face: top, at: DVec3::new(10.0, 10.0, 10.0) }, radius: 1.6, depth: None, insert: Some("M3 × 4".into()), exprs: exprs() }),
        FeatureKind::Thread(ThreadFeature { body, face: bore, standard: ThreadStandard::Print45, diameter: 6.0, pitch: 1.0, left_handed: false, length: Some(8.0), start: 1.0, flip: true, offset: 0.1, chamfer: Some(true), exprs: exprs() }),
        FeatureKind::CutPlane(CutPlaneFeature { body, plane: PlaneRef::Radial { axis: RadialAxis::Element { id: 7 }, angle_deg: 30.0, offset: Some(2.0) }, keep: -1, exprs: exprs() }),
        FeatureKind::Mirror(MirrorFeature { body, plane: MirrorPlane::Face { face: top }, join: true, exprs: exprs() }),
        FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Intersect, body, tool: island, keep: true }),
        FeatureKind::Split(SplitFeature { body, tools: vec![island], keep: SplitKeep::Between, tolerance: 0.2, overlap: 0.5, within: Some(3.0), exprs: exprs() }),
        FeatureKind::Pattern(PatternFeature { body, mode: PatternMode::Linear, axis: AxisRef::World { axis: WorldAxis::X }, count: 3, step: 12.0, axis2: Some(AxisRef::Element { id: 8 }), count2: Some(2), step2: Some(9.0), exprs: exprs() }),
        FeatureKind::Draft(DraftFeature {
            body,
            neutral: MirrorPlane::Plane(PlaneRef::Plane { id: 9, offset: Some(-1.0), up: Some(WorldAxis::Z) }),
            faces: vec![top],
            faces_where: face_picks.clone(),
            angle_deg: 3.0,
            groups: vec![DraftGroup { faces: vec![bore], angle_deg: 5.0 }],
            exprs: exprs(),
        }),
        FeatureKind::MoveFace(MoveFaceFeature { body, faces: vec![top], faces_where: vec![], offset: -0.5, exprs: exprs() }),
        FeatureKind::DeleteFace(DeleteFaceFeature { body, faces: vec![bore], faces_where: face_picks }),
        FeatureKind::OffsetBody(OffsetBodyFeature { body, distance: 0.4, keep: true, exprs: exprs() }),
    ]
}

#[test]
fn every_kind_goes_to_json_and_comes_back_the_same() {
    let mut part = Part::default();
    part.features.push(feature(1, "Sketch 1", sketch_kind()));
    for (i, k) in every_kind().into_iter().enumerate() {
        let mut f = feature(10 + i as u32, &format!("F{i}"), k);
        f.flags.suppressed = i % 3 == 0;
        f.flags.visible = i % 2 == 0;
        part.features.push(f);
    }
    part.elements = vec![
        Element { id: 7, name: "Cylinder 1".into(), geometry: ElementGeometry::Cylinder { center: DVec3::ZERO, axis: DVec3::Z, radius: 5.0, length: 20.0, coverage: 360.0 }, stats: Some(FitStats { rms: 0.01, sigma: 0.012, used_points: 900, region_size: 1000, form_error: Some(0.05), residual_min: None, residual_max: None }) },
        Element { id: 9, name: "Plane 1".into(), geometry: ElementGeometry::Plane { center: DVec3::ZERO, normal: DVec3::Z, basis_u: DVec3::X, basis_v: DVec3::Y, extent_u: 10.0, extent_v: 5.0 }, stats: None },
        Element { id: 11, name: "Section A".into(), geometry: ElementGeometry::Section { frame: frame() }, stats: None },
    ];
    part.parameters = vec![Parameter { id: 1, name: "wall".into(), kind: ParameterKind::Length, expression: "2.5".into(), comment: Some("the shell's wall".into()) }];
    part.step_files.insert(step_file_key("ISO-10303-21; A"), StepFile { name: "a.step".into(), size: 15, text: "ISO-10303-21; A".into() });
    part.body_styles.insert(island_id(2, 1), BodyStyle { name: Some("Header J1".into()), color: Some("#ff8800".into()) });
    part.body_styles.insert(BodyId::new(3), BodyStyle { name: None, color: Some("#00ff00".into()) });
    part.body_folders = vec![BodyFolder { id: 1, name: "Component 1".into(), parent: None, bodies: vec![BodyId::new(2), island_id(2, 1)] }, BodyFolder { id: 2, name: "Component 2".into(), parent: Some(1), bodies: vec![] }];
    part.notes = vec![Note { id: 1, by: Some(NoteBy::Agent), text: "Is this a fillet?".into(), at: DVec3::new(1.0, 2.0, 3.0), normal: Some(DVec3::Z), on: vec![NoteTarget::Scan, NoteTarget::Edge { body: BodyId::new(2), edge: edge(2) }, NoteTarget::Vertex { body: BodyId::new(2), point: DVec3::ONE }], done: true, reply: Some("Yes.".into()) }];
    part.rolled_to = Some(5);
    let text = part.to_json();
    let back = Part::from_json(&text).unwrap_or_else(|e| panic!("{e}\n{text}"));
    assert_eq!(back, part);
    // Written as ScanRuler writes it: the kind beside the fields, body ids as numbers.
    for kind in ["\"kind\": \"cutPlane\"", "\"kind\": \"moveBody\"", "\"kind\": \"offsetBody\"", "\"kind\": \"bodyFace\"", "\"kind\": \"toFace\"", "\"sketchId\": 1", "\"target\": 2.0009765625", "\"depth\": null", "\"given\": \"largest\"", "\"2.0009765625\": {", "\"basisU\"", "\"axis\": \"x\"", "\"mode\": \"force\"", "\"ref\": \"surface:5\""] {
        assert!(text.contains(kind), "{kind} not in the part's JSON");
    }
}

#[test]
fn a_feature_reads_from_scanrulers_json() {
    let text = r#"{"id":5,"kind":"fillet","name":"Rim round","body":2,"edges":[{"kind":"line","mid":[20,0,10],"dir":[1,0,0],"length":40,"faces":[],"origin":4}],"radius":1,"chain":false,"visible":true}"#;
    let f: Feature = serde_json::from_str(text).unwrap_or_else(|e| panic!("{e}"));
    assert_eq!(f.name, "Rim round");
    assert!(f.visible() && !f.suppressed());
    let FeatureKind::Fillet(x) = &f.kind else { panic!("not a fillet") };
    assert_eq!(x.body, BodyId::new(2));
    assert_eq!(x.edges[0].origin, 4);
    let mirror = r#"{"id":6,"kind":"mirror","name":"Mirror 1","body":2.0009765625,"plane":{"kind":"world","axis":"x","offset":0},"join":false,"visible":true}"#;
    let f: Feature = serde_json::from_str(mirror).unwrap_or_else(|e| panic!("{e}"));
    let FeatureKind::Mirror(m) = &f.kind else { panic!("not a mirror") };
    assert_eq!(m.body, island_id(2, 1));
    assert_eq!(m.plane, MirrorPlane::Plane(PlaneRef::World { axis: WorldAxis::X, offset: 0.0, up: None }));
}

#[test]
fn a_part_saved_by_the_earlier_model_is_refused_cleanly() {
    let old = r#"{"features":[{"id":1,"name":"Sketch 1","visible":true,"suppressed":false,"kind":{"Sketch":{"plane":{"World":{"axis":"Z","offset":0.0}},"sketch":{"entities":[],"constraints":[],"next_id":1}}}}],"next_id":2}"#;
    let e = Part::from_json(old).err().unwrap_or_default();
    assert!(e.contains("earlier version"), "{e}");
    let later = format!(r#"{{"version":{},"features":[],"nextId":1}}"#, PART_VERSION + 1);
    assert!(Part::from_json(&later).err().unwrap_or_default().contains("later version"));
    assert!(Part::from_json("not json").is_err());
    assert_eq!(Part::from_json(&Part::default().to_json()), Ok(Part::default()));
}

#[test]
fn body_ids_are_the_maker_and_a_fraction() {
    let id = island_id(2, 1);
    assert_eq!(id.as_number(), 2.0009765625);
    assert_eq!(body_maker(id), 2);
    assert_eq!(island_of(id), 1);
    assert_eq!(island_of(BodyId::new(7)), 0);
    assert_eq!(id.to_string(), "Body 2.0009765625");
    assert_eq!(BodyId::new(3).to_string(), "Body 3");
    assert_eq!(BodyId::from_number(5.0 + 3.0 / 1024.0), Some(island_id(5, 3)));
    assert_eq!(BodyId::from_number(-1.0), None);
    assert_eq!(serde_json::to_string(&vec![BodyId::new(2), id]).unwrap_or_default(), "[2,2.0009765625]");
    let back: Vec<BodyId> = serde_json::from_str("[2,2.0009765625,7.0]").unwrap_or_default();
    assert_eq!(back, vec![BodyId::new(2), id, BodyId::new(7)]);
}

#[test]
fn a_step_files_key_is_scanrulers_cyrb53() {
    // Computed with ScanRuler's own stepFileKey under node.
    assert_eq!(step_file_key(""), "0-wvjl67o803");
    assert_eq!(step_file_key("a"), "1-262p94epp1d");
    assert_eq!(step_file_key("ISO-10303-21; A"), "f-d39o9m8y72");
    assert_eq!(step_file_key("ISO-10303-21; B"), "f-234hxr0kc99");
    // UTF-16 units, as JavaScript counts them: the rocket is two.
    assert_eq!(step_file_key("Größe Ø 12 — 🚀 part"), "k-29f8271w2rf");
    assert_eq!(step_file_key(&"x".repeat(1000)), "rs-21ded0ooxff");
    assert_eq!(step_file_key("line1\r\nline2\n"), "d-2bohf7omig3");
    // A STEP file's worth of text: 85 000 units, its length past four base-36 digits.
    assert_eq!(step_file_key(&"#1=CARTESIAN_POINT('',(0.,0.,0.));".repeat(2500)), "1tl4-1yngvkpikho");
}

#[test]
fn a_nan_pitch_is_no_pitch_as_in_scanruler() {
    assert_eq!(revolve_max_deg(Some(f64::NAN)), FULL_TURN_DEG);
}

#[test]
fn a_circular_patterns_pitch_shares_a_full_turn_among_the_copies_and_an_open_span_among_the_gaps() {
    assert!((pattern_pitch(90.0, 4.0) - 30.0).abs() < 1e-12);
    assert!((pattern_pitch(360.0, 14.0) - 25.714285714).abs() < 1e-8);
    assert_eq!(pattern_pitch(90.0, 1.0), 0.0);
    assert!(full_turn(-360.0) && !full_turn(359.0));
    assert_eq!(revolve_max_deg(None), FULL_TURN_DEG);
    assert_eq!(revolve_max_deg(Some(0.0)), FULL_TURN_DEG);
    assert_eq!(revolve_max_deg(Some(1.5)), MAX_COIL_DEG);
}

#[test]
fn a_plane_moves_to_an_offset_with_a_faces_frame_and_a_fitted_one_takes_none() {
    let top = face(FaceKind::Plane, [0.0, 0.0, 5.0], [0.0, 0.0, 1.0], 2);
    let on_face = PlaneRef::BodyFace { body: BodyId::new(2), face: top, frame: frame(), offset: Some(1.0) };
    let moved = plane_at_offset(&on_face, 3.0);
    let PlaneRef::BodyFace { frame: f, offset, .. } = &moved else { panic!("not a face") };
    assert_eq!(*offset, Some(3.0));
    assert!((f.origin.z - 7.0).abs() < 1e-12);
    assert_eq!(plane_offset(&moved), 3.0);
    let world = PlaneRef::World { axis: WorldAxis::Z, offset: 4.0, up: None };
    assert_eq!(plane_offset(&world), 4.0);
    assert_eq!(plane_at_offset(&world, -2.0), PlaneRef::World { axis: WorldAxis::Z, offset: -2.0, up: None });
    let fitted = PlaneRef::Fitted { frame: frame(), label: None };
    assert!(!plane_takes_offset(&fitted));
    assert_eq!(plane_at_offset(&fitted, 9.0), fitted);
    assert_eq!(plane_offset(&fitted), 0.0);
    // A sketch moved: its frame carried by the change, its plane's offset set.
    let (plane, f) = with_plane_offset(&world, &frame(), 6.0);
    assert_eq!(plane, PlaneRef::World { axis: WorldAxis::Z, offset: 6.0, up: None });
    assert!((f.origin.z - 7.0).abs() < 1e-12);
}

#[test]
fn a_flat_face_is_a_sketch_plane_and_a_round_one_is_not() {
    let top = face(FaceKind::Plane, [3.0, 4.0, 10.0], [0.0, 0.0, 1.0], 2);
    let f = frame_on_face(&top).unwrap_or(frame());
    assert!((f.origin - DVec3::new(0.0, 0.0, 10.0)).length() < 1e-12);
    assert!((f.u - DVec3::X).length() < 1e-12);
    let bore = face(FaceKind::Cylinder, [3.0, 4.0, 5.0], [0.0, 0.0, 1.0], 4);
    assert!(frame_on_face(&bore).is_none());
    assert!(body_face_plane(BodyId::new(2), &bore).is_none());
    assert!(matches!(body_face_plane(BodyId::new(2), &top), Some(PlaneRef::BodyFace { .. })));
    assert!(movable_face(&top) && movable_face(&bore) && !movable_face(&face(FaceKind::Cone, [0.0; 3], [0.0, 0.0, 1.0], 1)));
    assert!(partial_round(Some(false)) && !partial_round(None) && !partial_round(Some(true)));
}

/// ScanRuler's cup: a block, a cavity cut in it, a lip on it.
fn cup() -> Vec<Feature> {
    vec![
        feature(1, "Cup sk", sketch_kind()),
        feature(2, "Cup", pad(1, BodyOp::New, 0.0, 20.0)),
        feature(3, "Cavity sk", sketch_kind()),
        feature(4, "Cavity", pad(3, BodyOp::Cut, 2.0, 20.0)),
        feature(5, "Lip sk", sketch_kind()),
        feature(6, "Lip slab", pad(5, BodyOp::Join, 5.0, 8.0)),
    ]
}

fn order(fs: &[Feature]) -> Vec<&str> {
    fs.iter().map(|f| f.name.as_str()).collect()
}

#[test]
fn a_feature_moves_up_the_list_with_what_it_builds_on_and_is_refused_before_it() {
    let fs = cup();
    let moved = moved_list(&fs, &[6, 5], Some(2)).unwrap_or_default();
    assert_eq!(order(&moved), ["Cup sk", "Cup", "Lip sk", "Lip slab", "Cavity sk", "Cavity"]);
    assert_eq!(moved_list(&fs, &[6], Some(1)).err().unwrap_or_default(), "Lip slab builds on Lip sk, which would then come after it — move Lip sk with it, or put Lip slab lower.");
    assert_eq!(moved_list(&fs, &[1], Some(2)).err().unwrap_or_default(), "Cup builds on Cup sk, which the move puts after it — move Cup with it, or put Cup sk higher.");
    assert!(moved_list(&fs, &[2], Some(2)).err().unwrap_or_default().contains("after names a feature that is moved"));
    // A round on an edge the cut made comes after the cut — by its query's madeBy, and by
    // its fingerprint's maker.
    let mut fs = fs;
    let rim = |edges: Vec<EdgeRef>, edges_where: Vec<StoredPick<EdgeRef>>| FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges, edges_where, radius: 1.0, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() });
    fs.push(feature(7, "Rim round", rim(vec![], vec![StoredPick { query: StoredWhere { made_by: Some(4), between: Some(vec![DVec3::X, DVec3::Y]), ..Default::default() }, found: vec![] }])));
    assert!(moved_list(&fs, &[7], Some(2)).err().unwrap_or_default().contains("Rim round builds on Cavity"));
    fs[6] = feature(7, "Rim round", rim(vec![edge(4)], vec![]));
    assert!(moved_list(&fs, &[7], Some(2)).err().unwrap_or_default().contains("Rim round builds on Cavity"));
    assert_eq!(order(&moved_list(&fs, &[5, 6], None).unwrap_or_default()), ["Lip sk", "Lip slab", "Cup sk", "Cup", "Cavity sk", "Cavity", "Rim round"]);
}

#[test]
fn what_a_feature_builds_on_takes_bodies_by_their_maker() {
    let fs = cup();
    assert_eq!(references_of(&fs[1]), vec![BodyId::new(1)]);
    assert_eq!(built_on(&fs[3]), vec![3]);
    let mut on_island = pad(5, BodyOp::Cut, 0.0, 3.0);
    if let FeatureKind::Extrude(e) = &mut on_island {
        e.target = Some(island_id(2, 1));
        e.extent = ExtrudeExtent::ToBody { from: 0.0, body: BodyId::new(4), reverse: false };
    }
    let f = feature(8, "Cut on the island", on_island);
    assert_eq!(references_of(&f), vec![BodyId::new(5), island_id(2, 1), BodyId::new(4)]);
    assert_eq!(built_on(&f), vec![5, 2, 4]);
    // A sketch on a body's face builds on the body.
    let mut on_face = sketch_kind();
    if let FeatureKind::Sketch(s) = &mut on_face {
        s.plane = PlaneRef::BodyFace { body: island_id(6, 2), face: face(FaceKind::Plane, [0.0; 3], [0.0, 0.0, 1.0], 6), frame: frame(), offset: None };
    }
    let sk = feature(9, "Sketch on top", on_face);
    assert_eq!(built_on(&sk), vec![6]);
    let mut fs = fs;
    fs.push(f);
    fs.push(sk);
    // What deleting a feature would orphan: a feature on a further body of its step too.
    assert_eq!(dependents_of(2, &fs), vec![8]);
    assert_eq!(dependents_of(6, &fs), vec![9]);
    assert_eq!(dependents_of(1, &fs), vec![2]);
    let boolean = feature(10, "Combine 1", FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Join, body: BodyId::new(2), tool: island_id(6, 1), keep: false }));
    assert_eq!(references_of(&boolean), vec![BodyId::new(2), island_id(6, 1)]);
    let held = FeatureKind::Surface(SurfaceFeature {
        region: vec![],
        fit: SurfaceFitParams { net: None, degree: 3, expand_pct: 0.0, smoothness: 0.0, robust: None, normal: None },
        shape: None,
        constrain: Some(ConstrainMode::Force { relation: Relation::On, reference: "surface:4".into() }),
        region_box: None,
        region_stale: None,
        exprs: Exprs::new(),
    });
    assert_eq!(references_of(&feature(11, "Surface 1", held)), vec![BodyId::new(4)]);
}

#[test]
fn a_part_names_its_features_by_kind_and_deletes_what_depends_on_one() {
    let mut part = Part::default();
    let s = part.add(sketch_kind());
    let e = part.add(pad(s, BodyOp::New, 0.0, 10.0));
    let c = part.add(FeatureKind::Boolean(BooleanFeature { op: BooleanOp::Cut, body: BodyId::new(e), tool: BodyId::new(e), keep: false }));
    let k = part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(e), plane: PlaneRef::World { axis: WorldAxis::Z, offset: 5.0, up: None }, keep: 1, exprs: Exprs::new() }));
    let names: Vec<&str> = part.features.iter().map(|f| f.name.as_str()).collect();
    assert_eq!(names, ["Sketch 1", "Extrude 1", "Combine 1", "Cut 1"]);
    assert_eq!((s, e, c, k), (1, 2, 3, 4));
    let gone = part.remove(s);
    assert_eq!(gone, vec![1, 2, 3, 4]);
    assert!(part.features.is_empty());
    // Ids are never given out again.
    assert_eq!(part.add(sketch_kind()), 5);
    assert_eq!(part.features[0].name, "Sketch 2");
}

#[test]
fn an_edited_feature_keeps_the_queries_its_picks_were_given_by() {
    let picks = vec![StoredPick { query: StoredWhere { made_by: Some(4), ..Default::default() }, found: vec![edge(4)] }];
    let round = |edges_where: Vec<StoredPick<EdgeRef>>, radius: f64| feature(7, "Fillet 1", FeatureKind::Fillet(FilletFeature { body: BodyId::new(2), edges: vec![edge(4)], edges_where, radius, chain: false, skip_failing: false, variable: vec![], exprs: Exprs::new() }));
    let old = round(picks.clone(), 1.0);
    let kept = with_kept_queries(&old, round(vec![], 2.0));
    let FeatureKind::Fillet(x) = &kept.kind else { panic!("not a fillet") };
    assert_eq!(x.edges_where, picks);
    assert_eq!(x.radius, 2.0);
    // Another kind keeps nothing of the old.
    let other = feature(7, "Cut 1", FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(2), plane: PlaneRef::World { axis: WorldAxis::Z, offset: 5.0, up: None }, keep: 1, exprs: Exprs::new() }));
    assert_eq!(with_kept_queries(&old, other.clone()), other);
}

#[test]
fn a_feature_added_while_the_list_is_rolled_back_goes_in_at_the_bar() {
    let mut part = Part::default();
    let s = part.add(sketch_kind());
    let e = part.add(pad(s, BodyOp::New, 0.0, 10.0));
    let late = part.add(sketch_kind());
    part.rolled_to = Some(2);
    let k = part.add(FeatureKind::CutPlane(CutPlaneFeature { body: BodyId::new(e), plane: PlaneRef::World { axis: WorldAxis::Z, offset: 5.0, up: None }, keep: 1, exprs: Exprs::new() }));
    // Built: the sketch, the extrude and the cut; under the bar, the later sketch as it was.
    assert_eq!(part.features.iter().map(|f| f.id).collect::<Vec<_>>(), vec![s, e, k, late]);
    assert_eq!(part.rolled_to, Some(3));
}
