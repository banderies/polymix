

# fits

## public_api
Everything below is re-exported flat: from crate::mesh_graph and from crate::fit3d. Vertex ids are u32. Seeds and selections are &[u32]. Every fit returns Result<RegionFit<XFit>, FitError>, where RegionFit<T> {fit: T, region: Vec<u32>, told: Option<FitTold>} derefs to T and converts into FitOutput with .into().

MESH GRAPH (crate::mesh_graph):
- struct MeshGraph {positions: Vec<DVec3>, indices: Vec<u32> (3 per triangle), normals: Vec<DVec3>, adj_offsets: Vec<u32>, adj_list: Vec<u32>, vertex_count, triangle_count, bbox_diag: f64}; .neighbours(v) -> &[u32], .position(v), .normal(v)
- MeshGraph::from_mesh(&mm_core::Mesh, keep_winding: bool) -> Result<MeshGraph, GraphError>
- build_mesh_graph(ParsedMesh{kind: ParsedKind::{Soup, Indexed}, positions: Vec<DVec3>, indices: Option<Vec<u32>>, keep_winding}) -> Result<MeshGraph, GraphError>; constructors ParsedMesh::soup(..) and ParsedMesh::indexed(..)
- build_mesh_graph_from_flat(kind, positions: &[f32], indices: Option<&[u32]>, keep_winding) -> Result<MeshGraph, GraphError>
- GraphError(pub String) (thiserror)
- build_adjacency(indices, vertex_count) -> Adjacency{offsets, list}
- compute_vertex_normals(&[DVec3], &[u32]) -> Vec<DVec3>
- orient_normals_outward(&[DVec3], &mut [u32], &mut [DVec3]) -> bool
- extent_of(&[DVec3]) -> Extent
- tetra_volume6(a, b, c, o) -> f64
- weld_triangle_soup(&[DVec3]) -> (Vec<DVec3>, Vec<u32>)
- filter_degenerate_triangles(&[u32]) -> Vec<u32>
- triangles_within(indices, vertices, vertex_count) -> Vec<u32>
- find_root(&mut [u32], usize) -> usize; UnionFind::new(n) with .find and .union; vertex_components(indices, n) -> UnionFind
- mean_curvature(&MeshGraph, smoothing: usize /*CURVATURE_SMOOTHING = 4*/) -> Vec<f32>
- curvature_range(&[f32], size: f64) -> f64
- curvature_scale(range) -> FieldScale{low, high, bands, valid_min, valid_max, reversed}
- CreaseMode::{Auto, On, Off}; CreaseSetting{mode, angle_deg}; crease_setting(mode, angle); crease_angle_of(&serde_json::Value) -> f64; crease_budget(n) -> usize
- coplanar_share(&[DVec3], &[u32]) -> f64; looks_tessellated(..) -> bool
- split_creases(positions, indices, wire_slots: &[u8], angle_deg, budget) -> Option<CreaseSplit{positions, normals, indices, wire_slots, copy_of}>
- crease_for(mode, positions, indices, wire_slots, angle_deg) -> (Option<CreaseSplit>, CreaseReport{added, skipped: Option<CreaseSkip::{Off, Scan, Budget}>})
- scan_render(&MeshGraph, CreaseSetting) -> ScanRender
- wire_slots(adj_offsets, adj_list, vertex_count) -> Vec<u8>; wire_conflicts(indices, slots) -> usize; WIRE_SLOT_COUNT
- Stamps {stamp, gen} with new, next_gen, seen and mark

REGION GROWING (crate::fit3d):
- flood_by_normal(g, seed: u32, max_angle_deg: f64, limit: Option<usize> /*None = vertex_count*/) -> Vec<u32>
- collect_patch(g, seeds, limit) -> Vec<u32>
- stranded_seeds(g, region, seeds) -> HashSet<u32>
- SEED_PATCHES = [1500, 6000, 24000]
- seeded_fit<C, T, R>(g, seeds, estimate: FnMut(&[u32], u32) -> Option<C>, grow: FnMut(C) -> Option<GrowResult<T>>, finish: FnMut(GrowResult<T>) -> Result<Option<R>, FitError>) -> Result<Option<R>, FitError>
- grow_{sphere,cylinder,torus,cone,plane}_region(g, seeds, init: Model, init_sigma, init_count, guide: Option<&FitGuide>) -> Option<GrowResult<Model>{region, model, sigma}>

GUIDE:
- FitGuide{side: Option<Side::{Inside, Outside}>, radius: Option<RadiusRange{min, max}>, axis: Option<DVec3>, within: Option<GuideBox{min, max}>} (serde, Copy, Default)
- has_guide, unit_axis, in_box, radius_in_range(g, r, rough), axis_agrees
- guide_patch(graph, patch, g) -> Vec<u32>
- side_share; keep_where
- guided_estimate<C: Inliers>(graph, patch, guide, estimate, judge, on, &mut Vec<GuideSeen>) -> Option<C>
- guided_seeds
- radial_from(c) and radial_off(p, a), each -> impl Fn(DVec3) -> Option<DVec3>
- judge_round(graph, guide, radius, inliers, outward, rough) -> Option<GuideSeen>
- axis_off(...) -> GuideSeen; guide_refusal(kind, g, &[GuideSeen]) -> String; GUIDED_TRIES

SEED AND SELECTION FITS (window and guide optional; plane and torus use only guide.within):
- fit_plane_from_seed(g, seeds, &FitSettings, Option<&AxialWindow>, Option<&FitGuide>) -> Result<RegionFit<PlaneFit>, FitError>
- fit_plane_on_selection(g, selection, &FitSettings)
- fit_sphere_from_seed(..) and fit_sphere_on_selection(..) -> RegionFit<SphereFit> (held diameter from settings.known_diameter)
- fit_cylinder_from_seed(..) -> RegionFit<CylinderFit> (told set when the region was parted)
- fit_cylinder_on_selection(g, selection, &FitSettings, Option<&AxialWindow>)
- fit_cone_from_seed(..) and fit_cone_on_selection(..) -> RegionFit<ConeFit>
- fit_torus_from_seed(..) and fit_torus_on_selection(..) -> RegionFit<TorusFit>; TORUS_MAX_R_DIAGS = 5.0

## integration_notes
Wiring:
1. Copy src/fit3d/ (the whole directory, replacing stage 1's) to crates/mm-reverse/src/fit3d/ and src/mesh_graph/ to crates/mm-reverse/src/mesh_graph/.
2. In mm-reverse's lib.rs, add `pub mod fit3d;` and `pub mod mesh_graph;`. The name mesh_graph does not clash with regions.rs, segment.rs, topo.rs or the other existing modules.
3. Drop this crate's src/lib.rs, src/fit2d.rs and src/linalg.rs. They are the verbatim mm-reverse copies from stage 1 and were not modified in stage 2.

Dependencies:
- No Cargo changes. mm-reverse already depends on mm-core, glam (serde), serde, serde_json and thiserror.
- serde_json is used at runtime by crease_angle_of.
- rayon is not used.

Dependencies on existing mm-reverse modules:
- crate::fit2d::{FitError, Mulberry32, solve_linear} (solve_linear for part_along's quadratic)
- crate::linalg::ortho_basis (plane patch extents)

Internal cross-module use:
- mesh_graph uses crate::fit3d::linalg3::{Extent, js_max, js_min}.
- fit3d uses crate::mesh_graph::{MeshGraph, Stamps}.
- fit3d/testkit.rs and fit3d/test_meshes.rs are #[cfg(test)] only.

Calling from Polymix:
- Build the graph once per mesh with MeshGraph::from_mesh(&mesh, false), or with true to keep the mesh's own winding. Rebuild it after any edit.
- Regions come back as mm_core vertex ids.
- For files, use build_mesh_graph or build_mesh_graph_from_flat.
- The clicked vertex is the seed (for example the nearest vertex of the triangle a Bvh raycast hits). fit_*_from_seed(&graph, &[v], &FitSettings::default(), None, None) returns RegionFit<XFit>; .into() gives a FitOutput for storage or the UI.
- The cylinder's `told` carries the parted second surface.
- FitError.0 carries ScanRuler's user-facing messages verbatim.

Status:
- cargo build and cargo build --all-targets show no warnings.
- cargo clippy --all-targets is clean apart from 4 neg_cmp_op_on_partial_ord notes in the unmodified verbatim fit2d.rs and linalg.rs.
- 136 tests pass in about 0.6 s (release): 58 from stage 1, 66 newly ported from ScanRuler and 11 new tests of ported code.

## not_ported
Assigned scope: everything is ported, including floodByNormal and the full curvature.ts and crease.ts. Files outside the list were pulled in because the listed files depend on them:
- weld.ts, wireSlots.ts and scratch.ts.
- tetraVolume6 and extentOf.
- the samplePart geometry (test-only, unposed).

Not ported:
- curvature.ts imports FieldScale from field/colormap.ts. Only a FieldScale struct is defined here; the colour maps are out of scope.
- The onProgress text callbacks (worker UI).
- samplePart's posed variant: samplePose plus rigidApplyToPoints (deviation/rigid). The floodByNormal test uses the unposed part.

Tests not ported:
- wireSlots 'colours a real scan without a single conflict' and ballbar.test.ts: they need the ballbar.stl fixture, which is absent and skipped in ScanRuler too, and the STL parser.
- cylinderStepped: the 4 tests using closedAlong, planeRegionReport or looksLikeCrown (segment.ts, not assigned). Its 6 fitCylinderFromSeed tests are ported.
- cylinderTwoSurfaces: all 6 use secondCylinder, wallTaper or planeRegionReport (segment.ts). Stage 1's two adapted versions remain.
- segmentPlanes, symmetry and reCompareSide: segment.ts, symmetry.ts and compareDetail/liveMap, all out of scope.

Stage-1 deferrals now covered:
- The 4 torusFit seed/selection tests.
- The 2 sphereKnownDiameter seed/selection tests.
- selectionFit, fitWindow and fitGuide in full.

## deviations
Representation:
- MeshGraph positions and normals are Vec<DVec3> (f64), matching mm_core and stage 1.
- ScanRuler keeps both in Float32Array. Its normals are summed and kept in f32; here they are summed and kept in f64.
- The test helpers round every soup or mesh coordinate to f32, so the fits get the same numbers. Welding on f64 bits is equivalent for those values.
- mean_curvature returns Vec<f32> like Float32Array, with f32 rounding between the smoothing passes.
- peel_transition's medians are taken over f32 values, sorted in JS TypedArray order (NaN last).

Implementation choices with identical results:
- acquireStamps (one buffer per graph in a WeakMap) becomes a per-walk Stamps. grow_region takes a new generation of one buffer each round, as the shared buffer did.
- weld uses std's HashMap instead of the custom open-addressing table. Ids still follow first occurrence, so the output is the same.

API shape:
- TS default parameters are explicit arguments: smoothing, angle_deg and budget, and flood_by_normal's limit as Option.
- seeded_fit's finish returns Result<Option<R>, FitError>, so the cylinder's window error propagates as TS's throw does. It works on GrowResult<T> directly instead of a generic G.
- The guide's `seen` and the torus `refused` live in a RefCell, because the estimate and finish closures both write them.
- PlaneFit & {region} and the like become RegionFit<T>.
- The Judged and PartAlong structs stand in for TS's union and object returns.

Robustness (only pathological inputs behave differently):
- Seeds and selections are filtered to ids < vertex_count before use. ScanRuler would read undefined coordinates there.
- flood_by_normal's seed is u32, so the TS test's `-1` became u32::MAX.
- part_along's two `cut += step` loops stop if the step is too small to move cut. In TS these would loop forever.
- wire_slots skips uncoloured neighbours when counting shared slots. TS wrote those counts out of bounds, which is silently ignored.
- crease_angle_of takes &serde_json::Value for `unknown` and reads strings as JS Number does (decimal, Infinity, 0x/0o/0b).

Text:
- Messages use exact JS toFixed and number-to-string semantics (js_text.rs).
- The exponent form JS uses for magnitudes below 1e-6 or of 1e21 and over is written out in full; no message reaches those.

Graph building:
- MeshGraph::from_mesh takes the centroid and box (which orient the winding and set bbox_diag) over live vertices only. positions keeps the dead slots so vertex ids match the mm_core mesh.
- Orienting flips only the graph's copy of the indices and normals.
- The progress callbacks of buildMeshGraph and weldTriangleSoup are not ported; they are worker UI text.

Tests:
- toBeCloseTo(x, d) is checked as |a - b| < 10^-d / 2.
- The regexes in cylinderStepped, fitGuide and the told note are checked with equivalent string parsing (no regex crate).
- 11 new tests cover graph building (flat, Polymix mesh, orientation, errors), weld, union-find, the guide refusal text, axis_off, js_text, stranded seeds and the torus refusal rounding.

## tests_passing
66

## failing_tests
[]

## review.remaining
["crate::fit2d::fit_circle_2d (the verbatim mm-reverse copy that stands in for circle2d.ts) uses f64::hypot where ScanRuler uses Math.hypot. It causes 11 of the 15 remaining differential mismatches, all in degenerate fits (huge-radius candidates on flats, a two-cylinder tie, a zero-half-angle cone sign, junk entries in refusal lists). The fix belongs in mm-reverse's fit2d.rs, in the two lines `let dist = du.hypot(dv)` and `let moved = (nu - cu).hypot(nv - cv)`, using a V8-exact hypot (for example by moving fit3d::linalg3::js_hypot into crate::linalg). I verified this fix brings the replay to 4 of 971 mismatches, then reverted it so the copy stays verbatim.", "Transcendental functions (cos, sin, tan, atan, atan2, acos, ln, powf) come from the platform libm and differ in the last place from V8's fdlibm on 3–36% of inputs. This leaves about 1e-8 relative differences in 4 cone/torus cases. Removing it would need an fdlibm port such as the libm crate, i.e. a new dependency; I did not add one.", 'Documented representation choice, unchanged: MeshGraph normals are summed and kept in f64, where ScanRuler uses Float32Array. Replaying with the Rust-built graphs, curvature differs by 1 f32 ulp and fit results by about 1e-8. No region or decision changed.', 'Documented robustness deviations, unchanged and harmless: seeds or selections past vertex_count are filtered (on_graph), part_along stops if the cut step cannot move, wire_slots skips uncoloured neighbours, and welding compares f64 bits (the same as f32 bits for f32-representable input).']

## review.tests_passing
144

## review.tests_failing
0


# fillet

## public_api
fillet_estimate:
- estimate_fillet_radius(points:&[DVec2], n1:DVec2, n2:DVec2, reach:f64, miss:Option<&mut Miss>) -> Option<FilletReading>
- estimate_chamfer(same args) -> Option<FilletReading>
- estimate_chamfer_setbacks(points, n1, n2, reach) -> Option<ChamferSetbacks>
- corner_window(chains:&[Vec<DVec2>], reach) -> Vec<DVec2>
- face_window(chains, reach, n:[DVec2;2], d:[DVec2;2]) -> FaceWindow
- corner_unreadable(&EdgeFingerprint) -> Option<String>
- corner_frame(&CornerEdge, at:DVec3) -> Option<CornerFrame> (methods to2, dir2)
- read_corner_cut(&CornerEdge, at, cut:&SectionCut, reach, ReadShape, Option<&mut Miss>) -> Option<FilletReading>
- read_corner_setbacks(&CornerEdge, at, &SectionCut, reach) -> Option<ChamferSetbacks>
- face_runs(n, other) -> DVec2
- read_on_window(&FaceWindow, impl FnOnce(&[DVec2], f64, &mut Miss) -> Option<FilletReading>, Option<&mut Miss>) -> Option<FilletReading>
- face_end_why(reach, &FaceEnd) -> String
- cut_chains(&SectionCut, impl Fn(DVec3) -> DVec2) -> Vec<Vec<DVec2>>
- combine_readings(&[FilletReading]) -> Option<FilletReading>
- few_why(points, reach) -> &'static str
- Types: FilletReading{radius, points, rms, sigma}; Miss{why:Option<String>}; ReadShape{Round, Chamfer} with other() and word().

curved_edge:
- is_curved_edge(&EdgeFingerprint) -> bool
- curved_unreadable(&CurvedEdge) -> Option<String>
- curved_stations(&CurvedEdge, n, share) -> Option<Vec<Station>>
- read_curved_cut(&CurvedEdge, &Station, &SectionCut, reach, ReadShape, Option<&mut Miss>) -> Option<CurvedRead>
- estimate_round_on_traces(points, &Trace2, &Trace2, reach) -> Option<FilletReading>
- face_triangles(&FacedMesh, hash:u32) -> Option<FaceTriangles>
- free_edge_of(polyline:&[DVec3], faces:&[u32], &FacedMesh) -> Option<FreeEdge>
- face_normal_near(&FaceTriangles, at) -> Option<DVec3>
- face_cut(&FaceTriangles, at, normal, x, y, reach) -> Vec<CutPoint>
- free_stations(&FreeEdge, n, share) -> Option<Vec<Station>>
- free_unreadable(&FreeEdge) -> Option<String>
- trace_from_cut(&[CutPoint], m:DVec2, reach) -> Option<Trace2>
- read_free_cut(&FreeEdge, &Station, &mut StationFaces, &SectionCut, reach, ReadShape, Option<&mut Miss>) -> Option<CurvedRead>
- Types: CurvedEdge{fingerprint:EdgeFingerprint, ends:Option<[DVec3;2]>, surfaces:Option<Vec<Option<EdgeSurface>>>}; EdgeSurface{kind:FaceKind, dir, radius, center} with From<&EdgeSide> and From<&FaceFingerprint>; FacedMesh{positions, normals, triangles:Vec<[u32;3]>, face_hash:Option<Vec<u32>>}; CurvedRead{reading, setbacks, countersink}.

fillet_read:
- read_edge_of<'a>(&EdgeInfo, &Topology, Option<&'a FacedMesh>) -> ReadEdge<'a>
- ReadEdge::of_fingerprint(EdgeFingerprint)
- read_edge_round(section:&mut Section, &ReadEdge, ReadOptions{stations:Option<usize>, reach, shape, widen}) -> EdgeRoundRead{round:Option<EdgeRound>, why, radii, places, reach, stations, linear}
- linear_law(&[EdgeStation]) -> Option<LinearLaw>
- round_reach(ReadShape, model_size) -> f64
- fillet_read_answer(section:&mut Section, body_id:u64, body_name:&str, edges:&[PickedEdge{index, edge}], ReadShape, stations:Option<usize>, reach:Option<f64>, model_size) -> serde_json::Value
- Section<'s> = dyn FnMut(origin:DVec3, normal:DVec3, min_length:f64) -> SectionCut + 's
- ScanSection::{new, of_mesh(&mm_core::Mesh), cut(origin, normal, min_length) -> SectionCut, size()}
- Constants: STATIONS, FREE_STATIONS, MAX_STATIONS, FILLET_READ_NAME, FILLET_READ_TITLE, FILLET_READ_DESCRIPTION.

fillet_corner:
- fillet_lines(&mut Sketch, a:EntityId, b:EntityId, radius:Option<f64>, room:Option<f64>) -> Result<FilletResult{arc, dimension, radius}, String>
- chamfer_lines(&mut Sketch, a, b, setback:Option<f64>) -> Result<ChamferResult{line, dimensions:[ConstraintId;2], setback}, String>
- corner_lines_at(&Sketch, Anchor) -> Option<[EntityId;2]>

round_corner:
- round_corner(&mut Sketch, a, b, radius:Option<f64>) -> Result<RoundCorner{arc, dimension:Option<ConstraintId>, radius, points, found:Option<Found>}, String>
- round_unread(points, Option<Found>) -> Option<String>
- reassign_corner_points(&mut Sketch, a, b, arc) -> usize
- round_radius_sigma(&Sketch, a, b, arc) -> Option<f64>

chamfer_sides:
- face_angle([DVec3;2]) -> f64
- angled_ask(SidedChamfer, ref_is_first:bool, normals:[DVec3;2], scale:f64) -> Result<AngledAsk{distance, angle_deg}, String>
- reference_faces(&[[u32;2]], flip) -> Vec<u32>

fillet_num:
- js_round, round4, js_num, to_precision, precision_value, fixed_value

## integration_notes
Steps:
1. Drop src/chamfer_sides.rs, curved_edge.rs, fillet_corner.rs, fillet_estimate.rs, fillet_read.rs, fillet_num.rs and round_corner.rs into crates/mm-reverse/src and add `pub mod` lines for each.
2. Add `#[cfg(test)] mod fillet_testkit;` (needed by the tests in fillet_estimate and fillet_read).
3. mm-reverse's lib.rs must also declare `pub mod topo;`. topo.rs is already in mm-reverse/src but was not declared there at the time of writing.
4. It also needs `pub mod nominal;`. My nominal.rs is a byte-for-byte copy of the sketchtools port's, so keep one.
5. The dependencies (glam, serde, serde_json, mm-core) are already in mm-reverse's Cargo.toml.

Copied mm-reverse files I used, all byte-for-byte: chain, corners, datums, fit2d, frame, linalg, segment, simplify, sketch, slice, solver, topo.
- mm-reverse is changing under me: frame.rs and topo.rs changed around 19:44, and I re-copied both.
- Compile again against the final topo.rs. I rely on EdgeFingerprint{kind, mid, dir, length, radius, faces:Vec<EdgeSide>, origin}, EdgeSide{kind, dir}, FaceFingerprint{kind, dir, radius, center}, EdgeInfo{faces:Vec<u32>, ends, polyline, fingerprint}, Topology::face(u32), and canonical_dir (tests only).

Helpers that may duplicate other ports:
- fillet_num (JS number formatting) may overlap with the expressions port's js_number.rs and the threads port's round.rs. They could be merged later.

Wiring in the app:
1. reverse.body.fillet_read:
   - Build `ScanSection::of_mesh(&scan_mesh)` (or wrap the app's own slicing in a closure).
   - model_size = the scan's bounding-sphere radius (ScanSection::size).
   - For each picked EdgeInfo: `read_edge_of(edge, &topology, Some(&faced_mesh))`. faced_mesh is the body's kernel mesh, with per-vertex normals and face_hash per triangle as the topology numbers faces (u32).
   - Pass them to `fillet_read_answer(section, body_id, body_name, &edges, shape, stations, reach, model_size)`.
   - Refuse with code no_scan when there is no scan.
2. The fillet panel's estimate and a fillet's radius 'fit': read_edge_round with widen = true when no reach is given.
3. Sketch fillet and chamfer tool: fillet_lines, chamfer_lines and corner_lines_at, then solve.
4. Agent sketch-edit rounds: round_corner, then solve_sketch, reassign_corner_points (up to 3 rounds), round_radius_sigma and round_unread.
5. Chamfer by two distances, or a distance and an angle, on SolidCraft: reference_faces picks each edge's reference face, then angled_ask with face_angle of the two faces' normals.

## not_ported
Command and app plumbing in reFilletRead.ts:
- scanSection(): the store's file check, the no_scan error 'There is no scan to read a round off.' and the mesh-worker lookup. ScanSection and the Section closure replace it.
- The JSON input schema and its validation (stations 1–9, reach > 0). Only MAX_STATIONS and the description text are carried.
- ready(), bodyNamed, edgesPicked, topologyOf, meshOf: the command run's body, pick and topology lookup.

ScanRuler tests not ported, because they need Open CASCADE, the command layer or the browser store:
- reFilletTry: all of it (Builder.filletTry). It does not exercise these modules.
- reFilletRead: 'reverse.sketch.edit rounds' (needs the section fit and the sketch-edit command; round_corner itself is covered by reRoundCorner) and the four section-picture and section.render tests.
- reFilletCurvedBody test 2 and reFilletLaw tests 2–3: a fillet's radius 'fit', refit and the variable-radius law. That feature-fit code is outside these files and needs the kernel.
- reFilletFree: the wheel tests built and meshed by the kernel. A box FacedMesh test of my own stands in.
- reCurvedChamfer test 1 ('coaxial takes the bore's rims'): pick queries and kernel.
- reRoundsHonest: the kernel and command tests (copy-of-body fillets, tangent-chain picks, body check).
- reChamferDelete: everything after 'the sides of a chamfer' (kernel chamfers, face delete, store boxes).
- reFilletCorner: the two 'Fillet tool in the store' tests (zustand store and UI).
- reAnswersK3 K5 test 3: goes through the command.
- reJoyconReads: needs the 128 MB real scan from JOYCON_DIR.

## deviations
Note: a previous, interrupted run of this step had already written most modules (no lib.rs, no fillet_read.rs, few tests). I reviewed each one line by line against the TypeScript, then finished the work. I changed the edge types to mm-reverse's topo types, wrote fillet_read.rs and lib.rs, and wrote or checked all the tests.

Test counts: 46 ScanRuler tests are ported, as Rust test functions covering 48 `it` blocks (reRoundCorner's 3 share one function because they share a module-level mulberry32(3)). All 46 pass. With 6 tests of my own, 1 in the nominal copy and the copied modules' own tests, `cargo test --release` passes 90 of 90. `cargo build` and the test build have 0 warnings. Clippy only flags `!(x > y)` comparisons, which ScanRuler and the mm-reverse files use deliberately for NaN.

Data types:
- The scan's cut is slice::SectionCut (chains of DVec3, f64), not ScanCut's flat Float32 arrays. Face meshes are Vec<DVec3> with [u32;3] triangles.
- Face and edge kinds are the topo::FaceKind/EdgeKind enums, not strings. Messages use .name(), so the text is unchanged. Face hashes are u32.
- Trace2 is {curve: TraceCurve::Line | Circle{q, r}, m, d} instead of a kind with optional q/r.
- The per-station face cache (a WeakMap keyed by Station in TS) is an explicit StationFaces passed to read_free_cut. read_edge_round keeps one per place.
- Miss is an Option<&mut Miss> out-parameter. Readings return Option, as ScanRuler's null.
- Sketch edits return Result<_, String> with ScanRuler's exact sentences (TS returns `| string`). They do not use FitError, because these failures are user-facing sentences, not fit errors.
- face_end_why takes (reach, &FaceEnd).
- read_edge_round is synchronous and takes a section closure `&mut dyn FnMut(origin, normal, min_length) -> SectionCut` instead of an async worker call.
- round_reach takes model_size (ScanRuler reads the store's modelSize, the bounding-sphere radius).
- fillet_read_answer returns serde_json::Value. Keys are not kept in insertion order. The body is given as an id and a name.
- intent.ts `rounding` is a private helper in fillet_read.
- nominal.rs is a byte-for-byte copy of the sketchtools port's file, so the two do not clash.

Small additions that guard edge cases ScanRuler would crash on or loop on:
- face_cut returns nothing for a reach of 0 (TS would loop forever).
- corner_frame and read_corner_cut return None, with a reason, for an edge with fewer than two faces.
- free_edge_of needs at least 2 polyline points (TS: 6 numbers).

chamfer_lines sets each dimension's placement right after adding the constraint, with the same values.

Test adaptations (each is labelled in its doc comment):
- reFilletCorner 'keeps a rectangle one closed region': mm-reverse regions.rs needs SolidCraft, so the test checks every end is joined exactly once and touching.
- reFilletRead tests 1–2, reFilletCurvedBody test 1, reFilletLaw test 1: the bodies built by the kernel are replaced by hand-made edge fingerprints with ends and surfaces. The welded scans are built directly as indexed meshes, rounded to f32. The command's answer is checked through fillet_read_answer.
- reFilletFree 'says to pick an edge of any shape…': uses a hand-made kind-Other fingerprint.
- reRoundsHonest test 1: uses a swept disc rounded R6 instead of the kernel's dome rounded R20. The reach-held why is checked as the single-reason form ('no round within 3 mm (reach): the faces not found in the cut'). The general 'larger reach' wording is checked exactly in the box test's gathered sharp row.
- reCurvedChamfer: a flat cut along an R40 drum, chamfered C1 and read through the curved reader against the drum's tangent. ScanRuler's version reads the rim off the kernel's mesh.

My own tests (not from ScanRuler): combine_readings, linear_law, round_reach, cutting an mm_core::Mesh, a free (any-shape) edge read off a 6-face box FacedMesh, and fillet_num.

## tests_passing
46

## failing_tests
[]

## review.remaining
["reFilletCorner 'keeps a rectangle one closed region' is still adapted: it checks joins and coincidence instead of findRegions == 1. mm-reverse/src/regions.rs needs solidcraft-geom and solidcraft-sketch (git dependencies), which this standalone crate does not take. Restore the exact assertion with crate::regions::find_regions after integration.", "Scan cut points are f64. ScanRuler's worker returns Float32Array points and f32 plane distances. This comes from mm-reverse's slice.rs (a copied module, identical upstream); the effect is about 1e-6 mm.", "serde_json writes keys in sorted (BTreeMap) order, not ScanRuler's insertion order. Matching it needs serde_json's preserve_order feature, which would apply across the whole workspace, so it is left as an integration decision.", 'Tests that need Open CASCADE, the command layer or the browser store are still not ported, as the porter reported: reFilletTry, sketch.edit rounds, section picture/render, the fit/refit/variable-law tests, the wheel tests built by the kernel (now partly stood in for by the dome and drum FacedMesh tests), the store tool tests, K5 test 3 through the command, and reJoyconReads (needs the 128 MB scan).', "Integration note update: mm-reverse's lib.rs now declares `pub mod topo` (the porter's step 3 is already done). All 12 copied files are still byte-identical to /Users/banderies/Documents/GitHub/polymix/crates/mm-reverse/src."]

## review.tests_passing
95

## review.tests_failing
0


# surface

## public_api
crate::surface_fit:
- Constants: NET_MIN=3, NET_MAX=20, GRID_PER_SPAN=3, GRID_RETRIES=[6,8], GRID_AREA_SLACK=1.02, PULL=0.01, HEIGHT_FIELD_DEG=70.0, MESH_ANGLE=0.15.
- Params: struct Net{u,v}; struct SurfaceFitParams{net:Option<Net>,degree:u32,expand_pct,smoothness,robust:bool,normal:Option<DVec3>}; const DEFAULT_SURFACE_FIT (Default impl too).
- struct SurfaceFitResult{origin,axis_u,axis_v,normal,u_range:[f64;2],v_range,degree:usize,knots_u:Vec<f64>,knots_v,nu,nv,heights:Vec<f64>,rms,max,net:Net,footprint:Option<Vec<DVec2>>}, with .plane() -> SurfacePlane{origin,axis_u,axis_v,u_range,v_range}.
- B-spline pieces: clamped_knots(n,p,a,b)->Vec<f64>; greville(&[f64],n,p)->Vec<f64>; find_span(&[f64],n,p,t)->usize; basis_functions(&[f64],span,p,t,&mut [f64]).
- auto_net(extent_u,extent_v,scatter)->Net; height_frame(&[DVec3;3],Option<DVec3>)->[DVec3;3].
- fit_surface(xyz:&[DVec3],&SurfaceFitParams,scatter:Option<f64>,pull:f64)->Result<SurfaceFitResult,FitError>. The TS defaults are scatter=None and pull=PULL.
- surface_residuals(&SurfaceFitResult,&[DVec3])->Vec<f64>.
- Reporting types: struct RegionFacing{dir,within_deg:Option<f64>}; struct Bounds{min,max}; struct Stray{points,bounds (serde \"box\"),off_mm:[f64;2],why}; struct StraysRefusal{message,strays}; enum SurfaceFitError{Fit(FitError),Strays(StraysRefusal)} with .strays(); struct WorstPoint{at,off}; struct RobustAt{med,limit}; struct SurfaceReport{fit,worst:Vec<WorstPoint>,dropped,strays:Option<Vec<Stray>>}.
- fit_surface_reported(&[DVec3],&SurfaceFitParams,facing:Option<&RegionFacing>)->Result<SurfaceReport,SurfaceFitError>.
- footprint_of(&SurfacePlane,&[DVec3])->Vec<DVec2> (rounded to f32, as ScanRuler's Float32Array).
- stray_groups(&SurfaceFitResult,&[DVec3],Option<RobustAt>)->Vec<Stray>.
- normal_spread(&[DVec3],DVec3)->f64 (the p95).
- run_away(&SurfaceFitResult,&[DVec3],expand_pct,asked:Option<DVec3>,facing:Option<&RegionFacing>)->Option<String>.
- mesh_triangles(&SurfaceFitResult,chord,angle)->f64.
- eval_surface(&SurfaceFitResult,u,v)->DVec3.
- struct Grid{points:Vec<DVec3>,rows,cols}; grid_of(&SurfaceFitResult,per_span)->Grid; grid_area(&Grid)->f64.
- struct Reach{u,v}; reach_over(&SurfacePlane,&[DVec3],margin)->Option<Reach>.
- carried_height(&SurfaceFitResult,u,v)->f64; reaching_grid_of(&SurfaceFitResult,u:[f64;2],v:[f64;2],per_span)->Grid.

crate::surface_region:
- PAST_REGION_MM=0.2.
- struct RegionField{u0,v0,cell,cols,rows,dist:Vec<f32>}; point_spacing(&[DVec2])->f64; region_field(&[DVec2],u:[f64;2],v:[f64;2],cell)->RegionField; field_at(&RegionField,u,v)->f64.
- struct PastRegion{off,at}; past_region(&SurfaceFitResult, near: impl Fn(DVec3)->Option<DVec3>)->Option<PastRegion>.
- struct OffPastRegion{max,area,at}; off_past_region(&SurfaceFitResult,points:&[DVec3],areas:&[f64],near:impl Fn(DVec3)->Option<DVec3>,tolerance)->Option<OffPastRegion>.
- struct RegionOutline{loops:Vec<Vec<DVec2>>,field,level}; region_outline(&SurfaceFitResult,margin)->Option<RegionOutline>.

crate::principal_frame:
- struct PrincipalFrame{centroid,axes:[DVec3;3],spread:[f64;3]}; principal_frame(&[DVec3])->PrincipalFrame.
- struct Eigen3{values:[f64;3],vectors:[DVec3;3]}; symmetric_eigen3(&[f64;9])->Eigen3.
- normalize_or_z(DVec3)->DVec3; cross(DVec3,DVec3)->DVec3; js_cmp(f64,f64)->Ordering.

crate::height_surface:
- enum SurfaceModel{Plane,Bilinear,Quadric}; SURFACE_MODELS.
- struct SurfaceFrame{origin,basis_u,basis_v,normal}; enum CurvatureShape; struct Curvature{k1,k2,r1,r2,axis1,axis2,shape}; struct HeightFit{model,coefficients:BTreeMap<String,f64>,rms,worst,curvature}; struct SurfaceModels{frame,fits,best}.
- patch_frame(&[DVec3])->Option<SurfaceFrame>; fit_height(&[DVec3],&SurfaceFrame,SurfaceModel)->Option<HeightFit>; fit_surface_models(&[DVec3],&[SurfaceModel])->Option<SurfaceModels>.

crate::sparse:
- struct CsrMatrix{n,row_ptr,cols,vals}; trait LinearOperator{size,apply,diagonal} (implemented by CsrMatrix); struct FnOperator{n,apply,diagonal}.
- struct Triplets with new/size/add/add_outer/to_csr.
- csr_from_triplets(n,&rows,&cols,&vals,count)->CsrMatrix; csr_multiply; csr_diagonal.
- struct CgOptions{x0,tolerance,max_iterations,precondition} (Default: 1e-10, None, true); struct CgResult{x,iterations,residual,converged}.
- conjugate_gradient(&dyn LinearOperator,&[f64],&CgOptions)->CgResult; biconjugate_gradient_stabilized(same)->CgResult.

Serde: the types ScanRuler stores or answers with serialise camelCase (expandPct, axisU, uRange, knotsU, offMm, withinDeg, basisU), so ScanRuler's JSON reads straight in.

## integration_notes
- Copy into mm-reverse/src: principal_frame.rs, height_surface.rs, sparse.rs, surface_fit.rs, surface_region.rs. Add `pub mod` lines for each to mm-reverse lib.rs.
- fit2d.rs and linalg.rs here are verbatim copies; drop them. The new modules use crate::fit2d::FitError and crate::linalg::{cholesky_solve, solve_linear, ortho_basis}.
- No new runtime dependencies (glam, serde, thiserror are already there). serde_json is used only in tests, and mm-reverse already depends on it.
- None of the five names clash with existing mm-reverse modules.
- Overlaps with sibling port packages, to resolve at integration:
  - ports/network/src/sparse.rs is byte-identical to this sparse.rs; keep one copy.
  - ports/fits/src/fit3d/height_surface.rs is a second port of heightSurface.ts (its types are SurfaceCurvature/SurfaceFitResult; mine are Curvature/HeightFit). surface_fit does not use height_surface at all, so keep whichever the fits package wires, and drop this height_surface.rs if fits lands.
  - ports/fits/src/fit3d/linalg3.rs also defines symmetric_eigen3(&[f64;9])->Eigen3 and normalize_or_z with the same signatures. principal_frame.rs could import those instead of its own copies. surface_fit and surface_region use crate::principal_frame::{principal_frame, symmetric_eigen3, js_cmp}.
  - ports/network/src/network_spline.rs has its own clamped_knots (ScanRuler imports it from surfaceFit); it could use crate::surface_fit::clamped_knots.
  - js_round/js_num-style helpers exist in expressions/js_number.rs and fillet_num.rs too; they could be unified later. Here they are pub(crate) in surface_fit.
- To wire Fit Surface into the build (ScanRuler build.ts), the Rust side needs:
  - A kernel 'surface through a regular grid' (OCCT GeomAPI_PointsToBSplineSurface equivalent on SolidCraft) taking Grid{points rows×cols, row-major along u}.
  - Surface area and mesh calls on that kernel.
  - A NearScan closure (nearest scan point within reach, e.g. mm_core Bvh::nearest) for past_region/off_past_region.
- Loop over per_span in [GRID_PER_SPAN, 6, 8], accepting when kernel_area / grid_area(&grid) <= GRID_AREA_SLACK. Refuse via mesh_triangles(fit, chord, MESH_ANGLE) against the build's triangle budget. Warn when past_region(...).off > PAST_REGION_MM. On SurfaceFitError::Strays, attach .strays() to the feature result.
- For a split: reach_over then reaching_grid_of. 'within' uses region_outline and field_at, and the cut-face check uses off_past_region.
- The feature model needs a Surface feature carrying SurfaceFitParams (camelCase serde matches ScanRuler's saved `fit` object) and RegionFacing (`regionBox.normal`).

## not_ported
- prealign.ts preAlignCandidates and frameToFrame: not used by the surface fit, and they need deviation/rigid.ts.
- linalg.ts symmetricEigenN, centroidOf, projectOntoPlane, rmsOver: not used here. mm-reverse already has symmetric_eigen_n.
- Kernel-dependent tests, which need Open CASCADE (ReKernel.surfaceThrough, surfaceArea, mesh) or the reverse Builder:
  - reSurfaceRunaway: 5 tests (triangle count within ×2 of the mesher, denser-grid retry, refusal before the kernel, refusal for too many triangles, chord after size).
  - reSurfaceStrays: 'answers them on the feature's result' (Builder).
  - reSplitWithin: the 4 split/pastRegion-through-build tests.
  - reSplitSides: all but the reachOver test.
  The functions those tests drive are fully ported; only the kernel side is missing.
- reSurfaceStrays: the lamp-housing test. It is skipIf without the RE_LAMP env file, and needs mesh-graph/queryVertices/STL parsing.
- reSurfaceRegion: the 4 app/worker/command tests (schema refusals, scan.query marking, edit).
- tests/surfaces.test.ts: it tests the main app's surfaces store (src/app/surfaces, PlaneFit), not surfaceFit, so it is out of scope.
- The build-side orchestration in build.ts lines ~500-700, ~2850-2960 and 3940 is not in these files: GRID_RETRIES loop, area check, SURFACE_TRIANGLES budget, the past-region warning, split within/offScan. It belongs to whoever ports the surface/split build steps on the SolidCraft kernel.

## deviations
- Vectors are &[DVec3] / Vec<DVec2>, not flat Float32/Float64Arrays. Grid.points is Vec<DVec3>; the kernel adapter flattens it. Callers that pass ScanRuler a Float32Array (region points) should round to f32 first; the ported tests do.
- footprint_of rounds (u, v) to f32, and RegionField.dist is Vec<f32>, so both keep ScanRuler's single precision.
- normal_spread returns the p95 as f64, not {p95}. mesh_triangles takes the angle explicitly; MESH_ANGLE = 0.15 is the TS default.
- fit_surface takes scatter: Option<f64> and pull: f64 explicitly; the TS defaults are null and PULL. degree is u32 and net counts are usize; TS rounds numbers that might be fractional.
- Errors: fit_surface returns FitError with ScanRuler's messages ('Too few points to fit a surface through.', 'The surface fit is singular — the region is too thin for this net.'). fit_surface_reported returns SurfaceFitError::Fit or ::Strays(StraysRefusal{message, strays}), with the message byte-identical to TS's away + strays words.
- Messages: js_round, js_num and to_fixed1 reproduce Math.round, Number-to-string and toFixed(1), ties included, so run_away and stray wording match ScanRuler character for character.
- Determinism: JS Map insertion order is kept with an internal OrderedMap and stable sorts, so stray order matches ScanRuler.
- Robustness additions that never change results on valid data:
  - eval_height returns NaN, instead of indexing out of bounds, when a fit's knots, net and heights disagree (for example a fit read back from a file). ScanRuler reads undefined there and gets NaN.
  - second_sheets returns no strays when its grid would exceed 2^53 cells (non-finite or absurd extents).
  - Voxel and cell neighbour keys in far_points and normal_spread use saturating adds, so debug builds cannot overflow-panic on infinite coordinates.
- NaN inputs: Rust's f64::min/max skip NaN where Math.min/max propagate it. This only matters for NaN coordinates.
- thin_loop uses an explicit stack instead of recursion. The kept set is identical because the runs are independent.
- The module doc's sentence 'A thin-plate term on second differences…' became 'A fairness term on the net…', because ScanRuler's own later comment says the thin-plate term was thrown out for third divided differences. The SurfaceFitParams.smoothness doc is reworded the same way.
- Adapted tests:
  - reSurfaceRegion 'a robust fit leaves the boss out by itself' runs in ScanRuler through the app, the worker, an STL and a facing query. Here the region is the skin's 51×41 vertices plus the boss top's 7×7, given straight to fit_surface_reported, with the same assertions (dropped >= 30, robust rms < plain/5, rms < 0.01, every worst point under 0.05).
  - reNetworkFit's 270° strip refusal builds the same gridScan cylinder (R20, 180×40, mulberry32(5) Gaussian noise 0.02 along the normal, f32) and takes the vertices at 0..270° and z 5..35 directly, not from a patch-network cell.
- Partial ports:
  - reSurfaceNormal's hump test drops its last line, which checks the app command schema (validate/featureOf).
  - heightSurface's five its run as one Rust test, because the TS file shares one module-level mulberry32(21) across them in order.
- Tests added beyond ScanRuler's (labelled in code): BiCGSTAB solve, grid_area/carried_height/reaching_grid_of, mesh_triangles monotonicity, JS number formatting, ScanRuler JSON round-trip, mismatched arrays answering NaN with non-finite points not panicking, off_past_region/field_at, eigen3 and principal_frame sanity.

## tests_passing
47

## failing_tests
[]

## review.remaining
["Math.acos (normal_spread angles, run_away 'apart') and Math.pow (pull weight) still use system libm; V8 uses fdlibm, and arm64 V8 builds contract it into FMA (FMA-fused fdlibm acos matched V8 1M/1M on this arm64 machine, plain fdlibm 99.6%), so JS itself is platform-dependent. libm acos is off in the last bit for ~10% of inputs and powf for ~10%. The values only feed angles rounded to whole degrees or compared with 70°, and the pull weight; across 228 differential cases the only visible effect was one 1.6e-16 relative difference.", 'NaN/Infinity coordinates (documented by the porters): Rust f64::min/max skip NaN where Math.min/max propagate it, and NaN grid/voxel keys cast to 0 in Rust where JS uses NaN keys (second_sheets, far_points, normal_spread, run_away, reach_over). Behaviour differs only on garbage input.', "Kernel/Builder/app-dependent tests are still not ported: reSurfaceRunaway's 5 kernel tests, reSplitWithin's 4 build tests, reSplitSides' kernel tests, the RE_LAMP lamp test (skipIf in TS), and the reSurfaceRegion app/worker tests. The functions they drive are ported.", "Integration caveat: if principal_frame switches to the fits package's symmetric_eigen3/normalize_or_z, or height_surface goes back to mm-reverse's linalg::ortho_basis, the last-bit match with ScanRuler is lost unless those also use js_hypot. sparse.rs is untouched and still byte-identical to ports/network's copy.", 'Clippy (not build) warnings are pre-existing: neg_cmp_op_on_partial_ord (deliberate NaN-faithful !(a > b) checks, also present in the verbatim fit2d/linalg copies) and needless_range_loop. cargo build is warning-free.']

## review.tests_passing
49

## review.tests_failing
0


# network

## public_api
network_fit:
  pub fn fit_network(input: &NetworkFitInput<'_>) -> Result<NetworkFitResult, FitError>
  pub fn read_network_fitted(scan: &ScanPaths, f: &NetworkFeature, bodies: Option<&dyn NetworkBodies>) -> Result<NetworkReading<NetworkFitResult>, FitError>   // read_network with fit_network as its fit; reading.fit: Option<NetworkFitResult>
  pub fn on_side(k: usize, f: f64) -> (f64, f64)
  pub struct NetworkFitResult { patches: Vec<FittedPatch>, curves: BTreeMap<u32, CurveFit>, shared: Vec<SharedCurve>, corners: Vec<IrregularCorner>, strips: Vec<Strip>, bodies: Vec<BodyReading>, rounds: usize, iterations: usize }
  pub struct FittedPatch { net: PatchNet, sides: Vec<FittedSide>, corners: Vec<u32>, seed: DVec3, points: usize, rms: f64, max: f64, misfit: f64, worst: Vec<WorstPoint>, dropped: Option<usize> }
  pub struct FittedSide { edges: Vec<HalfEdge>, poles: Vec<f64>, single: bool }
  pub struct SharedCurve { curve: u32, patches: [usize; 2], gap: f64, angle: f64, crease: bool, tee: bool }
  pub struct IrregularCorner { node: u32, at: DVec3, patches: usize, angle: f64 }   (serde)
  pub struct Strip { spans: usize, auto: bool, curves: Vec<u32> }   (serde)
  pub struct BodyReading { curve: u32, patch: usize, mode: SideMode, angle: f64, gap: f64 }   (serde)
  pub struct WorstPoint { at: DVec3, off: f64 }   (serde)

network_body:
  pub const GRID_PER_SPAN: usize = 3; pub const SEW_TOLERANCE: f64 = 1e-3; pub const PATCH_GRID: usize = 9
  pub trait NetworkKernel {
    type Shape;
    fn surface_through(&mut self, grid: &PatchGrid) -> Result<Self::Shape, FitError>;
    fn surface_area(&mut self, shape: &Self::Shape) -> Result<f64, FitError>;
    fn distance_to_point(&mut self, shape: &Self::Shape, point: DVec3) -> Result<f64, FitError>;
    fn copy(&mut self, shape: &Self::Shape) -> Result<Self::Shape, FitError>;
    fn sew_patches(&mut self, faces: &[Self::Shape], tolerance: f64, closed: bool) -> Result<Self::Shape, FitError>;
    fn is_closed_solid(&mut self, shape: &Self::Shape) -> Result<bool, FitError>;
    fn diagonal(&mut self, shape: &Self::Shape) -> Result<f64, FitError>;
    fn edges(&mut self, shape: &Self::Shape, deflection: f64) -> Result<Vec<KernelEdge>, FitError>;
    fn face_points(&mut self, shape: &Self::Shape) -> Result<Vec<FacePoint>, FitError>;
    fn release(&mut self, shape: Self::Shape) { drop(shape) }
  }
  pub struct KernelEdge { mid: DVec3, faces: usize }
  pub struct FacePoint { hash: u64, at: DVec3 }
  pub struct PatchGrid { points: Vec<DVec3>, rows: usize, cols: usize }   // row-major, u along rows (same fields as surface_fit::Grid)
  pub fn network_body<K: NetworkKernel>(k: &mut K, reading: &NetworkReading<NetworkFitResult>) -> Result<NetworkBody<K::Shape>, FitError>
  pub struct NetworkBody<S> { shape: S, report: NetworkReport, face_of: HashMap<u64, usize> }
  pub fn network_ghost(reading: &NetworkReading<NetworkFitResult>) -> Option<NetworkGhost>
  pub struct NetworkGhost { mesh: GhostMesh, report: NetworkReport, volume: f64 }
  pub struct GhostMesh { positions: Vec<f32>, normals: Vec<f32>, indices: Vec<u32> }
  pub fn empty_report(reading: &NetworkReading<NetworkFitResult>) -> NetworkReport
  pub fn patch_grid(patch: &FittedPatch, per_span: usize, sampler: &mut SideSampler<'_>, fewest: usize) -> PatchGrid
  pub struct SideSampler<'a>; SideSampler::new(fit: &NetworkFitResult); .side_samples(&mut self, patch: &FittedPatch, k: usize, count: usize) -> Vec<DVec3>
  pub fn patch_surfaces(fit: &NetworkFitResult) -> Vec<PatchSurface>
  pub struct PatchSurface { degree: usize, rows: usize, cols: usize, poles: Vec<DVec3>, knots_u: Vec<f64>, knots_v: Vec<f64>, seed: DVec3 }
  pub struct NetworkReport { patches: Vec<ReportPatch>, cells: Vec<CellOutcome>, unfilled: Vec<Unfilled>, curves: Vec<ReportCurve>, corners: Vec<IrregularCorner>, strips: Vec<Strip>, lost: Vec<u32>, spokes: Vec<ReportSpoke>, centres: Vec<ReportCentre>, bodies: Vec<BodyReading>, closed: bool, worst_angle: f64, worst_gap: f64, ms: ReportTimes }   (serde camelCase)
  pub struct ReportPatch { seed, rms, max, misfit, net: ReportNet { u, v }, points, worst: Vec<WorstPoint>, dropped: Option<usize>, sides: Vec<Vec<u32>>, corners: Vec<u32>, grid: Vec<f32> }
  pub struct ReportCurve { curve: u32, gap: f64, angle: f64, crease: bool }
  pub struct ReportSpoke { cell: usize, points: Vec<f32> }
  pub struct ReportCentre { cell: usize, at: DVec3 }
  pub struct ReportTimes { trace, cells, map, fit, faces: f64 }

sparse (unchanged; the fit's solver):
  conjugate_gradient(&dyn LinearOperator, &[f64], &CgOptions { x0, tolerance, max_iterations, precondition }) -> CgResult { x, iterations, residual, converged }

Stage-1 API: unchanged (see stage-1 report).

## integration_notes
1. **Files to copy into crates/mm-reverse/src.**
   - New: network_fit.rs and network_body.rs.
   - The stage-1 files, which are still needed: adjacency, scan_path, network, network_cells, network_creases, network_curve, network_feature, network_map, network_seed, network_spline, network_split, and the test-only network_meshes.
   - Re-copy these stage-1 files, which changed in stage 2: network.rs (doc), network_curve.rs (a clippy allow), and the tests in network_split.rs and network_seed.rs.
2. **lib.rs.**
   - Add `pub mod network_fit; pub mod network_body;` alongside the stage-1 modules, and `#[cfg(test)] pub(crate) mod network_meshes;`.
   - mm-reverse also needs `pub mod topo;` (and sparse) registered, as stage 1 noted.
3. **Shared modules.**
   - Do not copy fit2d.rs, linalg.rs or topo.rs; they are mm-reverse's own, verbatim. Stage 2 uses FitError, symmetric_eigen_n and the fingerprints from them.
   - sparse.rs is byte-identical to the surface package's copy; integrate one as crate::sparse. The fit uses conjugate_gradient, LinearOperator and CgOptions.
4. **Duplicates to fold in after integration.**
   - network_body's private GRID_RETRIES, GRID_AREA_SLACK and grid_area duplicate surface_fit's public ones.
   - Its mesh_deflection duplicates build::mesh_deflection.
   - PatchGrid has the same fields as surface_fit::Grid and could become it.
5. **Build step for FeatureKind::Network(NetworkFeature).**
   - Read: `let reading = read_network_fitted(&scan, &f, Some(&bodies))?;`. Here bodies is the builder's NetworkBodies impl (edge points and face normals for PN11).
   - Create: `network_body(&mut kernel, &reading)` gives the shape, the NetworkReport and face_of (each kernel face hash to its patch).
   - Preview: `network_ghost(&reading)`, which needs no kernel and gives a mesh, report.closed and a volume.
   - With no fit: `empty_report(&reading)`.
6. **The kernel.**
   - Implement NetworkKernel for the SolidCraft kernel. surface_through must interpolate the grid (rows along u, row-major) within about a micron, so the shared sample rows coincide.
   - If SolidCraft can make a clamped cubic B-spline face from poles and knots, patch_surfaces gives each patch's net. A shared curve's poles are the boundary row of both patches beside it, so their edges coincide exactly. Sew those faces at SEW_TOLERANCE, and the grid retries are not needed.
7. **Report.** NetworkReport serialises in camelCase like ScanRuler's (worstAngle, worstGap, and a 9×9×3 f32 grid per patch). It can go into the feature result and agent rows directly.
8. **Tests and timing.**
   - All 72 tests in the crate pass in about 2.6 s in release, warning-free: 40 ScanRuler ports across both stages, plus the crate's own checks. cargo clippy reports no errors, only style warnings.
   - In the bench, the map stage, stage 1's network_map, takes most of the time (2–3.7 s of 2.4–4.1 s).
   - rayon is still declared but unused.

## not_ported
- **Kernel calls themselves.** surfaceThrough, sewPatches, isClosedSolid, distanceToPoint, edges, facePoints, bounds and copy are Open CASCADE calls in ScanRuler. They are left to integration behind NetworkKernel; everything up to them is ported.
- **reNetworkBody, 'thickens the sheet in and out, and refuses both ways'.** This needs the thicken feature, the Builder and the kernel.
- **reNetworkBody, kernel-only assertions.** STEP export and verify, the builder's body volume and face count, and the 0.005 mm gap that only a real surface through the grid can reach.
- **reNetworkFit, Fit Surface refusal.** It belongs to the separate surface package.
- **UI and app tests.** reNetworkDraft, reNetworkCommands, reNetworkPick and reNetworkProject are out of scope, as noted in stage 1.

No logic in networkFit.ts, networkBody.ts or sparse.ts is left unported. sparse.ts was already complete from stage 1 and was checked again here.

## deviations
**fit_network**
1. It returns Result<_, FitError>. It first checks that every patch has four sides, each side's curves are in input.curves, each corner is in input.nodes, and xyz covers every (u, v). The messages are mine; ScanRuler has none there and would fail on a `!`. read_network always hands it valid input.
2. read_network_fitted is new. ScanRuler's readNetwork calls fitNetwork itself; stage 1's read_network takes the fit as a closure.
3. NetworkFitResult.curves is a BTreeMap keyed by id. ScanRuler uses a Map in input order; only lookups by id use it.
4. SharedCurve.tee is a bool defaulting to false; ScanRuler's is optional.

**Numerics**
5. symmetricEigen3 is linalg::symmetric_eigen_n(3, ..): the same Jacobi sweeps, pair order, gate and ascending order. Its cap is 64 sweeps against 32, which does not matter for a 3×3.
6. Math.hypot is a square root of a sum of squares. `**` is powf, and a square is x*x.
7. ScanRuler's `!(x > y)` tests are kept, so NaN behaves as in JS. Where JS Math.max/min would carry a NaN on, Rust's f64::max/min ignore it; this only matters on NaN poles. The ratio clamp and the (u, v) clamp in project use f64::clamp, which keeps NaN as JS does.
8. ScanRuler's insertion-ordered Maps and Sets (leaving, sharedBy, cornerPatches, irregular, torn, tangentTo, notSewn) are kept as Vec plus HashMap. This keeps the order ties are made and unknowns numbered, and so the CG's arithmetic order.
9. The CG operator is a FitOperator with RefCell scratch, because LinearOperator::apply takes &self.

**network_body and the kernel**
10. The kernel is the NetworkKernel trait. Every call returns Result<_, FitError>, network_body returns Err where ScanRuler throws, and the faces are released on every path (ScanRuler's `finally`). `release` drops the shape by default.
11. GRID_RETRIES, GRID_AREA_SLACK and grid_area (from surfaceFit.ts) and mesh_deflection (from buildTypes.ts) are private local copies, because this crate cannot depend on the surface package or on mm-reverse's build module.
12. PatchGrid holds Vec<DVec3> in place of a flat Float64Array; it has the same fields as surface_fit::Grid.
13. patch_surfaces is an addition. ScanRuler's kernel takes only grids.
14. network_ghost returns None when there is no fit; ScanRuler would fail.

**Stage-1 files touched**
15. network_curve::fit_curve got a clippy allow for the empty CURVE_ROUNDS loop.
16. network.rs: doc only.

**How the tests were adapted**
- reNetworkFit, '270° strip': the check that Fit Surface refuses the same points (fitSurfaceReported, which is in the separate surface package) is left out. Everything else in reNetworkFit is ported with the same inputs, seeds and bounds, including the exact [6, 4] and [9, 4] span counts and misfit == 0.
- reNetworkBodies: the body is an analytic half-cylinder behind NetworkBodies (its straight edge, and radial face normals) in place of a kernel-built sketch and extrude found by fingerprint. The assertions are unchanged. Tangent reads 0.0002°, contact 6.04°, and the edge gap is about 1e-11 mm.
- reNetworkBody, both tests: the kernel is a test-only GridKernel. Its faces are the grids' facets, and it sews sides whose samples coincide.
- reNetworkBody, 2×2 sheet: worstGap must be under 0.1 mm here, where ScanRuler requires 0.005, because facet sag is not the kernel's micron. The STEP export and verify checks and the builder's body.surface and faces are not ported. Face mapping and a JSON round trip of the report are checked in their place.
- reNetworkBody, closed sphere: report.closed, patches 6, corners 8 and face_of are checked. For volume, the ghost's must be within 0.005 of the sphere's (toBeCloseTo(1, 2)); ScanRuler compares a kernel body volume, which there is none of here. Ghost RMS equal to report RMS is checked across a second read. Ghost indices > 36 and ghost worstGap ≤ report worstGap + 1e-9 are kept.
- reNetworkBench: it is gated on RE_NETWORK_BENCH as in ScanRuler (the test returns at once without it) and uses GridKernel for the body. On 480k triangles it took 2.4–4.1 s against the 10 s bound, with worst edge 0.021° and RMS 0.0194–0.0220 mm.

## tests_passing
15

## failing_tests
[]

## review.remaining
["reNetworkFit '270° strip': the assertion that Fit Surface (fitSurfaceReported, in the separate surface package) refuses the same points is not ported, because this crate cannot depend on that package.", "reNetworkBody: the 2×2 sheet's worstGap bound is 0.1 mm instead of 0.005, because the test-only GridKernel's faces are facets and their sag shows there. The STEP export and verify checks, the builder body's volume and face count, and the whole 'thickens the sheet' test need the real kernel and Builder. The sphere test checks the ghost's volume against the sphere's within 0.005, where ScanRuler compares it against the kernel body's volume.", 'reNetworkBodies uses an analytic half-cylinder behind NetworkBodies instead of a kernel-built extrude found by fingerprint. Its assertions are unchanged.', 'fit_network validates its input and read_network_fitted returns Err; ScanRuler has no such checks and would crash or produce NaN. read_network never builds such input.', "Last-bit numeric deviations remain, as the reports already state: Math.hypot is a square root of a sum of squares, ** with an exponent is powf, and Rust's f64::max/min ignore NaN where JS's Math.max/min carry it on. Defensive guards (out-of-range indices filtered, empty holes skipped, missing curves or nodes skipped) differ only where the TS would yield NaN, undefined or a crash.", 'Other cosmetic or edge differences: NetworkFitResult.curves is a BTreeMap and lost_why is a BTreeMap, where ScanRuler uses Maps in insertion order (only the iteration order differs); the trace cache is keyed on f64 bits, so -0 and +0 are not shared, but the result is the same.', 'rayon is still declared and unused. 54 clippy style warnings in the library code predate this review (e.g. needless_range_loop); a plain cargo build/test is warning-free.']

## review.tests_passing
73

## review.tests_failing
0


# expressions

## public_api
expression:
  enum ParameterKind { Length, Angle, Integer, Number } (.dim(), .id())
  enum Dim { Of { mm: f64, deg: f64 }, Free }; const UNITLESS, LENGTH, ANGLE: Dim
  fn dim_of_kind(kind: ParameterKind) -> Dim
  struct Value { value: f64, dim: Dim }
  enum ScopeEntry { Value(Value), Error(String) }
  trait Scope { fn lookup(&self, name: &str) -> Option<ScopeEntry> }  (implemented for any Fn(&str) -> Option<ScopeEntry>, for ParameterTable, and for NoNames)
  struct ExpressionError { message: String, at: (usize, usize) }; type Evaluated = Result<Value, ExpressionError>
  fn evaluate(src: &str, scope: &dyn Scope) -> Evaluated
  fn evaluate_for(src: &str, kind: ParameterKind, scope: &dyn Scope, whole: bool) -> Evaluated
  fn references(src: &str) -> Vec<String>
  fn rename_in(src: &str, from: &str, to: &str) -> String
  fn is_plain_number(src: &str) -> bool
  fn plain_number(src: &str) -> Option<f64>
  fn valid_name(name: &str) -> bool; fn is_reserved(name: &str) -> bool; fn reserved_names() -> impl Iterator<Item=&'static str>
  fn describe_dim(d: Dim) -> String; fn describe_kind(k: ParameterKind) -> &'static str
  fn format_number(value: f64, places: usize) -> String
  fn format_value(value: f64, kind: ParameterKind, units: DisplayUnits) -> String

parameters:
  struct Parameter { id: u32, name: String, kind: ParameterKind, expression: String, comment: Option<String> }; PARAMETER_KINDS
  struct ExprMap (ordered path -> expression: get / insert / remove / iter / paths / exprs / len / is_empty; FromIterator; serde as a JSON object)
  fn negated_expr(e: &str) -> String
  fn exprs_from_command(kind: Option<&str>, given: serde_json::Map<String, Value>) -> serde_json::Map<String, Value>
  fn exprs_for_command(kind: Option<&str>, exprs: &ExprMap) -> ExprMap
  struct ParameterReading { id, name, kind, value: Option<f64>, error: Option<String> }
  struct ParameterTable { pub readings: Vec<ParameterReading>, .. } with fn scope(&self, name: &str) -> Option<ScopeEntry>, fn reading(&self, id) -> Option<&ParameterReading>; impl Scope
  fn evaluate_parameters(parameters: &[Parameter]) -> ParameterTable
  fn name_problem(name: &str, others: &[Parameter], self_id: Option<u32>) -> Option<String>
  struct Slot<T> { pub path: String, pub label: String, pub kind: ParameterKind }
    with Slot::new(path, label, kind, get: impl Fn(&T)->f64, set: impl Fn(&T,f64)->T), .get(&T) -> f64, .set(&T, f64) -> T, .lift(inner, put)
  trait Slotted: Clone + PartialEq { fn slots(&self) -> Vec<Slot<Self>>; fn exprs(&self) -> Option<&ExprMap>; fn with_exprs(&self, Option<ExprMap>) -> Self }
  type ExprErrors = BTreeMap<String, String>
  fn apply_exprs<T>(x: &T, exprs: &ExprMap, scope: &dyn Scope, slots: &dyn Fn(&T) -> Vec<Slot<T>>) -> ExprsApplied<T> { next: Cow<T>, kept: Option<ExprMap>, errors }   (the generic "slot map plus closure" entry point)
  fn apply_slots<T: Slotted>(x: &T, scope: &dyn Scope) -> SlotsApplied<T> { next: Cow<T>, errors }; fn apply_slots_with(x, scope, slots)
  fn with_slot<T: Slotted>(x: &T, path: &str, value: f64, expr: Option<&str>) -> T; fn with_slot_with(.., slots)
  fn unlink_slot<T: Slotted>(x: &T, path: &str) -> Cow<T>; fn slot_at<T: Slotted>(x: &T, path: &str) -> Option<Slot<T>>
  fn dimension_kind(rule: &Rule) -> ParameterKind
  fn apply_sketch_expressions(sketch: &Sketch, frame: &SectionFrame, scope: &dyn Scope) -> SketchApplied { next: Cow<Sketch>, errors }
  struct SketchInset { of: u32, by: f64, keep: Vec<u32> }
  trait ParametricFeature: Slotted { id, name, sketch, sketch_frame, with_sketch, source_sketch, inset, with_inset }
  fn replace_sketch<F>(features: &[F], id: FeatureId, sketch: &Sketch, rekey: &mut dyn FnMut(&F) -> Option<F>) -> Vec<F>
  enum PickGiven { All, Largest, Picks(Vec<PickItem>) }; enum PickItem { Key(String), At { at: DVec2 } }; struct RegionPick { given, keys }
  trait RegionPicks; trait RegionMatch { key, match_region, regions_by_pick, was_region, region_by_place, split_parts }
  fn rekeyed<T: RegionPicks>(f: &T, regions: &dyn RegionMatch) -> Cow<T>
  type ReplaceSketch<F> = dyn FnMut(&[F], FeatureId, Sketch) -> Vec<F>; type InsetAgain<F> = dyn FnMut(&F, &[F]) -> Result<Sketch, String>; type InsetChains = dyn Fn(&Sketch, &SketchInset, &[DatumTrace], &str) -> Result<Sketch, String>
  fn apply_parameters<F: ParametricFeature>(features: &[F], scope: &dyn Scope, replace_sketch: &mut ReplaceSketch<F>, inset_again: &mut InsetAgain<F>) -> ParametersApplied<F> { features: Vec<F>, changed: bool, errors: BTreeMap<FeatureId, ExprErrors> }
  fn inset_again<F>(f: &F, features: &[F], inset_chains: &InsetChains) -> Result<Sketch, String>
  fn follow_parameters<F>(features: &[F], table: &ParameterTable, renamed: Option<(&str, &str)>, replace_sketch, inset_again) -> ParametersApplied<F>
  fn feature_expression_errors<F>(f: &F, scope: &dyn Scope) -> ExprErrors
  fn feature_reads<F>(f: &F, name: &str) -> bool
  fn users_of<F>(name: &str, parameters: &[Parameter], features: &[F]) -> Vec<ParameterUser { kind: UserKind, id, name }>
  fn fields_reading<F>(f: &F, name: &str) -> Vec<String>
  fn idle_parameters<F>(parameters: &[Parameter], features: &[F]) -> Vec<&Parameter>
  fn rename_in_feature<F>(f: &F, from, to) -> Cow<F>; fn rename_in_sketch(sketch, from, to) -> Cow<Sketch>; fn rename_in_parameters(parameters, id, to) -> Vec<Parameter>
  fn new_parameter(parameters: &[Parameter], id: u32, kind: ParameterKind) -> Parameter
  struct ParameterPatch { name, kind, expression, comment: Option<..> }; struct ParameterEdit { parameters, table, renamed: Option<(String, String)> }
  fn edit_parameter(parameters: &[Parameter], before: &ParameterTable, id: u32, patch: &ParameterPatch) -> Result<ParameterEdit, String>
  fn removal_problem<F>(parameters: &[Parameter], features: &[F], id: u32, drafts_reading: &[&str]) -> Option<String>; const READ_BY_FEATURE_DRAFT, READ_BY_SKETCH_DRAFT

param_slots:
  struct JsonFeature { sketch: Option<Sketch>, exprs: Option<ExprMap>, fields: serde_json::Map } (from_value, kind, get, with; implements Slotted, ParametricFeature, RegionPicks)
  fn json_slots(x: &JsonFeature, kind: Option<&str>) -> Vec<Slot<JsonFeature>>; fn json_slot_at(x, path, kind) -> Option<Slot<JsonFeature>>
  fn no_slots_why(kind: Option<&str>, x: &JsonFeature) -> &'static str
  fn plane_takes_offset(&Value) -> bool; fn plane_offset(&Value) -> f64; fn plane_at_offset(&Value, f64) -> Value; fn with_plane_offset(&JsonFeature, f64) -> JsonFeature
  fn frame_of(Option<&Value>) -> Option<SectionFrame>; fn frame_json(&SectionFrame) -> Value; fn json_number(f64) -> Value
  fn revolve_max_deg(pitch: f64) -> f64; const FULL_TURN_DEG, MAX_COIL_DEG
  fn model_slots(kind: &FeatureKind) -> Vec<Slot<FeatureKind>>
  struct ModelFeature { feature: model::Feature, exprs: Option<ExprMap> } (new; implements Slotted, ParametricFeature, RegionPicks)

length_fields:
  fn len_bare(mm, units, mm_places) -> String; fn len_short(mm, units, mm_places) -> String; fn field_text(value, kind, places, units) -> String
  struct FieldRead { value: f64, expr: Option<String> }; fn read_field(text: &str, kind, scope: &dyn Scope, units) -> Result<FieldRead, String>
  fn parameter_text(expression, kind, units) -> String; fn parameter_expression(typed, kind, units) -> String

display_units / js_number: DisplayUnits, to_display, from_display, length_decimals, unit_word, fmt_len, length_step, length_input_text; to_fixed, js_string, js_round, js_pow, fixed_trimmed.

## integration_notes
Copying the files in:
- Copy src/expression.rs, parameters.rs, param_slots.rs, length_fields.rs, display_units.rs and js_number.rs into crates/mm-reverse/src and add a `pub mod` line for each.
- The copied datums.rs, fit2d.rs, frame.rs, linalg.rs, model.rs, sketch.rs and solver.rs are identical to mm-reverse's (checked with cmp), so drop them.
- Nothing new is needed in Cargo.toml: glam, serde, serde_json and thiserror only.

Overlap with sibling ports:
- The threads port has a partial `units.rs` (DisplayUnits, to_display, length_decimals taking i32). The fillet port's `fillet_num.rs` has its own js_round and to_precision.
- Merge these into display_units.rs and js_number.rs, which are the complete port of units.ts plus the JavaScript number formatting.

Wiring into mm-reverse:
1. Give model::Feature an `#[serde(default, skip_serializing_if = "Option::is_none")] exprs: Option<ExprMap>`. Then implement Slotted, ParametricFeature and RegionPicks on Feature directly, using model_slots(&self.kind) in the same way as ModelFeature, and drop the wrapper. Sketch dimensions already keep `Constraint.expr`.
2. Keep `parameters: Vec<Parameter>` (and a next id) in the Part or project and save them. Keep a ParameterTable in app state, rebuilt with evaluate_parameters.
3. On a parameter edit, call edit_parameter. If it succeeds:
   - Call follow_parameters on the features and on the rolled-back features, and rebuild when `changed` is true.
   - Open drafts follow with rename_in_feature plus apply_slots_with. A sketch draft follows with rename_in_sketch plus apply_sketch_expressions.
   - Before removing a parameter, call removal_problem.
4. The replace_sketch closure should call parameters::replace_sketch with a rekey that runs rekeyed against a RegionMatch built on mm-reverse's regions. That needs regions.ts's matchRegion, regionsByPick, regionByPlace and splitParts ported. Until then, `|_| None` keeps keys as they are.
5. The inset_again closure should call parameters::inset_again with a port of stations.ts insetChains, once loft inner stations exist.
6. UI number fields: read with length_fields::read_field, store the value, and store the expression with with_slot. A plain number typed over an expression, or a dragged grip, calls unlink_slot. Show a feature's errors (feature_expression_errors, or the errors map from apply_parameters) on its row.
7. Commands: exprs_from_command / exprs_for_command handle the extrude draftDeg/taper sign; no_slots_why finishes refusal messages.

JsonFeature is a faithful, fully tested port of ScanRuler's whole slot table. It is useful for reading ScanRuler projects or command JSON, and as the reference to follow when the model gains ScanRuler's other feature kinds.

Totals:
- `cargo build --release` gives no warnings.
- `cargo test --release`: 82 tests, all passing. That is 53 ported from ScanRuler (16 reExpression, 15 reParameters, 6 reParameterStore core-level, 5 reExprsEveryKind slot-level, 10 reUiUnits, 1 reCoreUnits), 15 added by the port, and 14 in the copied modules.
- Clippy notes left in my files: one `!(x > 0.0)` in display_units.rs, mirroring ScanRuler's NaN-aware check.

## not_ported
Outside this scope (needs the command layer, the app store, the UI or the kernel):
- reFeatureInput's withExpressions and applyAll, reFeatureCommands' loftExprsIn and noNumbers. noSlotsWhy itself is ported.
- reStore's undo/redo around parameter edits, setDraftExpr, the grip and key-parameter actions, and setDimensionForm (radius to diameter with '2 * (bore)').
- ui/featureReading (keyParameter, bodyFeatureReading) and viewer/sketchDimText.

Not ported because they live in other ScanRuler modules and are reached through traits or closures here:
- regions.ts matching: matchRegion, regionsByPick, regionByPlace, splitParts. mm-reverse's regions.rs depends on SolidCraft and has none of these.
- stations.ts insetChains.

Tests not ported:
- reExprsEveryKind as written: it needs the headless app, the commands and the kernel, and checks built bodies' bounds. Its five cases are ported at slot level instead: cut and mirror offsets, the mirror-on-a-face refusal, a loft station's wall and offset, the surface box with regionStale, and a kind with no numbers.
- reParameterStore: 'a round piece's dimension', 'reopens with its expression … grip dragged', 'follows the parameter while the sketch is open' (sketch-draft history), and the undo/redo and rebuild-counter parts. Their effect on feature numbers is ported.
- reUiUnits 'feature rows' (featureReading), reSketchUnits (viewer dimension text, planSnap), reAppUnits (store keyParameterOf).

## deviations
1. Return types and sameness. Results replace ScanRuler's {ok}/{error} unions. Where ScanRuler hands back "the same object" when nothing changed, this hands back a borrowed Cow. "Changed" is decided by comparing values, not object identity. So a number written over with the value it already held (for example a revolve clamped back to 360°), or an inset restored after a failed move, does not count as a change here. ScanRuler would mark it changed and rebuild.
2. Nesting limit. The parser refuses expressions nested more than 1000 deep with "is nested too deeply to read". This includes a flat chain of more than 1000 operations, which ScanRuler would evaluate; ScanRuler has no limit of its own and relies on JavaScript's stack. The limit keeps Rust's stack safe.
3. Error positions count characters, not UTF-16 units. They are the same for every character the language reads; they differ only for characters outside the Basic Multilingual Plane (emoji and the like).
4. A JavaScript quirk is not reproduced: TypeScript's `in` and object indexing would treat names like `constructor` or `toString` as units or constants.
5. Slots are generic. JsonFeature runs ScanRuler's own table over a feature's JSON fields. Numbers missing where ScanRuler falls back to 0 read as 0; numbers missing where it has no fallback read as NaN, which no expression ever writes over (the same effect as `undefined` in JavaScript). An extrude with no extent at all offers no extent slots, where ScanRuler would throw. JSON numbers are written as JavaScript writes them (whole numbers as integers, -0 as 0, NaN as null), and JsonFeature compares numbers by value.
6. model_slots maps the same table onto Polymix's FeatureKind, with these differences: a through-all extent has no `extent.from` (the model's has no start); revolve has no pitch slots; chamfer has one distance; every model plane takes an offset. Polymix's `taper_deg` already uses ScanRuler's displayed sign (taper = −draftDeg), so the `taper` slot reads and writes it directly.
7. ExprErrors is a BTreeMap sorted by path, not in insertion order. Only matters if a message lists several errors in a row.
8. Inputs that live elsewhere become arguments:
   - apply_parameters takes `replace_sketch` and `inset_again` closures, because region matching (regions.ts) and the inset move (stations.ts insetChains) are not in this crate.
   - replace_sketch takes a `rekey` closure and does not take ScanRuler's `patch` argument; the caller writes the new name.
   - inset_again takes an `inset_chains` closure.
   - rekeyed works against a RegionMatch trait.
9. apply_sketch_expressions never fills `errors.sketch`. Polymix's solver returns a report instead of throwing, and ScanRuler's solver never throws either.
10. inset_again returns its own error for a feature with no inset; ScanRuler just asserts one exists.
11. The store actions setParameter, addParameter and removeParameter become pure functions (edit_parameter, new_parameter, removal_problem) with no undo snapshot. removal_problem takes the names of open drafts that read the parameter.
12. A rekeying test with toy regions, and own tests for JSON round-trip, more kinds, model slots, signed-number reading and JavaScript number formatting were added beyond ScanRuler's tests.

## tests_passing
53

## failing_tests
[]

## review.remaining
["model_slots and ModelFeature are written against the 8-kind model copied into this crate. mm-reverse's live model.rs now carries ScanRuler's whole feature list with `exprs` per kind, and its frame.rs uses ScanRuler's field names. These copies must not be dropped as identical. The slots need mapping onto the live model, with json_slots (ScanRuler's whole table, faithful and tested) as the reference.", "Deviation 1 stays: 'changed' is value equality, not object identity. A clamped revolve written back, or an inset restored after a failed move, is no change here where ScanRuler would rebuild.", "The stack budget makes the refusal depth for pathological input depend on the build. On a 2 MiB thread, debug reads a 123-link parameter chain, 202 brackets and 102 nested calls; release reads 425, 1056 and 727. ScanRuler reaches roughly 1500 and 3000 before throwing. In a chain that is cut, the parameter where the budget ran out has its error cached. The budget counts from the evaluator's first entry on the thread; a caller with less than about 1 MiB of stack left is not covered.", "ParametersApplied.errors stays a BTreeMap keyed by feature id; ScanRuler's Map iterates in list order. Lookups by id are unaffected, only the iteration order.", "Deviations kept as documented by the porter: error positions count chars rather than UTF-16 units; JavaScript's prototype-key quirk (`constructor` as a unit) is not reproduced; JsonFeature's `sketch` is Polymix's sketch form, not ScanRuler's.", "Not ported, as the porter listed: regions.ts matching (stands behind the RegionMatch trait), stations.ts insetChains (a closure), the store's undo/redo and draft histories, setDimensionForm, featureReading, and the kernel-level checks of reExprsEveryKind."]

## review.tests_passing
95

## review.tests_failing
0


# spline

## public_api
// flat_spline (src/core/flat/spline.ts)
pub const HANDLE_REACH: f64 = 1.0/3.0; pub const POLYLINE_STEPS: usize = 24;
pub struct FlatSplineFit { pub points: Vec<DVec2>, pub tangents: Vec<DVec2>, pub fixed: Vec<bool>, pub closed: bool, pub length: f64, pub sigma: f64, pub used_points: usize }  // Serialize/Deserialize
pub enum HandleEnd { A, B }
pub struct SplineHandle { pub at: DVec2, pub a: DVec2, pub b: DVec2, pub fixed: bool }
pub struct SplineBezierForm { pub poles: Vec<DVec2>, pub knots: Vec<f64> }
pub struct NearestSegment { pub segment: usize, pub dist: f64 }
pub fn spline_segment_count(point_count: usize, closed: bool) -> usize
pub fn fit_spline_points(points: &[DVec2], handles: &[Option<DVec2>], closed: bool) -> Result<FlatSplineFit, FitError>
pub fn eval_spline(fit: &FlatSplineFit, k: usize, u: f64) -> DVec2
pub fn eval_spline_derivative(fit: &FlatSplineFit, k: usize, u: f64) -> DVec2
pub fn spline_polyline(fit: &FlatSplineFit, per_segment: usize) -> Vec<DVec2>
pub fn spline_midpoint(fit: &FlatSplineFit) -> DVec2
pub fn spline_handles(fit: &FlatSplineFit) -> Vec<SplineHandle>
pub fn spline_bezier_form(fit: &FlatSplineFit) -> SplineBezierForm
pub fn nearest_spline_segment(fit: &FlatSplineFit, p: DVec2, per_segment: usize) -> NearestSegment

// sketch_spline (sketchSpline.ts + spline parts of sketch.ts and profiles.ts)
pub const NEAREST_STEPS: usize = 16;
pub struct SplineShape { pub pts: Vec<DVec2>, pub closed: bool, pub start_dir: Option<f64>, pub end_dir: Option<f64> }  // serde; SplineShape::open(pts), SplineShape::closed(pts)
pub type Bezier = [DVec2; 4];
pub struct BSpline<P = DVec2> { pub poles: Vec<P>, pub knots: Vec<f64>, pub mults: Vec<usize> }  // serde
pub trait Pole: Copy + Add<Output=Self> + Mul<f64, Output=Self> { const DIM: usize; fn coord(self, i: usize) -> f64 }  // impl for DVec2, DVec3
pub struct SplineSample { pub t: f64, pub p: DVec2 }
pub struct NearestOnSpline { pub t: f64, pub point: DVec2, pub dist: f64 }
pub struct StretchBeziers { pub pieces: Vec<Bezier>, pub spans: Vec<f64> }
pub struct SplineSoftPoints { pub points: Vec<DVec2>, pub scatter: f64 }
pub fn spline_segments_of(e: &SplineShape) -> usize
pub fn is_closed(e: &SplineShape) -> bool
pub fn spline_curve(e: &SplineShape) -> FlatSplineFit
pub fn spline_at(fit: &FlatSplineFit, t: f64) -> DVec2
pub fn spline_deriv_at(fit: &FlatSplineFit, t: f64) -> DVec2
pub fn spline_dir_at(fit: &FlatSplineFit, t: f64) -> DVec2
pub fn spline_end(fit: &FlatSplineFit) -> f64
pub fn spline_samples(fit: &FlatSplineFit, per_segment: usize) -> Vec<SplineSample>
pub fn nearest_on_spline(fit: &FlatSplineFit, p: DVec2, per_segment: usize) -> NearestOnSpline
pub fn spline_stretch_beziers(fit: &FlatSplineFit, t0: f64, t1: f64) -> StretchBeziers
pub fn bezier_chain_to_bspline<P: Pole>(pieces: &[[P; 4]], spans: &[f64]) -> Option<BSpline<P>>
pub fn spline_stretch(fit: &FlatSplineFit, t0: f64, t1: f64) -> Option<BSpline>
pub fn bspline_at<P: Pole>(b: &BSpline<P>, t: f64) -> Option<P>
pub fn bspline_samples<P: Pole>(b: &BSpline<P>, count: usize) -> Vec<P>
pub fn lift_bspline(frame: &SectionFrame, b: &BSpline) -> BSpline<DVec3>
pub fn lift_spline(frame: &SectionFrame, e: &SplineShape, t0: Option<f64>, t1: Option<f64>) -> Option<BSpline<DVec3>>
pub fn fit_spline_to_points(e: &SplineShape, slice: &[DVec2]) -> Option<SplineSoftPoints>
pub fn spline_scatter(e: &SplineShape, points: &[DVec2]) -> f64
pub fn fit_points_along(chain: &[DVec2], count: usize, closed: bool) -> Option<Vec<DVec2>>
pub fn points_along(fit: &FlatSplineFit, points: &[DVec2], reach: f64) -> Vec<DVec2>
pub fn spline_point_index(e: &SplineShape, at: AnchorAt) -> Option<usize>      // 'a'/'b'/'center'
pub fn spline_fit_point_index(e: &SplineShape, i: usize) -> Option<usize>     // 'p{i}'
pub fn spline_anchor(e: &SplineShape, i: usize) -> Option<AnchorAt>          // None = named 'p{i}'
pub fn spline_anchor_point(e: &SplineShape, index: Option<usize>) -> DVec2   // fit point, else the middle
pub fn spline_anchor_order(e: &SplineShape) -> Vec<usize>                     // anchorsOf as indices
pub fn spline_has_ends(e: &SplineShape) -> bool                               // endsOf
pub fn nearest_on_spline_piece(e: &SplineShape, p: DVec2) -> (DVec2, f64)
pub fn spline_piece_polyline(e: &SplineShape, steps: usize) -> Vec<DVec2>
pub struct SplineJoint { pub end: AnchorAt, pub at: AnchorAt }
pub fn spline_joint(constraints: &[Constraint], spline: EntityId, other: EntityId) -> Option<SplineJoint>

// spline_solve (spline parts of solver.ts)
pub const REFIT_ROUNDS: usize = 24; pub const REFIT_SETTLED: f64 = 1e-7;
pub fn spline_param_count(e: &SplineShape) -> usize
pub fn spline_params(e: &SplineShape) -> Vec<f64>
pub fn spline_with_params(e: &SplineShape, x: &[f64], at: usize) -> SplineShape
pub fn mean_chord(e: &SplineShape) -> f64
pub fn spline_stay_units(e: &SplineShape) -> Vec<f64>
pub fn spline_size(e: &SplineShape) -> f64
pub fn spline_end_dir(s: &SplineShape, end: AnchorAt) -> DVec2
pub struct HeldEnds { pub a: bool, pub b: bool }
pub fn held_ends(constraints: &[Constraint], spline: EntityId, e: &SplineShape) -> HeldEnds
pub fn with_held_ends(s: &SplineShape, held: HeldEnds) -> SplineShape
pub struct SplineSoftData; impl { pub fn new(e: &SplineShape, offset: usize, points: &[DVec2]) -> Self; pub fn params(&self) -> Range<usize>; pub fn residual_count(&self) -> usize; pub fn feet(&self) -> Vec<f64>; pub fn eval(&self, xs: &[f64]) -> Vec<f64>; pub fn refresh(&self, xs: &[f64]) -> f64 }
pub enum JointPiece<'a> { Line { a: DVec2, b: DVec2 }, Arc { start: f64, sweep: f64 }, Spline(&'a SplineShape), Other }
pub fn spline_tangent_residual(s: &SplineShape, joint: SplineJoint, other: JointPiece, scale: f64) -> f64
pub fn point_on_spline_residual(e: &SplineShape, p: DVec2) -> f64

## integration_notes
- Copied mm-reverse files: fit2d.rs, linalg.rs, frame.rs, datums.rs, sketch.rs and chain.rs. All are byte-identical to mm-reverse (checked with cmp). Drop them and add `pub mod flat_spline; pub mod sketch_spline; pub mod spline_solve;` to mm-reverse/src/lib.rs. No new dependencies.
- sketch.rs (spline piece):
  - Add a spline piece. Either drop Copy from `Shape` and add `Shape::Spline(SplineShape)` with ShapeKind::Spline, or keep splines out of `Shape`.
  - Add `AnchorAt::P(u32)` for fit points, mapped with spline_fit_point_index and spline_anchor. ScanRuler's message for a bad name is "has no point “p7” — it has a, b, p1, p2".
  - anchor_point → spline_anchor_point(e, spline_point_index(e, at) or spline_fit_point_index(e, i)).
  - anchors_of → spline_anchor_order; ends_of → spline_has_ends; nearest_on → nearest_on_spline_piece; polyline → spline_piece_polyline(e, steps) (bounds uses it).
  - relation_takes "tangent": a spline counts as curved, but circle + spline is not allowed.
  - infer_relations: join an open spline's ends.
- solver.rs (spline support):
  - Run every spline through `with_held_ends(e, held_ends(&constraints, id, e))` and write the result into the sketch before laying out params.
  - Params: params_of/with_params → spline_params/spline_with_params.
  - Stay-put units: spline_stay_units.
  - Scale: `scale = scale.max(spline_size(e))`.
  - Soft data: `Rc::new(SplineSoftData::new(&held_e, at, &thin(points)))`, weight 1/scatter.max(1e-3), BlockKind::Soft; Block needs an optional refresh hook.
  - After the first full descend, when not quick: up to REFIT_ROUNDS rounds. Take the max of refresh(&x); stop below REFIT_SETTLED; otherwise descend all blocks again (MAX_ITERATIONS, 1e-9, 1e-12). This comes before the pull-release pass and the CLOSE_FROM pass.
  - Tangent with a spline on either side: `spline_joint(&constraints, spline, other)`. With no joint, add no block. Otherwise add a hard block over both pieces' params whose eval is spline_tangent_residual(read spline, joint, JointPiece::{Line, Arc{start,sweep}, Spline, Other}, l_scale).
  - PointOn Curve(spline): point_on_spline_residual. A bounded pointOn on a spline gives 0.
  - Equal: a spline's size is 0. Concentric: a spline's centre is pts[0].
  - finish: no normalisation for splines. heldByItself is unchanged.
- Kernel and profiles:
  - A loop's spline piece is lift_spline(frame, e, t0, t1), giving BSpline<DVec3> {poles, knots, mults}: clamped cubic, mults 4/1…1/4, or 4/3…3/4 when the chain is not C2. Convert this to SolidCraft's B-spline edge in build.rs; I did not check the SolidCraft API, because this crate cannot depend on it.
  - sweepPath runs a spline from the end it is entered at: lift_spline(frame, e, Some(0.0), Some(end)) or (Some(end), Some(0.0)). Its start tangent is spline_dir_at.
- Fitting (state layer):
  - fitChain's spline option: view_chain(chain), count = options.spline.max(3 if closed else 2), refuse when there are fewer than 2*count points, then fit_points_along(points, count, closed).
  - Fit to slice: fit_spline_to_points(e, slice), solve, then reset scatter to spline_scatter(e, &points).
- flat_spline also serves the 2D workspace: handles, Bézier form for SVG, DXF and STEP, and the hit test.

## not_ported
- reSpline "regions a spline bounds" (4 tests): need regions.ts with spline pieces (findRegions, loopGeometry, splitting at T-joins). Out of this package's scope; regions should call `spline_stretch(fit, t0, t1)` for a loop's spline piece.
- reSpline "a drawn spline's ends are joined where they meet another piece": needs sketch.ts inferRelations with spline ends, and sketch.rs may not be edited here.
- reSpline "a spline in the part" (2 tests, extrude and STEP B_SPLINE_CURVE_WITH_KNOTS) and reSplineCommands.test.ts (3 tests): need the OCCT/SolidCraft kernel build, STEP export and the headless command layer.
- Spline cases in other test files exercise modules outside this package: flatDatum, flatDxf, flatSvg, flatReport, sectionLift, exportStep, reStations, reOffsetFaces, reRevolvePoleStep, reRound23/24.
- Not ported because their files are outside this scope: profiles.ts sweepPath (the closed-spline refusal and the spline's start tangent), drag.ts moveAnchor/translateEntity, the state-layer fitChain and fitSplinesToSlice (store glue), and solver.ts measureDimension (a spline's pointOf is null).
- There are no reSketchSpline* or reSplineFit* files in ScanRuler, no reSolver test file, and no spline cases in reSketch.test.ts.
- Everything in the assigned TS files that is pure geometry has been ported.

## deviations
- Spline data: the Rust sketch has no spline piece, and `Shape` derives Copy, so it cannot hold a Vec. The spline is therefore its own struct, `SplineShape` (pts, closed, start_dir, end_dir). The soft data (points, scatter) stays on `Entity` as it does for the other pieces. Because of this, `fit_spline_to_points` returns `SplineSoftPoints {points, scatter}` instead of a copy of the entity. Serde field names are snake_case like the rest of sketch.rs, not ScanRuler's startDir/endDir.
- Fit-point anchors: the copied `AnchorAt` has only A, B and Center, with nothing for `p{i}`. The sketch.ts helpers therefore take an index: `spline_point_index` handles a/b/center, `spline_fit_point_index` handles p{i}, and `spline_anchor` returns None where TS returns `p${i}`. `spline_anchor_order` gives anchorsOf as indices.
- `withHeldEnds(s, sketch)` is split into `held_ends(constraints, id, e)` and `with_held_ends(e, held)`. The soft-data block is a struct, `SplineSoftData`, with eval and refresh. The feet sit in a RefCell so the solver can share the block through an Rc, the way solver.rs shares `rests`.
- Default arguments are now explicit parameters, with constants for the defaults: POLYLINE_STEPS = 24 and NEAREST_STEPS = 16. The solver's foot search keeps ScanRuler's literal 8.
- `fit_points_along` takes `count: usize`. TS takes any number and applies Math.round.
- Panic-free guards where TS would throw a TypeError or loop forever:
  - `bezier_chain_to_bspline` returns Option: None for an empty chain or when spans and pieces differ in length.
  - `bspline_at` returns Option: None when there are too few poles or knots.
  - `spline_stretch_beziers` returns nothing when t0 or t1 is not finite.
  - `spline_at`, `spline_samples` and `spline_polyline` return zero or empty for a fit with no points.
  - Segment indices wrap modulo n, and a stored fit with too few tangents reads zero.
- `FlatSplineFit` drops the `kind` tag and the optional `formError`, which ScanRuler never sets on a spline.
- `point_on_spline_residual` builds the curve once. TS builds it twice, with identical results.
- Clamp patterns use `f64::clamp`, which matches JavaScript's NaN behaviour.
- Test adaptations:
  - The reSpline solver cases need a sketch solver with splines, and the Rust solver has none. They run on a test-only harness that follows solveSketch step by step and calls these functions where the solver will. Its LM `descend` is copied from mm-reverse solver.rs.
  - The `p1` coincidence is modelled as a harness "fit pin", because AnchorAt cannot name it.
  - drag.ts `moveAnchor` (spline case) is done by setting pts[i] directly.
  - In the chain test, fitChain's spline branch is inlined: view_chain, then fit_points_along, then a closed spline with points and scatter. The final `findRegions(s).length == 1` assertion is dropped because regions are out of scope.
- Test counts: 20 ScanRuler tests ported (11 from flatSpline, 9 from reSpline), all passing. I added 4 tests of my own: the lift through a frame, a kinked chain on triple knots, anchors and joints, and a parameter round trip. The copied modules bring 9 of their own. In total, 33 tests pass.

## tests_passing
20

## failing_tests
[]

## review.remaining
["nearest_on_spline (and so the solver's feet, point_on_spline_residual and spline_scatter at the last digits) can differ from ScanRuler by about 1e-8 in t. This is floating-point noise in the golden-section search, mostly V8 Math.hypot against libm hypot, not a logic difference. A zero-length segment can tie-break to a different t at the same point.", "spline_anchor_order returns anchorsOf as indices, so a degenerate open spline differs from TS: with 0 points it gives [] where TS gives ['a','b'], and with 1 point it gives [0,0], which cannot name 'b' apart from 'a'. Indices are unavoidable until AnchorAt gets a P(i) variant.", "NaN handling differs where Rust's f64::max/min drop NaN and JS Math.max/min propagate it: locate holds a NaN t to 0 (TS would throw), the fit_spline_to_points scatter floor and the spline_stay_units chord floor give 0.01 and 1e-3 (TS gives NaN), and solve_linear's tiny is never NaN. This only matters on NaN input.", "Serde: FlatSplineFit has no 'kind' tag or 'formError', and FlatSplineFit and SplineShape use snake_case (used_points, start_dir, end_dir), not ScanRuler's camelCase. Files saved by ScanRuler would need renaming to load.", "Still out of scope and unported, as the porter reported: the regions with spline pieces (4 tests), inferRelations joining spline ends (1 test), relationTakes' spline rule, the kernel and STEP extrude tests (2 tests) and reSplineCommands (3), drag.ts moveAnchor and translateEntity, the profiles.ts sweepPath spline branch and loop mending of spline poles, project.ts spline projection, solver.ts measureDimension, and the state-layer fitChain and fitSplinesToSlice glue.", 'clippy, not rustc, flags 11 style warnings: nonminimal_bool on deliberate `!(x > y)` NaN guards, plus one needless_range_loop. Most are in my files; the rest are in the copied mm-reverse modules. cargo test is warning-free.']

## review.tests_passing
35

## review.tests_failing
0


# threads

## public_api
thread.rs:
  enum ThreadStandard { Metric, Unc, Unf, Print45 } (serde "metric"|"unc"|"unf"|"print45"); fn id(self)->&'static str; fn label(self)->&'static str; fn from_id(&str)->Option<ThreadStandard>
  const THREAD_STANDARDS: [ThreadStandard; 4]
  struct ThreadSize { label: String, diameter: f64, pitches: Vec<f64> }
  fn thread_sizes(ThreadStandard) -> &'static [ThreadSize]
  fn size_of(ThreadStandard, diameter: f64) -> Option<&'static ThreadSize>
  fn threads_per_inch(pitch: f64) -> f64
  fn thread_designation(ThreadStandard, diameter: f64, pitch: f64) -> Option<String>
  fn thread_name(ThreadStandard, diameter: f64, pitch: f64) -> String
  fn thread_depth(pitch: f64, ThreadStandard) -> f64; fn minor_diameter(d, p, ThreadStandard) -> f64; fn pitch_diameter(d, p, ThreadStandard) -> f64
  struct ThreadSpec { standard, diameter, pitch, offset } (serde camelCase)
  struct ThreadCut { #[serde(flatten)] spec: ThreadSpec, left_handed: bool, length: Option<f64>, start: f64, flip: bool, chamfer: bool }
  enum FaceAgainst { Wider, Narrower }; fn face_against_thread(&ThreadSpec, internal: bool, face_diameter: f64) -> Option<FaceAgainst>
  fn max_thread_offset(pitch: f64, ThreadStandard) -> f64
  struct ThreadChoice { diameter, pitch }; fn size_for_face(ThreadStandard, face_diameter: f64, internal: bool) -> ThreadChoice
  fn thread_problem(&ThreadSpec, units: DisplayUnits) -> Option<String>
  struct ToothOutline { points: Vec<DVec2> /* x=r, y=z, CCW */, crest: f64, root: f64, pointed: bool }
  fn tooth_outline(&ThreadSpec, internal: bool, reach: f64, face: Option<f64>) -> ToothOutline
  fn fill_radius(&ThreadSpec, internal: bool, face_r: f64) -> Option<f64>
  fn chamfer_radius(&ThreadSpec, internal: bool) -> f64
  struct FaceMesh<'a> { positions: &'a [DVec3], triangles: &'a [[u32;3]], tri_face: &'a [u32] }
  fn round_face_is_boss(FaceMesh, face: u32, center: DVec3, axis: DVec3) -> Option<bool>
  struct RoundFacePick { face: Option<u32>, center: Option<DVec3>, radius: Option<f64>, dir: DVec3 }
  struct ThreadFaceReading { internal: bool, diameter: Option<f64>, pitch: Option<f64> }
  fn thread_face_reading(FaceMesh, &RoundFacePick, ThreadStandard, sized: bool) -> Option<ThreadFaceReading>
  struct ThreadSpan { from, to }; fn thread_span(lo: f64, hi: f64, length: Option<f64>, start: f64, flip: bool) -> Option<ThreadSpan>
thread_cut.rs:
  const MAX_THREAD_TURNS: u32 = 400; const THREAD_FACE_NEEDS: &str
  struct ThreadCutError(pub String) (Display "Cutting the thread: {0}"); fn ThreadCutError::not_a_cylinder() -> Self
  fn thread_note(&ThreadCut, internal: bool, face_r: f64, pointed: bool, unfilled: bool) -> Option<String>
  struct RoundFace { center: DVec3, axis: DVec3, radius: f64, convex: bool, lo: f64, hi: f64, area: f64 }
  struct Annulus { center, axis, outer: f64, inner: Option<f64>, from: f64, to: f64 }
  struct ThreadSetup { cut, face, internal, span, wanted: Option<f64>, square: bool }
    fn new(&ThreadCut, &RoundFace) -> Result<ThreadSetup, ThreadCutError>; fn wants_hull(&self) -> bool; fn fill(&self, hull: bool) -> Result<ThreadFill, ThreadCutError>
  struct ThreadFill { setup, fill: Option<f64>, face_r: f64, hull: bool, sleeve: Option<Annulus> }
    fn plan(&self, contains: impl FnMut(DVec3) -> bool) -> Result<ThreadPlan, ThreadCutError>
  struct HelixSweep { plane_origin, plane_x, plane_y: DVec3, profile: Vec<DVec2>, profile_world: Vec<DVec3>, axis_origin, axis: DVec3, pitch: f64, turns: u32, left_handed: bool, turn_by_turn: bool }
    fn angle(&self) -> f64; fn right_handed_axis_and_pitch(&self) -> (DVec3, f64)  // for SolidCraft sweep_helix: LH = (-axis, -pitch)
  struct ChamferRing { at: f64, out: f64, section: [DVec2;3] /* (r, s into body) */, world: [DVec3;3], center, axis }
  struct ThreadPlan { internal, tooth: ToothOutline, band: Annulus, helix: HelixSweep, free_lo, free_hi: bool, chamfers: Vec<ChamferRing>, keep_to_hull: bool, note: Option<String> }
thread_read.rs:
  struct ThreadReadInput<'a> { points: &'a [DVec3], normals: &'a [DVec3], origin, dir: DVec3, radius, band, half_length: f64 }
  struct ThreadReading { points, off_wall: usize, span: [f64;2], radius: ReadRadius{mean,low,high}, minor, major, depth, internal: Option<bool>, pitch: Option<f64>, strength, contrast: Option<f64>, noise, left_handed: Option<bool>, hand_confidence, windows: Vec<ThreadWindow>, agree: usize, threaded: Option<[f64;2]>, profile: Vec<ProfilePoint{z,r}> } (serde camelCase)
  fn read_thread(&ThreadReadInput) -> Option<ThreadReading>
  fn is_thread(&ThreadReading) -> bool
  struct ThreadCandidate { standard, size, designation, diameter, pitch, pitch_off, diameter_off, score }; struct ThreadMatch { candidates: Vec<ThreadCandidate>, suggested: Option<ThreadCandidate> }
  fn match_thread(&ThreadReading, standards: &[ThreadStandard]) -> ThreadMatch
  fn basic_depth(pitch: f64, ThreadStandard) -> f64; const ALL_STANDARDS: [ThreadStandard; 4]
inserts.rs:
  struct InsertSize { label: &'static str, length, hole, wall: f64 }; const INSERT_SIZES: [InsertSize; 12]
  fn insert_size(&str) -> Option<&'static InsertSize>; struct InsertHole { radius, depth }; fn insert_hole(&InsertSize) -> InsertHole
teeth.rs:
  enum Envelope { Arc { center: DVec2, radius, radius_sigma, rms, worst }, Line { point: DVec2, dir: DVec2, rms, worst } } (serde tag "kind"); fn is_arc(&self)->bool
  enum PitchAlong { U, V, Trend }; enum ToothCorner { Crest, Trough }; enum CrestsBy { Material, Side }
  struct CornerEnvelope { #[serde(flatten)] envelope: Envelope, corners: Vec<DVec2> }
  struct TeethRead { trend, crests, troughs, crests_by, pitch, along, spread, crest_to_trough, riser, facet, depth: TeethDepth{mean,min,max}, profile: Vec<DVec2>, kinds: Vec<ToothCorner>, start, fit: TeethFit{points,rms,worst} }
  struct TeethOptions<'a> { outward: Option<&'a [Option<DVec2>]> }
  fn envelope_of(&[DVec2]) -> Result<Envelope, FitError>; fn off_envelope(&Envelope, DVec2) -> f64
  fn read_teeth(chain: &[DVec2], TeethOptions) -> Result<TeethRead, FitError>
round.rs: fn js_round(f64)->f64; fn round_to(f64, i32)->f64; fn round3(f64)->f64; fn round4(f64)->f64; fn number_text(f64)->String
units.rs: enum DisplayUnits { Mm, In }; const MM_PER_INCH; fn to_display(mm, DisplayUnits)->f64; fn from_display(v, DisplayUnits)->f64; fn length_decimals(i32, DisplayUnits)->i32; fn unit_word(DisplayUnits, power: u8)->String

## integration_notes
- Copying in. Copy src/thread.rs, thread_cut.rs, thread_read.rs, teeth.rs, inserts.rs, round.rs and units.rs into crates/mm-reverse/src, and add `pub mod inserts; pub mod round; pub mod teeth; pub mod thread; pub mod thread_cut; pub mod thread_read; pub mod units;` to lib.rs.
- Copied modules. src/fit2d.rs and src/chain.rs are cmp-identical copies of mm-reverse's own (crate::fit2d::{FitError, solve_linear, fit_circle_2d, Mulberry32}, crate::chain::view_chain). Do not copy them back.
- Possible name clashes. If another package also ported src/core/round.ts or units.ts, merge with it. round.rs adds js_round and number_text beyond round.ts.
- Dependencies. mm-reverse already has glam (serde), serde, thiserror and serde_json. The tests use serde_json, which mm-reverse already has as a normal dependency.
- Wiring a Thread feature in build.rs on SolidCraft:
  1. Run thread_problem(&cut.spec, DisplayUnits::Mm) at resolve time.
  2. Read the picked cylinder into RoundFace { center, axis, radius, convex (outward normal points away from the axis), lo, hi (ends along the axis from center), area }. On failure, use ThreadCutError::not_a_cylinder() or the readRound messages with THREAD_FACE_NEEDS.
  3. Call ThreadSetup::new. If wants_hull(), try deleting the face (body with the face removed and its neighbours extended); pass whether that worked to fill(hull).
  4. If sleeve is Some, extrude that Annulus from..to along the axis and fuse it.
  5. Call plan(|p| body.contains(p)) on the fused body.
  6. Build the band Annulus (a disc when inner is None) extruded from..to.
  7. Sweep the tooth: solidcraft_kernel::helix::sweep_helix(&Plane { origin: plane_origin, x: plane_x, y: plane_y }, &Region2 from helix.profile, axis_origin, axis', pitch', turns as f64), with (axis', pitch') = helix.right_handed_axis_and_pitch(). sweep_helix only turns right-handed, so a left-hand thread uses the reversed axis and negated pitch.
  8. cutter = band minus the swept tooth.
  9. Revolve each ChamferRing.world triangle a full turn about center/axis and cut it from the body.
  10. Cut the cutter from the body.
  11. If keep_to_hull, intersect with the hull and keep the result only if volume(cut) - volume(kept) <= volume(sleeve) + 1e-9.
  12. Show plan.note in the feature's row.
- Turn limits. MAX_THREAD_TURNS is 400 but sweep_helix allows at most 200 turns. Integration must either split the sweep or lower the limit. helix.turn_by_turn records that ScanRuler builds the helix one edge per turn; sweep_helix's B-spline walls may need the same.
- The swept tooth is always narrower than the pitch, which sweep_helix's self-intersection check requires.
- Bolt or nut. Pass a BodyMesh to round_face_is_boss / thread_face_reading as FaceMesh { positions: &m.positions, triangles: &m.triangles, tri_face: &m.tri_face }.
- Scan reading. Collect scan positions and vertex normals as DVec3 within band of the radius and ±half_length along the axis, then call read_thread, is_thread and match_thread. reThreadCommand.ts defaults band to max(THREAD_BAND_MM, THREAD_BAND_SHARE × r).
- Inserts. A Hole feature with an insert uses insert_size and insert_hole for its radius and depth; INSERT_SIZES feeds the UI list.
- read_teeth plugs into a periodic section fit with slice chains in sheet (u, v) coordinates and optional per-point outward normals.
- Model. model.rs needs FeatureKind::Thread holding a ThreadCut plus the face reference, and an insert field on the Hole feature.
- Lints. cargo build and cargo test --no-run show no warnings. clippy reports only neg_cmp_op_on_partial_ord: 13 instances, 3 of them in the verbatim fit2d.rs. These are deliberate !(x > 0) NaN checks; mm-solid allows this lint.

## not_ported
- B-rep construction in kernelThread.ts / kernel.ts. This needs Open CASCADE, and the task leaves it to integration:
  - the band extrude, the screw sweep and the band-minus-tooth cutter;
  - the sleeve fuse and the chamfer revolve-and-cut;
  - deleteFaces for the hull, and the common-with-hull step with its volume guard;
  - the kernel's face reading (readFace/readRound), the cylinder check and uvBounds.
  The plan describes all of these; only the not-a-cylinder message and THREAD_FACE_NEEDS are provided for the face-reading errors.
- reThread.test.ts tests that need the OCCT kernel or the app:
  - 'a thread cut in the kernel', 'a thread on a face drawn another size' and 'a thread's chamfer': volume, containsPoint and isClosedSolid checks on real solids. Only their plan-level parts are adapted.
  - 'the Thread feature' (Builder/resolveFeatures) and 'the Thread box' (reStore UI state).
- reTeeth.test.ts 'reverse.section.fit { periodic }': a headless app command test (scan.open, section.cut, session).
- App-side callers: reThreadCommand.ts (collecting scan points in a box about the axis, the coaxial run of faces), reSectionCommands periodicOut, FeatureEditor/reFeatureInput insert UI. These are app/ files, outside the core scope.
- teeth.ts is not gear teeth. It reads chains of regular teeth (Fresnel rings, racks, knurls); there is no gear code to port.

## deviations
- Input types. Scan points and normals come in as &[DVec3], not Float32Array. The tests round coordinates to f32 to match Float32Array.
- Face-tagged mesh. round_face_is_boss and thread_face_reading take FaceMesh { positions: &[DVec3], triangles: &[[u32;3]], tri_face: &[u32] }, which is mm-reverse's BodyMesh shape, with face: u32. ScanRuler uses positions/indices/faceHash with a numeric hash. A triangle whose vertex index is out of range is skipped; the TS would produce NaN instead.
- Face pick. thread_face_reading takes RoundFacePick { face: Option<u32>, center, radius, dir } in place of {hash, fingerprint}. hash < 0 maps to face: None.
- thread_span takes (lo, hi, ...) instead of a {lo, hi} object.
- THREAD_STANDARDS is an array of the enum with .id() and .label(), not an array of {id, label} objects.
- kernelThread.threadTracked is split into three stages (ThreadSetup::new, then fill(hull), then plan(contains)) so the kernel operations happen in between:
  - deleteFaces for the hull, between new and fill;
  - fusing the sleeve, between fill and plan;
  - containsPoint, through the contains closure passed to plan.
  The order of operations, the conditions and the probe points are unchanged.
- Kernel errors are ThreadCutError, displayed as "Cutting the thread: <message>". That reproduces the prefix ScanRuler's run() wrapper adds.
- The turns check also refuses NaN turns. ScanRuler would hand NaN on to the kernel's screw, which then fails. Here turns is a u32 checked against MAX_THREAD_TURNS.
- JavaScript Math.round (halves up) and String(number) are emulated in round.rs, so names and messages come out character for character.
- units.rs ports only the part of src/core/units.ts that the thread messages need, plus from_display.
- Extra tests beyond the ScanRuler ones:
  - JSON shape: ThreadCut round-trips ScanRuler's project JSON, and TeethRead serialises in the command's shape;
  - round.rs behaviour;
  - insert label lookup;
  - fit2d's 4 tests, which run as part of the verbatim copy.
- Test count: 46 tests run and pass. Of these, 37 are counted as ported:
  - 25 straight ports, two of them adapted:
    - reThread.test.ts pure tests: 13.
    - 'tells a boss from a bore' (counted in the 25, adapted): it uses a hand-tessellated tube instead of an Open CASCADE mesh, with the same three expectations plus thread_face_reading.
    - reCoreUnits 'a thread's problem': 1.
    - reBodyTools insert-hole numbers (counted in the 25, adapted): only the insertHole/insertSize assertion; the drilling is the kernel's.
    - reThreadRead: all 5.
    - reTeeth: 4 of 5.
  - 12 thread_cut tests adapted from the reThread kernel tests. They check the plan level only: refusals, notes, free ends, chamfers, band, turns and handedness, on bodies given as point-inside closures. They do not check B-rep volumes.

## tests_passing
37

## failing_tests
[]

## review.remaining
["By design, still not ported, all requiring Open CASCADE or the app: the B-rep steps (band extrude, screw sweep, cutter boolean, sleeve fuse, chamfer revolve and cut, deleteFaces hull, common-with-hull volume guard), the kernel's face reading (readFace/readRound/uvBounds/getSurfaceArea), the kernel-volume and containsPoint parts of the reThread kernel tests, the Thread feature, the Thread box and the reTeeth 'reverse.section.fit { periodic }' command test.", "Deliberate deviations that differ from ScanRuler only on bad input. round_face_is_boss skips a triangle with an out-of-range index; ScanRuler gets a NaN vote and so reads the face as a bore. along_trend skips unwinding a non-finite angle, where ScanRuler would loop forever on an infinite 'near'. read_thread's sector index clamps a NaN angle to 0, where ScanRuler throws on bySector[NaN]. median and fold-based max/min ignore NaN, where JS Math.max/min would spread it.", "The turns < 1 refusal reproduces kernel.screw's message. ScanRuler would extrude the band before screwing (extrude sorts from and to, so that succeeds), so the error matches. The band-extrude 'no length' failure when from == to exactly is left to the integration's own extrude. Resolve-time thread_feature_problem prevents both.", "clippy now reports 16 neg_cmp_op_on_partial_ord lints (13 before, 3 from my new !(x > 0) checks). They are the same deliberate NaN-rejecting checks as ScanRuler's. cargo build and cargo test --no-run produce no warnings.", 'Integration info for the scan-read command (app side, not ported): reThreadCommand.ts uses THREAD_BAND_MM = 0.6 and THREAD_BAND_SHARE = 0.2 for band = max(0.6, 0.2 × r), and halfLength = (length ?? span) / 2.']

## review.tests_passing
53

## review.tests_failing
0


# sketchtools

## public_api
share_fit: share_pieces(&Sketch)->Vec<&Entity>; share_chain(chain:&[DVec2], pieces:&[&Entity], share:f64, deg:Option<f64>)->ChainShare{shared:Vec<SharedStretch{piece,points,off_mm:OffMm{mean,max,rms}}>, runs:Vec<OpenRun{points,from:Option<EntityId>,to}>, whole, corners}; take_shared(&mut Sketch,&[SharedStretch])->Vec<EntityId>; land_end(&mut Sketch, end:Anchor, on:EntityId, share:f64)->Option<Landing{end,on,held:Held(End|On|Free),at,collinear,constraints}>; let_go(&mut Sketch, &mut [Landing], solve: impl FnMut(&mut Sketch)->Option<HashMap<ConstraintId,f64>>, rounds:usize /*TS default 8*/); past_piece(&Sketch, p:Anchor, on:OnTarget)->Option<f64>.
chain_outline: chain_closed/chain_length/chain_area(&[DVec2]); unjoined(&[Vec<DVec2>])->Vec<JoinedChain{chain,from,gaps,largest}>; join_chain_ends(&[Vec<DVec2>], gap)->Vec<JoinedChain>; bridged_of(&[JoinedChain])->Bridged; runs_within(&[DVec2], SheetBox)->Vec<Vec<DVec2>>; side_of(&SidedChains,&SideRule)->SideVerdicts; runs_kept(&[DVec2], &KeepOptions{sheet_box,side,test})->Vec<Vec<DVec2>>; within_on_sheet(&WithinBox,&SectionFrame)->OnSheet{sheet_box, world:Option<WorldBoxTest>}; chains_span(_world); box_missed(&WithinBox,&[Vec<DVec2>],Option<&SectionFrame>,sheet:&str)->String; pick_chain(&[JoinedChain], ChainPick::{Index(usize),Largest,At(DVec2)})->Option<usize>; chain_at(&[JoinedChain],DVec2)->ChainAt{index:Option<usize>,by:ChainBy,area,enclosing}; filter_chains(&[Vec<DVec2>],&ChainFilter)->Filtered{kept,left:LeftOut}; outline_of(&[DVec2],&OutlineOptions{tolerance,snap_deg,max_points,simplify_deg})->Outline{points,closed,tolerance,max_deviation,snapped,merged}.
section_skin: sided_chains(points,offsets,normals)->SidedChains; side_mask(&[Vec<DVec2>],&[Vec<DVec2>],&SideRule)->Vec<Vec<bool>>; runs_where(&[DVec2], keep, min_points); run_indices(count, closed, keep, min_points); side_runs(&SidedChains,&SideRule,min_points)->Vec<SideRun>.
sketch_near_miss: SheetSketch{id,name,frame,sketch}::of(&model::Feature)->Option; sketch_near_misses(&SheetSketch,&[SheetSketch])->NearMisses{misses,found}; collinear_pieces(&Sketch)->Vec<CollinearJoin{pieces:[EntityId;2],at}>; kinked_joins(&Sketch)->Vec<Kink{pieces,deg,at}>; kink_fix(&SheetSketch,&[Kink])->Option<String>; near_miss_fix(&SheetSketch,&[NearMiss],&[SheetSketch])->String.
symmetry_plane: symmetry_accepted(&SymmetryPlane)->bool; symmetry_plane_ref(&SymmetryPlane)->Option<model::PlaneRef> (Fitted{frame, label:Some(SYMMETRY_LABEL)}).
body_move: rotation_of(DVec3)->[f64;9]; move_matrix(&Placement{pivot,translate,rotate})->Affine=[f64;12]; matrix4_of; move_point; move_dir; moves_nothing(translate,rotate); ring_axes(DVec3)->[DVec3;3]; moved_pivot; merge_meshes(&[MoveMesh])->MoveMesh; moved_mesh(&MoveMesh,&Affine); pivot_of(&[(DVec3,DVec3)])->DVec3; moved_face_mesh(&MoveMesh,&[topo::FaceInfo],offset,Option<Lean>)->Option<MoveMesh>.
pattern_copies: copy_mark_of(&Origins=HashMap<i64,i64>, hash:u32)->Option<CopyMark>; mark_copy(&mut Origins,u32,Option<CopyMark>); is_face_key(i64); copy_motions(&PatternMotion{mode,origin,dir,count,step,grid})->Vec<DAffine3>; trace_copies(before:&[FaceInfo], after:&[FaceInfo], motions:&[DAffine3], diagonal)->TracedCopies{history:HashMap<u32,Vec<u32>>, copies:HashMap<u32,usize>}.
auto_constrain: constrain_plane(&[DVec3], &[ConstrainRef{key,kind:RefKind,name,origin,dir}], &model::ConstrainMode)->Option<ConstrainedPlane>; constrain_cylinder(...)->Option<ConstrainedCylinder>; fit_cylinder_free(&[DVec3])->Option<Cylinder{point,axis,r}>; fit_cylinder_on_axis; refine_cylinder_geometric(points, init, max_iter /*TS 60*/); relation_words(FitShape,Relation,RefKind)->&str; relations_for; plane_patch(points,origin,normal,expand_pct)->PlanePatch; cylinder_patch(points,origin,axis,expand_pct)->CylinderPatch.
Modules from the earlier run: align_to_axes, merge_aligned_lines; candidates_of, read_pieces, line_stats, round_stats, sketch_residual, rounding; symmetric_pairs, mirror_point; dim_shape, dim_place_at, dim_strokes; pieces_in_box; move_anchor, translate_entity, detached, rotate_entity, rotate_sketch, rigid_cluster; tangent_arc, dimension_for, coincident_for, dim_pick_at, rel_pick_at, rectangle_corners, circle_from, arc_about, draw_clicks; snap_sketch_point, snap_hold; placed_image, scale_image, move_image, level_image, turn_image, image_point, image_corners; add_polygon, add_rectangle_centre, pattern_pieces, renumbered, infer_relations, fix_id, fixed_entity_of, fixed_pieces.

## integration_notes
1. Drop these new modules into mm-reverse/src and add `pub mod` lines for them: auto_constrain, axis_align, body_move, box_select, chain_outline, dim_layout, dim_strokes, drag, draw, intent, mirror_relation, nominal, pattern_copies, section_skin, share_fit, sketch_image, sketch_near_miss, sketch_shapes, snap, symmetry_plane.

2. mm-reverse's lib.rs (as of 19:51) does not yet declare topo or pick_query, and model.rs needs both. Add them too.

3. Copied verbatim (no edits) and used:
   - Used directly by my modules: chain (view_chain), corners (PlacedSegment), datums, fit2d (fit_circle_2d, solve_linear, Mulberry32), frame (SectionFrame), linalg (ortho_basis, symmetric_eigen_n), model (current version), simplify (simplify_run), sketch, solver (solve_sketch, used in tests and drag), topo (FaceInfo, FaceFingerprint, face_distance, transform_fingerprint).
   - Copied only so the crate compiles: segment, pick_query and model/tests.rs. regions.rs was not needed.

4. Name clashes with sibling ports:
   - nominal.rs also exists in ports/fillet, with a different body. That one adds `rounding` and depends on crate::fillet_num. Keep one; my intent.rs has its own `rounding`, `Rounding` and `round4`, so it only needs nominal_step and nominal_of.
   - The local Cylinder and cylinder fits in auto_constrain.rs copy fits' fit3d::cylinder::{Cylinder{point, axis, r}, fit_cylinder_on_axis, refine_cylinder_geometric}. Once fits is merged, they can be swapped for that.
   - section_skin.rs ports src/core/section/skin.ts. If another port also brings skin.ts, keep a single copy.

5. Wiring for the app:
   - body_move::MoveMesh maps one-to-one onto build::BodyMesh (positions, normals, triangles, tri_face as `face`).
   - pattern_copies expects a per-body origins map HashMap<i64, i64> (face hash to maker, marks under negative keys). trace_copies needs topo::FaceInfo for the body before and after the pattern, with copy_motions(&PatternMotion) built from the PatternFeature.
   - share_fit::let_go wants a closure that solves and returns the unmet constraint ids with their misses, e.g. from solve_sketch(..).unmet.
   - symmetry_plane_ref needs the mesh-side symmetry search, which is not ported here.

6. Test results: `cargo test --release` passes all 183 tests in the crate: 121 ported ScanRuler tests, 9 tests of my own and 53 from the copied mm-reverse modules. The build is warning-free in release and in the test build.

7. Part of the work, from axis_align through sketch_shapes, came from an earlier run of this same task whose files were already in the directory. I reviewed them against the TS exports and test lists, added 5 missing reSketchShapes tests, and adapted sketch_image to model::SketchImage.

## not_ported
Logic: nothing in the assigned files was left as a stub. Only browser and store parts were left out: symmetryPlane's describePlaneRef label check (features.ts UI) and the store glue.

Tests not ported, and why:
- rePatternCopies.test.ts, all 4 tests: they build bodies on the OCCT kernel.
- rePatternStep.test.ts: also kernel builds and project migration. Only its patternPitch assertions are ported.
- reChainOutline, the 'a sketch fitted with a chain too short to fit' block (6 tests): needs app/reSketchInput fitDraft/acceptOnDraft and the zustand store.
- reImportMove: the gizmo-grips test needs preview.ts moveGrips, out of scope; the Import/Move/project tests need the kernel and the store.
- reDraw: the Line-tool, click-tool, selection and draw-mode tests drive the store (useReverse/useRePrefs). The core ones are ported: tangent arc, dimension picks, geometry of each way, rigid cluster.
- reDimLayout: the chamferLines tests (filletCorner.ts, not in scope) and the store tests.
- reIntent: the 'accept' test needs core/accept.ts.
- reSnapDraft, reSnapEdit, reSnapFace: they test snapEdit.ts, snapFace.ts and the store, not snap.ts. reSnap.test.ts is fully ported.
- reCompare: the describePlaneRef assertion.
- reSketchNearMiss.test.ts does not exist. collinearPieces and kinkedJoins are ported from the asserts in reRound22 and reRound23Sketch. sketchNearMisses and nearMissFix are only exercised through commands in reStepsCommands, so they are covered by a test of my own.

## deviations
- Moving target: mm-reverse's model.rs, frame.rs, topo.rs and pick_query.rs were being rewritten while I worked. I re-copied their 19:44–19:51 versions, plus model/tests.rs because model.rs declares `mod tests;`. I then adapted my modules to them:
  - sketch_image uses model::SketchImage, so width and height are u32.
  - auto_constrain uses model::{Relation, ConstrainMode}.
  - pattern_copies re-exports model::{PatternMode, pattern_pitch(step, count: f64), full_turn}.
  - symmetry_plane_ref returns PlaneRef::Fitted{frame, label}.
  - SheetSketch reads SketchFeature.frame.
  If those files change again, these call sites need rechecking.
- auto_constrain takes points as &[DVec3], where the TS takes a Float32Array. The tests round the generated points through f32 so the inputs match the TS exactly; the eigenvector signs, and the exact `normal == [0,0,1]` checks, then come out the same.
- auto_constrain has its own Cylinder{point, axis, r} plus fit_cylinder_on_axis and refine_cylinder_geometric (from cylinder.ts). These copy what the `fits` port has in fit3d::cylinder.
- chain_outline:
  - side_of keeps each point's verdict under its coordinate bits (point_key). The TS keys it by the point object itself, so two different points with identical coordinates now share one verdict.
  - pick_chain and chain_at return Option<usize> where the TS returns -1.
  - Options are structs: OutlineOptions, ChainFilter, KeepOptions.
  - The TS SheetFrame is replaced by SectionFrame.
- share_fit:
  - share_chain takes (share, deg: Option) instead of an options object.
  - let_go's solve closure is handed &mut Sketch, and `rounds` must be passed explicitly (TS default 8).
  - Landing.held is an enum, and collinear is a bool.
- sketch_near_miss: the sketch feature is read through SheetSketch{id, name, frame, sketch}. kink_fix returns None for an empty list where the TS would throw. near_miss_fix keeps the TS text, including "undefined" when there are no misses.
- body_move: the mesh is MoveMesh{positions, normals: Vec<DVec3>, triangles: Vec<[u32;3]>, face: Option<Vec<u32>>}, shaped like build.rs's BodyMesh rather than the TS f32 KernelMesh. moves_nothing takes (translate, rotate). pivot_of takes (min, max) tuples. moved_face_mesh takes topo::FaceInfo and skips any triangle that names a missing vertex.
- pattern_copies:
  - Motions are glam DAffine3; the turn is built from the same quaternion formula as rigidFromAxisAngle.
  - Faces are topo::FaceInfo, with hashes as u32.
  - Origins is HashMap<i64, i64>; PatternMotion.count is usize.
- section_skin::sided_chains takes flat points, offsets and normals in place of EdgeChains.
- sketch_shapes and intent: mm-reverse's sketch has no splines, so spline handling is left out (in pattern_pieces, and the spline residual in intent).
- Every free-standing helper uses JS-style rounding (a local js_round), so rounded values come out as in the TS.
- 9 extra tests are marked "Not a ScanRuler test". They cover code no portable ScanRuler test reaches: trace_copies and copy marks, sketch_near_misses and near_miss_fix, skin masks, let_go, moved_face_mesh, box_missed, and plane_patch/cylinder_patch.

## tests_passing
121

## failing_tests
[]

## review.remaining
['sketch_shapes.rs: the findRegions assertions in two ScanRuler tests (polygon closes 1 region, pattern grid gives 6) are still dropped. The tests are otherwise intact; the assertions need regions.rs, which pulls in solidcraft_geom and so does not fit this standalone crate. Restore them on integration into mm-reverse, which has find_regions.', 'nominal.rs: nominalText (chip text with fmtLen/DisplayUnits) is not ported; it is UI-only.', "Still not ported, as the porter's report says: tests that need mesh slicing (the sliceSkin arch tests), the OCCT kernel (rePatternCopies, rePatternStep builds), the zustand store (reDraw/reDimLayout/reSketchImage store tests, reChainOutline fitDraft), accept.ts and chamferLines; plus the describePlaneRef UI check.", 'Documented deviations stay as they are: side_of keys verdicts by point coordinates instead of object identity; kink_fix returns None for an empty list where TS throws; Option<usize> replaces -1; spline handling is omitted because the sketch model has no splines.', "In the verbatim-copied sketch.rs (not changed here, to keep it identical to mm-reverse), measure_rule's rem_euclid can return 180 deg for an angle a hair below 0, where TS returns 0. Fix it in mm-reverse if exact parity is wanted. It does not affect any ported caller's results.", 'Last-ulp differences remain: Math.hypot vs sqrt-of-sum, and to_radians/to_degrees vs (x*PI)/180.']

## review.tests_passing
192

## review.tests_failing
0


# sweeps

## public_api
// sweep_route
pub struct SweepRoute { pub points: Vec<DVec3>, pub bend_radius: f64, pub radii: Option<Vec<Option<f64>>> }
pub struct RoutePieces { pub pieces: Vec<LoopPiece>, pub start: DVec3, pub tangent: DVec3, pub bends: Vec<RouteBend{index, radius}> }
pub fn route_pieces(route: &SweepRoute, from: DVec3) -> Result<RoutePieces, RouteError /* String message */>

// loop_piece
pub enum LoopPiece { Line{a,b}, Arc{a,m,b}, Circle{center,normal,radius} }   // .reversed(), .kind_name()
pub struct Profile { pub outer: Vec<LoopPiece>, pub holes: Vec<Vec<LoopPiece>> }
pub fn circle_through(a: DVec3, b: DVec3, c: DVec3) -> Option<(DVec3 /*centre*/, f64 /*radius*/, DVec3 /*axis*/)>
pub fn piece_points(piece: &LoopPiece, step: f64) -> Vec<DVec3>;  pub const PIECE_STEP: f64

// taper
pub struct EdgeLoop { pub points: Vec<DVec2>, pub edges: Vec<usize> }
pub enum TaperLoop { Plain(Vec<DVec2>), Edges(EdgeLoop) }  // From<Vec<DVec2>>, From<EdgeLoop>
pub struct OutlineEdge { pub polyline: Vec<DVec3>, pub ends: [DVec3; 2] }
pub fn face_outline(edges: &[OutlineEdge], origin: DVec3, u: DVec3, v: DVec3) -> Result<Vec<EdgeLoop>, OutlineError>
pub enum InsetBreak { Vanishes{edge}, Folds{edge}, Crosses{edges:[usize;2]} }  // .edges(), .why()
pub fn inset_break(loops: &[TaperLoop], inset: f64) -> Option<InsetBreak>
pub fn inset_holds(loops: &[TaperLoop], inset: f64) -> bool
pub struct TaperReach { pub reach: f64, pub broke: Option<TaperBreak{ broke: InsetBreak, at: Vec<DVec2> }> }
pub fn taper_limit(loops: &[TaperLoop], lean_in: f64, length: f64) -> TaperReach
pub fn taper_reach(loops: &[TaperLoop], lean_in: f64, length: f64) -> f64

// silhouette
pub enum SilhouetteView { Dir(DVec3), Eye(DVec3) }
pub fn silhouette_segments(positions: &[f32], normals: &[f32], indices: &[u32], ranges: &[(usize /*first tri*/, usize /*count*/)], view: SilhouetteView) -> Vec<f32> /* XYZ XYZ per segment */

// loft_fold
pub trait FoldFace { fn hash(&self)->u64; fn is_plane(&self)->bool; fn uv_bounds(&self)->UvBounds{u_min,u_max,v_min,v_max}; fn point_on_surface(&self,u,v)->DVec3; fn is_out(&self,u,v)->bool }
pub struct LoftFoldOptions { pub normal: Option<DVec3>, pub faces: Option<HashSet<u64>>, pub trimmed: bool }
pub struct LoftFold { pub at: DVec3, pub normal: Option<DVec3>, pub folded: Vec<FoldedCut{cut, height: Option<f64>}>, pub of: usize, pub ms: f64 }
pub fn loft_fold<F: FoldFace>(faces: &[F], options: &LoftFoldOptions) -> Option<LoftFold>
pub fn sections_normal(profiles: &[Profile]) -> Option<DVec3>
pub fn fold_words(fold: &LoftFold) -> String
pub struct NamedSection { pub name: String, pub profile: Profile }
pub fn loft_fold_refusal(fold: &LoftFold, sections: &[NamedSection]) -> String
pub const FOLD_CUTS: usize = 40

// tube_trace
pub trait TubeCutter { fn cut(&mut self, origin: DVec3, normal: DVec3) -> SectionCut }  // blanket impl for FnMut(DVec3, DVec3) -> SectionCut
pub struct TraceInput { pub start: DVec3, pub dir: DVec3, pub both: bool, pub step_mm: Option<f64>, pub max_length_mm: Option<f64> }  // TraceInput::new(start, dir)
pub fn trace_tube(cutter: &mut dyn TubeCutter, input: TraceInput) -> TubeTrace
pub struct TubeTrace { stations: Vec<TubeStation{s,at,dir,major,minor,major_dir,size}>, stops: Vec<TubeStop{kind: TubeStopKind, s, at, says}>, route: Option<SweepRoute>, runs: Vec<TubeRun /* Straight{..} | Bend{..} */>, residual: Option<TubeResidual{rms,max,at}>, warnings: Vec<String>, cuts: usize }
pub fn route_miss(centres: &[DVec3], route: &SweepRoute) -> (f64 /*rms*/, f64 /*max*/)
pub const SIZE_JUMP: f64

// outline
pub struct MaskGrid { cols, rows, cell, u0, v0 };  pub struct MaskLoop { points, hole, area }
pub fn mask_loops(mask: &[u8], g: &MaskGrid, min_area: f64) -> Vec<MaskLoop>
pub fn loop_shape(traced: &[DVec2], tolerance: f64, cell: f64) -> LoopShape { points, rounds, arcs, circle, worst, worst_at }

// shadow
pub enum ShadowPiece { Line{from,to}, Arc{from,to,center,radius,sweep}, Circle{center,radius} }
pub struct ShadowLoop { hole, pieces };  pub struct Span { min, max }
pub struct ShadowShape { loops, area, play, tolerance, smooth_deg, margin: Span, cell, widened, extent: Span, frame: SectionFrame }
pub struct ShadowOptions { tolerance, play, smooth_deg: Option<f64>, max_cells: Option<usize> }
pub struct ShadowCut { pub cut: SectionCut, pub shadow: ShadowShape }
pub struct ShadowOutline { pub exact: Vec<ExactLoop{points,hole}>, pub shape: ShadowShape, pub cut: ShadowCut }
pub fn shadow_mesh(positions: &[DVec3], triangles: &[[u32;3]], frame: &SectionFrame, opts: &ShadowOptions) -> ShadowCut
pub fn shadow_outline(positions: &[DVec3], triangles: &[[u32;3]], frame: &SectionFrame, opts: &ShadowOptions) -> ShadowOutline
pub fn shift_shadow(cut: &ShadowCut, to: &SectionFrame) -> Option<ShadowCut>
pub fn describe_shadow(sh: &ShadowShape, units: DisplayUnits /*Mm|In*/) -> ShadowWords { loops, area, band }
pub fn shadow_flat_fits(sh: &ShadowShape) -> Vec<ShadowFlatFit /* Line(LineFit)|Arc(ArcFit)|Circle(CircleFit) */>
pub fn shadow_regions(sh: &ShadowShape) -> Vec<ShadowRegion { outer: ShadowLoop, holes: Vec<ShadowLoop> }>
pub fn outline_points(pieces: &[ShadowPiece], arc_step: f64) -> Vec<DVec2>
pub fn loop_area(pieces: &[ShadowPiece]) -> f64;  pub fn reversed_pieces(pieces: &[ShadowPiece]) -> Vec<ShadowPiece>
pub const SHADOW_TOLERANCE: f64 = 0.1;  pub const SHADOW_SMOOTH_DEG: f64 = 30.0

// shadow_pieces
pub struct ShadowSource<'a> { pub name: Option<&'a str>, pub is_shadow: bool, pub shadow: Option<&'a ShadowShape> }
pub struct ShadowPrism { pub profiles: Vec<Profile>, pub dir: DVec3, pub from: f64, pub to: f64 }
pub fn shadow_primitive(sec: Option<&ShadowSource>) -> Result<ShadowPrism, ShadowError>
pub fn shadow_sketch(sh: &ShadowShape, frame: &SectionFrame, first_id: u32) -> Sketch
pub fn shadow_piece_count(sh: &ShadowShape) -> usize

// stations
pub struct StationPiece { id: EntityId, kind: ShapeKind, from: AnchorAt, to: AnchorAt, fitted: bool };  pub struct StationLoop { pieces: Vec<StationPiece>, closed: bool }
pub fn station_loop(entities: &[Entity], name: &str, any_runs: bool) -> Result<StationLoop, FitError>
pub fn station_faults(template: &Sketch, lp: &StationLoop, made: &[Entity], ids: Option<&[EntityId]>) -> Vec<String>
pub fn like_station(template: &Sketch, lp: &StationLoop, chain: &[DVec2], tolerance: Option<f64>) -> Result<LikeStation{sketch: Sketch, reversed: bool}, FitError>
pub fn gaps_round(chains: &[Vec<DVec2>], from: &[usize], close_gaps: f64) -> GapsRound { bridged: Vec<f64>, open: Vec<f64> }
pub struct RunFacts<'a> { run, joined_chain, joined_from, chains, runs, fit: RunFit{close_gaps: Option<f64>, within, skin, facing: bool} }
pub fn run_facts_words(lp: &StationLoop, f: &RunFacts) -> String
pub fn inset_keeps(lp: &StationLoop, sketch: Option<&Sketch>) -> Vec<EntityId>
pub fn inset_sketch(sketch: &Sketch, lp: &StationLoop, by: f64, keep: &HashSet<EntityId>, datums: &[DatumTrace]) -> Result<Inset{sketch, moved, dropped: Vec<Dropped{id, kind}>}, FitError>
pub fn inset_chains(sketch: &Sketch, ask: &InsetChainsAsk) -> Result<ChainsInset{sketch, moved, dropped, chains: Vec<ChainInset{pieces,by,kept}>, kept, open}, FitError>
pub struct InsetChainsAsk<'a> { by: InsetBy<'a> /*Fixed(f64)|ByDepth(&dyn Fn(usize)->f64)*/, keep: Option<&'a dyn Fn(&StationLoop)->HashSet<EntityId>>, only: Option<&'a HashSet<EntityId>>, keep_points: bool, name: &'a str, datums: &'a [DatumTrace] }  // InsetChainsAsk::new(by, name)
pub fn rule_kind(rule: &Rule) -> &'static str;  pub const STATION_MIN_PIECE_MM: f64 = 0.05;  pub const STATION_FOLD_HELP: &str

## integration_notes
FILES TO COPY into crates/mm-reverse/src: js_fmt.rs, loop_piece.rs, sweep_route.rs, taper.rs, silhouette.rs, loft_fold.rs, tube_trace.rs, outline.rs, shadow.rs plus the directory shadow/ (tests.rs), shadow_pieces.rs, and stations.rs plus the directory stations/ (tests.rs). Add the matching `pub mod` lines to mm-reverse's lib.rs.

DO NOT COPY:
- regions_standin.rs, or the `#[path = "regions_standin.rs"] pub mod regions;` line in lib.rs. mm-reverse's own regions.rs replaces it.
  - stations.rs calls `crate::regions::find_regions(&Sketch)` and reads only `.len()` and `.area`.
  - shadow_pieces' sketch test reads `.holes.len()` and `.area`, and expects the regions largest first. mm-reverse's Region has all of these.
- The copied modules (chain, corners, datums, fit2d, frame, linalg, segment, simplify, sketch, slice, solver). They are byte-identical to mm-reverse's current files. I re-copied them at the end because frame.rs gained serde renames and radial_frame during the work.

NAME CLASHES: none of the new module names (js_fmt, loop_piece, sweep_route, taper, silhouette, loft_fold, tube_trace, outline, shadow, shadow_pieces, stations) clash with mm-reverse's current modules (build, chain, compare, corners, datums, fit2d, frame, linalg, model, pick_query, regions, section, segment, simplify, sketch, slice, solver, topo). If another package also ports src/core/geometry/outline.ts, keep one `outline` module.

CROSS-MODULE DEPENDENCIES:
- stations uses `crate::taper::signed_area`, which is pub(crate).
- stations uses corners::place_segments, segment::kinds_of_chain, chain::view_chain, sketch::measure_rule and datums::trace_of.

WIRING NEEDED IN mm-reverse:
1. build.rs (SolidCraft) has to consume `loop_piece::LoopPiece` and `Profile`. That covers the sweep path from `route_pieces`, extract prisms from `shadow_primitive` (ShadowPrism: profiles stood along `dir` from `from` to `to`), and loft profiles.
2. Implement `loft_fold::FoldFace` for SolidCraft faces (hash, is_plane, uv bounds, point at (u, v), outside-the-trim test), then call `loft_fold` after a smooth loft and refuse with `loft_fold_refusal`.
3. Before a tapered extrude, build `taper::OutlineEdge` values from the face's kernel edges (polyline plus two ends), run `face_outline`, then `taper_limit(&loops, tan(draft), length)`. The message that names the broken edges ("closes the profile … mm …") lives in ScanRuler's build step and still has to be written in build.rs from TaperReach.broke.
4. For tube tracing, pass a closure such as `|o, n| slice_mesh(&pos, &tris, o, n, 0.5)` to `trace_tube`.
5. For a shadow section, call `shadow_mesh(&positions, &triangles, &frame, &opts)`. `shift_shadow` re-places it along its normal; `shadow_sketch` seeds a sketch; `shadow_flat_fits` feeds the 2D exports.
6. Viewport rims come from `silhouette_segments(positions_f32, normals_f32, indices, &[(first, count)], view)` on the round faces' triangle ranges.

DEPENDENCIES: no new ones beyond mm-reverse's own (glam, serde, thiserror). mm-core is only needed by the copied slice.rs test.

## not_ported
- **Open CASCADE parts:**
  - Not ported: loftFold.ts's direct OCCT calls (getSubShapes, surfaceType, uvBounds, pointOnSurface, classifyPointOnFace, release). The FoldFace trait replaces them, and the kernel must implement it.
  - Not ported: LoopPiece's `spline` variant and piecePoints for splines, which need sketchSpline's bsplineSamples. LoopPiece's optional `entity` field is also dropped.
- **Tests that need the kernel, the app's commands or the browser store:**
  - reTaper: the "an extrude with a taper" group (8) and the faceOutline-from-kernel-edges test.
  - reTaperEdges: the blade extrudes and their refusal messages (7). Those messages are composed by the build step, not taper.ts.
  - reLoftChecks: all of it (profiles.ts, resolve, build, commands).
  - reLoftFold: the kernel-built loft tests, adapted as described.
  - reSweepRoute: the 7 sweep-through-kernel tests.
  - reStations: the 2 moulded-wing end-to-end tests.
  - reTubeTrace: the env-gated real-scan test.
  - sectionShadow: the DXF/SVG writers, the STEP builder, the sectionModel keys and parameters, and the store save/load/move tests (6). Their shadow parts are kept as described.
  - reShadowExtract: the builder and store parts, and the "offered to Extract" UI test.
  - reShadowExtractCommands and reOffsetLoft: commands and kernel.

Everything else in the assigned TypeScript files is ported in full.

## deviations
TEST COUNTS: `cargo test --release` runs 92 tests, all passing, in about 1 s, and the crate builds with no warnings. Of the 92, 58 are ported or adapted from ScanRuler tests: sweep_route 4, taper 10, silhouette 3, loft_fold 3, tube_trace 6, stations 14, shadow 16, shadow_pieces 2. 6 are my own additions: one inset_chains test, 2 for outline, and one each for js_fmt, loop_piece and the regions stand-in. The other 28 belong to the copied mm-reverse modules.

ERRORS AND SIGNATURES:
- Where ScanRuler returns a refusal string, the port returns Result with the same text: FitError for stations, RouteError for sweep_route, OutlineError for taper::face_outline, ShadowError for shadow_pieces. face_outline returns its error instead of throwing.
- trace_tube is synchronous. The cutter is a TubeCutter trait, implemented for any FnMut(DVec3, DVec3) -> SectionCut, and reads mm-reverse's SectionCut (chains and normals per chain) rather than flat arrays.
- shadow_mesh and shadow_outline take positions as &[DVec3] and triangles as &[[u32;3]], the same convention as slice_mesh. They return a ShadowCut, which holds a SectionCut (chains closed by repeating the first point, normals per point, f64 rather than Float32) next to the ShadowShape. Triangles that name a missing vertex are left out.
- silhouette_segments takes flat f32 arrays and (first, count) ranges, and returns flat f32.
- InsetBreak is an enum rather than {why, edges}. TaperLoop is an enum over a plain point loop or an EdgeLoop.
- station_loop takes an entity slice plus an `any_runs` flag.
- RunFacts splits `joined` into joined_chain and joined_from. Its `fit` is RunFit, which only records whether within, skin and facing were given; the words never use their values.
- ShadowSource uses an is_shadow flag. shadow_primitive returns ShadowPrism, the prism member of ScanRuler's Primitive union.
- shadow_flat_fits returns fit2d's LineFit, ArcFit and CircleFit with sigma, used_points and form_error all 0.
- describe_shadow uses a local DisplayUnits enum (Mm, In) and copies units.ts's formatting rules.
- loft_fold reads faces through the FoldFace trait instead of Open CASCADE.
- like_station returns an error for an empty profile where ScanRuler would crash on stretch[0].

TEST ADAPTATIONS:
- The stationLoop spline refusal uses a circle, because mm-reverse's sketch has no spline. It gets the same "lines and arcs" refusal.
- reStationsOpenRun needs joinChainEnds from chainOutline.ts, which is outside this package. I ported it as a test-only helper.
- sectionShadow's DXF/SVG test keeps only its shadow_flat_fits assertions (4 lines; 4 quarter-turn arcs with radius between 0.5 and 0.75) and adds checks on describe_shadow. Its STEP-prism test keeps only the regions it stands on: 3 loops giving 2 regions with 1 hole, and extent 0 to 4.
- reLoftFold has two adapted tests:
  - The refusal-words test builds the LoftFold by hand from the hump profiles. It checks the full message for an assumed fold: crossing first at z 9.1, folded over z 9.1–17.4 and 35.4–36.1, with the hump sections 3 mm apart named as the close pair.
  - The fold check runs on synthetic FoldFace surfaces (a skin that sinks through itself), including the face filter by hash and the trimmed option.
- reShadowExtract's block is checked as ShadowPrism data rather than a kernel body. Its findRegions assertions run on the regions stand-in.
- The shadow tests weld the triangle soup by exact position instead of buildMeshGraph. The shadow does not depend on winding.
- vitest's toBeCloseTo(x, d) is ported as |diff| < 0.5·10^-d.
- One test helper carries #[allow(clippy::approx_constant)] because ScanRuler's test uses 6.28 for a full turn.

## tests_passing
92

## failing_tests
[]

## review.remaining
["Precision is different by design. mm-reverse's SectionCut, the shadow's lifted cut and tube_trace work in f64, while ScanRuler uses Float32Array points and normals. The silhouette keeps f32 in and out, as the TS does.", 'shadow_outline drops triangles that name a missing vertex before rasterising. If every triangle is invalid it returns blank(tol/2, false), where ScanRuler would rasterise nothing and return blank(cell, widened) with a possibly widened cell.', 'Degenerate inputs where ScanRuler would produce NaN or throw are handled gracefully in Rust: an empty sampled loop in lift_loops, a SegmentField with no segments (box set to 0 instead of ±Infinity), a hole with no pieces in shadow_regions, missing entity ids in stations/holds, and NaN coordinates in the raster (Rust f64::min/max ignore NaN where Math.min/max propagate it). None of these is reachable from valid geometry.', 'describe_shadow: `grouped` writes a rounded -0 area as "0", where JavaScript\'s (-0).toLocaleString gives "-0". This is a trivial edge case.', "Adaptations the porter already listed: the LoopPiece spline variant and its `entity` field are not ported; loft_fold reads faces through the FoldFace trait instead of OCCT; the reStations spline refusal is tested with a circle; reLoftFold's message test builds the fold by hand.", "Tests that need the kernel, the app's commands or the store are still not ported, as the porter listed: the reTaper extrudes, the reTaperEdges blade builds, reLoftChecks, the reSweepRoute sweeps, the moulded-wing end-to-end tests, the env-gated real tube scan, the sectionShadow DXF/SVG/STEP/store tests, the reShadowExtract builder/UI test, reShadowExtractCommands and reOffsetLoft.", 'Code style only: `cargo clippy --all-targets` reports pre-existing style warnings. Most are deliberate NaN-faithful `!(a > b)` comparisons, plus needless_range_loop and cloned_ref_to_slice_refs in tests. The cargo build and tests have no warnings.', 'Cargo.toml still lists serde_json and rayon, which nothing uses. They are harmless and not part of the integration.']

## review.tests_passing
95

## review.tests_failing
0


# SPEC SUMMARY
I wrote the spec to `/private/tmp/claude-501/-Users-banderies-Documents-GitHub-polymix/89dcd24a-cc95-4ee2-b482-c14523a3a492/scratchpad/ports/docs/ui-layout-spec.md` (1,663 lines, about 150 KB, labels and tooltips copied exactly from the source). Nothing was run; everything comes from reading the source files.

Layout summary:
1. **Screen:** a top bar with the `Reverse engineer` tab, a left panel 336 px wide (282 px on narrow screens, the only scrolling area), the 3D stage, and a 30 px status strip at the bottom. While a sketch is open, a 2D sheet covers the stage behind a 58 % stage-colour veil, and the 3D camera follows the sheet.
2. **Stage overlays:**
   * view bar in the bottom-left corner;
   * map legend down the right edge;
   * hint chip and build chip at top centre (the build chip only after 300 ms);
   * playback caption at top left;
   * context chip bar and note card floating 12 px from the click;
   * Fit options window fixed at 352/96, 400 px wide.
3. **Panel order:** Model (scan slot, kernel status) → Parameters (folded) → Create feature (hidden while a box is open) → the one open editor box → Features list → Bodies list → Notes for the agent → `Export body STEP` / `Compare in Deviation`.
4. **Create feature:** a 4-column grid of 26 icon-over-word keys. Section 3.3 gives each key's enabled condition, tooltip and what it opens.
5. **Context chips:** a card of readings (body line with Hide / Hide others / Make component, a title, measurement rows), then action chips that depend on what was clicked: region, element, coordinate plane, sketch line, face, edge, or several picks.
6. **Features list:** one row per feature (kind icon, name, ⚠ / `fit` reading, edit, eye or suppress key, bin), suppressed and rolled-back row styles, a draggable 3 px accent rollback bar with a `To end` key, Play with its playback bar, and the right-click Suppress menu.
7. **Bodies list:** XY/XZ/YZ plane rows first, then components and bodies (colour swatch, double-click to rename, volume, eye), multi-select like a file browser, drag-and-drop into components, a selection bar, and a context menu. Hovering a row lights the body blue in 3D.
8. **Every editor box** has the same frame: a 3 px accent left border, title, Name field when editing, its own fields, a preview line, the live Deviation-map row (`Range ±` defaults to 0.1 mm), then Create and Cancel. Number fields take expressions and offer a `→ round` value chip.
9. **Sketch box:** Finish at the top and bottom, plane offset, Level, Picture row, the Fit all / Infer corners / Close row, then the tool keys with L/R/C/S/F shortcuts and flyouts listing the ways to draw each shape. Below them: a 10-key row of Dimension plus the relations, edit keys, Pattern box, inferred chips, the dimension and relation lists, a `Needs input` / `Solved` status window, and Extrude / Revolve keys.
10. **Extrude, Revolve and all 24 feature kinds** (Shell through Patch network) are listed control by control with their defaults, ⓘ card text, hints, and Create-button wording.
11. **View bar:** Scan, Bodies and Elements keys with hover menus (Transparent, Overlaps, Tangent edges, Curvature), then Deviation, then Section. While a sketch is open it becomes `No part` / `Section` / `Full part`.
12. **3D drawing:** body opacity states, edges 1.1 px `#0c0f13`, hover white, picked `#3cb371`, ghosts (cuts in red `#e0564f`), grips at fixed screen size (arrows 44 px, Move gizmo rings 90 px in X/Y/Z colours) with a typed number field, sketches drawn through the part in `#3d8fd6`, plane and axis hints in amber `#ffb020`, note pins.
13. **2D sheet:** the full dark and light palette, line weights, snap marks (✕ ✚ ▽ ◇ – □), relation glyphs, dimensions drawn like a technical drawing, and window/crossing box selection.
14. **Interaction tables** for idle, Extrude/Revolve, body-feature boxes, the sketch tools, Patch network, note placing and playback, plus the full key map. Escape never closes a sketch; Enter or a middle click confirms.
15. Section 9 has short suggestions for building each piece in egui.


# SPEC SUMMARY
Both files are written and contain the same 460 rows.

- /private/tmp/claude-501/-Users-banderies-Documents-GitHub-polymix/89dcd24a-cc95-4ee2-b482-c14523a3a492/scratchpad/ports/docs/parity-inventory.md
- /private/tmp/claude-501/-Users-banderies-Documents-GitHub-polymix/89dcd24a-cc95-4ee2-b482-c14523a3a492/scratchpad/ports/docs/parity-inventory.json (an array of rows with the fields id, package, area, capability, sources, status, polymix_file, missing)

**Rows by status:** done 86, partial 95, missing 279.

**Packages:**

| ID | Package | Rows | Done | Partial | Missing | Depends on |
|---|---|---:|---:|---:|---:|---|
| BUILD | Build engine, picks & topology | 20 | 11 | 4 | 5 | — |
| HIST | Feature list, history, rollback & playback | 22 | 5 | 8 | 9 | BUILD, PARAM |
| PROJ | Project saving, import & export | 14 | 6 | 5 | 3 | BUILD, PARAM, LIST, NOTE |
| PLANE | Elements, sketch planes & datums | 16 | 3 | 7 | 6 | BUILD, PICK |
| SKT | Sketch tools | 48 | 14 | 10 | 24 | PLANE, REL, BUILD |
| REL | Relations & dimensions | 26 | 11 | 9 | 6 | SKT, PARAM |
| FIT | Fit all chains & reading the slice | 26 | 8 | 4 | 14 | PLANE, SKT, REL |
| EXT | Extrude & revolve | 28 | 9 | 6 | 13 | BUILD, PLANE, PICK, PARAM |
| RND | Fillet, chamfer, shell & delete face | 21 | 6 | 5 | 10 | BUILD, PICK |
| HOLE | Holes & threads | 19 | 0 | 0 | 19 | BUILD, PLANE, PICK, PARAM |
| DRAFT | Draft, move face & snap to scan | 13 | 0 | 0 | 13 | BUILD, PICK, PLANE, SKT |
| LOFT | Loft, sweep, pattern & move body | 20 | 1 | 7 | 12 | EXT, FIT, PICK, PARAM |
| BOOL | Mirror, combine, cut by plane & offset body | 12 | 4 | 2 | 6 | BUILD, PLANE, SURF |
| SURF | Sheets, split, thicken, trim, fit surface & extract | 23 | 0 | 2 | 21 | PLANE, PICK, CMP, BUILD |
| NET | Patch network | 21 | 0 | 0 | 21 | SURF, VIEW, CMP |
| PARAM | Parameters & expressions | 12 | 0 | 3 | 9 | BUILD |
| LIST | Body list, components & measurements | 15 | 0 | 6 | 9 | BUILD, PICK |
| PICK | Viewport picking & context chips | 22 | 0 | 4 | 18 | BUILD, LIST |
| VIEW | View bar, maps & section view | 17 | 6 | 4 | 7 | BUILD, CMP |
| CMP | Deviation, compare & checks | 9 | 2 | 1 | 6 | BUILD |
| NOTE | Notes & review | 12 | 0 | 2 | 10 | PICK, AGENT, PROJ, CMP |
| AGENT | Agent commands | 44 | 0 | 6 | 38 | BUILD, HIST, PARAM, FIT, CMP, EXT, LIST |

- **Suggested order:** the Markdown file proposes PARAM and PICK first, then PLANE, then the feature packages in parallel, with AGENT and NOTE last.
- **Partly ported packages:** six feature types build but have no editor: pattern, move body, import, extract, cut by plane and delete face. Parameters, notes, components, body names and colours, and the sketch picture exist only as data in `model.rs`. The rollback bar works in the build but has no UI.
- **Agent commands:** the Reverse workspace has no agent tools at all yet. Only test script steps exist.
- **No element source:** Polymix has no 3D Measure equivalent, so nothing can create the measured planes, cylinders and other elements that features refer to.
- **Ports staged in the scratchpad:** where a row's code already exists as a standalone port in `scratchpad/ports/<crate>` but is not in `mm-reverse` yet, its "missing" text says so. Those crates are expressions, fillet, fits, network, sketchtools, spline, surface, sweeps and threads.

All ScanRuler and Polymix file paths in the rows were checked and exist. The generator script is /private/tmp/claude-501/-Users-banderies-Documents-GitHub-polymix/89dcd24a-cc95-4ee2-b482-c14523a3a492/scratchpad/scripts/gen_parity.py; rerun it with `python3 -I`, giving the docs directory as its argument.
