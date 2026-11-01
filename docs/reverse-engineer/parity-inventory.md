# Reverse engineer: ScanRuler → Polymix feature parity inventory

Every user-facing capability of ScanRuler's Reverse engineer workspace (`scanruler/plugins/reverse`), with where Polymix's port stands. Polymix's port is `crates/mm-reverse/src` (scan-reading core, model, build on SolidCraft) and `crates/polymix/src/reverse` (workspace UI). Read on 2026-10-31 from the source; ScanRuler was not run.

- **done** — the capability works in Polymix as ScanRuler has it (differences noted in *missing*).
- **partial** — some of it is there: data model only, build only (no editor), editor only (no build), or a narrower version.
- **missing** — nothing usable in Polymix yet. A stub file is still named where one exists.

ScanRuler sources are relative to `scanruler/plugins/reverse/`; Polymix files to the repository root. "Staged" names a standalone port already in the session scratchpad (`scratchpad/ports/<crate>`) that is not yet in `mm-reverse`.

**460 rows** — done 86, partial 95, missing 279.

## Work packages

| Package | Title | Rows | Done | Partial | Missing | Depends on |
|---|---|---:|---:|---:|---:|---|
| [BUILD](#build) | Build engine, picks & topology | 20 | 11 | 4 | 5 | — |
| [HIST](#hist) | Feature list, history, rollback & playback | 22 | 5 | 8 | 9 | BUILD, PARAM |
| [PROJ](#proj) | Project saving, import & export | 14 | 6 | 5 | 3 | BUILD, PARAM, LIST, NOTE |
| [PLANE](#plane) | Elements, sketch planes & datums | 16 | 3 | 7 | 6 | BUILD, PICK |
| [SKT](#skt) | Sketch tools | 48 | 14 | 10 | 24 | PLANE, REL, BUILD |
| [REL](#rel) | Relations & dimensions | 26 | 11 | 9 | 6 | SKT, PARAM |
| [FIT](#fit) | Fit all chains & reading the slice | 26 | 8 | 4 | 14 | PLANE, SKT, REL |
| [EXT](#ext) | Extrude & revolve | 28 | 9 | 6 | 13 | BUILD, PLANE, PICK, PARAM |
| [RND](#rnd) | Fillet, chamfer, shell & delete face | 21 | 6 | 5 | 10 | BUILD, PICK |
| [HOLE](#hole) | Holes & threads | 19 | 0 | 0 | 19 | BUILD, PLANE, PICK, PARAM |
| [DRAFT](#draft) | Draft, move face & snap to scan | 13 | 0 | 0 | 13 | BUILD, PICK, PLANE, SKT |
| [LOFT](#loft) | Loft, sweep, pattern & move body | 20 | 1 | 7 | 12 | EXT, FIT, PICK, PARAM |
| [BOOL](#bool) | Mirror, combine, cut by plane & offset body | 12 | 4 | 2 | 6 | BUILD, PLANE, SURF |
| [SURF](#surf) | Sheets, split, thicken, trim, fit surface & extract | 23 | 0 | 2 | 21 | PLANE, PICK, CMP, BUILD |
| [NET](#net) | Patch network | 21 | 0 | 0 | 21 | SURF, VIEW, CMP |
| [PARAM](#param) | Parameters & expressions | 12 | 0 | 3 | 9 | BUILD |
| [LIST](#list) | Body list, components & measurements | 15 | 0 | 6 | 9 | BUILD, PICK |
| [PICK](#pick) | Viewport picking & context chips | 22 | 0 | 4 | 18 | BUILD, LIST |
| [VIEW](#view) | View bar, maps & section view | 17 | 6 | 4 | 7 | BUILD, CMP |
| [CMP](#cmp) | Deviation, compare & checks | 9 | 2 | 1 | 6 | BUILD |
| [NOTE](#note) | Notes & review | 12 | 0 | 2 | 10 | PICK, AGENT, PROJ, CMP |
| [AGENT](#agent) | Agent commands | 44 | 0 | 6 | 38 | BUILD, HIST, PARAM, FIT, CMP, EXT, LIST |

### Suggested order

1. **PARAM** and **PICK** first: almost every other package needs expressions in its fields and object-first face/edge/corner picks with chips.
2. **PLANE** next (an element source standing in for 3D Measure): Extract, Hole along an element, Revolve about an element, Trim with elements, Mirror's symmetry plane and sketches on sections all wait on it.
3. Then the feature packages in parallel — **EXT**, **RND**, **HOLE**, **DRAFT**, **LOFT**, **BOOL** — each mostly self-contained on the build engine, with its staged port to integrate.
4. **SKT**, **REL**, **FIT** can run alongside (they touch the sketcher and `mm-reverse` sketch/solver only).
5. **SURF** before **NET** (Thicken, surface bodies, live map against sheets).
6. **HIST**, **LIST**, **VIEW**, **CMP**, **PROJ** are UI/infrastructure packages that can be taken at any time.
7. **AGENT** and **NOTE** last: the commands are thin wrappers over the panel's code paths, so they follow the packages they expose.

## BUILD

**Build engine, picks & topology** — 20 rows (done 11, partial 4, missing 5).

How the feature list becomes bodies: join/cut/new semantics, islands, fingerprints and where-queries, incremental and parallel builds, build stop, failure diagnosis.

- **Depends on:** nothing beyond the current build engine

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| BUILD-01 | Feature model | Feature list stored by reference as JSON: ordered features, ids never reused, names from the kind's count, every kind's fields | core/features.ts; app/projectSection.ts | done | crates/mm-reverse/src/model.rs | — |
| BUILD-02 | Build | Features played in order on the CAD kernel; a failed feature is reported by name and the rest still build | core/build.ts; core/resolve.ts | done | crates/mm-reverse/src/build/mod.rs | — |
| BUILD-03 | Body semantics | Join merges with every solid it touches (bridging several under the oldest id); cut goes through every solid met and fails when none; New keeps its own body | core/build.ts | done | crates/mm-reverse/src/build/mod.rs (place, settle) | — |
| BUILD-04 | Body semantics | One body is one solid: pieces a step leaves become bodies of their own; split words say which piece kept the id and where each lies | core/build.ts; core/splitWords.ts | partial | crates/mm-reverse/src/model.rs (island_id); crates/mm-reverse/src/build/mod.rs | Islands are made; the narration of which piece kept the id / where each lies (splitWords) is not |
| BUILD-05 | Body semantics | Legacy target: a join/cut saved with a named body works on that body alone; null lets it go | app/reSchemas.ts (targetIn); core/build.ts | partial | crates/mm-reverse/src/build/mod.rs (place_on_target) | Built; no UI to set or clear the target |
| BUILD-06 | Topology | Face/edge fingerprints (kind, centroid/mid, dir, area/length, radius, maker) found again on every rebuild; a lost pick fails its feature by name | core/topo.ts | done | crates/mm-reverse/src/topo.rs; crates/mm-reverse/src/build/mod.rs | — |
| BUILD-07 | Topology | Every face carries the feature that first made it, through each step's kernel history | core/kernelOps.ts; core/kernelMath.ts | done | crates/mm-reverse/src/build/mod.rs (origins_after) | — |
| BUILD-08 | Topology | Where-queries as picks (box, near, through, kind, between normals, madeBy, length/area/radius/diameter, faces, loop, coaxial, exclude, optional, sharp, sides, nested) stored and re-run at every rebuild | core/where.ts; app/reSchemas.ts (whereIn) | done | crates/mm-reverse/src/pick_query.rs | — |
| BUILD-09 | Topology | Tangent chain of picked edges (a round runs on along tangent edges) | core/topo.ts | done | crates/mm-reverse/src/topo.rs (tangent_chain) | — |
| BUILD-10 | Topology | Seams and same-surface split lines neither drawn nor pickable | core/edgeLines.ts; core/topo.ts | done | crates/mm-reverse/src/topo.rs (split_edges); crates/mm-reverse/src/build/mod.rs (body_mesh) | — |
| BUILD-11 | Profiles | Sketch regions lifted into profiles; picked regions remembered by their pieces and middle and found again after the sketch is edited | core/profiles.ts; core/regions.ts | done | crates/mm-reverse/src/regions.rs; crates/mm-reverse/src/build/mod.rs (regions_by_pick) | — |
| BUILD-12 | Preview | A feature box builds the part as it stood before the feature (picks taken there) and the part with the feature (preview) | state/reFeatureSlice.ts; core/build.ts | done | crates/mm-reverse/src/build/mod.rs (build_before); crates/polymix/src/reverse/mod.rs (request_build) | — |
| BUILD-13 | Performance | Incremental build: a step's outcome reused when what it read is unchanged; only what a change reaches is rebuilt; rollback bar moved back is a look-up | core/build.ts; core/buildReads.ts; core/buildSteps.ts | missing | — | Every change rebuilds the whole list (only unchanged meshes are shared) |
| BUILD-14 | Performance | Parallel build: independent body groups built on helper kernels at the same time and taken over | core/buildGroups.ts; core/reParallel.ts; core/reHelper.ts; core/buildTransfer.ts | missing | — | Single-threaded build per request; rayon is available |
| BUILD-15 | Performance | Latest-wins build queue: changes during a build wait; only the newest is built; overtaken builds/previews left unplayed | core/reQueue.ts; core/reClient.ts | partial | crates/polymix/src/reverse/mod.rs (request_build, poll_build) | Stale results are discarded by generation, but each change spawns a new thread and old builds run to completion |
| BUILD-16 | Robustness | Build stop: a build that never finishes is stopped, the kernel restarted and the feature in play suppressed | app/reBuildStop.ts; core/reEvents.ts | missing | — | No cancellation of a running build |
| BUILD-17 | Robustness | Boolean volume guard: a join never loses volume, a cut never gains, a combine keeps the volume it should | core/booleanGuard.ts | partial | crates/mm-reverse/src/build/mod.rs (cut skips bodies only touched) | Only the touch check on cuts; no volume sanity guard on join/combine |
| BUILD-18 | Robustness | Failure diagnosis: why a round, shell, draft or offset was refused (round larger than a face, wall thinner than the shell) and the trapped edges named | core/diagnose.ts; core/trapEdges.ts | missing | — | Only the kernel's message is shown |
| BUILD-19 | Robustness | Cut-elsewhere warning: a cut extent that also nicks material far from where it was drawn | core/cutElsewhere.ts | missing | — | — |
| BUILD-20 | Meshing | Bodies meshed for display with every face drawn (bare faces re-meshed) | core/kernelMesh.ts | done | crates/mm-reverse/src/build/mod.rs (body_mesh) | SolidCraft display mesh; no bare-face check |

## HIST

**Feature list, history, rollback & playback** — 22 rows (done 5, partial 8, missing 9).

The feature list as the user works it: rows and their key fields, suppress menus, rollback bar, insert/reorder, undo, playback for the camera.

- **Depends on:** BUILD, PARAM

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| HIST-01 | Rows | Feature row: name, kind, status (ok / failed with message / warnings / notes), the bodies it joined or cut | ui/ReversePanel.tsx; ui/featureReading.ts | partial | crates/polymix/src/reverse/mod.rs (feature_list) | Error/warnings/touched only in a hover tooltip; no op word or status icons |
| HIST-02 | Rows | Row key-number field (10 mm, 360°, R 2 mm, 1.35 mm wall): typed over, the features rebuild without opening the box | ui/ReversePanel.tsx; state/reDrafts.ts; ui/ExprInput.tsx | missing | — | — |
| HIST-03 | Rows | Open a feature's box from its row (double click) — reopened box shows the body before it with picks lit | ui/ReversePanel.tsx | done | crates/polymix/src/reverse/mod.rs (edit_feature) | Single click opens it |
| HIST-04 | Rows | Delete a row; refused while another feature is made on it | ui/FeatureMenu.tsx; app/reFeatureCommands.ts (remove) | partial | crates/polymix/src/reverse/mod.rs; crates/mm-reverse/src/model.rs (Part::remove) | Polymix deletes the feature and every dependent instead of refusing |
| HIST-05 | Rows | Sketch row eye; a sketch is hidden once a feature is made from it and drawn while a feature on it is open | state/reFeatureHelpers.ts; ui/ReversePanel.tsx | partial | crates/polymix/src/reverse/mod.rs (toggle_visible) | Eye works; auto-hide on use and show-while-open not ported |
| HIST-06 | Suppress | Suppress key on every non-sketch row; suppressed rows greyed and struck through, picks and numbers kept | ui/FeatureMenu.tsx | done | crates/polymix/src/reverse/mod.rs (toggle_suppress) | — |
| HIST-07 | Suppress | Row menu (right click): Suppress / Build again | ui/FeatureMenu.tsx | missing | — | — |
| HIST-08 | Suppress | Row menu: Suppress every <kind> (n) / Build every <kind> again, and every round and chamfer at once | ui/FeatureMenu.tsx; app/reFeatureCommands.ts (suppress) | missing | — | — |
| HIST-09 | Rollback | Rollback bar under the list: drag up to set features aside (greyed, not built), To end, undo step, waits while a box is open, saved with the project | ui/ReversePanel.tsx; state/reStore.ts | partial | crates/mm-reverse/src/model.rs (Part::rolled_to); crates/mm-reverse/src/build/mod.rs (build_upto) | Honoured by the build and saved; no bar in the UI |
| HIST-10 | Rollback | New features go in at the bar (a fillet before the shell); a feature that lost what it stood on says so | state/reStore.ts | partial | crates/mm-reverse/src/model.rs (add_named) | No UI insertion point; Part::add appends at the end |
| HIST-11 | Order | Reorder features (drag rows / feature.move) keeping ids, names and picks | app/reFeatureCommands.ts (move) | partial | crates/mm-reverse/src/model.rs (moved_list) | List reorder function exists; no UI or command |
| HIST-12 | Order | Rename a feature (names unique across sketches and features) | app/reSchemas.ts (nameIn); state/reFeatureHelpers.ts | partial | crates/mm-reverse/src/model.rs | Names stored and unique; no rename UI |
| HIST-13 | Undo | Undo/redo of the feature list (Ctrl+Z, Ctrl+Shift+Z / Ctrl+Y), each change one step | state/reFeatureHelpers.ts; state/reStore.ts | done | crates/polymix/src/reverse/mod.rs (checkpoint, undo, redo) | — |
| HIST-14 | Undo | Sketch's own undo/redo while it is open | state/reSketchSlice.ts | done | crates/polymix/src/reverse/sketcher.rs (undo, redo) | — |
| HIST-15 | Confirm | Enter or middle click confirms (finish sketch / create feature); Escape backs out one step; Finish on an empty sketch makes nothing | app/keyboard.ts | done | crates/polymix/src/reverse/view.rs (confirm, keys) | — |
| HIST-16 | Confirm | Tab walks from one number field to the next in a box | ui/ExtrudeEditor.tsx; app/keyboard.ts | missing | — | — |
| HIST-17 | Clear | Remove every feature in one undo step, parameters kept | app/reFeatureCommands.ts (clear) | partial | crates/polymix/src/reverse/mod.rs (New part) | New part resets the whole part, parameters and notes included |
| HIST-18 | Fitted marks | Rows mark numbers fitted to the scan this session; Refit fits them again after alignment or an upstream change | app/reFitMarks.ts; app/reFit.ts | missing | — | — |
| HIST-19 | Playback | Play in the list heading: bar to the top (scan alone or bare stage), down one feature at a time, each held 0.5–5 s once built, row lit, sketches drawn at their step | app/playback.ts; ui/Playback.tsx | missing | — | — |
| HIST-20 | Playback | Playback caption in the viewport's top left (Step 4 / 6, Extrude 2, Join · 10 mm), switchable | ui/Playback.tsx (PlaybackCaption) | missing | — | — |
| HIST-21 | Playback | Skip sketches: only geometry-changing steps, the caption counting those | ui/Playback.tsx; state/rePrefs.ts | missing | — | — |
| HIST-22 | Playback | Playback keys: Space hold/resume, arrows step, Escape stops; bar restored; not an undo step; agent commands wait | app/keyboard.ts; app/playback.ts | missing | — | — |

## PROJ

**Project saving, import & export** — 14 rows (done 6, partial 5, missing 3).

Part file, autosave, STEP import/export (names, colours, mended loops), STL/3MF, blank stage, preferences.

- **Depends on:** BUILD, PARAM, LIST, NOTE

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| PROJ-01 | Saving | Features saved by reference (sketches, picks as fingerprints, planes/axes by the element they name); bodies rebuilt on load | app/projectSection.ts | done | crates/mm-reverse/src/model.rs (Part::to_json/from_json); crates/polymix/src/reverse/mod.rs (save/open .rpart) | — |
| PROJ-02 | Saving | One project archive with scan, model and notes (and the other workspaces) | app/projectSection.ts; app/notesSection.ts | partial | crates/polymix/src/reverse/mod.rs | Part saved alone as .rpart; the scan stays in the Mesh workspace |
| PROJ-03 | Saving | Version check: a part from an older or newer model refused with a sentence | app/projectSection.ts (validation) | done | crates/mm-reverse/src/model.rs (PART_VERSION) | — |
| PROJ-04 | Saving | Autosave and recovery of the part | (session persistence) | done | crates/polymix/src/reverse/mod.rs (autosave, emergency_save) | — |
| PROJ-05 | Saving | Start over / new project; opening a part is undoable | app/projectSection.ts | done | crates/polymix/src/reverse/mod.rs (New part, open_part_dialog) | — |
| PROJ-06 | Blank stage | Model with no scan: blank stage sized for a hand-size part; a scan loaded later joins the model | README; app/useReverseWorkspace.ts | done | crates/polymix/src/reverse/mod.rs (set_scan) | — |
| PROJ-07 | Export | Export body STEP: every visible body as a solid, each a part under its name | app/reBodyCommands.ts (step); core/kernel.ts | done | crates/mm-reverse/src/build/mod.rs (export_step); crates/polymix/src/reverse/mod.rs (export_step_dialog) | — |
| PROJ-08 | Export | STEP part colours from the body's colour | core/kernelStep.ts; app/reBodyCommands.ts (step) | missing | — | export_step takes names only |
| PROJ-09 | Export | STEP mending: loop orientation for Parasolid readers, round-curve ends a hair off their corner | core/kernelStep.ts; core/stepLoops.ts | missing | — | Needs a check of SolidCraft's writer in Onshape/SolidWorks |
| PROJ-10 | Export | Export STL (binary, chord) or 3MF (each body an object), visible solids only | app/reBodyCommands.ts (stl) | partial | crates/polymix/src/reverse/mod.rs (send_to_mesh) | Bodies go to the Mesh workspace, which exports STL; no direct STL/3MF or chord option |
| PROJ-11 | Import | Import STEP: a body per solid in the file's frame, named after its part (count suffix) | core/kernelStep.ts; app/reFeatureCommands.ts (import.step) | partial | crates/mm-reverse/src/build/import.rs; crates/polymix/src/reverse/editors/import.rs (stub) | Built from Part::step_files; no file dialog/editor |
| PROJ-12 | Import | Import box: file shown as a ghost before Import; row named after the file; Another file… replaces it | ui/FeatureEditor.tsx (import-*) | missing | crates/polymix/src/reverse/editors/import.rs (stub) | — |
| PROJ-13 | Import | STEP files travel in the project and are read once a session | app/projectSection.ts (STEP_MEMBER) | partial | crates/mm-reverse/src/model.rs (step_files) | Stored in the part JSON; re-read on every build |
| PROJ-14 | Preferences | Preferences remembered: fit options, live map and its range, draw modes, polygon sides, grid snap, relation glyphs, playback hold/caption/skip | state/rePrefs.ts | partial | crates/polymix/src/reverse/mod.rs (save: fit options, deviation range) | Only fit options and the deviation range are stored |

## PLANE

**Elements, sketch planes & datums** — 16 rows (done 3, partial 7, missing 6).

What a sketch or a feature can stand on: coordinate planes, sections, measured elements (Polymix has no 3D Measure, so a substitute element source is needed), radial planes, body faces, fitted planes, coordinate-plane rows, alignment warning.

- **Depends on:** BUILD, PICK
- **Staged ports to integrate:** ports/fits (fit3d: plane/cylinder/cone/sphere/torus fits, region grow)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| PLANE-01 | Sketch on | Sketch on a coordinate plane (XY, XZ, YZ) at zero | ui/PlaneSelect.tsx; app/reSchemas.ts (worldPlane) | done | crates/polymix/src/reverse/mod.rs (idle_box, chosen_plane) | — |
| PLANE-02 | Sketch on | Sketch on a coordinate plane through the scan's centre | ui/PlaneSelect.tsx | done | crates/polymix/src/reverse/mod.rs (chosen_plane) | — |
| PLANE-03 | Sketch on | Sketch on a section (cut through the part), by id or name | ui/PlaneSelect.tsx; app/reSchemas.ts | partial | crates/mm-reverse/src/model.rs (PlaneRef::Section); crates/mm-reverse/src/build/mod.rs (resolve_plane_frame) | Resolved if present; Polymix has no way to make a section |
| PLANE-04 | Sketch on | Sketch on a measured plane element | ui/PlaneSelect.tsx | partial | crates/mm-reverse/src/model.rs (PlaneRef::Plane) | Model/resolve only; no element source |
| PLANE-05 | Sketch on | Plane through an axis at an angle (radial: half section of a turned part); angle editable later | app/reSchemas.ts (radialPlane) | partial | crates/mm-reverse/src/frame.rs (radial_frame); crates/mm-reverse/src/model.rs (PlaneRef::Radial) | No UI |
| PLANE-06 | Sketch on | Plane fitted through a clicked face of the scan (to sketch on) | app/reSchemas.ts; core/snapFace.ts | partial | crates/mm-reverse/src/model.rs (PlaneRef::Fitted) | Model only; plane fit staged in ports/fits |
| PLANE-07 | Sketch on | Sketch on a flat face of a body (Sketch here) that follows the face by fingerprint when an earlier feature moves it | ui/ContextChips.tsx; core/build.ts (placeSketch) | partial | crates/mm-reverse/src/build/sketch.rs; crates/mm-reverse/src/model.rs (PlaneRef::BodyFace) | Build follows the face; no chip/UI to start one |
| PLANE-08 | Sketch on | Face's outline projected into a sketch on that face (fixed closed profile; project: false leaves it out) | core/project.ts; app/reSchemas.ts (sketchIn.project) | missing | — | — |
| PLANE-09 | Sketch on | Plane offset under the sketch's name, as a number or an expression the sketch follows | ui/SketchEditor.tsx (sketch-offset) | partial | crates/polymix/src/reverse/view.rs (sketch_box Offset) | Number only; expressions wait on PARAM |
| PLANE-10 | Sketch on | Plane's `up` axis (u = up × normal) | app/reSchemas.ts (upIn) | partial | crates/mm-reverse/src/build/mod.rs (up_u) | No UI |
| PLANE-11 | Sheet | Sheet zero is the part's origin projected; datum dot; coordinate planes' lines across the sheet as dimension targets | core/datums.ts | done | crates/mm-reverse/src/datums.rs; crates/polymix/src/reverse/sketcher.rs (paint) | — |
| PLANE-12 | Sketch on | Hover a Sketch-on entry: that plane drawn on the part with its name | ui/PlaneSelect.tsx | missing | — | — |
| PLANE-13 | Coordinate planes | XY/XZ/YZ planes at the origin drawn faint, listed at the head of the body list with eyes (hidden over a scan); a click offers a sketch | viewer/coordinatePlanes.ts; ui/ReversePanel.tsx (plane-row) | missing | — | — |
| PLANE-14 | Elements | Element source: planes, cylinders, cones, spheres, tori, lines, circles, points fitted on the scan (ScanRuler's 3D Measure) | (host src/core/fit); app/reSchemas.ts | missing | crates/mm-reverse/src/model.rs (Element, data only) | No element creation in Polymix; fits staged in ports/fits |
| PLANE-15 | Elements | Measured elements and sections drawn on the part as datums | viewer/reScene.ts | missing | — | — |
| PLANE-16 | Alignment | First sketch on a part still in the scanner's frame warns and offers Align part; alignment carries sketches and notes | ui/SketchEditor.tsx (sketch-align) | missing | — | Polymix has no 3-2-1 alignment |

## SKT

**Sketch tools** — 48 rows (done 14, partial 10, missing 24).

Drawing tools and their modes, snapping, dragging, box selection, polygon/spline/shapes, sketch fillet/chamfer, Project, sketch Pattern, Picture, Align.

- **Depends on:** PLANE, REL, BUILD
- **Staged ports to integrate:** ports/sketchtools (snap, drag, draw, box_select, pattern_copies, sketch_image, body_move); ports/spline (sketch_spline, flat_spline, spline_solve); ports/fillet (fillet_corner, round_corner)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| SKT-01 | Select | Select tool: click a piece; Shift/Ctrl adds | ui/SketchEditor.tsx; state/reSketchSlice.ts | done | crates/polymix/src/reverse/sketcher.rs (click_select) | — |
| SKT-02 | Drag | Drag an end, a centre or a whole piece; constraints hold; dragged point snaps; an end dropped on another end joins it | core/drag.ts; app/reSketchInput.ts | done | crates/polymix/src/reverse/sketcher.rs (select_pointer) | — |
| SKT-03 | Drag | A line dragged by its body moves square to itself; Ctrl moves it freely | core/drag.ts | partial | crates/polymix/src/reverse/sketcher.rs | Whole piece follows the cursor in any direction |
| SKT-04 | Drag | A fitted piece dragged by hand lets go of its slice points | core/drag.ts | done | crates/polymix/src/reverse/sketcher.rs (select_pointer) | — |
| SKT-05 | Box select | Selection box on the sheet: right = window (inside), left = crossing (touching), lit while dragged, Shift/Ctrl adds | core/boxSelect.ts | missing | — | Staged: ports/sketchtools box_select.rs |
| SKT-06 | Line | Line tool (L) draws a chain; ends on Escape, Enter, a second click on the last end, or landing on another end | ui/SketchEditor.tsx; state/reSketchSlice.ts | done | crates/polymix/src/reverse/sketcher.rs (draw_click) | — |
| SKT-07 | Line | Press on the chain's last end and drag: an arc tangent to the piece before it | core/draw.ts | missing | — | Staged: ports/sketchtools draw.rs |
| SKT-08 | Line | Line within 3° of level/upright made so; inferred relation shown as a chip under the tools, one click takes it off | core/sketch.ts; ui/SketchEditor.tsx (sketch-inferred-chip) | partial | crates/polymix/src/reverse/sketcher.rs; crates/mm-reverse/src/sketch.rs (axis_relation_of) | Inferred; removable from the relation list, no chip row |
| SKT-09 | Rectangle | Rectangle: two corners (R) | ui/SketchEditor.tsx (DRAW_MODES corners) | done | crates/polymix/src/reverse/sketcher.rs | — |
| SKT-10 | Rectangle | Rectangle: centre and a corner (construction diagonals and centre point) | ui/SketchEditor.tsx (center) | missing | — | — |
| SKT-11 | Rectangle | Rectangle: 3-point (one side, then the far side at any angle) | ui/SketchEditor.tsx (threePoint) | partial | crates/mm-reverse/src/sketch.rs (add_turned_rectangle) | Model helper only, no tool mode |
| SKT-12 | Circle | Circle: centre and rim (C) | ui/SketchEditor.tsx (center) | done | crates/polymix/src/reverse/sketcher.rs | — |
| SKT-13 | Circle | Circle: 2-point (ends of a diameter) | ui/SketchEditor.tsx (twoPoint) | missing | — | — |
| SKT-14 | Circle | Circle: 3 points on the rim | ui/SketchEditor.tsx (threePoint) | missing | — | — |
| SKT-15 | Arc | Arc: 3-point (start, end, through) | ui/SketchEditor.tsx (threePoint) | done | crates/polymix/src/reverse/sketcher.rs | — |
| SKT-16 | Arc | Arc: centre, start, end (shorter way round) | ui/SketchEditor.tsx (center) | missing | — | — |
| SKT-17 | Draw modes | Mode picker: rest on a key and its ways open as pictures; the chosen one becomes the key's picture and is remembered | ui/SketchEditor.tsx (sketch-draw-modes); state/rePrefs.ts | missing | — | — |
| SKT-18 | Polygon | Polygon inscribed: centre and a corner; sides field (3–64); corners on a construction circle; sides Equal; fully defined | ui/SketchEditor.tsx; core/loops.ts | partial | crates/mm-reverse/src/sketch.rs (add_polygon) | Model helper only, no tool |
| SKT-19 | Polygon | Polygon circumscribed: centre and middle of a side (sized across flats like a hex nut) | ui/SketchEditor.tsx (circumscribed) | partial | crates/mm-reverse/src/sketch.rs (add_polygon circumscribed) | Model helper only, no tool |
| SKT-20 | Spline | Spline tool (S): smooth curve through clicked points; end open (re-click/double/Enter/Esc), close on the first point; fit points snap, drag, take relations | core/sketchSpline.ts; state/reSketchDraft.ts | missing | — | Sketch model has no spline piece; staged ports/spline |
| SKT-21 | Spline | Spline held tangent where joined end-to-end to a line, arc or spline | core/sketchSpline.ts; core/solver.ts | missing | — | Staged ports/spline spline_solve.rs |
| SKT-22 | Spline | Fit spline to slice: slice points along it become soft data | ui/SketchEditor.tsx (sketch-spline-fit) | missing | — | — |
| SKT-23 | Spline | Spline handed to the kernel as one B-spline edge (in the STEP) | core/sketchSpline.ts; core/kernel.ts | missing | — | — |
| SKT-24 | Shapes | Rounded rectangle (radius; degenerates to slot or circle) with dimensioned rounds | app/reSchemas.ts (shapeIn rectangle.radius); app/reSketchInput.ts | missing | — | — |
| SKT-25 | Shapes | Slot (two straights, half-rounds tangent and equal) by ends or centre/length/angle | app/reSchemas.ts (slot) | missing | — | — |
| SKT-26 | Shapes | Polyline shape (open or closed) | app/reSchemas.ts (polyline) | missing | — | — |
| SKT-27 | Shapes | Ellipse as one closed spline (points count; deviation from a true ellipse reported) | app/reSchemas.ts (ellipse) | missing | — | — |
| SKT-28 | Shapes | Shapes placed in world coordinates (`world: true`), fixed, construction, centre+size, angle | app/reSchemas.ts (shapeCommon) | missing | — | — |
| SKT-29 | Snap | Every click snaps: to an end (held coincident), a piece (held on), a slice point, the origin, a fitted piece | core/snap.ts | done | crates/polymix/src/reverse/sketcher.rs (snap, hold) | — |
| SKT-30 | Snap | Snap markers: cross on an end, diamond on the slice, square on the grid | core/snap.ts; app/useSketchSceneSync.ts | partial | crates/polymix/src/reverse/sketcher.rs (paint) | Generic marker only |
| SKT-31 | Snap | Snap to grid (off by default); millimetre grid anchored on the origin, spacing following the zoom | core/snap.ts; app/useSketchSceneSync.ts | missing | — | No sheet grid drawn |
| SKT-32 | Snap | Ctrl places a point exactly at the cursor | core/snap.ts | done | crates/polymix/src/reverse/sketcher.rs (exact) | — |
| SKT-33 | Hover | What a tool would take is lit under the cursor before the click (piece, end, plane line, both lines of a corner) | app/useSketchSceneSync.ts; state/reSketchDraft.ts | partial | crates/polymix/src/reverse/sketcher.rs (pick_at, hover) | Piece/end hover only; no plane-line or corner-pair lighting |
| SKT-34 | Corner tools | Sketch Fillet (F): click a corner or two lines; arc with its radius as a dimension; next starts with the last size | core/filletCorner.ts; state/reSketchDraft.ts | missing | — | Staged ports/fillet fillet_corner.rs |
| SKT-35 | Corner tools | Sketch Chamfer: corner or two lines; even or uneven distances as dimensions | core/filletCorner.ts | missing | — | Staged ports/fillet |
| SKT-36 | Corner tools | Sharp corner: remove a fillet arc or chamfer line and carry its sides to the corner | core/corners.ts (sharpen) | done | crates/polymix/src/reverse/sketcher.rs (sharpen); crates/mm-reverse/src/corners.rs (sharpen_corner) | — |
| SKT-37 | Corner tools | Round a fitted corner where the scan is round (arc tangent to any two pieces; sketch.edit rounds) | core/roundCorner.ts | missing | — | Staged ports/fillet round_corner.rs |
| SKT-38 | Project | Project tool: a body edge or another sketch's piece clicked under the sheet comes in as a fixed magenta piece; straight edges and circles flat or square to the plane; slanted circles refused | core/project.ts; app/reSketchInput.ts | missing | crates/mm-reverse/src/sketch.rs (Entity.projected flag) | — |
| SKT-39 | Project | Projections follow their source on every build (edge by fingerprint, piece by id) and the sketch is solved again; a lost source is reported on the row | core/project.ts; core/build.ts | missing | — | — |
| SKT-40 | Pattern | Sketch Pattern: selected pieces in a row, grid or round a centre, copies previewed until applied, their relations copied | core/patternCopies.ts; ui/SketchEditor.tsx (sketch-pattern-*) | missing | — | Staged ports/sketchtools pattern_copies.rs |
| SKT-41 | Picture | Picture: photo/drawing under the sketch, scaled by two points, moved by a point, levelled by two points, faded, saved with the sketch, removed | core/sketchImage.ts; ui/SketchEditor.tsx (sketch-image-*) | partial | crates/mm-reverse/src/model.rs (SketchImage) | Data only; staged ports/sketchtools sketch_image.rs |
| SKT-42 | Align | Align: turn the sheet so the selected line lies along X (frame turns underneath) | core/drag.ts; ui/SketchEditor.tsx (sketch-align) | missing | — | — |
| SKT-43 | Edit keys | Fix (lock) and its removable row; Construction toggle; Delete (Del/Backspace) | ui/SketchEditor.tsx (sketch-fix, sketch-construction, sketch-delete) | done | crates/polymix/src/reverse/sketcher.rs (toggle_fix, toggle_construction, delete_selection) | — |
| SKT-44 | Sheet | The sheet has no edge: a click beyond the card lands on the plane | app/useSketchSceneSync.ts | done | crates/polymix/src/reverse/view.rs (sheet_point) | — |
| SKT-45 | Status | Closed regions filled faintly; status Needs input / Solved with the solver's words in a tooltip; loose pieces counted and coloured | ui/SketchEditor.tsx (sketch-status, sketch-loose) | partial | crates/polymix/src/reverse/sketcher.rs (status, paint) | Status and loose colour shown; regions not filled, no loose count |
| SKT-46 | Finish | Extrude/Revolve offered as soon as a region closes (finish and open the feature on it) | ui/SketchEditor.tsx (sketch-extrude, sketch-revolve) | done | crates/polymix/src/reverse/view.rs (sketch_box) | — |
| SKT-47 | Keys | Tool letters D, L, R, C, S, F; Escape backs out one step; Delete removes | app/keyboard.ts | partial | crates/polymix/src/reverse/view.rs (keys); crates/polymix/src/reverse/sketcher.rs (Tool::key) | D, L, R, C (and A) only; S and F wait for their tools |
| SKT-48 | Overlay | Finished sketches drawn on the part in their plane; regions as click targets | viewer/sketchOverlay.ts | done | crates/polymix/src/reverse/view.rs (paint_overlays) | — |

## REL

**Relations & dimensions** — 26 rows (done 11, partial 9, missing 6).

Every relation kind and how it is applied, every dimension kind, the dimension's placement/drawing/editing, nominal chips, solver status.

- **Depends on:** SKT, PARAM
- **Staged ports to integrate:** ports/sketchtools (dim_layout, nominal, mirror_relation)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| REL-01 | Relations | Coincident | core/solver.ts; ui/SketchEditor.tsx | done | crates/polymix/src/reverse/sketcher.rs (apply_relation); crates/mm-reverse/src/solver.rs | — |
| REL-02 | Relations | Horizontal | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs | — |
| REL-03 | Relations | Vertical | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs | — |
| REL-04 | Relations | Parallel | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs | — |
| REL-05 | Relations | Perpendicular | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs | — |
| REL-06 | Relations | Tangent (line–arc/circle, arc–arc; spline end tangency) | core/solver.ts | partial | crates/polymix/src/reverse/sketcher.rs | No spline tangency (no splines) |
| REL-07 | Relations | Concentric | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs | — |
| REL-08 | Relations | Equal (two lines one length; two round pieces one radius) | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs | — |
| REL-09 | Relations | Symmetric: two pieces then the mirror line (construction line or a coordinate plane's line); lines end for end, circles by centres, arcs by both; prompt says which pick is next | core/mirrorRelation.ts; state/reSketchDraft.ts | partial | crates/mm-reverse/src/solver.rs (Rule::Symmetric) | Solver has it; not offered in the UI; staged ports/sketchtools mirror_relation.rs |
| REL-10 | Relations | Point on a line, a round piece, a spline or a coordinate plane's line; bounded (T-junction) option | core/solver.ts; app/reSchemas.ts (pointOn) | partial | crates/polymix/src/reverse/sketcher.rs ('on piece') | Datum-line target and bounded not offered |
| REL-11 | Relations | Collinear: a line along another line or a coordinate plane's line | core/solver.ts | partial | crates/mm-reverse/src/solver.rs (Rule::Collinear) | Not offered in the UI |
| REL-12 | Relations | Two points level or upright | core/solver.ts | done | crates/polymix/src/reverse/sketcher.rs ('horizontal points') | Polymix uses a construction line |
| REL-13 | Applying | Relation tool armed then the next pieces clicked, or applied to the selection made beforehand | ui/SketchEditor.tsx (RELATIONS); state/reSketchDraft.ts | partial | crates/polymix/src/reverse/view.rs (sketch_box) | Selection-first only |
| REL-14 | Glyphs | Relation glyphs on the sheet, off until the eye key asks; hovering a glyph lights it and its pieces | ui/SketchEditor.tsx (sketch-show-relations); app/useSketchSceneSync.ts | missing | — | Relations only listed in the panel |
| REL-15 | Dimension | Dimension tool (D): line length, arc radius, circle diameter, two pieces or ends → distance between parallels, angle, point–line, point–point, centres, to a coordinate plane's line | core/draw.ts; state/reSketchDraft.ts | done | crates/polymix/src/reverse/sketcher.rs (dimension_click) | — |
| REL-16 | Dimension | New dimension's number follows the cursor; click puts it down and opens it for typing; Enter uses the layout; Escape takes it back | state/reSketchDraft.ts (dimMade); app/keyboard.ts | missing | crates/polymix/src/reverse/sketcher.rs | Polymix places it and opens the field at once |
| REL-17 | Dimension | While a length/radius/diameter follows the cursor, a click on another piece, end or plane line turns it into a distance or angle | state/reSketchDraft.ts | missing | — | — |
| REL-18 | Dimension | Dimension drawn as a drawing carries it: extension lines, arrows (turned in when short), number in a gap; radius leader; diameter through the centre; angle arc | core/dimLayout.ts | missing | crates/polymix/src/reverse/sketcher.rs (paint_dimensions) | Number label only; staged ports/sketchtools dim_layout.rs |
| REL-19 | Dimension | Drag the number to where it reads best; it keeps that place against what it measures | core/dimLayout.ts; core/sketch.ts (place) | partial | crates/mm-reverse/src/sketch.rs (DimPlace) | Stored, not draggable |
| REL-20 | Dimension | Click a number on the sheet to type; Ø12 over an R or R6 over a Ø turns it over | viewer/sketchDimText.ts; ui/SketchEditor.tsx | done | crates/polymix/src/reverse/view.rs (dimension_editor); crates/polymix/src/reverse/sketcher.rs (set_dimension) | — |
| REL-21 | Dimension | Row chip →Ø / →R turns a round dimension over with the size held | ui/SketchEditor.tsx (sketch-dimension-row) | missing | — | — |
| REL-22 | Dimension | Nominal chip beside every measured number (halves under 10 mm, whole above, radii to the half, angles to 5°), one click takes it | core/nominal.ts | partial | crates/polymix/src/reverse/sketcher.rs (round_value) | New dimensions auto-round when within 0.5 %; no chip; staged ports/sketchtools nominal.rs |
| REL-23 | Rows | Relation and dimension rows in the box, selectable, value editable, removable; inferred rows marked | ui/SketchEditor.tsx (sketch-relation-row, sketch-dimension-row) | done | crates/polymix/src/reverse/view.rs (sketch_box relations list) | — |
| REL-24 | Solver | Every relation and dimension met exactly; unmet reported (not split); redundant relations counted | core/solver.ts | partial | crates/mm-reverse/src/solver.rs (SolveReport); crates/polymix/src/reverse/view.rs | Unmet shown in red; redundant count not displayed |
| REL-25 | Solver | Hover a dimension or relation row/number lights the pieces it holds | app/useSketchSceneSync.ts | partial | crates/polymix/src/reverse/view.rs (hover Pick::Constraint) | Row hover only |
| REL-26 | Parameters | A dimension driven by a parameter wears ƒ on the sheet; dimension expressions | core/parameters.ts; viewer/sketchDimText.ts | missing | crates/mm-reverse/src/sketch.rs (Constraint.expr) | Expr stored, never evaluated (PARAM) |

## FIT

**Fit all chains & reading the slice** — 26 rows (done 8, partial 4, missing 14).

Turning the scan's slice into a sketch: fit options, axis alignment, corners, gap closing, levels and draft offset, share fit, skins, arcs/spline modes, stations, candidates and accept, inset, teeth, shadow outlines.

- **Depends on:** PLANE, SKT, REL
- **Staged ports to integrate:** ports/sketchtools (axis_align, intent, share_fit, chain_outline, section_skin, sketch_near_miss); ports/sweeps (taper, stations, shadow); ports/threads (teeth); ports/spline

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| FIT-01 | Slice | The plane's slice through the scan drawn as edge chains on the sheet | app/useReverseWorkspace.ts; app/useSketchSceneSync.ts | done | crates/mm-reverse/src/section.rs (section_chains); crates/polymix/src/reverse/sketcher.rs (paint) | — |
| FIT-02 | Slice | Slice under a sketch on a part modelled without a scan (the bodies cut as a scan is) | README §2; viewer/reScene.ts | missing | crates/polymix/src/reverse/mod.rs (slice) | Slice reads the scan only |
| FIT-03 | Fit all chains | Fit all chains: lines and arcs, corners joined, closed slice → closed profile; pieces keep their slice points and the solver holds them | core/segment.ts; core/corners.ts; state/reSketchDraft.ts | done | crates/mm-reverse/src/section.rs (fit_all_chains); crates/mm-reverse/src/segment.rs; crates/mm-reverse/src/corners.rs | — |
| FIT-04 | Fit all chains | Robust cut: a bowing wall is one line, a fillet stays, a corner blob is cut out, a small step's riser stays a line | core/segment.ts | done | crates/mm-reverse/src/segment.rs | — |
| FIT-05 | Fit options | Detail slider (coarse … fine) | ui/FitOptionsModal.tsx (fit-detail); core/fitOptions.ts | done | crates/polymix/src/reverse/view.rs (Fit options…) | — |
| FIT-06 | Fit options | Fit tolerance typed or from the scan's noise | ui/FitOptionsModal.tsx (sketch-tolerance) | done | crates/polymix/src/reverse/view.rs | — |
| FIT-07 | Fit options | Join the corners toggle | ui/FitOptionsModal.tsx (fit-join) | done | crates/polymix/src/reverse/view.rs | — |
| FIT-08 | Fit options | Align lines to the sheet's axes (within N°, and 45° if asked), merging one wall's pieces; inferred relation one click takes off | core/axisAlign.ts; ui/FitOptionsModal.tsx (fit-align*) | missing | — | Staged ports/sketchtools axis_align.rs |
| FIT-09 | Fit options | A change refits the chains already fitted as one undo step; options remembered; Reset | ui/FitOptionsModal.tsx (fit-options-reset) | partial | crates/polymix/src/reverse/sketcher.rs (refit); crates/polymix/src/reverse/mod.rs | Refit and remembering done; no reset, not one undo step |
| FIT-10 | Corners | Infer corners: join ends within 2 mm, tangency where a line runs into a fillet | core/corners.ts | done | crates/polymix/src/reverse/sketcher.rs (infer_corners); crates/mm-reverse/src/corners.rs | — |
| FIT-11 | Corners | Close: two lines carried to their intersection, anything else bridged straight | core/corners.ts | done | crates/polymix/src/reverse/sketcher.rs (close); crates/mm-reverse/src/corners.rs (close_chain) | — |
| FIT-12 | Chains | Close gaps between chain ends before fitting (holes in the scan); open chains report their ends, gap and a closeGaps to try | core/chainOutline.ts; app/reSchemas.ts (fit.closeGaps) | missing | — | Staged ports/sketchtools chain_outline.rs |
| FIT-13 | Chains | Which chains: all / largest / the one at a point / by index; only within a box (one wall) | app/reSchemas.ts (fit.chains, fit.within) | partial | crates/mm-reverse/src/section.rs (fit_all_chains skip) | All (with skips) only |
| FIT-14 | Levels | Level: cut the scan this far off the sketch plane and fit that level into the same sketch (stacked parts) | ui/SketchEditor.tsx (sketch-level); app/reSchemas.ts (fit.level) | missing | — | — |
| FIT-15 | Levels | draftDeg with a level: chains moved out by level·tan(draft) so a tapered extrude lands on the wall | core/taper.ts; app/reSchemas.ts (fit.draftDeg) | missing | — | Staged ports/sweeps taper.rs |
| FIT-16 | Share | Share fit: slice points near pieces already drawn/fitted/projected go to them; the rest fitted as open runs landing on them | core/shareFit.ts | missing | — | Staged ports/sketchtools share_fit.rs |
| FIT-17 | Skins | Skin (outer / inner of a thin wall) and facing-direction filters | src/core/section/skin.ts; app/reSchemas.ts (skinIn, facingIn) | missing | — | Staged ports/sketchtools section_skin.rs |
| FIT-18 | Modes | Arcs mode: a chain as N arcs end to end | core/segment.ts (arcs); app/reSchemas.ts (fit.arcs) | partial | crates/mm-reverse/src/segment.rs (arcs_of_chain) | Core only, not reachable |
| FIT-19 | Modes | Spline mode: a chain as one spline through N fit points settled onto the slice | app/reSchemas.ts (fit.spline) | missing | — | Needs SKT spline |
| FIT-20 | Stations | Fit like a station sketch: the run cut into the same pieces in the same order (loft stations), drawn pieces copied, relations copied | core/stations.ts | missing | — | Staged ports/sweeps stations.rs |
| FIT-21 | Candidates | Candidates: each piece's residual, uncertainty and nominal; near-relations (level, upright, parallel, square, in line, round angle, equal, concentric, tangent, on datum, mirrored) with off and sigmas | core/intent.ts | missing | — | Staged ports/sketchtools intent.rs |
| FIT-22 | Candidates | Accept candidates within N sigmas (kinds, maxMoveMm), reporting conflict / implied / moved | core/accept.ts | missing | — | — |
| FIT-23 | Inset | Inset: every closed chain moved inward by a wall (a shell's inner station), kept pieces, refusals by chain | app/reSchemas.ts (sketchIn.inset); app/reSketchInput.ts | partial | crates/mm-reverse/src/model.rs (SketchInset) | Data only |
| FIT-24 | Teeth | A chain of regular teeth (Fresnel rings, rack, knurl) read for its design | core/teeth.ts | missing | — | Staged ports/threads teeth.rs |
| FIT-25 | Near miss | Pieces nearly coinciding with another sketch's on a parallel plane flagged | core/sketchNearMiss.ts | missing | — | Staged ports/sketchtools sketch_near_miss.rs |
| FIT-26 | Shadow | Shadow section's outline (lines, arcs, circles as read) as a sketch's start | core/shadowPieces.ts; core/silhouette.ts | missing | — | Staged ports/sweeps shadow.rs |

## EXT

**Extrude & revolve** — 28 rows (done 9, partial 6, missing 13).

Every extent, taper, oblique direction, face extrude, height clicks, grips and frame-by-frame ghost, revolve axes, coil.

- **Depends on:** BUILD, PLANE, PICK, PARAM
- **Staged ports to integrate:** ports/sweeps (taper)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| EXT-01 | Profiles | Regions shown on the part as faces, clicked to toggle; one region preselected; an unpicked inner loop is a hole | state/reSolidSlice.ts; viewer/sketchOverlay.ts | done | crates/polymix/src/reverse/editors/extrude.rs (pick_region); crates/polymix/src/reverse/editors/mod.rs | — |
| EXT-02 | Extent | Distance: From and To either side of the plane | ui/ExtrudeEditor.tsx; core/buildExtrude.ts | done | crates/polymix/src/reverse/editors/extrude.rs; crates/mm-reverse/src/build/extrude.rs | — |
| EXT-03 | Extent | Symmetric total | ui/ExtrudeEditor.tsx | done | crates/polymix/src/reverse/editors/extrude.rs; crates/mm-reverse/src/build/extrude.rs | — |
| EXT-04 | Extent | Up to a plane (from an offset) | ui/ExtrudeEditor.tsx; core/buildExtrude.ts | partial | crates/mm-reverse/src/build/extrude.rs | Built for planes parallel to the sketch; no editor; slanted planes refused |
| EXT-05 | Extent | Between two planes | ui/ExtrudeEditor.tsx | partial | crates/mm-reverse/src/build/extrude.rs | As above |
| EXT-06 | Extent | To a face of a body (flat as its plane; curved face as itself; refused where it does not lie under the whole profile) | ui/ExtrudeEditor.tsx (extrude-to-face); core/buildExtrude.ts | missing | crates/mm-reverse/src/build/extrude.rs | Fails by name |
| EXT-07 | Extent | To a body's surface (solid skin or sheet), reverse | app/reSchemas.ts (toBody); ui/ExtrudeEditor.tsx (extrude-to-body) | missing | — | — |
| EXT-08 | Extent | To next: until the profile first meets the body it joins or cuts (cut through the first wall); whole profile must land | ui/ExtrudeEditor.tsx (extrude-until) | missing | — | — |
| EXT-09 | Extent | To last / through all (a cut a hair beyond) | ui/ExtrudeEditor.tsx | done | crates/polymix/src/reverse/editors/extrude.rs; crates/mm-reverse/src/build/extrude.rs (ToLast) | — |
| EXT-10 | Extent | To scan: end at the scan face capping the region, offset fitted (near), stored as a distance | app/reSchemas.ts (toScan); app/reFit.ts | missing | — | Not in the model |
| EXT-11 | Direction | Reverse direction (flip the extrude; next/last look the other way) | ui/ExtrudeEditor.tsx (extrude-flip) | done | crates/polymix/src/reverse/editors/extrude.rs | — |
| EXT-12 | Taper | Taper typed: positive leans out, negative in; outline stays the section at the sketch plane; refused if it closes the profile, saying how far it gets | core/taper.ts; core/buildExtrude.ts | partial | crates/polymix/src/reverse/editors/extrude.rs; crates/mm-reverse/src/build/extrude.rs | Only from the sketch plane outward; no closing-distance message |
| EXT-13 | Taper | Taper on an extent reaching both ways or ending on a plane/face leans all the way to it | core/buildExtrude.ts | missing | crates/mm-reverse/src/build/extrude.rs | Refused (taper has to start at the plane) |
| EXT-14 | Direction | Oblique direction: along a measured line/axis, a coordinate axis or a typed vector; ends parallel to the sketch; taper about it | ui/DirectionField.tsx; core/buildExtrude.ts | missing | crates/mm-reverse/src/build/extrude.rs | Refused when slanted |
| EXT-15 | Height | With a Distance, a click on the scan/body/corner puts To (in front) or From (behind) at that height; Symmetric reaches it with a half | state/reSolidSlice.ts | partial | crates/polymix/src/reverse/editors/extrude.rs (pick Height) | Distance only |
| EXT-16 | Face extrude | Extrude a flat face of a body as it stands (outline made a sketch; out adds, in cuts; Cancel takes the sketch back) | ui/ExtrudeEditor.tsx (extrude-face-profile); state/reSolidSlice.ts | missing | crates/mm-reverse/src/build/extrude.rs | Refused by name |
| EXT-17 | Result | Join / Cut / New body (no body chosen) | ui/featureFields.tsx; core/build.ts | done | crates/polymix/src/reverse/editors/mod.rs (op_line) | — |
| EXT-18 | Ghost | Ghost of the body; a cut's ghost red | viewer/reScene.ts | done | crates/polymix/src/reverse/view.rs (BODY_PREVIEW, BODY_CUT) | — |
| EXT-19 | Grips | Arrow on each face to drag the distance, one on the far edge for the taper, ring for a revolve's angle; constant size on screen; a field on the grip takes typing | core/preview.ts; state/reGripSlice.ts | missing | — | — |
| EXT-20 | Grips | Ghost follows a dragged grip frame by frame (last ghost stretched; one request out at a time) | core/preview.ts; core/reQueue.ts | missing | — | — |
| EXT-21 | Readouts | Box reads the ghost's volume; From/To/Total/Angle offer round-value chips | ui/ExtrudeEditor.tsx; core/nominal.ts | missing | — | Volume only in the body list |
| EXT-22 | Planes | Plane lists show the plane under the cursor on the part | ui/PlaneSelect.tsx | missing | — | — |
| EXT-23 | Revolve axis | Revolve about a coordinate axis | ui/RevolveEditor.tsx | done | crates/polymix/src/reverse/editors/revolve.rs; crates/mm-reverse/src/build/revolve.rs | — |
| EXT-24 | Revolve axis | Revolve about a line of the sketch (construction line) | ui/RevolveEditor.tsx | done | crates/polymix/src/reverse/editors/revolve.rs | — |
| EXT-25 | Revolve axis | Revolve about a measured element's axis (line, cylinder, cone, circle) | ui/RevolveEditor.tsx | partial | crates/mm-reverse/src/build/mod.rs (resolve_axis) | Resolved; no element source or list entry |
| EXT-26 | Revolve axis | Click the axis line on the part to choose it; hovering the list draws the axis | ui/RevolveEditor.tsx; viewer/reScene.ts | missing | — | — |
| EXT-27 | Revolve | Angle (≤ 360°) typed or dragged on a ring; presets | ui/RevolveEditor.tsx | partial | crates/polymix/src/reverse/editors/revolve.rs | Typed and presets; no ring grip |
| EXT-28 | Coil | Coil: pitch, angle past 360° as degrees / turns / height, left-handed, reverse or negative pitch; defaults (1.5× reach, 3 turns); refused when self-touching or touching the axis | ui/RevolveEditor.tsx (revolve-coil, revolve-left); core/buildExtrude.ts | missing | crates/mm-reverse/src/build/revolve.rs | Refused by name |

## RND

**Fillet, chamfer, shell & delete face** — 21 rows (done 6, partial 5, missing 10).

Rounds and bevels (constant/variable/fit, modes, reference face, skipFailing), shell (thickness map, strategies), delete face (whole feature, tangent faces).

- **Depends on:** BUILD, PICK
- **Staged ports to integrate:** ports/fillet (fillet_estimate, curved_edge, fillet_read, chamfer_sides)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| RND-01 | Fillet | Fillet at a constant radius on picked edges | ui/FeatureEditor.tsx; core/kernel.ts | done | crates/polymix/src/reverse/editors/fillet.rs; crates/mm-reverse/src/build/fillet.rs | — |
| RND-02 | Fillet | Tangent propagation switch (on), the edges the chain will reach lit lighter as soon as one is picked | ui/FeatureEditor.tsx; viewer/bodyPicks.ts | partial | crates/polymix/src/reverse/editors/fillet.rs (Along tangent edges) | Switch only; chain not previewed |
| RND-03 | Fillet | Radius read off the scan across the first edge (section square to the edge, circle tangent to both faces); box says what it found or that there is no round | core/filletEstimate.ts; core/curvedEdge.ts | missing | — | Staged ports/fillet fillet_estimate.rs, curved_edge.rs |
| RND-04 | Fillet | Variable radius: each edge with Start and End radii, ⇄ swaps them, Start/End labels with radii on the part | ui/FeatureEditor.tsx (fillet-varying*); viewer/varyingEnds.ts | missing | crates/mm-reverse/src/build/fillet.rs | Refused by name |
| RND-05 | Fillet | Radius "fit" (median of the edges read), variable "fit" law, reach | app/reFit.ts; app/reFilletRead.ts | missing | — | Staged ports/fillet fillet_read.rs |
| RND-06 | Fillet | skipFailing: round what the kernel can, the note naming the chains left out | app/reSchemas.ts (skipFailing); core/build.ts | partial | crates/mm-reverse/src/build/fillet.rs (what_holds) | Built; no UI switch |
| RND-07 | Fillet | A feature of several solids rounded solid by solid (<name> 1..n, all or none) | app/reSchemas.ts (solidsAsBody) | missing | — | — |
| RND-08 | Fillet | Rounds taking the whole face between their edges made a hair under size, the sliver taken off | core/kernelRounds.ts | missing | — | — |
| RND-09 | Chamfer | Chamfer Equal distance | ui/FeatureEditor.tsx (chamfer-mode) | done | crates/polymix/src/reverse/editors/chamfer.rs; crates/mm-reverse/src/build/chamfer.rs | — |
| RND-10 | Chamfer | Chamfer Two distances (first on the reference face) | core/chamferSides.ts | done | crates/polymix/src/reverse/editors/chamfer.rs | — |
| RND-11 | Chamfer | Chamfer Distance and angle | core/chamferSides.ts | done | crates/polymix/src/reverse/editors/chamfer.rs | — |
| RND-12 | Chamfer | Reference face = the face most edges share (a rim's top); Flip takes the other | core/chamferSides.ts; ui/FeatureEditor.tsx (chamfer-flip) | partial | crates/mm-reverse/src/build/chamfer.rs | Reference is the face facing up most; staged ports/fillet chamfer_sides.rs |
| RND-13 | Chamfer | Chamfer distance "fit" from the scan | app/reFit.ts | missing | — | — |
| RND-14 | Shell | Shell: faces to take away (none = closed hollow), wall thickness | ui/FeatureEditor.tsx (shell-faces) | done | crates/polymix/src/reverse/editors/shell.rs; crates/mm-reverse/src/build/shell.rs | — |
| RND-15 | Shell | Shell wall from the Thickness map's reading (2 mm otherwise); arrow on the opened face to drag it | ui/FeatureEditor.tsx (shell-source); core/preview.ts | missing | — | — |
| RND-16 | Shell | Shell strategies: kernel thick solid, offset core cut out, core rebuilt with sharp inside corners | core/kernelShell.ts | partial | crates/mm-reverse/src/build/shell.rs | Kernel shell only |
| RND-17 | Delete face | Delete face: faces taken off and the faces round them run on over the gap | ui/FeatureEditor.tsx (delete-face-faces) | partial | crates/mm-reverse/src/build/delete_face.rs; crates/polymix/src/reverse/editors/delete_face.rs (stub) | Built; no editor |
| RND-18 | Delete face | One click takes the whole feature: faces it is split from, faces it closes in (blind hole's floor, boss top); second click lets it go | core/faceGroup.ts | missing | — | No `whole` field in the model |
| RND-19 | Delete face | Tangent faces: the rounds running on from a round (a fillet's whole chain) | ui/FeatureEditor.tsx (delete-face-tangent) | missing | — | — |
| RND-20 | Preview | A feature on a body previews as the body it will leave (solid, with edges) | viewer/reScene.ts | done | crates/polymix/src/reverse/mod.rs (shown_part) | — |
| RND-21 | Picking | Shift+drag window picks edges/faces into an open Fillet, Chamfer, Shell or Draft (window/crossing, hidden edges included) | viewer/bodyPicks.ts | missing | — | — |

## HOLE

**Holes & threads** — 19 rows (done 0, partial 0, missing 19).

Hole along element/round face/flat face, heat-set inserts, modelled threads (ISO/UNC/UNF/print45, sizes, hand, length, offset, chamfer, face sizing), thread read off the scan.

- **Depends on:** BUILD, PLANE, PICK, PARAM
- **Staged ports to integrate:** ports/threads (thread, thread_cut, thread_read, inserts)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| HOLE-01 | Hole | Hole along a cylinder, circle or line element's axis | ui/FeatureEditor.tsx (hole-*); core/resolve.ts | missing | crates/mm-reverse/src/build/hole.rs (stub) | — |
| HOLE-02 | Hole | Hole along a round face of the body (its radius by default) | ui/FeatureEditor.tsx | missing | crates/mm-reverse/src/build/hole.rs (stub) | — |
| HOLE-03 | Hole | Hole into a flat face at a point or its centre, square to it (a standoff's top) | app/reSchemas.ts (hole.axis point) | missing | — | — |
| HOLE-04 | Hole | Through everything, or to a depth each way along the axis (into the body alone from a flat face) | ui/FeatureEditor.tsx (hole-through) | missing | — | — |
| HOLE-05 | Hole | Insert list (CNC Kitchen M2–M8, ¼"-20): hole Ø, 1 mm deeper than the insert, wall to leave shown | core/inserts.ts; ui/FeatureEditor.tsx (hole-insert, hole-insert-wall) | missing | — | Staged ports/threads inserts.rs |
| HOLE-06 | Thread | Thread on a full round face: bolt on a boss or nut in a bore, told by the face's side | core/thread.ts; core/kernelThread.ts | missing | crates/mm-reverse/src/build/thread.rs (stub) | Staged ports/threads thread.rs, thread_cut.rs |
| HOLE-07 | Thread | ISO metric coarse and fine | core/thread.ts | missing | — | — |
| HOLE-08 | Thread | UNC and UNF | core/thread.ts | missing | — | — |
| HOLE-09 | Thread | 3D printing form print45: 45° flanks, flat crest/root p/5, 0.3 p deep, Ø3–Ø64 at the deep pitch or ISO coarse | core/thread.ts | missing | — | — |
| HOLE-10 | Thread | Size from the clicked face: a boss's major Ø; a bore drawn at a size's Ø that size, otherwise its tap drill | core/thread.ts; ui/FeatureEditor.tsx (thread-size) | missing | — | — |
| HOLE-11 | Thread | Size by designation (M10x1, 1/4-20 UNC, Ø10) or diameter + pitch typed | core/thread.ts; ui/FeatureEditor.tsx (thread-designation) | missing | — | — |
| HOLE-12 | Thread | Right or left hand | ui/FeatureEditor.tsx (thread-hand) | missing | — | — |
| HOLE-13 | Thread | Full length, or a length from a start in from either end (flip) | ui/FeatureEditor.tsx (thread-full, thread-flip) | missing | — | — |
| HOLE-14 | Thread | Offset clearance for printing; refused at √3/8 p (√2/8 for print45) | core/thread.ts | missing | — | — |
| HOLE-15 | Thread | Face brought to the thread (bore filled/bored to minor, boss built up/turned down to major); box and row say so past the thread depth | core/kernelThread.ts; ui/FeatureEditor.tsx (thread-problem) | missing | — | — |
| HOLE-16 | Thread | Chamfer the free ends (45°, from 0.1 p past the root); refused where it comes out free nowhere | ui/FeatureEditor.tsx (thread-chamfer) | missing | — | — |
| HOLE-17 | Thread | Thread stops at a shoulder or bore floor, runs out past a free end | core/kernelThread.ts | missing | — | — |
| HOLE-18 | Thread | Thread read off the scan along a cylinder element, round face or axis: pitch, minor/major, bore or boss, hand, nearest standard size, feature ready to add | core/threadRead.ts; app/reThreadCommand.ts | missing | — | Staged ports/threads thread_read.rs |
| HOLE-19 | Expressions | Hole diameter/depth and thread diameter/pitch/length/start/offset driven by expressions | app/reSchemas.ts (exprs) | missing | — | Needs PARAM |

## DRAFT

**Draft, move face & snap to scan** — 13 rows (done 0, partial 0, missing 13).

Draft (neutral, groups, fit), Move face (flat/round, several faces), Snap to scan with plane/cylinder fits, change-what-made-it edits, draft-it-as-well.

- **Depends on:** BUILD, PICK, PLANE, SKT
- **Staged ports to integrate:** ports/fits (fit_plane_from_seed, fit_cylinder_from_seed)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| DRAFT-01 | Draft | Draft: neutral plane by the flat face that stays or a chosen plane; faces to lean; angle (in positive, out negative, under 89°) | ui/FeatureEditor.tsx (draft-*); core/kernelFaces.ts | missing | crates/mm-reverse/src/build/draft.rs (stub); crates/polymix/src/reverse/editors/draft.rs (stub) | — |
| DRAFT-02 | Draft | Draft groups: more faces each at its own angle; tangent-joined faces passed over or refused | app/reSchemas.ts (draft.groups); ui/FeatureEditor.tsx (draft-groups) | missing | — | — |
| DRAFT-03 | Draft | Draft angle "fit" from the scan (plane fitted inside each face, median; coverage warned under 50 %) | app/reFit.ts | missing | — | — |
| DRAFT-04 | Move face | Move face: a flat face along its outward normal (out adds, in cuts), a round face by radius; box reads the new diameter | ui/FeatureEditor.tsx (move-face-*); core/kernelFaces.ts | missing | crates/mm-reverse/src/build/move_face.rs (stub) | — |
| DRAFT-05 | Move face | Several faces moved by one distance as a whole (a cylinder's wall and ends) | core/kernelFaces.ts (offsetFacesTracked) | missing | — | — |
| DRAFT-06 | Move face | Distance typed, dragged by its arrow, rounded by its chip | core/preview.ts; core/nominal.ts | missing | — | — |
| DRAFT-07 | Snap to scan | Snap to scan: pick a body face, click the scan where it should lie; plane fit for a flat face, cylinder for a round; σ, tilt, axis offset reported; > 20° off refused | core/snapFace.ts; ui/FeatureEditor.tsx (move-face-snap, move-face-reading) | missing | — | Fits staged in ports/fits |
| DRAFT-08 | Snap to scan | While clicking the scan: bodies as tint and edges, the scan solid, a ghost of the face where it will lie | viewer/reScene.ts | missing | — | — |
| DRAFT-09 | Snap to scan | Change what made it instead: extrude cap → From/To; wall → its sketch line; bore/boss → sketch circle radius; revolve section lines; nothing added to the list | core/snapEdit.ts; ui/FeatureEditor.tsx (move-face-history) | missing | — | snapEdit.ts / snapFace.ts not staged anywhere yet |
| DRAFT-10 | Snap to scan | Why there is no offer (fillet/mirror face, dimension holds the wall, extrude to a plane, line would overlap another profile) | core/snapEdit.ts | missing | — | — |
| DRAFT-11 | Move face | Move face knows only the face's outline: walls must stand square, rim rounds do not follow — box says so | ui/FeatureEditor.tsx (move-face-plan) | missing | — | — |
| DRAFT-12 | Snap to scan | Draft it as well, by N°: when the scan's face leans past the fit's scatter, a Draft hinged across the face's middle is added after the Move face; one Ctrl+Z | ui/FeatureEditor.tsx (move-face-lean) | missing | — | — |
| DRAFT-13 | Move face | Several picked faces that all trace back are moved in their makers in one edit | core/snapEdit.ts | missing | — | — |

## LOFT

**Loft, sweep, pattern & move body** — 20 rows (done 1, partial 7, missing 12).

Loft (ruled/smooth, stations fitted from the scan, inner loft), sweep (path sketch or 3D route), tube trace, body pattern (circular/linear/grid/sheet), Move body with gizmo and copies.

- **Depends on:** EXT, FIT, PICK, PARAM
- **Staged ports to integrate:** ports/sweeps (stations, loft_fold, sweep_route, tube_trace); ports/sketchtools (body_move)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| LOFT-01 | Loft | Loft through two or more sketches' largest regions, ruled or smooth | ui/FeatureEditor.tsx (loft-*); core/profiles.ts | missing | crates/mm-reverse/src/build/loft.rs (stub); crates/polymix/src/reverse/editors/loft.rs (stub) | — |
| LOFT-02 | Loft | Stations fitted from the scan on planes like a sketch (sections + fit.like) into “<name> station k” | core/stations.ts; app/reSchemas.ts (loft.sections, loft.fit) | missing | — | Staged ports/sweeps stations.rs |
| LOFT-03 | Loft | Inner loft: every station inset by the wall into “<name> inner k” and lofted as a body of its own | app/reSchemas.ts (loft.inner) | missing | — | — |
| LOFT-04 | Loft | Station offsets and inner wall driven by expressions | app/reSchemas.ts (loft.exprs) | missing | — | — |
| LOFT-05 | Loft | Smooth loft fold detection (skin crossing itself between close sections) | core/loftFold.ts | missing | — | Staged ports/sweeps loft_fold.rs |
| LOFT-06 | Sweep | Sweep a sketch's largest region along another sketch's chain of lines and arcs, section square to the path (profile within 1° square at the start) | ui/SweepFields.tsx; core/profiles.ts | missing | crates/mm-reverse/src/build/sweep.rs (stub) | — |
| LOFT-07 | Sweep | Route: 3D centreline points with a bend radius (and per-bend radii); a bend that does not fit refused with its largest | core/sweepRoute.ts; ui/SweepFields.tsx (sweep-route) | missing | — | Staged ports/sweeps sweep_route.rs |
| LOFT-08 | Sweep | Swap profile and path | ui/SweepFields.tsx (sweep-swap) | missing | — | — |
| LOFT-09 | Tube | Tube trace: a bent tube's centreline read from the scan, stopping at a flange, bellows or junction | core/tubeTrace.ts | missing | — | Staged ports/sweeps tube_trace.rs |
| LOFT-10 | Pattern | Body pattern circular about an axis: count, angle spanned (360° spreads evenly) | ui/FeatureEditor.tsx (pattern-mode); core/patternCopies.ts | partial | crates/mm-reverse/src/build/pattern.rs; crates/polymix/src/reverse/editors/pattern.rs (stub) | Built; no editor |
| LOFT-11 | Pattern | Body pattern linear along a direction with a spacing | ui/FeatureEditor.tsx | partial | crates/mm-reverse/src/build/pattern.rs | Built; no editor |
| LOFT-12 | Pattern | Grid: second direction with count and spacing | ui/FeatureEditor.tsx (pattern-axis2-clear) | partial | crates/mm-reverse/src/build/pattern.rs | Built; no editor |
| LOFT-13 | Pattern | Pattern of a sheet (one window outline patterned into cutters) | README §7 | missing | — | Needs SURF sheets |
| LOFT-14 | Pattern | Copies apart are bodies of their own; copies' faces keep their makers | core/patternCopies.ts | partial | crates/mm-reverse/src/build/pattern.rs | Islands made; maker tracking of copies not verified |
| LOFT-15 | Move body | Move body: selected bodies, the one there is, or clicked once open; Shift+drag window; click a picked one to let it go | ui/FeatureEditor.tsx (move-body-*); state/reFeatureSlice.ts | partial | crates/mm-reverse/src/build/move_body.rs; crates/polymix/src/reverse/editors/move_body.rs (stub) | Built; no editor |
| LOFT-16 | Move body | Gizmo: arrows along X/Y/Z and gimbal rings, constant screen size, field on the grip | core/preview.ts; state/reGripSlice.ts | missing | — | Staged ports/sketchtools body_move.rs |
| LOFT-17 | Move body | Six numbers: Move (mm) and Turn (deg) about X then Y then Z through the stored pivot (middle of the bodies as picked); parameters may drive them | core/move.ts | partial | crates/mm-reverse/src/build/move_body.rs; crates/mm-reverse/src/build/mod.rs (move_matrix) | Built; no box, no expressions |
| LOFT-18 | Move body | Copy: copies moved as bodies of their own, originals left | ui/FeatureEditor.tsx (move-body-copy) | partial | crates/mm-reverse/src/build/move_body.rs | Built; no UI |
| LOFT-19 | Move body | Ghost rides the gizmo at once (meshes carried, nothing built); live map against where the bodies go | state/reFeatureSlice.ts | missing | — | — |
| LOFT-20 | Move body | A moved body's faces keep their makers (later picks still find them) | core/move.ts | done | crates/mm-reverse/src/build/move_body.rs | — |

## BOOL

**Mirror, combine, cut by plane & offset body** — 12 rows (done 4, partial 2, missing 6).

Mirror (planes, face, scan symmetry plane), Combine (join/cut/intersect, keep tool, multi-piece tools, sheets), Cut by plane, Offset body.

- **Depends on:** BUILD, PLANE, SURF
- **Staged ports to integrate:** ports/sketchtools (symmetry_plane)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| BOOL-01 | Mirror | Mirror about a plane: coordinate (with offset), section, measured plane, radial | ui/FeatureEditor.tsx (mirror-*) | partial | crates/polymix/src/reverse/editors/mirror.rs; crates/mm-reverse/src/build/mirror.rs | Editor offers coordinate planes and the face only |
| BOOL-02 | Mirror | Mirror about a flat face of the body | ui/ContextChips.tsx (mirror) | done | crates/polymix/src/reverse/editors/mirror.rs; crates/mm-reverse/src/build/mirror.rs | — |
| BOOL-03 | Mirror | Join with the original (default) or a body of its own | ui/FeatureEditor.tsx (mirror-join) | done | crates/polymix/src/reverse/editors/mirror.rs | — |
| BOOL-04 | Mirror | A new Mirror opens on the scan's symmetry plane (measured, or mirror registration of the principal planes accepted ≤ 0.1 mm RMS over half the samples; σ and counts; refused by name) | core/symmetryPlane.ts; ui/FeatureEditor.tsx (mirror-symmetry*) | missing | — | Staged ports/sketchtools symmetry_plane.rs |
| BOOL-05 | Mirror | Keep the found symmetry plane as a plane element | ui/FeatureEditor.tsx (mirror-symmetry-keep) | missing | — | Needs PLANE element source |
| BOOL-06 | Combine | Combine two bodies: join, cut, intersect | ui/FeatureEditor.tsx (boolean-op) | done | crates/polymix/src/reverse/editors/boolean.rs; crates/mm-reverse/src/build/boolean.rs | — |
| BOOL-07 | Combine | Keep the tool body beside the result | ui/FeatureEditor.tsx (boolean-keep) | done | crates/polymix/src/reverse/editors/boolean.rs | — |
| BOOL-08 | Combine | A tool feature left in pieces goes in with all of them (a plug patterned into copies cuts every window); two profiles extruded together combined either way round | core/build.ts | missing | crates/mm-reverse/src/build/boolean.rs | One tool body only |
| BOOL-09 | Combine | Sheet as the body: intersect keeps it inside a solid tool, cut outside, join sews two sheets edge to edge | ui/FeatureEditor.tsx (boolean-sheet) | missing | — | Needs SURF |
| BOOL-10 | Cut by plane | Cut a body by a plane and keep one side (flip); plane offset by expression | ui/FeatureEditor.tsx (cut-plane-flip) | partial | crates/mm-reverse/src/build/cut_plane.rs; crates/polymix/src/reverse/editors/cut_plane.rs (stub) | Built; no editor |
| BOOL-11 | Offset body | Offset body: grow or shrink all round by a distance (convex edges rounded); kept as a body of its own (default) or replacing it | ui/FeatureEditor.tsx (offset-body-keep) | missing | crates/mm-reverse/src/build/offset_body.rs (stub) | — |
| BOOL-12 | Offset body | Keep-out workflow: offset body + Combine Cut (tool kept) → a pocket with clearance | README §7 | missing | — | — |

## SURF

**Sheets, split, thicken, trim, fit surface & extract** — 23 rows (done 0, partial 2, missing 21).

The surface route: sheets, split (keep scan/largest/smallest/between/within), thicken, Fit surface (region flood/marking, freeform/plane/cylinder, Auto Constrain, Hold to), Trim, Extract of elements and shadow sections.

- **Depends on:** PLANE, PICK, CMP, BUILD
- **Staged ports to integrate:** ports/surface (surface_fit, surface_region, height_surface); ports/fits (region_grow, mesh_graph); ports/sketchtools (auto_constrain); ports/sweeps (shadow, shadow_pieces)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| SURF-01 | Sheet | Sheet: a sketch's pieces (an open chain will do) extruded as a surface body (from/to), marked surface, never exported alone | ui/FeatureEditor.tsx (sheet-sketch) | missing | crates/mm-reverse/src/build/sheet.rs (stub) | — |
| SURF-02 | Sheet | Surface bodies: never joined or cut; marked surface in the body list; live map against them | core/build.ts; ui/BodyList.tsx | missing | — | — |
| SURF-03 | Split | Split by sheets/bodies keeping the fragments on the scan (sampled by area, tolerance, overlap share) | ui/FeatureEditor.tsx (split-*) | missing | crates/mm-reverse/src/build/split.rs (stub) | — |
| SURF-04 | Split | Split keep largest / smallest (volume, a sheet's area) | ui/FeatureEditor.tsx (split-keep) | missing | — | — |
| SURF-05 | Split | Split keep between: the wall between two fitted skins; refused where they meet | app/reSchemas.ts (split.keep between) | missing | — | — |
| SURF-06 | Split | Split within: freeform surfaces trimmed to their region's outline + margin; surfaces carried on to the body's sides; offScan reported | core/surfaceRegion.ts; app/reSchemas.ts (split.within) | missing | — | Staged ports/surface surface_region.rs |
| SURF-07 | Split | A sheet as the split body: trimmed, the pieces kept sewn into one sheet | app/reSchemas.ts (split) | missing | — | — |
| SURF-08 | Thicken | Thicken a sheet or fitted surface: thickness, side in/out/both, New/Join/Cut, target; both refused for a multi-patch sheet | ui/FeatureEditor.tsx (thicken-*); core/kernelThicken.ts | missing | crates/mm-reverse/src/build/thicken.rs (stub) | — |
| SURF-09 | Thicken | Why a sheet does not thicken (a face bending tighter than the thickness), the wall reached along its edge | core/diagnoseThicken.ts | missing | — | — |
| SURF-10 | Fit surface | Region flood from a clicked scan face over edges turning less than the angle (1°); Ctrl-click takes a flood away | ui/FeatureEditor.tsx (surface-region); app/reSurfaceRegion.ts | missing | crates/mm-reverse/src/build/surface.rs (stub) | Staged ports/fits region_grow.rs |
| SURF-11 | Fit surface | Region marked with the tools (brush, window, lasso), or by box / normal / exclude boxes / limit | ui/FeatureEditor.tsx (surface-paint); app/reSurfaceRegion.ts | missing | — | — |
| SURF-12 | Fit surface | Freeform fit: degree-3 B-spline, control net from extent and noise or typed 3–20, Expand by 35 %, Smoothness 0.01, robust (strays named), height direction | core/surfaceFit.ts | missing | — | Staged ports/surface surface_fit.rs |
| SURF-13 | Fit surface | Fit report: σ, worst point, net used, past-region standoff; live map against the surface | core/surfaceFit.ts; core/surfaceRegion.ts | missing | — | — |
| SURF-14 | Fit surface | Shape: Plane or Cylinder instead of freeform, grown by Expand by | ui/FeatureEditor.tsx (surface-shape) | missing | — | — |
| SURF-15 | Fit surface | Auto Constrain: within 0.5° parallel/square to XY/YZ/ZX, measured or earlier-fitted references refitted held; within 1 mm put on it; says what held and σ before/after | core/autoConstrain.ts; ui/featureFields.tsx (surface-held) | missing | — | Staged ports/sketchtools auto_constrain.rs |
| SURF-16 | Fit surface | Hold to: force a relation however far off, or Nothing (the free fit) | ui/FeatureEditor.tsx (surface-constrain) | missing | — | — |
| SURF-17 | Fit surface | Region box driven by expressions, the points read again when a parameter moves it | app/reSchemas.ts (surface.exprs) | missing | crates/mm-reverse/src/model.rs (RegionBox, RegionStale) | — |
| SURF-18 | Trim | Trim: surfaces, sheets, solid bodies and measured planes/cylinders/cones/spheres/tori; a box round the scan cut by all; faces lying on the scan kept (Tolerance from 3× median σ, Overlap) | ui/FeatureEditor.tsx (trim-*) | missing | crates/mm-reverse/src/build/trim.rs (stub) | — |
| SURF-19 | Trim | Closed → solid (faces named by their surfaces, filletable, exportable); open → surface body, counts, open edges drawn red | core/build.ts | missing | — | — |
| SURF-20 | Trim | Planes and cylinders cut as the whole plane/cylinder, not the patch drawn | core/build.ts | missing | — | — |
| SURF-21 | Extract | Extract a measured plane (slab of a thickness), cylinder, cone, sphere at its extents; new/join/cut; element put out of view | ui/FeatureEditor.tsx (extract-element); core/primitives.ts | partial | crates/mm-reverse/src/build/extract.rs; crates/mm-reverse/src/build/mod.rs (primitive_of) | Built; no editor; no element source |
| SURF-22 | Extract | Torus element extracted as a ring body; torus in the trim's set | core/primitives.ts | partial | crates/mm-reverse/src/build/mod.rs (primitive_of) | Ring built; trim missing |
| SURF-23 | Extract | Extract a shadow section: its outline stood up through the part's thickness, following the shadow | core/primitives.ts; core/shadowPieces.ts | missing | crates/mm-reverse/src/build/extract.rs | Refused by name; staged ports/sweeps shadow |

## NET

**Patch network** — 21 rows (done 0, partial 0, missing 21).

QuickSurface-style patch network: placing, editing and filling patches on the scan, creases, strips and spans, the joint tangent fit, the body it makes.

- **Depends on:** SURF, VIEW, CMP
- **Staged ports to integrate:** ports/network (stage 1 + stage 2); ports/fits (mesh_graph curvature/crease)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| NET-01 | Placing | Click a patch's corners on the scan; its sides follow the scan between them; shaded as it grows | state/reNetworkDraft.ts; viewer/networkLayer.ts | missing | crates/mm-reverse/src/build/network.rs (stub); crates/polymix/src/reverse/editors/network.rs (stub) | Staged ports/network |
| NET-02 | Placing | Close with a click on the first corner, Enter, double click or middle click; fitted at once | state/reNetworkDraft.ts; app/keyboard.ts | missing | — | — |
| NET-03 | Placing | Backspace takes the last corner back; Esc lets the patch go; Enter with two corners makes a line | state/reNetworkDraft.ts | missing | — | — |
| NET-04 | Placing | A corner on a corner, an edge (split there) or a point of an edge joins the neighbours (one shared edge) | state/reNetworkDraft.ts | missing | — | — |
| NET-05 | Placing | Drag an outer edge across the scan to make the next patch (corners landing on corners near) | state/reNetworkDraft.ts | missing | — | — |
| NET-06 | Editing | Corners and points dragged along the scan; a shared edge bent by dragging (point inserted); Shift-click puts a point in | state/reNetworkDraft.ts | missing | — | — |
| NET-07 | Editing | Click picks a patch (its strips' spans in the box); Delete patch removes it and its unshared edges | ui/NetworkEditor.tsx (network-delete-patch) | missing | — | — |
| NET-08 | Editing | Delete a point, corner or edge under the cursor; C makes an edge a crease (dashed) | app/useNetworkLayer.ts | missing | — | — |
| NET-09 | Editing | Wheel over an edge sets its strip's span count | app/useNetworkLayer.ts | missing | — | — |
| NET-10 | Editing | Three-, five- and six-sided patches split from a draggable centre into four-sided ones | core/networkSplit.ts | missing | — | Staged ports/network network_split.rs |
| NET-11 | Editing | Network editor's own Undo/Redo | ui/NetworkEditor.tsx (network-undo, network-redo) | missing | — | — |
| NET-12 | Display | Scan coloured by curvature while the box is open | app/useNetworkLayer.ts | missing | — | Needs VIEW curvature |
| NET-13 | Bodies | A body's edge as a boundary: Contact, or Tangent to one of its faces | ui/NetworkEditor.tsx (network-bodies) | missing | — | — |
| NET-14 | Creases | Suggest creases: sharp edges and folds (not rounds) dashed violet; click one or Accept all | core/networkCreases.ts; ui/NetworkEditor.tsx (network-suggest*) | missing | — | Staged ports/network network_creases.rs |
| NET-15 | Fill | Fill: click a patch; Ctrl-click empties; click inside a region closed another way; Fill all; refusals drawn where (not closed, 2 or 7 sides, crossing, scan rim) | core/networkCells.ts; ui/NetworkEditor.tsx (network-fill-all, network-refused) | missing | — | Staged ports/network network_cells.rs |
| NET-16 | Fit | Joint fit: shared curve poles exact, tangency pole by pole, T-junctions by weight; corners of 3/5/6 near tangent with angle reported | core/networkFit.ts | missing | — | Staged ports/network network_fit.rs |
| NET-17 | Fit | Auto spans (4, up to 20 where a patch misses by > Tolerance 0.01 beyond noise); Smoothness 0.001; Leave out stray points | core/networkFit.ts; ui/NetworkEditor.tsx (network-robust) | missing | — | — |
| NET-18 | Readout | Status line (patches, unfilled regions, worst tangent angle, worst gap, worst corner) and a row per patch (σ, worst, net) | ui/NetworkEditor.tsx (network-status, network-patches) | missing | — | — |
| NET-19 | Body | Result: a sheet (surface) or a solid where it closes; exported as a skin; thickened by Thicken | core/networkBody.ts | missing | — | Staged ports/network network_body.rs |
| NET-20 | Performance | Preview without kernel faces; traces and maps kept between edits (fit on screen in ~0.25 s) | core/network.ts; app/useNetworkLayer.ts | missing | — | — |
| NET-21 | Live map | Live map against the patches | core/liveMap.ts | missing | — | — |

## PARAM

**Parameters & expressions** — 12 rows (done 0, partial 3, missing 9).

The expression language every number field takes, the Parameters section, kinds, renames and removals, feature expression slots, display units.

- **Depends on:** BUILD
- **Staged ports to integrate:** ports/expressions (expression, parameters, param_slots, display_units, length_fields)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| PARAM-01 | Expressions | Arithmetic in every number field: + − × ÷ ^, parentheses, sin cos tan asin acos atan atan2 sqrt abs min max floor ceil round pow, pi, e; degrees | core/expression.ts | missing | — | Staged ports/expressions expression.rs |
| PARAM-02 | Expressions | Units on numbers (in, cm, deg, rad); a bare number read in the field's unit | core/expression.ts | missing | — | — |
| PARAM-03 | Expressions | What was typed stays in the field; the value it comes to is written where the unit was | ui/ExprInput.tsx | missing | — | — |
| PARAM-04 | Parameters | Parameters section: named length / angle / integer / number, each an expression that may read the others | ui/ParameterList.tsx; core/parameters.ts | partial | crates/mm-reverse/src/model.rs (Parameter) | Data only; no UI, no evaluator |
| PARAM-05 | Parameters | Any field takes a parameter by name; a change rebuilds every number that reads it (sketches re-solved, profiles kept) | core/parameters.ts; state/reFeatureHelpers.ts | missing | — | Staged ports/expressions parameters.rs, param_slots.rs |
| PARAM-06 | Parameters | Kinds kept apart (length + angle refused, area refused, integer must be whole); error under the field, old number kept | core/expression.ts; ui/ParameterList.tsx (parameter-error) | missing | — | — |
| PARAM-07 | Parameters | A plain number typed (or a grip dragged) over a parameter-driven field lets the parameter go | core/parameters.ts | missing | — | — |
| PARAM-08 | Parameters | Rename a parameter renames it in every expression | core/parameters.ts | missing | — | — |
| PARAM-09 | Parameters | Remove refused while something reads it; the row names the readers | core/parameters.ts | missing | — | — |
| PARAM-10 | Saving | Parameters saved with the project, expressions beside the numbers they last came to | app/projectSection.ts | partial | crates/mm-reverse/src/model.rs (Part.parameters, exprs maps) | Saved; numbers never refreshed from expressions |
| PARAM-11 | Slots | Feature expression slots per kind (extent.from/to/total, draftDeg, angleDeg, pitch, turns, height, thickness, radius, distance, distance2, count, step, translate.x…, plane.offset, sections.k.offset, region.box…) | app/reSchemas.ts (exprsIn); core/parameters.ts | partial | crates/mm-reverse/src/model.rs (Exprs on every kind) | Stored, not evaluated |
| PARAM-12 | Units | Display units (inch-aware length fields) | ui/reLength.ts; src/core/units.ts | missing | — | Staged ports/expressions display_units.rs, length_fields.rs |

## LIST

**Body list, components & measurements** — 15 rows (done 0, partial 6, missing 9).

Body rows (name, colour, visibility), hover linking, components (folders), multi-row selection, selection readouts and two-pick measurements, clearance, overlap hatching.

- **Depends on:** BUILD, PICK

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| LIST-01 | Rows | Body rows: eye, name, faces/triangles count | ui/BodyList.tsx (body-row) | partial | crates/polymix/src/reverse/mod.rs (body_list) | Eye, name, volume; no face/triangle counts |
| LIST-02 | Rows | Name a body by double click (its STEP part name); cleared → Solid n | ui/BodyList.tsx (body-name-input) | partial | crates/mm-reverse/src/model.rs (BodyStyle); crates/mm-reverse/src/build/mod.rs (name_bodies) | Names applied; no UI |
| LIST-03 | Rows | Colour a body from swatches its dot opens; Auto restores the maker's colour | ui/BodyList.tsx (body-color, body-swatch*) | partial | crates/mm-reverse/src/model.rs (BodyStyle.color) | Data only; bodies drawn one colour |
| LIST-04 | Hover | Pointer on a row lights the body in blue over everything; cursor on a body lights its row | ui/BodyList.tsx; viewer/reScene.ts | missing | — | — |
| LIST-05 | Visibility | Hide / Hide others / Show all (row menu and chips) | ui/BodyList.tsx (body-menu); ui/ContextChips.tsx | partial | crates/polymix/src/reverse/mod.rs (hidden_bodies) | Per-body eye only |
| LIST-06 | Components | Components: Component key makes one; drag bodies/components in, drop on the strip to take out; nesting | core/bodyFolders.ts; ui/BodyList.tsx (body-folder*, body-drop-top) | partial | crates/mm-reverse/src/model.rs (BodyFolder) | Data only |
| LIST-07 | Components | Component eye shows/hides all it holds; fold; double-click rename; Remove component; New component inside; Move to … | ui/BodyList.tsx | missing | — | — |
| LIST-08 | Components | Components saved with the project and undone like features | app/projectSection.ts | partial | crates/mm-reverse/src/model.rs (Part.body_folders) | Saved; nothing edits them |
| LIST-09 | Selection | Rows picked as in a file browser (click, Ctrl+click, Shift+click range), lit in the viewport, dragged together; Move / Component / Hide others keys; Escape clears | ui/BodyList.tsx (body-selection-*) | missing | — | — |
| LIST-10 | Selection | While a Move is open a row click puts the body in or takes it out | ui/BodyList.tsx (body-list-move-hint) | missing | — | — |
| LIST-11 | Readouts | Pick readouts: a face's area and normal (round: radius, axis), an edge's length and ends (round: radius, centre), a corner's coordinates | core/selectionInfo.ts; ui/ContextChips.tsx | missing | — | — |
| LIST-12 | Measure | Two picks: corner–corner distance with ΔX ΔY ΔZ, corner to edge/axis/face, parallel faces/edges, angle between, two holes' axes; more picks: lengths and areas summed | core/selectionInfo.ts | missing | — | — |
| LIST-13 | Clearance | Clearance between two bodies: closest gap, or the volume they share | ui/ContextChips.tsx (Clearance); app/reBodyCommands.ts (clearance) | missing | — | — |
| LIST-14 | Overlaps | Bodies running into each other show the shared volume hatched red (through see-through bodies and on the section cut); Overlaps switch, kept in the project | core/overlapMesh.ts; viewer/reScene.ts | missing | — | — |
| LIST-15 | Components | Picks on several parts offer Make component; the chip line heads with the body and its component path | ui/ContextChips.tsx | missing | — | — |

## PICK

**Viewport picking & context chips** — 22 rows (done 0, partial 4, missing 18).

Object-first picking of faces/edges/corners/elements/planes, Ctrl+click, Shift+drag windows, hover lighting, and every context chip offered by the click.

- **Depends on:** BUILD, LIST

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| PICK-01 | Picking | With no box open a click on a body takes a face, an edge or a corner | viewer/bodyPicks.ts; viewer/reScene.ts (bodyItemAt) | partial | crates/polymix/src/reverse/view.rs (pointer_input) | Idle click only picks sketch regions; face/edge picks exist inside boxes; no corners |
| PICK-02 | Picking | Ctrl+click adds a pick | viewer/bodyPicks.ts | missing | — | — |
| PICK-03 | Picking | Shift+drag selection window (right: inside, solid; left: touching, dashed; hidden edges included) into the selection | viewer/bodyPicks.ts | missing | — | — |
| PICK-04 | Picking | Hover lights the face/edge a click would take; bodies not in play dimmed while a box is open | viewer/bodyPicks.ts; viewer/reScene.ts | partial | crates/polymix/src/reverse/view.rs (hover_edge, hover_face) | No dimming |
| PICK-05 | Picking | Picks through the scan's veil (a face seen through it is picked) | viewer/bodyPicks.ts | partial | crates/polymix/src/reverse/view.rs (pick_hit) | Not verified against see-through scan |
| PICK-06 | Chips | Sketch region: Extrude, Revolve | ui/ContextChips.tsx (sketchRegion) | partial | crates/polymix/src/reverse/view.rs (pointer_input) | A click opens Extrude directly; no chips |
| PICK-07 | Chips | Element with an axis or sketch line: Revolve about this | ui/ContextChips.tsx | missing | — | — |
| PICK-08 | Chips | Cylinder/circle element: Hole; element with a volume: Extract as body; plane element or section: Sketch | ui/ContextChips.tsx | missing | — | Needs PLANE elements |
| PICK-09 | Chips | Coordinate plane: Sketch, Hide | ui/ContextChips.tsx (hide-plane) | missing | — | — |
| PICK-10 | Chips | Body face: Edit <the feature that made it> | ui/ContextChips.tsx (edit) | missing | — | — |
| PICK-11 | Chips | Flat face: Extrude (the face), Sketch here | ui/ContextChips.tsx | missing | — | — |
| PICK-12 | Chips | Face: Move face, Snap to scan (not on a fillet's part turn) | ui/ContextChips.tsx | missing | — | — |
| PICK-13 | Chips | Face: Shell, Delete face | ui/ContextChips.tsx | missing | — | — |
| PICK-14 | Chips | Round face: Hole, Thread (no Thread on a partial round) | ui/ContextChips.tsx | missing | — | — |
| PICK-15 | Chips | Flat face: Mirror (about it), Draft (as the neutral plane) | ui/ContextChips.tsx | missing | — | — |
| PICK-16 | Chips | Face, edge or selection: Move body | ui/ContextChips.tsx (moveChip) | missing | — | — |
| PICK-17 | Chips | Edge(s) of one body: Fillet, Chamfer | ui/ContextChips.tsx | missing | — | — |
| PICK-18 | Chips | Several faces: Move faces, Shell, Delete faces | ui/ContextChips.tsx | missing | — | — |
| PICK-19 | Chips | Picks on two bodies: Clearance, Move bodies | ui/ContextChips.tsx | missing | — | — |
| PICK-20 | Chips | Body line: name and component path, Hide, Hide others / Show all, Component | ui/ContextChips.tsx (chip-hide-others) | missing | — | — |
| PICK-21 | Chips | Selection: Note (one note on all picks) | ui/ContextChips.tsx (Note) | missing | — | Needs NOTE |
| PICK-22 | Chips | Escape or the next click puts the chips away | ui/ContextChips.tsx | missing | — | — |

## VIEW

**View bar, maps & section view** — 17 rows (done 6, partial 4, missing 7).

Scan/Bodies/Elements keys and their menus (transparent, curvature, tangent edges, overlaps), silhouettes, deviation key and legend, sketch part view (no part/section/full), section view and its render layer.

- **Depends on:** BUILD, CMP
- **Staged ports to integrate:** ports/fits (mesh_graph curvature); ports/sweeps (silhouette)

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| VIEW-01 | Keys | Scan key: put the scan away | app/useReversePlugin.tsx | done | crates/polymix/src/reverse/mod.rs (view_box Scan) | — |
| VIEW-02 | Keys | Bodies key: put the bodies away with the map left on the scan | app/useReversePlugin.tsx | done | crates/polymix/src/reverse/mod.rs (view_box Bodies) | — |
| VIEW-03 | Keys | Elements key: put the measured elements and sections away | app/useReversePlugin.tsx | missing | — | Needs PLANE elements |
| VIEW-04 | Menus | Scan Transparent/solid; the scan a see-through veil once there is a body | app/useReversePlugin.tsx | partial | crates/polymix/src/reverse/mod.rs (See-through) | Manual checkbox; no automatic veil |
| VIEW-05 | Menus | Bodies Transparent/solid (both solid shows where each stands proud) | app/useReversePlugin.tsx | missing | — | Bodies only go see-through with Deviation on |
| VIEW-06 | Bodies | Bodies solid with thin dark edges that stay behind what hides them | viewer/reScene.ts; core/edgeLines.ts | done | crates/polymix/src/reverse/view.rs (render) | — |
| VIEW-07 | Bodies | Silhouettes of round faces from the view direction (a boss's sides, a ball's rim) | core/silhouette.ts | missing | — | Staged ports/sweeps silhouette.rs |
| VIEW-08 | Menus | Tangent edges: Drawn / Dimmed / Off (off ones not pickable) | app/useReversePlugin.tsx (TANGENT_EDGE_CHOICES) | missing | — | — |
| VIEW-09 | Menus | Curvature map (mean curvature, scale as the radius 9 in 10 stay under, dark caps); one map at a time with Deviation | app/useReversePlugin.tsx (curvatureSwitch) | missing | — | Staged ports/fits mesh_graph curvature.rs |
| VIEW-10 | Keys | Deviation key against the committed bodies | app/useReversePlugin.tsx | done | crates/polymix/src/reverse/mod.rs (view_box Deviation) | — |
| VIEW-11 | Legend | Legend beside the viewport reads the ramp; click either end figure to retype the range | app/useReversePlugin.tsx (MapLegend) | partial | crates/polymix/src/reverse/view.rs (legend); crates/polymix/src/reverse/mod.rs (range) | Range set in the panel, not on the legend |
| VIEW-12 | Sketch view | No part / Section / Full part under a sketch | app/useReversePlugin.tsx (SKETCH_PART_VIEWS) | partial | crates/polymix/src/reverse/view.rs (cut_away) | Always cut; Scan/Bodies/Slice checkboxes instead |
| VIEW-13 | Sketch view | Part cut open at the sketch plane with a solid slate cap; bodies closed where cut, each in its own cut colour | viewer/reScene.ts; viewer/sketchPalette.ts | partial | crates/polymix/src/reverse/view.rs (globals.cut_plane) | Clip only, no caps |
| VIEW-14 | Sketch view | Camera swings to face the sketch plane and back when it is put away; the part follows the sheet's pan and zoom | app/useSketchSceneSync.ts | done | crates/polymix/src/reverse/view.rs (face_plane, restore_after_sketch) | — |
| VIEW-15 | Section view | Section view honoured by the reverse scene: part cut at the section's planes, solid bodies capped in cut colours | viewer/reScene.ts | missing | — | Polymix has no section view in this workspace |
| VIEW-16 | Section view | Section render layer: bodies cut by a section's plane as outlines and sketches with piece ids (section.render pictures) | app/reSectionLayer.ts | missing | — | — |
| VIEW-17 | Stage | Stage follows the bodies as they grow; fit view | app/useReverseWorkspace.ts | done | crates/polymix/src/reverse/mod.rs (want_fit_bodies); crates/polymix/src/reverse/view.rs (fit_view) | — |

## CMP

**Deviation, compare & checks** — 9 rows (done 2, partial 1, missing 6).

Live map and deviation map, compare in Deviation, volume compare, compare by face/feature/patch, body check and topology audit.

- **Depends on:** BUILD

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| CMP-01 | Live map | Live map while a ghost is up: scan near it green/red/blue at ± range (0.1 mm, remembered), darker caps, > 10 ranges left as is; switch in the box | core/liveMap.ts; app/useReversePlugin.tsx | partial | crates/mm-reverse/src/compare.rs; crates/polymix/src/reverse/mod.rs (update_deviation) | Ramp, caps, 10-range reach and remembered range are there; the map reads all bodies (preview included) only when the View box turns Deviation on — no switch in the feature box, not on by default while a ghost is up |
| CMP-02 | Live map | Count of scan points near the ghost/bodies within the range | core/liveMap.ts | done | crates/polymix/src/reverse/mod.rs (view_box) | — |
| CMP-03 | Deep inside | Scan lying deeper inside the bodies than the reach painted violet and counted (a pocket the bodies lack) | core/liveMap.ts | done | crates/mm-reverse/src/compare.rs (deep); crates/polymix/src/reverse/mod.rs | — |
| CMP-04 | Deviation | Compare in Deviation: bodies (0.01 mm chord) as the Deviation workspace's reference, measured at once, no best fit; scale, histogram, pins, report | core/compare.ts; app/reBodyCommands.ts (compare_in_deviation) | missing | — | Polymix has no Deviation workspace; Send to Mesh is the nearest |
| CMP-05 | Volume | Volume error of the bodies against the scan as solids | app/reVolumeCompare.ts | missing | — | — |
| CMP-06 | Compare | Comparison told by where: patches over tolerance on one face each, worst spots with face and feature, far inside/outside, by feature | core/compareDetail.ts; app/reBodyCommands.ts (compare) | missing | crates/mm-reverse/src/compare.rs | Map only |
| CMP-07 | Compare | One feature compared against the scan (the faces it made) | app/reCompareFeature.ts | missing | — | — |
| CMP-08 | Check | Body check: valid, closed, solids, volume/area, tiny/narrow/untessellated faces, free/non-manifold edges, shells and voids | app/reBodyCommands.ts (check); core/kernelEdges.ts | missing | — | — |
| CMP-09 | Audit | Topology audit: kinks, steps between two heights, faces nearly parallel/on an axis, short edges — traced to the sketch piece with a repair | core/audit.ts; core/stepsBetween.ts; core/stepWords.ts | missing | — | — |

## NOTE

**Notes & review** — 12 rows (done 0, partial 2, missing 10).

Notes pinned on the part between person and agent (pins, list, agent questions, resolve, persistence by scan), the design-loop review and plan.

- **Depends on:** PICK, AGENT, PROJ, CMP

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| NOTE-01 | Pinning | Note key, then a click on a face/edge/corner or the scan pins a note; a box by the pin takes the sentence (Enter / middle click keeps, Escape leaves) | app/useNotePins.ts; state/noteStore.ts | missing | — | — |
| NOTE-02 | Pinning | One note on a selection of faces, edges or corners | ui/ContextChips.tsx (Note) | missing | — | — |
| NOTE-03 | Pins | Numbered pins: dot and amber badge with text label, constant pixel size, faint facing away, drawn into the canvas for pictures | viewer/notePins.ts | missing | — | — |
| NOTE-04 | List | Notes list: rows, open, done, delete, Delete all (numbers carry on) | ui/NoteList.tsx | missing | — | — |
| NOTE-05 | Targets | A note keeps its pin, normal and targets by fingerprint; found again on the bodies (first body, then others, joins); gone reported; nearest body face | core/notes.ts; app/reNoteCommands.ts | partial | crates/mm-reverse/src/model.rs (Note, NoteTarget) | Data only |
| NOTE-06 | Agent | Agent questions: blue pin, “The agent asks”, an answer field written into reply | ui/NoteList.tsx (note-card-question, note-card-reply) | missing | — | — |
| NOTE-07 | Agent | Agent resolves a note with a line shown under it; pin goes grey | app/reNoteCommands.ts | missing | — | — |
| NOTE-08 | Persistence | Notes belong to the scan by file name, set aside for another scan, follow alignment, saved under reverseNotes; a project brings its own | app/notesSection.ts | partial | crates/mm-reverse/src/model.rs (Part.notes) | Saved with the part; no per-scan keying |
| NOTE-09 | Visibility | Notes only while the agent link is on; pins in every workspace with the 3D view, clicks only here | app/useNotePins.ts | missing | — | — |
| NOTE-10 | Review | Design-loop review: volume error and scan comparison at T merged into places ranked by wrong material, with the plan's verdict and next place | app/reReview.ts; core/reviewPlaces.ts | missing | — | — |
| NOTE-11 | Review | Review pins in the viewport for its picture | viewer/reviewPins.ts | missing | — | — |
| NOTE-12 | Plan | The plan: archetype, region inventory (function, route, place), ticks/skips, replaced removals, plan gate on commands | app/rePlan.ts | missing | — | — |

## AGENT

**Agent commands** — 44 rows (done 0, partial 6, missing 38).

Every reverse.* command an agent models through, the session readout, guards, answers, schemas and guides. Polymix's reverse workspace has no agent tools yet (agents.rs says so); only test script steps exist.

- **Depends on:** BUILD, HIST, PARAM, FIT, CMP, EXT, LIST

| ID | Area | Capability | ScanRuler | Status | Polymix | Missing |
|---|---|---|---|---|---|---|
| AGENT-01 | Commands | reverse.parameter.list / add / set / rename / remove | app/reParameterCommands.ts | missing | — | Needs PARAM |
| AGENT-02 | Commands | reverse.feature.list (rows, status, key value, rollback bar; one feature or detail full with regions) | app/reFeatureCommands.ts (list); app/reReadout.ts | missing | — | — |
| AGENT-03 | Commands | reverse.feature.add (single or features[], after, through the panel's box path; picks by index/fingerprint/where) | app/reFeatureCommands.ts (add); app/reFeatureInput.ts; app/reSketchInput.ts | missing | — | Model JSON matches ScanRuler's shape, which helps |
| AGENT-04 | Commands | reverse.feature.schema (per-kind schema with $defs/refs and worked examples) | app/reFeatureCommands.ts (schema); app/reSchemas.ts | missing | — | — |
| AGENT-05 | Commands | reverse.feature.update (set fields, exprs, visible, suppressed; edits below the bar) | app/reFeatureCommands.ts (update) | missing | — | — |
| AGENT-06 | Commands | reverse.feature.refit (numbers made with "fit" fitted again) | app/reFeatureCommands.ts (refit); app/reFit.ts | missing | — | — |
| AGENT-07 | Commands | reverse.feature.remove | app/reFeatureCommands.ts (remove) | missing | — | — |
| AGENT-08 | Commands | reverse.feature.move | app/reFeatureCommands.ts (move) | missing | crates/mm-reverse/src/model.rs (moved_list) | — |
| AGENT-09 | Commands | reverse.feature.suppress (by kind or name) | app/reFeatureCommands.ts (suppress) | missing | — | — |
| AGENT-10 | Commands | reverse.feature.clear | app/reFeatureCommands.ts (clear) | missing | — | — |
| AGENT-11 | Commands | reverse.feature.rollback | app/reFeatureCommands.ts (rollback) | missing | — | — |
| AGENT-12 | Commands | reverse.sketch.edit (add pieces/shapes/relations/patterns/fit/project/rounds; remove by id; moved report) | app/reFeatureCommands.ts (edit); app/reSketchInput.ts | missing | — | — |
| AGENT-13 | Commands | reverse.sketch.candidates | app/reSectionCommands.ts (candidates); core/intent.ts | missing | — | Staged intent.rs |
| AGENT-14 | Commands | reverse.build (kernel up, every feature's result, where a failure was) | app/reFeatureCommands.ts (build) | missing | — | — |
| AGENT-15 | Commands | reverse.build.stop | app/reBuildStop.ts | missing | — | — |
| AGENT-16 | Commands | reverse.body.describe (faces/edges with fingerprints, index, kind, size, side; where; before) | app/reBodyCommands.ts (describe); core/where.ts | partial | crates/mm-reverse/src/topo.rs; crates/mm-reverse/src/pick_query.rs | Core present; no command |
| AGENT-17 | Commands | reverse.body.set (name, colour, visible) | app/reBodyCommands.ts (set) | missing | — | — |
| AGENT-18 | Commands | reverse.body.clearance | app/reBodyCommands.ts (clearance) | missing | — | — |
| AGENT-19 | Commands | reverse.body.compare (score, patches, far, by face/feature; within; feature) | app/reBodyCommands.ts (compare); core/compareDetail.ts | partial | crates/mm-reverse/src/compare.rs | Map only; no command |
| AGENT-20 | Commands | reverse.body.compare_in_deviation | app/reBodyCommands.ts (compare_in_deviation) | missing | — | — |
| AGENT-21 | Commands | reverse.body.volume_compare | app/reVolumeCompare.ts | missing | — | — |
| AGENT-22 | Commands | reverse.body.review | app/reReview.ts | missing | — | — |
| AGENT-23 | Commands | reverse.body.check (with audit) | app/reBodyCommands.ts (check); core/audit.ts | missing | — | — |
| AGENT-24 | Commands | reverse.body.fillet_read | app/reFilletRead.ts | missing | — | Staged ports/fillet fillet_read.rs |
| AGENT-25 | Commands | reverse.body.fillet_try (sizes tried on the body before the feature) | app/reFilletTry.ts | missing | — | — |
| AGENT-26 | Commands | reverse.export.step (bytes and name) | app/reBodyCommands.ts (step) | partial | crates/mm-reverse/src/build/mod.rs (export_step) | Core present; no command |
| AGENT-27 | Commands | reverse.export.stl (binary STL to a chord, or 3MF) | app/reBodyCommands.ts (stl) | missing | — | — |
| AGENT-28 | Commands | reverse.import.step | app/reFeatureCommands.ts (import.step) | missing | crates/mm-reverse/src/build/import.rs | — |
| AGENT-29 | Commands | reverse.plane.describe (origin, normal, u, v in numbers and words; a point converted) | app/reSectionCommands.ts (describe) | partial | crates/mm-reverse/src/build/mod.rs (describe_plane_ref); crates/mm-reverse/src/frame.rs | Words and frames present; no command |
| AGENT-30 | Commands | reverse.section.fit (chains as lines/arcs with residual, uncertainty, nominal and candidates; chain filters) | app/reSectionCommands.ts (fit) | partial | crates/mm-reverse/src/section.rs; crates/mm-reverse/src/segment.rs | Fit core present; no evidence, no command |
| AGENT-31 | Commands | reverse.section.stack (several offsets, trends: tilt, draft, taper, round radius) | app/reSectionCommands.ts (stack) | missing | — | — |
| AGENT-32 | Commands | reverse.section.outline (one chain as a polyline shape) | app/reSectionCommands.ts (outline); core/chainOutline.ts | missing | — | Staged chain_outline.rs |
| AGENT-33 | Commands | reverse.scan.tube_trace | app/reSectionCommands.ts (tube_trace); core/tubeTrace.ts | missing | — | Staged tube_trace.rs |
| AGENT-34 | Commands | reverse.thread.read | app/reThreadCommand.ts | missing | — | Staged thread_read.rs |
| AGENT-35 | Commands | reverse.note.list / add / resolve | app/reNoteCommands.ts | missing | — | — |
| AGENT-36 | Commands | reverse.network.create / edit / read | app/reNetworkCommands.ts | missing | — | — |
| AGENT-37 | Commands | reverse.network.suggest_creases | app/reNetworkCommands.ts | missing | — | — |
| AGENT-38 | Commands | reverse.plan.set / update / tick / replace | app/rePlan.ts | missing | — | — |
| AGENT-39 | Readout | session.state carries the workspace: kernel, build, parameters with values, features with status, bodies with boxes, notes waiting | app/reReadout.ts | partial | crates/polymix/src/shell.rs (agent_status) | Only running/busy per workspace |
| AGENT-40 | Guards | Commands refused while a box holds work or the kernel builds; each command one named undo step (Agent: add thread); boxes it opened shut; build awaited | app/reCommandCore.ts | missing | — | — |
| AGENT-41 | Answers | Answers at detail none/brief/full: the feature row, a sketch's regions by a point, bodies made/changed/removed, open ends of an unclosed sketch | app/reCommandAnswers.ts | missing | — | — |
| AGENT-42 | Headless | Commands that run headless (no UI) for scripted clients | app/reCommands.ts (HEADLESS) | missing | — | — |
| AGENT-43 | Pictures | view.render / view.set take scan and bodies in show while the workspace is on screen | app/useReversePlugin.tsx (registerViewToggles) | missing | — | — |
| AGENT-44 | Guides | Modelling guidance for agents: scan-to-cad guide, model reference, the cad-stl-reverse-engineering skill | guides/*.md; guides/guides.ts; skills/cad-stl-reverse-engineering | missing | — | — |
