# Polymix roadmap

The work queue for reaching (and passing) Meshmixer parity. Items are worked **in order**,
top to bottom; each is ticked in the commit that finishes it. Status and notes for every
Meshmixer feature live in [feature_parity.md](feature_parity.md).

Order: first what every session touches (navigation, input, hotkeys), then the daily
selection → repair workflow, then deformation, analysis, display, files, sculpting, and
modeling extras.

**Out of scope — 3D printing** (modern slicers do this better): Print / printers / printer bed,
Overhangs & supports, Orientation, Layout/Packing, Slicing, Make Slices, Stability, Strength,
Thickness, Hollow escape holes, Export SVG. The Inspector (Analysis) stays — it is done.

---

## M1 — Navigation, input and hotkeys

- [ ] **M1.1 Navigation modes.** Preference *Fusion 360* (MMB pan, Shift+MMB tumble, wheel zoom;
      the user's Meshmixer setting, default) / *Meshmixer* (RMB or Alt+LMB tumble, Alt+MMB /
      Shift+RMB / Alt+Shift+LMB pan, Alt+RMB and Alt+⌘+LMB zoom). Behaviour checked against
      the live Meshmixer.
- [ ] **M1.2 3D mouse (3Dconnexion SpaceMouse).** 6-DoF camera navigation (pan/zoom/tumble
      around the view's axes, matching the free-rotation setting), device buttons (Fit, views),
      sensitivity and axis-invert settings; macOS first via the 3Dconnexion driver, with a
      device abstraction for Windows/Linux later.
- [ ] **M1.3 Brush size on the mouse wheel** in Select/Sculpt where Meshmixer does it (per
      navigation mode), RMB-drag expand/contract of the selection.
- [ ] **M1.4 Hotkeys.** `A` accept; Alt+←/→ undo/redo; Alt+Shift+←/→ undo/redo skipping tool
      changes; ⌘⇧O object browser; selection: `B` Smooth Boundary, ⌘F Smooth, `Y` Separate,
      `J` Join, `P` Fit Primitive, ⌘B Bridge (bound as the tools land); Remesh/Reduce in-tool
      D/F density ±10, T adaptive, G preserve groups.
- [ ] **M1.5 View commands.** Recenter at cursor, Previous/Next view, Show grid.

## M2 — Selection

- [ ] **M2.1 Smooth Boundary** (Smoothness, Preserve Shape, Iterations, Border Rings, Preserve
      Mesh Boundary, Preserve Group Borders, Create New Groups).
- [ ] **M2.2 Erase & Fill parity**: Flat Minimal / Flat Remeshed / Smooth MVC, Refine, Smooth,
      Scale, Bulge, Create New Groups.
- [ ] **M2.3 Modify menu**: Select Visible, Expand to Same Groups (non-connected).
- [ ] **M2.4 Brush modes**: Sphere / SphereDisc / Unwrap (geodesic) brush; Allow Back Faces and
      Restrict To Exterior Faces for the brush; Symmetry plane.
- [ ] **M2.5 Expand modes**: Geodesic Distance / Crease Angle / Vertex Color with Crease Angle
      Threshold (drives RMB-drag expand and brush spread).
- [ ] **M2.6 Lasso Mode**: multi-curve loop drawn on the surface, rotating between curves.
- [ ] **M2.7 Extract** (Shift+D) and Extrude/Offset parity (Harden, Density, Direction Normal /
      Constant / Axis, EndType Offset / Flat, Soft Transition, Connected, Preserve Boundary /
      Groups, Accuracy, Regularity).
- [ ] **M2.8 Magic wand** selection.

## M3 — Repair and topology

- [ ] **M3.1 Close Cracks** (weld coincident open edges).
- [ ] **M3.2 Weld Boundaries** and **Join** (selection boundaries).
- [ ] **M3.3 Bridge between two selected regions** (Meshmixer style) and **Tube Handle**.
- [ ] **M3.4 Generate Face Groups** (Edge Angle mode, Angle / Size thresholds, …).
- [ ] **M3.5 Remesh parity**: Linear Subdivision, Regularity, Transition, Threshold.
- [ ] **M3.6 Reduce / Smooth parity**: Uniform reduce; Uniform Triangles, Preserve Group Borders,
      Per-Group Membranes smoothing; Smoothing Scale, Constraint Rings.
- [ ] **M3.7 Plane Cut parity**: Fixed Fill, Slice Groups.
- [ ] **M3.8 Make Solid parity**: Sharp Edge Preserve, Offset Distance, colour/group transfer,
      Cull Edges Threshold.
- [ ] **M3.9 Exact mesh booleans** (Precise / Max Quality) alongside the volumetric one.

## M4 — Transform, deformation and targets

- [ ] **M4.1 Transform parity**: World / Local frame, snapping, absolute coords/sizes.
- [ ] **M4.2 Transform selection** (Face Transform, T with a selection).
- [ ] **M4.3 Soft Transform** (Shift+T) and **Warp**.
- [ ] **M4.4 Align** (source/destination picking: base point, surface, world origin/axes).
- [ ] **M4.5 Target surface**: Set/Clear Target (menu + object browser), Transparent Target,
      Align To Target, Attract To Target, "Project to Target" options.
- [ ] **M4.6 Fit Primitive** (Square, Rectangle, Cylinder, Spheres, Disc, Linear Sweep, Convex
      Hull, Height Field).
- [ ] **M4.7 Add Tube** (spline tube between points; vessel bypass/repair).

## M5 — Analysis and measurement

- [ ] **M5.1 Measure parity**: point coordinates, normal, axis X/Y/Z, min/max interior/exterior,
      fit cylinder, snapping (vertices / edges / group borders), set target length.
- [ ] **M5.2 Mesh Query**: per-triangle/vertex info, curvature and normal colour modes.
- [ ] **M5.3 Units/Dimensions from measurements** (set scale from a measured length).
- [ ] **M5.4 Clearance** (minimum distance between objects).

## M6 — Display and shading

- [ ] **M6.1 Normal modes**: Vertex / Face / Group normals (Preferences + Hotbox).
- [ ] **M6.2 Vertex colours**: display, colour mode (Group / Vertex / Constant), OBJ/PLY export.
- [ ] **M6.3 Hotbox parity**: camera drag buttons, background colour, brush Size/Strength.
- [ ] **M6.4 Material library** (clay variants, glossy, metal, resin).
- [ ] **M6.5 View cube** (own design) with clickable faces/edges.
- [ ] **M6.6 Meshmixer-style light theme** and an original icon set for the palette and tools.

## M7 — Files and application shell

- [ ] **M7.1 Preferences window** (navigation, colour/normal modes, view, file and dialog
      options).
- [ ] **M7.2 Flip Z-Y axis on import/export** (on by default, as in the user's install).
- [ ] **M7.3 Import prompts**: Append / Replace, auto-fix scale / position of appended meshes,
      "remember my choice".
- [ ] **M7.4 Project files**: save/open a scene (objects, groups, patch names, view, history-free).
- [ ] **M7.5 Autosave and crash recovery.**
- [ ] **M7.6 Formats**: AMF import/export; COLLADA, VRML, SMESH export; export selection only;
      Import Bunny / Sphere / Plane.
- [ ] **M7.7 History**: Discard Undo History, Copy.

## M8 — Sculpting

- [ ] **M8.1 Brush set**: Draw, Draw2, Draw3, DrawMax, Move, Push, Pinch, Spikes, Nudge,
      ZipperEdge, Attract, BubbleSmooth, ShrinkSmooth.
- [ ] **M8.2 Dynamic refinement** while sculpting (Refine, Reduce, Smooth, Adaptivity; R / T).
- [ ] **M8.3 Volume vs Surface brushes**, Depth, Lazyness, Flow, Volumetric, falloff curves,
      secondary brush (Shift), toggles Shift+S / B / G / A, hotkeys 1 / 2 / 3.
- [ ] **M8.4 Mesh density brushes**: Refine, Reduce, AdaptiveReduce.
- [ ] **M8.5 Symmetry and SymSnap.**

## M9 — Modeling extras

- [ ] **M9.1 Stamp tool** (polygon stencils).
- [ ] **M9.2 Parts library** (Meshmix) and Drop Part; Convert To Open/Solid Part, Stamp.
- [ ] **M9.3 Make Pattern.**
- [ ] **M9.4 Vertex painting** (PaintVertex, SmoothColor).
- [ ] **M9.5 Unwrap / UV layout**, Generate Complex, Hair, Pivots.
