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

- [x] **M1.1 Navigation modes.** Preference *Fusion 360* (MMB pan, Shift+MMB tumble, wheel zoom;
      the user's Meshmixer setting, default) / *Meshmixer* (RMB or Alt+LMB tumble, Alt+MMB /
      Shift+RMB / Alt+Shift+LMB pan, Alt+RMB and Alt+⌘+LMB zoom). View → Navigation; mapping per
      Meshmixer's hotkeys.pdf (live check pending: Meshmixer was closed).
- [x] **M1.2 3D mouse (3Dconnexion SpaceMouse).** 6-DoF camera navigation (pan/zoom/tumble
      around the view's axes, matching the free-rotation setting), device buttons (Fit, views),
      sensitivity and axis-invert settings; macOS first via the 3Dconnexion driver, with a
      device abstraction for Windows/Linux later. Done on macOS (View → 3D mouse…,
      `polymix --probe-3dmouse` checks the axis mapping on a device); Windows (3DxWare) and
      Linux (spacenavd) backends still to add.
- [x] **M1.3 Brush size on the mouse wheel** in Select/Sculpt where Meshmixer does it (per
      navigation mode), RMB-drag expand/contract of the selection. Wheel (Meshmixer nav) or
      Shift+wheel (Fusion 360 nav); right-drag grows/shrinks by surface distance in Select and
      sets strength in Sculpt.
- [x] **M1.4 Hotkeys.** `A` accept; Alt+←/→ undo/redo; Alt+Shift+←/→ undo/redo skipping tool
      changes; ⌘⇧O object browser; selection: `B` Smooth Boundary, ⌘F Smooth, `Y` Separate,
      `J` Join, `P` Fit Primitive, ⌘B Bridge (bound as the tools land); Remesh/Reduce in-tool
      D/F density ±10, T adaptive, G preserve groups. Done except `P` (M4.6).
- [x] **M1.5 View commands.** Recenter at cursor (Alt+click / View menu), Previous/Next view (⌘[ ⌘]),
      Show grid.

## M2 — Selection

- [x] **M2.1 Smooth Boundary** (Smoothness, Preserve Shape, Iterations, Border Rings, Preserve
      Mesh Boundary, Preserve Group Borders, Create New Groups).
- [x] **M2.2 Erase & Fill parity**: Flat Minimal / Flat Remeshed / Smooth MVC, Refine, Smooth,
      Scale, Bulge, Create New Groups.
- [x] **M2.3 Modify menu**: Select Visible, Expand to Same Groups (non-connected).
- [x] **M2.4 Brush modes**: Sphere / SphereDisc / Unwrap (geodesic) brush; Allow Back Faces and
      Restrict To Exterior Faces for the brush; Symmetry plane (mirror across the object's centre on
      X, Y or Z; arbitrary planes later).
- [x] **M2.5 Expand modes**: Geodesic Distance / Crease Angle / Vertex Color with Crease Angle
      Threshold (drives RMB-drag expand and brush spread). Vertex Color mode comes with vertex
      colours (M6.2).
- [x] **M2.6 Lasso Mode**: multi-curve loop drawn on the surface, rotating between curves.
- [x] **M2.7 Extract** (Shift+D) and Extrude/Offset parity (Harden, Density, Direction Normal /
      Constant / Axis, EndType Offset / Flat, Soft Transition, Connected, Preserve Boundary /
      Groups, Accuracy, Regularity). Offset's Connected / Accuracy / Preserve Groups wait for a
      live check of their meaning in Meshmixer.
- [x] **M2.8 Magic wand** selection (Select → Wand: click a face, angle threshold).

## M3 — Repair and topology

- [x] **M3.1 Close Cracks** (weld coincident open edges, T-junctions; Edit → Close Cracks).
- [x] **M3.2 Weld Boundaries** and **Join** (selection boundaries; `J`).
- [x] **M3.3 Bridge between two selected regions** (Meshmixer style, ⌘B) and **Tube Handle**.
- [x] **M3.4 Generate Face Groups** (Edge Angle / Normal Angle, Angle / Size thresholds).
- [x] **M3.5 Remesh parity**: Linear Subdivision, Regularity, Transition, Threshold, Meshmixer density
      scale and defaults, Preserve vs Smooth group borders.
- [x] **M3.6 Reduce / Smooth parity**: Uniform reduce; Uniform Triangles, Preserve Group Borders,
      Per-Group Membranes smoothing; Smoothing Scale, Constraint Rings.
- [x] **M3.7 Plane Cut parity**: Fixed Fill, Slice Groups.
- [x] **M3.8 Make Solid parity**: Sharp Edge Preserve, Offset Distance, group transfer,
      Cull Edges Threshold. (Colour Transfer Mode needs vertex colours and moves to M6.2.)
- [x] **M3.9 Exact mesh booleans** (Precise / Max Quality) alongside the volumetric one.

## M4 — Transform, deformation and targets

- [x] **M4.1 Transform parity**: World / Local frame, snapping, absolute coords/sizes.
- [x] **M4.2 Transform selection** (Face Transform, T with a selection).
- [x] **M4.3 Soft Transform** (Shift+T) and **Warp**.
- [x] **M4.4 Align** (source/destination picking: base point, surface, world origin/axes).
- [x] **M4.5 Target surface**: Set/Clear Target (menu + object browser), Transparent Target,
      Align To Target, Attract To Target, "Project to Target" options.
- [x] **M4.6 Fit Primitive** (Square, Rectangle, Cylinder, Spheres, Disc, Linear Sweep, Convex
      Hull, Height Field).
- [x] **M4.7 Add Tube** (spline tube between points; vessel bypass/repair).

## M5 — Analysis and measurement

- [x] **M5.1 Measure parity**: point coordinates, normal, axis X/Y/Z, min/max interior/exterior,
      fit cylinder, snapping (vertices / edges / group borders), set target length.
- [x] **M5.2 Mesh Query**: per-triangle/vertex info, curvature and normal colour modes.
- [x] **M5.3 Units/Dimensions from measurements** (set scale from a measured length).
- [x] **M5.4 Clearance** (minimum distance between objects).

## M6 — Display and shading

- [x] **M6.1 Normal modes**: Vertex / Face / Group normals (Preferences + Hotbox).
- [x] **M6.2 Vertex colours**: display, colour mode (Group / Vertex / Constant), OBJ/PLY export;
      Make Solid's Color Transfer Mode (No Colors / Vertex-to-Vertex / Material-to-Vertex).
- [x] **M6.3 Hotbox parity**: camera drag buttons, background colour, brush Size/Strength.
- [x] **M6.4 Material library** (clay variants, glossy, metal, resin).
- [x] **M6.5 View cube** (own design) with clickable faces/edges.
- [x] **M6.6 Meshmixer-style light theme** and an original icon set for the palette and tools.

## M7 — Files and application shell

- [x] **M7.1 Preferences window** (navigation, colour/normal modes, view, file and dialog
      options).
- [x] **M7.2 Flip Z-Y axis on import/export** (on by default, as in the user's install).
- [x] **M7.3 Import prompts**: Append / Replace, auto-fix scale / position of appended meshes,
      "remember my choice".
- [x] **M7.4 Project files**: save/open a scene (objects, groups, patch names, view, history-free).
- [x] **M7.5 Autosave and crash recovery.**
- [x] **M7.6 Formats**: AMF import/export; COLLADA, VRML, SMESH export; export selection only;
      Import Bunny / Sphere / Plane.
- [x] **M7.7 History**: Discard Undo History, Copy.

## M8 — Sculpting

- [x] **M8.1 Brush set**: Draw, Draw2, Draw3, DrawMax, Move, Push, Pinch, Spikes, Nudge,
      ZipperEdge, Attract, BubbleSmooth, ShrinkSmooth.
- [x] **M8.2 Dynamic refinement** while sculpting (Refine, Reduce, Smooth, Adaptivity; R / T).
- [x] **M8.3 Volume vs Surface brushes**, Depth, Lazyness, Flow, Volumetric, falloff curves,
      secondary brush (Shift), toggles B / G / A, hotkeys 1 / 2 / 3 (Shift+S comes with M8.5).
- [x] **M8.4 Mesh density brushes**: Refine, Reduce, AdaptiveReduce.
- [x] **M8.5 Symmetry and SymSnap.**

## M9 — Modeling extras

- [x] **M9.1 Stamp tool** (polygon stencils).
- [x] **M9.2 Parts library** (Meshmix) and Drop Part; Convert To Open/Solid Part, Stamp.
- [x] **M9.3 Make Pattern.**
- [x] **M9.4 Vertex painting** (PaintVertex, SmoothColor).
- [x] **M9.5 Unwrap / UV layout**, Generate Complex, Hair, Pivots.

## M10 — Loft (join two holes with a tube)

Erase & Fill's *Loft* fill type, picked automatically when the selection is the edge rings of
two holes (double-click each open edge, then `F`).

- [x] **M10.1 Loft**: shapes Minimal / Straight / Smooth; End 1 / End 2 continuity G0
      (position) or G1 (leaves along the surrounding wall, per boundary vertex) with tension;
      twist, bulge, refine; End 1 / End 2 marked in the view; warning when the tube runs into
      the surface.
- [x] **M10.2 G2 ends**: the vessel's bend behind each opening (curvature of the ring
      centres, tighter on the inside of the bend) continued by quintic Hermite columns; G1
      directions corrected for the bend, tension 1 = circular arc. *Fair*: thin-plate or
      triharmonic (new order-3 fairing) smoothing of the tube's middle with the end rows kept
      and the row radii restored; *Round*: circular middle cross-section.
- [ ] **M10.3 One loft for every tube join**: ⌘B Tube Handle and Vessel → Bridge boundaries use
      the loft, so all three behave the same.
