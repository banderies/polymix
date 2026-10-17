# Meshmixer 3.5 → Polymix feature parity

Every user-facing feature found in Meshmixer 3.5 (macOS), with its status in Polymix.
Sources: Meshmixer 3.5's menus, tool panels, help pages and hotkey sheet, as seen in the
running application.

**Status:** ✅ done · 🟡 partial (see notes) · ❌ missing · ⚪ low priority (art, parts library) ·
⛔ out of scope (3D printing — modern slicers cover it; the Inspector stays).

The ordered work queue is [ROADMAP.md](ROADMAP.md).

---

## 1. Application shell

### Left palette (Meshmixer order)

| Meshmixer | Polymix | Notes |
|---|---|---|
| Import | ✅ | Polymix palette: Import · Select · Sculpt · Edit · Vessel · Analysis · Export |
| Meshmix (parts library: drop parts) | ⚪ ❌ | parts in Arms/Ears/Heads/… |
| Select | ✅ | |
| Sculpt | 🟡 | see §3 |
| Stamp | ⚪ ❌ | |
| Edit | 🟡 | see §4 |
| Analysis | 🟡 | see §5 |
| Shaders (materials) | 🟡 | see §6 |
| Export | ✅ | |
| Print (3D print prep) | ⛔ | |
| Large icon buttons with labels (64/128 px, `*_center`, `*_highres`) | ❌ | Polymix uses text buttons; draw our own icons |

### File menu

| Meshmixer | Polymix | Notes |
|---|---|---|
| Open / Save / Save As scene (`.mix`) | ❌ | No project file yet (objects, groups, patch names, view) |
| Import (OBJ, PLY, STL, AMF, 3MF, OFF, MIX) | 🟡 | OBJ, PLY, STL, OFF, 3MF ✅; AMF, MIX ❌ |
| Import → *append or replace* prompt | ❌ | Polymix always appends |
| Auto-fix prompt when an appended mesh is much smaller/larger/far away | ❌ | |
| Import Bunny / Sphere / Plane | ❌ | Primitives exist in `mm_core::primitives`, not exposed |
| Import Reference / Import Parts Folder | ⚪ ❌ | |
| Recent files | ✅ | |
| Drag-and-drop import | ✅ | |
| Export: OBJ, STL ASCII/binary, PLY, 3MF | ✅ | plus OFF |
| Export: OBJ with per-vertex colour | ❌ | Colours are read on import but not shown or written |
| Export: "Sequence, one file per region" (OBJ/STL) | 🟡 | One STL per face group via Boundary patches |
| Export: AMF, COLLADA (.dae), VRML, SMESH | ❌ | |
| Export SVG | ⛔ | |
| Print / Send to printer | ⛔ | |
| Preferences (Alt+T / ⌘,) | 🟡 | View options persist; no Preferences window |
| Pref: Flip Z-Y axis on import/export (on in your install) | ❌ | Polymix has a Z-up *view* but does not convert file axes |
| Pref: Mesh Normal Mode Vertex / Face / Group normals | 🟡 | Smooth (+ sharp edges) and flat ✅; group normals ❌ |
| Pref: Mesh Color Mode Group / Vertex / Constant | 🟡 | Group and constant ✅; vertex ❌ |
| Pref: Export selection only vs whole scene; OBJ face groups ↔ groups | 🟡 | Export active object or all visible; OBJ groups ✅ |
| Pref: Import dialogs (append/replace, scale models) | ❌ | |
| Autosave + crash recovery | ❌ | |

### Actions menu

| Meshmixer | Polymix | Notes |
|---|---|---|
| Undo / Redo (⌘Z, ⇧⌘Z, Ctrl+Y, Alt+←/→) | ✅ | |
| Undo/Redo (Skip Tools) Alt+Shift+←/→ | ✅ | Same as undo/redo: Polymix tools add no history entries of their own |
| Discard Undo History | ❌ | |
| Accept (Enter **or A**) / Cancel (Esc) | ✅ | |
| Switching tools accepts the current tool | ✅ | Hotkey, palette, menu; only Esc/Cancel discards |
| Set as Target / Clear Target (target surface) | ❌ | Used by Attract/Align to Target, "Project to target" options |
| Copy | ❌ | |
| Toggle visibility / Show all | ✅ | Object browser |

### View menu

| Meshmixer | Polymix | Notes |
|---|---|---|
| Recenter View | ✅ | F / Focus |
| Recenter At Cursor | ✅ | Alt+click, or View menu |
| Previous View / Next View | ✅ | ⌘[ / ⌘] |
| Orthographic View | ✅ | |
| Show Grid | ✅ | Floor grid under the scene |
| Show Printer Bed | ⛔ | |
| Transparent Target | ❌ | |
| Show Wireframe | ✅ | W |
| Show Boundaries | ✅ | |
| Show Objects Browser (Ctrl+Shift+O) | ✅ | |
| — | ✅ | Polymix extras: Sharp edges (crease-angle shading), face group colours, Y/Z up, standard views, statistics |

### Navigation (hotkeys.pdf page 1)

| Meshmixer | Polymix | Notes |
|---|---|---|
| Tumble: RMB or Alt+LMB | ✅ | Free trackball (Hotbox "Free" / View → Free rotation), or turntable |
| Pan: Alt+MMB, Shift+RMB, Alt+Shift+LMB | ✅ | plus MMB |
| Zoom: Alt+RMB (Maya) | ✅ | In Meshmixer navigation mode |
| Zoom: Alt+Ctrl/⌘+LMB (pen-friendly) | ✅ | |
| Wheel zooms (outside brush tools) | ✅ | Zooms toward the cursor |
| "Fusion 360" navigation mode (Shift+MMB tumble, MMB pan, wheel zoom) | ✅ | View → Navigation; default, as in your Meshmixer |
| ViewCube / SteeringWheels | ⚪ ❌ | Autodesk components; Polymix has an axis gizmo |
| 3D mouse (3Dconnexion) | ✅ | Polymix addition: View → 3D mouse… (macOS) |

### Hotbox (Space) — hotkeys.pdf page 5

| Meshmixer | Polymix | Notes |
|---|---|---|
| Camera: Snap, Free | ✅ | |
| Camera: Tumble / Pan / Zoom drag buttons | ❌ | |
| Camera: Focus | ✅ | |
| Camera: background Color | ❌ | |
| Mesh → Normals: smooth / faceted / face-group normals | 🟡 | Smooth, Faceted ✅; per-group normals ❌ |
| Mesh → Color: vertex colours / face-group colours | 🟡 | Group colours ✅; vertex colours ❌ |
| Brushes: Size, Strength sliders | ❌ | |
| — | ✅ | Polymix extras: Y-up / Z-up |

### Object browser — hotkeys.pdf page 4

| Meshmixer | Polymix | Notes |
|---|---|---|
| List, active object, visibility eye | ✅ | |
| Set/clear target per object | ❌ | |
| Duplicate, Delete buttons | ✅ | |
| Rename | ✅ | |
| Show only this / Show all | ✅ | |
| Pivots (show/hide) | ⚪ ❌ | |

---

## 2. Select tool (S)

| Meshmixer | Polymix | Notes |
|---|---|---|
| Paint Mode: sphere brush, Shift deselect | ✅ | Brush keeps its size on the model |
| Brush types: Sphere, SphereDisc, Unwrap (geodesic) | ✅ | Unwrap is the default, as in your Meshmixer |
| **Mouse wheel resizes the brush** (also `[` `]`) | ✅ | Wheel in Meshmixer navigation, Shift+wheel in Fusion 360 navigation |
| RMB-drag expands / contracts the selection | ✅ | With a selection; geodesic distance (other expand modes: ROADMAP M2.5) |
| Laser Lasso (curve from background; closed = inside, open = smaller side, through the model) | ✅ | |
| Lasso Mode (multi-curve loop drawn *on the surface*, rotate between curves) | ❌ | |
| Double-click: connected part | ✅ | |
| Double-click an open-edge triangle: hole-border ring | ✅ | Polymix addition (requested) |
| S inside Select clears the selection | ✅ | Polymix addition (requested) |
| Expand Mode: Geodesic / Crease Angle / Vertex Color + Crease Angle Thresh | 🟡 | Geodesic, Crease Angle and the threshold ✅ (brush and growth stop at creases); Vertex Color with M6.2 |
| Allow Back Faces | ✅ | Brush filter; the lasso has "selects through the model" |
| Restrict To Exterior Faces | ✅ | Winding-number test: skips faces buried inside other parts |
| Symmetry plane (Set / Reset) | 🟡 | Mirror across the object's centre on X/Y/Z; free plane placement ❌ |
| Magic wand | ❌ | |
| **Modify Selection** menu: Select All, Expand/Contract Ring, Optimize Boundary, Invert, Invert (Connected), Expand to Connected, Expand to Groups | ✅ | Ctrl+A, < >, O, I, Shift+I, E, G |
| Modify: Expand to Same Groups (non-connected) | ✅ | "Same groups"; `G` now expands through connected faces only, as in Meshmixer |
| Modify: Select Visible | ✅ | |
| **Edit** menu: Discard (X) | ✅ | |
| Edit: Erase & Fill (F) | ✅ | Flat Minimal / Flat Remeshed / Smooth MVC, Refine, Smooth, Scale, Bulge, Create New Groups (Meshmixer's defaults) |
| Edit: Flip Normals | ✅ | |
| Edit: Create / Clear FaceGroup (Ctrl+G / Ctrl+Shift+G) | ✅ | |
| Edit: Extrude (D) | ✅ | |
| Edit: Extract (Shift+D) — offset copy into a new object | ❌ | |
| Edit: Offset (Ctrl+D) | 🟡 | Distance ✅; Soft Transition, Connected ❌ |
| Edit: Remesh (R) / Reduce (Shift+R) on the selection | ✅ | In-tool D/F density ±10, T adaptive (Remesh), G preserve groups |
| Edit: Bridge (two selected regions) | 🟡 | Polymix bridges two boundary loops |
| Edit: Join (along boundary edges) | ❌ |  |
| Edit: Weld Boundaries | ❌ |  |
| Edit: Tube Handle | ❌ | Connect two selected regions with a tube |
| Edit: Align To Target / Attract To Target | ❌ | Needs target surface |
| Edit: Separate (Y) / Duplicate selection | ✅ | |
| Edit menu hotkeys Join (J), Fit Primitive (P), Bridge (⌘B) | 🟡 | ⌘B ✅; J and P with their tools |
| Edit: Fit Primitive | ❌ | |
| Edit: Grow Hair | ⚪ ❌ | |
| **Deform** menu: Smooth (⌘F) | ✅ | |
| Deform: Smooth Boundary ("Smooth Loop", B) | ✅ | Meshmixer's parameters and defaults; remesher smooths the border curve |
| Deform: Soft Transform (Shift+T) / Transform selection (T) | ❌ | Polymix Transform is whole-object only |
| Deform: Warp | ❌ | |
| **Convert To**: Open Part / Solid Part / Stamp | ⚪ ❌ | |

## 3. Sculpt

| Meshmixer | Polymix | Notes |
|---|---|---|
| Volume brushes (3) vs Surface brushes (2), Smoothing brush (1) | 🟡 | One brush tool, surface-style |
| Brushes: RobustSmooth, ShrinkSmooth (Smooth), BubbleSmooth, Inflate, Flatten, Drag | 🟡 | RobustSmooth, Smooth, Inflate, Flatten, Drag ✅; BubbleSmooth ❌ |
| Brushes: Draw, Draw2, Draw3, DrawMax, Move, Push, Pinch, Spikes, Nudge, Zipper, Attract | ❌ | |
| Brushes: Refine, Reduce, AdaptiveReduce (mesh density brushes) | ❌ | |
| Brushes: PaintVertex, SmoothColor | ⚪ ❌ | |
| Dynamic refinement while sculpting (Refinement, Adaptivity; R, T) | ❌ | |
| Size / Strength | ✅ | Size is a % of the model; wheel size, right-drag strength |
| Spacing, Flow, Lazyness, AlignToStroke, Falloff curve, Stencils, brush textures | ❌ | |
| Primary / Secondary (Shift) / Invert (Ctrl/⌘) | 🟡 | Invert ✅ for Inflate; Secondary brush ❌ |
| Toggles: Symmetry (Shift+S), Hold boundary (B), Preserve groups (G), Attract (A) | 🟡 | Hold boundary checkbox ✅ (no hotkey) |

## 4. Edit category

| Meshmixer | Polymix | Notes |
|---|---|---|
| Mirror | ✅ | Weld at the plane |
| Duplicate | ✅ | |
| Transform (T) | 🟡 | Gizmo, numeric, center, drop to ground ✅; World/Local frame, snapping ❌ |
| Align (N) — align surfaces/axes to world | 🟡 | Principal-axes alignment only |
| Create Pivot | ⚪ ❌ | |
| Plane Cut | 🟡 | Cut/Slice, No/Minimal/Remeshed fill ✅ (+ vessel cross-section mode); Fixed Fill, Slice Groups ❌ |
| Combine | ✅ | |
| Boolean Union / Difference / Intersection | 🟡 | Volumetric (SDF) result; exact mesh boolean ("Precise", "Max Quality") ❌ |
| Close Cracks | ❌ | Weld near-coincident open edges |
| Generate Face Groups | ❌ | Segment by crease angle / region |
| Make Solid | 🟡 | Blocky / Fast / Accurate, accuracy, density, min thickness, close boundaries ✅; Sharp Edge Preserve, offset, colour/group transfer ❌ |
| Hollow | ✅ | Hollow wall ✅; escape holes ⛔ (printing) |
| Make Pattern | ⚪ ❌ | |
| Separate Shells (Shift+Y) | ✅ | |
| Add Tube | ❌ | Possibly useful for vessel bypass/repair |
| Unwrap / UV layout / SetUV | ⚪ ❌ | |
| Generate / Filter / Split Complex, Convex Decomposition | ⚪ ❌ | |
| Make Slices | ⛔ | |
| Remesh (whole object) | 🟡 | Relative/Adaptive density, target edge length, Fixed/Refined/Free boundary, sharp edges, groups ✅; Linear Subdivision, Regularity, Transition ❌ |
| Reduce | 🟡 | Percentage / triangle budget / max deviation ✅; Uniform vs Shape Preserving type ❌ |
| Smooth (whole object) | 🟡 | Shape Preserving / Membrane / Max Smoothness ✅; Uniform Triangles, Per-Group Membranes ❌ |

## 5. Analysis category

| Meshmixer | Polymix | Notes |
|---|---|---|
| Inspector (I, Q = Auto Repair All) | ✅ | Minimal/Flat/Smooth fill, small-part threshold, click markers; + keep flat vessel ends |
| Units / Dimensions | ✅ | |
| Measure | 🟡 | Distance, cross-section diameter ✅; point coords, normal, axis X/Y/Z, min/max interior/exterior, fit cylinder, snapping ❌ |
| Mesh Query | 🟡 | Statistics panel (counts, size, area, volume, open edges) |
| Thickness (identify / thicken thin regions) | ⛔ | 3D printing wall check |
| Deviation | ✅ | Signed colour map + statistics |
| Clearance | ❌ | |
| Stability, Strength, Orientation, Overhangs/Support, Slicing, Layout/Packing | ⛔ | 3D printing |
| Mass properties | 🟡 | Area, volume (centre of mass ❌) |

## 6. Shaders / display

| Meshmixer | Polymix | Notes |
|---|---|---|
| Material library (clay variants, metals, glossy, resin…) | 🟡 | One clay material + per-object colour |
| Vertex colour display | ❌ | |
| Textured / textured-unlit / UV checker / X-ray | ⚪ ❌ | |
| Wireframe | ✅ | |
| Face group colours | ✅ | |
| Light UI theme (warm greys: panels `#ede8e4`, text `#6b6864`, hover `#a5a19d`/`#b3afab`, scrollbar `#888480`, dark text `#444240`, disabled `#d0cac7`) | ❌ | Polymix is dark-themed; a "Meshmixer light" theme is easy to add |

## 7. Polymix features Meshmixer does not have

Trim branch (square vessel cut + cap) · Flow extensions · Cap open ends · Centerline with
branch lengths/diameters (CSV/VTK) · Check for CFD (incl. self-intersections, triangle
quality) · Boundary patches + multi-solid STL · Measure cross-section diameter ·
crease-angle sharp-edge shading · laser selection on 1M+ triangle meshes · Z-up (RAS) mode ·
free/turntable camera toggle.

