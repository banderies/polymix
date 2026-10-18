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
| Meshmix (parts library: drop parts) | ⚪ 🟡 | Own generated parts — solid: Sphere, Cube, Cylinder, Cone, Capsule, Torus; open: Disc, Dome, Boss, Spike, Ring — plus My parts. Drop Part: open parts are welded into the surface (follow surface, blend width), solid parts Boolean Union / Difference / Append / New Object. Meshmixer's art library (Arms, Ears, Heads…) is Autodesk content and not included |
| Select | ✅ | |
| Sculpt | 🟡 | see §3 |
| Stamp (4) | ⚪ 🟡 | Sculpt → Stamp: polygon stencils (Circle, Square, Triangle, Hexagon, Star, Arrow, Heart) cut into the surface facing the stamp, inside → new face group or hole; rotation, Snap Dimension. Meshmixer's stencil library and Convert To Stamp ❌ |
| Edit | 🟡 | see §4 |
| Analysis | 🟡 | see §5 |
| Shaders (materials) | 🟡 | see §6 |
| Export | ✅ | |
| Print (3D print prep) | ⛔ | |
| Large icon buttons with labels (64/128 px, `*_center`, `*_highres`) | 🟡 | Palette buttons with Polymix's own drawn icons and labels (Meshmixer's artwork is not used) |

### File menu

| Meshmixer | Polymix | Notes |
|---|---|---|
| Open / Save / Save As scene (`.mix`) | ✅ | Own `.polymix` project (zip: manifest + lossless meshes): objects, face groups, vertex colours, patch names, colours, materials, frames, visibility, active object, target, view. ⌘S / ⇧⌘S, File → Open project, drag & drop. `.mix` files are not read |
| Import (OBJ, PLY, STL, AMF, 3MF, OFF, MIX) | ✅ / ⚪ | OBJ, PLY, STL, OFF, 3MF, AMF (plain or zipped, volumes → groups) ✅; Meshmixer's .mix ⚪ (Polymix projects instead) |
| Import → *append or replace* prompt | ✅ | Asked when a scene is open; Preferences: Ask / Append / Replace |
| Auto-fix prompt when an appended mesh is much smaller/larger/far away | ✅ | Models under 1 or over 10 000 units across: prompt to rescale; Preferences: Ask / Never / Always (factor) |
| Import Bunny / Sphere / Plane | ✅ / ⚪ | File → Add shape: Sphere, Plane, Cube, Cylinder; no bundled bunny |
| Import Reference / Import Parts Folder | ⚪ ❌ | |
| Recent files | ✅ | |
| Drag-and-drop import | ✅ | |
| Export: OBJ, STL ASCII/binary, PLY, 3MF | ✅ | plus OFF |
| Export: OBJ with per-vertex colour | ✅ | OBJ (`v x y z r g b`) and PLY (uchar RGB) read and write vertex colours |
| Export: "Sequence, one file per region" (OBJ/STL) | 🟡 | One STL per face group via Boundary patches |
| Export: AMF, COLLADA (.dae), VRML, SMESH | ✅ | AMF keeps groups (volumes) and colours; SMESH writes face groups as TetGen boundary markers; plus File → Export selected triangles |
| Export SVG | ⛔ | |
| Print / Send to printer | ⛔ | |
| Preferences (Alt+T / ⌘,) | ✅ | ⌘, or File → Preferences: General (navigation, colour / normal modes, up axis), View (free rotate, reset camera on scene change, view cube, object browser, grid, sharp edges, theme), File (export scope, STL ASCII, OBJ groups) |
| Pref: Flip Z-Y axis on import/export (on in your install) | ✅ | On by default; a rotation, so a round trip keeps the file's coordinates |
| Pref: Mesh Normal Mode Vertex / Face / Group normals | ✅ | View menu and hotbox; Group (default) keeps group borders crisp; Vertex still honours Sharp edges |
| Pref: Mesh Color Mode Group / Vertex / Constant | ✅ | View menu and hotbox; Vertex (default) shows vertex colours when the mesh has them |
| Pref: Export selection only vs whole scene; OBJ face groups ↔ groups | ✅ | ⌘E follows the export scope; OBJ group export and import are switchable |
| Pref: Import dialogs (append/replace, scale models) | ✅ | Append / Replace prompt (with Flip Z-Y and Remember my choice); size prompt for models under 1 or over 10 000 units (m / cm / in → mm, keep); Preferences → Dialogs. Models are never moved, so coordinates are kept |
| Autosave + crash recovery | ✅ | Changed scenes autosave in the background every 5 min (Preferences → File; 0 = off) to the user data folder; after a crash the next launch offers Recover / Discard. Running two instances at once can show a false recovery offer |

### Actions menu

| Meshmixer | Polymix | Notes |
|---|---|---|
| Undo / Redo (⌘Z, ⇧⌘Z, Ctrl+Y, Alt+←/→) | ✅ | |
| Undo/Redo (Skip Tools) Alt+Shift+←/→ | ✅ | Same as undo/redo: Polymix tools add no history entries of their own |
| Discard Undo History | ✅ | Edit menu |
| Accept (Enter **or A**) / Cancel (Esc) | ✅ | |
| Switching tools accepts the current tool | ✅ | Hotkey, palette, menu; only Esc/Cancel discards |
| Set as Target / Clear Target (target surface) | ✅ | Edit menu and object list; used by Attract / Align To Target and Project to Target |
| Copy | ✅ | ⌘C / Edit → Copy: the open tool's readout (Measure, Mesh Query, Units) or the active object's statistics as text |
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
| Transparent Target | ✅ | View menu, on by default |
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
| — | ✅ | Polymix addition: Custom navigation mode — Preferences → General → Mouse buttons picks Orbit / Pan / Zoom for each middle and right drag, plain, with Shift or with Alt (editing a preset's table switches to Custom) |
| ViewCube / SteeringWheels | ✅ / ⚪ | Own view cube: click a face, edge or corner for that view, drag to tumble (View → Show view cube); SteeringWheels not planned |
| 3D mouse (3Dconnexion) | ✅ | Polymix addition: View → 3D mouse… (macOS) |

### Hotbox (Space) — hotkeys.pdf page 5

| Meshmixer | Polymix | Notes |
|---|---|---|
| Camera: Snap, Free | ✅ | |
| Camera: Tumble / Pan / Zoom drag buttons | ✅ | |
| Camera: Focus | ✅ | |
| Camera: background Color | ✅ | Persisted; right-click the swatch to reset |
| Mesh → Normals: smooth / faceted / face-group normals | ✅ | |
| Mesh → Color: vertex colours / face-group colours | ✅ | Colours follow edits (interpolated on splits) |
| Brushes: Size, Strength sliders | ✅ | For the open brush tool (Select: size; Sculpt: size and strength) |
| — | ✅ | Polymix extras: Y-up / Z-up |

### Object browser — hotkeys.pdf page 4

| Meshmixer | Polymix | Notes |
|---|---|---|
| List, active object, visibility eye | ✅ | |
| Set/clear target per object | ✅ | Right-click an object; the target is marked ◎ |
| Duplicate, Delete buttons | ✅ | |
| Rename | ✅ | |
| Show only this / Show all | ✅ | |
| Pivots (show/hide) | ⚪ ✅ | View > Show pivots: a ball with its frame axes, drawn on top of the model |

---

## 2. Select tool (S)

| Meshmixer | Polymix | Notes |
|---|---|---|
| Paint Mode: sphere brush, Shift deselect | ✅ | Brush keeps its size on the model |
| Brush types: Sphere, SphereDisc, Unwrap (geodesic) | ✅ | Unwrap is the default, as in your Meshmixer |
| **Mouse wheel resizes the brush** (also `[` `]`) | ✅ | Shift+wheel (or Control / Option / Command, Preferences → General → Brush size scroll) in every navigation mode, one step per notch with no easing lag (the plain wheel always zooms — your preference over Meshmixer's plain wheel); also Cmd/Ctrl+middle-drag |
| RMB-drag expands / contracts the selection | ✅ | With a selection; geodesic distance (other expand modes: ROADMAP M2.5) |
| Laser Lasso (curve from background; closed = inside, open = smaller side, through the model) | ✅ | |
| Lasso Mode (multi-curve loop drawn *on the surface*, rotate between curves) | ✅ | Closes when a curve ends near the start, or Enter; Esc clears |
| Double-click: connected part | ✅ | |
| Double-click an open-edge triangle: hole-border ring | ✅ | Polymix addition (requested) |
| S inside Select clears the selection | ✅ | Polymix addition (requested) |
| Expand Mode: Geodesic / Crease Angle / Vertex Color + Crease Angle Thresh | ✅ | Brush and growth stop at creases; Vertex Color grows into similarly coloured faces |
| Allow Back Faces | ✅ | Brush filter; the lasso has "selects through the model" |
| Restrict To Exterior Faces | ✅ | Winding-number test: skips faces buried inside other parts |
| Symmetry plane (Set / Reset) | 🟡 | Mirror across the object's centre on X/Y/Z; free plane placement ❌ |
| Magic wand | ✅ | Select → Wand: connected faces within an angle of the clicked face's normal |
| **Modify Selection** menu: Select All, Expand/Contract Ring, Optimize Boundary, Invert, Invert (Connected), Expand to Connected, Expand to Groups | ✅ | Ctrl+A, < >, O, I, Shift+I, E, G |
| Modify: Expand to Same Groups (non-connected) | ✅ | "Same groups"; `G` now expands through connected faces only, as in Meshmixer |
| Modify: Select Visible | ✅ | |
| **Edit** menu: Discard (X) | ✅ | |
| Edit: Erase & Fill (F) | ✅ | Flat Minimal / Flat Remeshed / Smooth MVC, Refine, Smooth, Scale, Bulge, Create New Groups (Meshmixer's defaults). Pieces the erase cuts loose inside the region (e.g. the far side of a spike the brush skipped) are removed instead of being filled into floating blobs |
| Edit: Flip Normals | ✅ | |
| Edit: Create / Clear FaceGroup (Ctrl+G / Ctrl+Shift+G) | ✅ | |
| Edit: Extrude (D) | ✅ | Harden, Density, Direction Normal / Constant / X / Y / Z, End type Offset / Flat, Preserve groups |
| Edit: Extract (Shift+D) — offset copy into a new object | ✅ | New closed solid object; Direction and End type |
| Edit: Offset (Ctrl+D) | 🟡 | Distance, Soft Transition, Preserve Boundary, Regularity ✅; Connected, Accuracy, Preserve Groups ❌ (meaning to confirm in Meshmixer) |
| Edit: Remesh (R) / Reduce (Shift+R) on the selection | ✅ | In-tool D/F density ±10, T adaptive (Remesh), G preserve groups |
| Edit: Bridge (two selected regions) | ✅ | ⌘B; Edit → Bridge boundaries also joins two existing openings |
| Edit: Join (along boundary edges) | ✅ | `J`: stitches the two open boundaries next to the selection |
| Edit: Weld Boundaries | ✅ | Close Cracks restricted to the selection's open edges, tolerance in edge lengths |
| Edit: Tube Handle | ✅ | Curved tube between two selected regions |
| Edit: Align To Target / Attract To Target | ✅ | Align: ICP on the selected (or all) faces, Translate and Rotate / Translate Only. Attract: offset, smooth, normal threshold, preserve mesh / group boundaries, optional refinement (density). Project to Target in Erase & Fill and Smooth Boundary |
| Edit: Separate (Y) / Duplicate selection | ✅ | |
| Edit menu hotkeys Join (J), Fit Primitive (P), Bridge (⌘B) | ✅ | |
| Edit: Fit Primitive | ✅ | Square, Rectangle, Cylinder, Sphere from plane / best-fit, Disc, Linear Sweep, Convex Hull, Height Field; Mesh Density, Single Primitive, Create New Objects; fitted sizes shown. The per-primitive adjust gizmo ❌ (use Transform) |
| Edit: Grow Hair | ⚪ ✅ | Select > Edit > Grow Hair: NumHairs, Points, Length, Curl Radius / Pitch, HairStiff / TipStiff, Gravity, Obj Collisions, Base / Tip Radius, Show Tubes; live preview, Accept adds tapered tubes as a new face group (Fill Density / Hair Patch modes ❌) |
| **Deform** menu: Smooth (⌘F) | ✅ | |
| Deform: Smooth Boundary ("Smooth Loop", B) | ✅ | Meshmixer's parameters and defaults; remesher smooths the border curve |
| Deform: Transform selection (T) | ✅ | Face Transform: T in Select with a selection; frame along the selection normal; Shared Frame, Fixed Boundary |
| Deform: Soft Transform (Shift+T) | ✅ | Linear / Nonlinear (ARAP) Deformation, Flat / Stretchy Transition; Boundary Continuity Boundaries / SoftPosition / SoftTangent; Falloff, Fixed Boundary, Ignore Groups, Shared Frame. Refine Mesh ❌ |
| Deform: Warp | ✅ | Right-click adds/removes handles, drag to warp; smooth (biharmonic) or detail-preserving (ARAP); handle size |
| **Convert To**: Open Part / Solid Part / Stamp | ✅ | Saved to the user's library (data directory `parts/`, `stencils/`); open parts need one border loop, solid parts a closed shell |

## 3. Sculpt

| Meshmixer | Polymix | Notes |
|---|---|---|
| Volume brushes (3) vs Surface brushes (2), Smoothing brush (1) | ✅ | Volume: dynamic refinement + volumetric footprint, Draw / BubbleSmooth; Surface: fixed triangles, footprint limited to the stroked surface; 1 opens RobustSmooth. Switchable in the panel |
| Brushes: RobustSmooth, ShrinkSmooth (Smooth), BubbleSmooth, Inflate, Flatten, Drag | ✅ | BubbleSmooth is a volume-keeping (Taubin) smooth |
| Brushes: Draw, Draw2, Draw3, DrawMax, Move, Push, Pinch, Spikes, Nudge, Zipper, Attract | ✅ | Own brush definitions (hover a brush for what it does); Attract pulls onto the target surface |
| Brushes: Refine, Reduce, AdaptiveReduce (mesh density brushes) | ✅ | Refine / Reduce step the edge length per dab (Refine bounded by the brush size); AdaptiveReduce is a quadric reduction that thins flat areas first |
| Brushes: PaintVertex, SmoothColor | ⚪ ✅ | Colour picker in the panel; Cmd/Ctrl paints the default grey back; strokes upload only the painted vertices |
| Dynamic refinement while sculpting (Refinement, Adaptivity; R, T) | ✅ | Per dab: split / collapse / relax under the brush; Refine, Reduce, Smooth, Adaptive + Adaptivity; R and T toggle |
| Size / Strength | ✅ | Size is a % of the model; Shift+wheel size (modifier set in Preferences), Cmd/Ctrl+right-drag strength (a plain right-drag always navigates) |
| Depth, Lazyness, Flow, Volumetric, Falloff curve | ✅ | Falloff Smooth / Linear / Sharp / Dome / Flat (own curve shapes) |
| Spacing, AlignToStroke, Stencils, brush textures | ❌ | |
| Primary / Secondary (Shift) / Invert (Ctrl/⌘) | ✅ | Secondary is a choice of smoothing brush; Invert for the draw brushes, Inflate, Push and Pinch |
| Toggles: Symmetry (Shift+S), Hold boundary (B), Preserve groups (G), Attract (A) | ✅ | Attract pulls toward the target after each dab. Symmetry mirrors every dab (and Move grabs) across an X / Y / Z plane through the object's centre, with an offset and Reset; free plane placement ❌ |
| SymSnap | ✅ | Own reading (Meshmixer has no help text for it): vertices that sat on the symmetry plane stay on it, so the halves stay joined |

## 4. Edit category

| Meshmixer | Polymix | Notes |
|---|---|---|
| Mirror | ✅ | Weld at the plane |
| Duplicate | ✅ | |
| Transform (T) | ✅ | Gizmo (move / rotate / per-axis + uniform scale), Local Frame (default; objects remember their frame) / World Frame, Translate / Rotate / Scale / Size X/Y/Z, Uniform Scaling, snapping (step; 15° rotations) with Absolute Coords/Sizes, center, drop to ground |
| Align (N) — align surfaces/axes to world | ✅ | Source Center / Base / Surface Point / Surface Scribble (left-click); Destination World Origin X/Y/Z-up, Ground Plane, or a surface (shift-click); Translate Only / Translate and Rotate; Flip. Principal-axes alignment in Transform |
| Create Pivot | ⚪ 🟡 | Edit > Create Pivot: Placement Mode (Snap To Vertex / Edge / Edge Midpoint / Face Center / Group Center / Group Border Center / Boundary Center, First Midpoint Along Ray / Normal, Bounding Box), Frame Mode (Geometry / Eye Ray / World). Pivots move with Transform and can be its centre; kept in projects. Existing Pivot / From Last Tool, Create Symmetric, Link To Target and pivot-to-pivot Align ❌ |
| Plane Cut | ✅ | Cut/Slice; No / Minimal / Fixed / Remeshed Fill (Remeshed also remeshes a band across the cut edge); Slice Groups; plus a vessel cross-section mode |
| Combine | ✅ | |
| Boolean Union / Difference / Intersection | ✅ | Fast Approximate (volumetric), Precise (exact, keeps triangles away from the seam; Auto-Reduce Result), Max Quality (seam band remesh: Target Edge Scale, Merge Border Rings). Precise needs closed meshes; coplanar faces are nudged automatically. Search Depth / Use Intersection Curves / Preview Iterations not exposed |
| Close Cracks | ✅ | Welds coincident open boundaries incl. T-junctions; tolerance slider with live preview |
| Generate Face Groups | ✅ | Edge Angle / Normal Angle, angle and size thresholds; shows group colours while open |
| Make Solid | 🟡 | Blocky / Fast / Accurate / Sharp Edge Preserve, accuracy, density, offset, min thickness, Cull Edges Threshold, close boundaries, Transfer Groups, Color Transfer Mode (Automatic / No Colors / Vertex-to-Vertex) ✅; Material-to-Vertex ⚪ (no materials/textures) |
| Hollow | ✅ | Hollow wall ✅; escape holes ⛔ (printing) |
| Make Pattern | ⚪ ✅ | All 10 pattern types (Edges, Dual Edges, Mesh + Delaunay variants, Tiled Tubes / Spheres, Lattice, Random Primitives Volume / Surface, FaceGroup Borders), Regular / Hex tiling, Subtract / Intersect, Clip To Surface, Grid Smoothing (joint fillets), Grid Resolution, Reduce, Line / Sphere gradient. Elements are balls and tubes (no Primitive Type choice); Mesh Resolution is folded into Reduce |
| Separate Shells (Shift+Y) | ✅ | |
| Add Tube | ✅ | Line / Spline (Inside, Outside), AutoRoute Minimal / Smooth (elastic band keeping the collision radius); Combine Boolean (union outside, difference inside) / Append / New Object; Direction Any / Face Normal; radius, end radius, tangent, length scale, solve steps; drag ends to move |
| Unwrap / UV layout / SetUV | ⚪ 🟡 | Edit > Unwrap: By Groups / By Components (cut into discs automatically), Conformal (LSCM) / Rigid (ARAP), Stacked / Line / Optimized packing on a Width × Height sheet with Border; layout drawn in the panel; UVs exported with OBJ and kept in projects, dropped by topology edits. UV Layout editing and SetUV ❌ |
| Generate / Filter / Split Complex, Convex Decomposition | ⚪ 🟡 | Generate Complex: cells of the overlapping visible closed objects (A − B, A ∩ B, B − A, …) by exact booleans, one face group per cell; Split Complex: one object per face group (Polymix's reading; Meshmixer documents neither). Convex Decomposition ✅: plane splits at the piece's own vertex positions until each piece is nearly convex; the hulls become a new object. Filter Complex ❌ |
| Make Slices | ⛔ | |
| Remesh (whole object) | ✅ | All four modes incl. Linear Subdivision; Density (%), Regularity, Threshold, Iterations, Transition, Preserve / Smooth group borders, sharp edges, boundary modes — Meshmixer's defaults |
| Reduce | ✅ | Percentage / triangle budget / max deviation; Uniform vs Shape Preserving (T) |
| Smooth (whole object) | ✅ | Shape Preserving / Uniform Triangles / Max Smoothness / Preserve Group Borders / Per-Group Membranes; Smoothing, Smoothing Scale, Constraint Rings (our reading: soft blend at the selection edge) |

## 5. Analysis category

| Meshmixer | Polymix | Notes |
|---|---|---|
| Inspector (I, Q = Auto Repair All) | ✅ | Minimal/Flat/Smooth fill, small-part threshold, click markers; + keep flat vessel ends |
| Units / Dimensions | ✅ | X/Y/Z dimensions in mm / cm / m / in (editable), unit conversion, bounding box with labels; Set From Measurements: drag to measure, real lengths, least-squares scale with avg / max deviation. Grid Size ❌ (automatic grid) |
| Measure | ✅ | Point to point (clicks or drag), point coordinates, min/max interior/exterior along the normal or axis X/Y/Z, fit cylinder, plus vessel cross-section diameter; snapping to vertices / edges / group borders; set target length (scales the object) |
| Mesh Query | ✅ | Color Mode: Vertex Position / Surface Normal / Aspect Ratio / Internal Angle / Gaussian / Mean Curvature with Color Range; click for triangle id, group, aspect ratio, min/max angle, nearest edge / vertex and curvatures. Per-triangle colours (Meshmixer blends per vertex) |
| Thickness (identify / thicken thin regions) | ⛔ | 3D printing wall check |
| Deviation | ✅ | Signed colour map + statistics |
| Clearance | ✅ | Min clearance threshold, Bi-Directional colouring, min distance with the closest points, intersection detection |
| Stability, Strength, Orientation, Overhangs/Support, Slicing, Layout/Packing | ⛔ | 3D printing |
| Mass properties | 🟡 | Area, volume (centre of mass ❌) |

## 6. Shaders / display

| Meshmixer | Polymix | Notes |
|---|---|---|
| Material library (clay variants, metals, glossy, resin…) | ✅ | Shaders palette: Clay, Soft clay, Satin, Glossy, Ceramic, Tissue, Resin, Gold, Silver, Copper (own lighting presets, per object, or apply to all) |
| Vertex colour display | ✅ | |
| Textured / textured-unlit / UV checker / X-ray | ⚪ 🟡 | View > Color: UV checker ✅; textures and X-ray ❌ |
| Wireframe | ✅ | |
| Face group colours | ✅ | |
| Light UI theme (warm greys: panels `#ede8e4`, text `#6b6864`, hover `#a5a19d`/`#b3afab`, scrollbar `#888480`, dark text `#444240`, disabled `#d0cac7`) | ✅ | View → Theme: Dark (default) / Light; palette uses Polymix's own vector icons |

## 7. Polymix features Meshmixer does not have

Trim branch (square vessel cut + cap) · Flow extensions · Cap open ends · Centerline with
branch lengths/diameters (CSV/VTK) · Check for CFD (incl. self-intersections, triangle
quality) · Boundary patches + multi-solid STL · Measure cross-section diameter ·
crease-angle sharp-edge shading · laser selection on 1M+ triangle meshes · Z-up (RAS) mode ·
free/turntable camera toggle.

