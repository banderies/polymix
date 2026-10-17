# Polymix

An open-source mesh editor in the spirit of Autodesk Meshmixer, written in Rust and aimed at
cleaning up medical segmentations (e.g. cerebral vessels from 3D Slicer) for CFD.

* Native and GPU-rendered (wgpu/Metal); stays interactive on multi-million-triangle meshes,
  and all heavy operations run in the background with live previews.
* Meshmixer-style workflow, panels and hotkeys (brush/lasso selection, Inspector, Remesh,
  Reduce, Smooth, Erase & Fill, Plane Cut, Make Solid, Transform, Sculpt, Space hotbox…).
* Vessel / CFD tools Meshmixer never had: square branch trimming, flow extensions, centerline
  with branch lengths and diameters, CFD readiness check (incl. self-intersections), deviation
  analysis against the original segmentation, and named boundary patches with multi-solid STL
  export.

## Build & run

Requires Rust (<https://rustup.rs>).

```sh
cargo run -p polymix --release                 # start the app
cargo run -p polymix --release -- vessel.stl   # open a file directly
scripts/make_app.sh                            # build dist/Polymix.app (double-clickable)
cargo test --workspace --release               # run the test suite
```

Synthetic test meshes (a messy branching vessel with an aneurysm, holes and floating
fragments, plus a 1.3M-triangle sphere):

```sh
cargo run -p polymix --example gen_test_meshes --release -- test_meshes
```

## Typical vessel → CFD workflow

1. **Import** the segmentation (STL/OBJ/PLY/OFF/3MF; drag & drop works). For scanner (RAS)
   coordinates use *View → Up axis → Z up*. Optionally *Edit → Duplicate object* first to keep
   the original for a later deviation check.
2. **Analysis → Inspector** (`I`): click markers to fix individual holes / non-manifold
   vertices / floating parts, or **Auto Repair All** (`Q`). Fill mode Minimal / Flat / Smooth.
3. Remove unwanted pieces: **Edit → Separate Shells** (`Shift+Y`), or **Select** (`S`), paint
   or lasso the region, then `X` (Discard) or `F` (Erase & Fill: delete and fill smoothly).
4. **Smooth**: select a bumpy region and press `Cmd+F` (Max Smoothness fairs it into its
   surroundings), smooth the whole model (Shape Preserving, volume-preserving), or use the
   *Robust smooth* sculpt brush.
5. **Vessel → Trim branch**: click a branch where an inlet/outlet should be. It is cut
   perpendicular to the vessel axis (only that branch) and capped; caps become patches named
   `outlet_N`. **Plane Cut** is available for general cuts (drag a line; Shift+click a vessel to
   cut it square; the red half is removed).
6. **Vessel → Flow extensions**: extrude straight, optionally circular extensions (length in
   diameters) from the selected ends; caps keep their patch names.
7. **Edit → Remesh** (`R`) for uniform, well-shaped triangles. **Make Solid** rebuilds a
   guaranteed closed, manifold, non-self-intersecting surface when the input is messy.
8. **Vessel → Check for CFD**: watertight, manifold, single shell, outward orientation,
   self-intersections, triangle quality, patches — each with a one-click fix.
9. **Vessel → Boundary patches**: name wall/inlet/outlets (or *Auto-name*: largest cap =
   inlet) and export a multi-solid ASCII STL (OpenFOAM `snappyHexMesh`, Fluent, CFX,
   SimVascular) or one STL per patch. Plain export: **Export** (`Cmd+E`).

Analysis extras: **Centerline** (vessel tree, per-branch length and diameters, CSV / VTK export
for ParaView/VMTK), **Measure** (cross-section equivalent diameter, distances), **Deviation**
(signed distance to a reference with colour map, mean/RMS/95th percentile/Hausdorff — handy for
reporting how much a cleanup changed the geometry), **Units/Dimensions**.

Work queue: [docs/ROADMAP.md](docs/ROADMAP.md) · Meshmixer parity status: [docs/feature_parity.md](docs/feature_parity.md).

## Navigation & hotkeys

| Action | Mouse / key |
|---|---|
| Tumble (free trackball in every direction; *View → Free rotation* off = turntable) | Shift + middle-drag (Fusion 360 mode), right-drag, or Alt + left-drag |
| Pan | Middle-drag, Shift + right-drag, or Alt + Shift + left-drag |
| Zoom | Scroll / pinch, Alt + right-drag (Meshmixer mode), or Alt + Cmd + left-drag |
| Navigation scheme | *View → Navigation*: Fusion 360 (default) or Meshmixer, as in Meshmixer's preferences |
| 3D mouse (3Dconnexion SpaceMouse, macOS) | Move the cap the way you want the model to move; Menu button = Hotbox, Fit = focus. Settings and a live readout under *View → 3D mouse…*; needs the 3DxWare driver |
| Focus / wireframe / hotbox | `F` / `W` / `Space` |
| Select / Inspector / Transform | `S` / `I` / `T` |
| Accept / Cancel / Undo / Redo | `Enter` / `Esc` / `Cmd+Z` / `Shift+Cmd+Z` |
| Switching tools | Opening another tool (hotkey, palette, menu) accepts the current one, as in Meshmixer; only `Esc` / Cancel discards |
| With a selection | `T` transform the selected faces, `Shift+T` soft transform, `X` discard, `F` erase & fill, `D` extrude, `Cmd+D` offset, `R` remesh, `Shift+R` reduce, `Cmd+F` smooth, `B` smooth boundary, `Y` separate, `Cmd+B` bridge, `E` connected, `I` invert, `G` groups, `O` optimise boundary, `<` `>` shrink/grow, `[` `]` brush size, `S` clear selection |
| Select tool | Drag on the model: paint · Drag from the background: lasso (closed loop selects inside) or laser cut (open curve selects the smaller side, through the model) · Double-click a triangle at an open edge: select the ring around that hole (then `F` to erase & fill it cleanly) · Double-click elsewhere: connected part · Shift: deselect |

Shading: *View → Sharp edges* (on by default) keeps hard corners on CAD parts crisp: edges
that bend more than the crease angle (default 40°) are shaded sharp, so flat faces render flat.
Smooth organic surfaces such as segmentations are unaffected; turn it off to smooth everything.

## Tools

| Category | Tools |
|---|---|
| Select | Sphere brush (fixed size on the model), lasso and laser lasso (through or visible only), hole-border ring, grow/shrink, connected, invert, groups, optimise boundary · Discard, Erase & Fill, Extrude / Offset, Separate, Duplicate, Flip normals, face groups · Smooth, Remesh, Reduce on the selection |
| Sculpt | Robust smooth, Smooth, Inflate, Flatten, Drag |
| Edit | Transform (gizmo + numeric, local / world frame, sizes, snapping, centre, drop to ground, principal axes; Face Transform on a selection), Align (picked face to ground / origin / another surface), Plane Cut, Mirror, Make Solid, Remesh, Reduce, Smooth, Offset / Hollow, Boolean (union/difference/intersection; exact or volumetric), Bridge, Separate Shells, Combine, Duplicate, Flip normals |
| Vessel | Trim branch, Flow extensions, Cap open ends, Measure diameter, Centerline, Boundary patches / CFD export, Check for CFD |
| Analysis | Inspector, Measure, Units / Dimensions, Check for CFD, Deviation |

## Project layout

| Crate | Contents |
|---|---|
| `mm-core` | Dynamic triangle mesh (edge split/flip/collapse, vertex split), BVH, selections, topology, self-intersection test, primitives |
| `mm-io` | STL (binary/ASCII), OBJ, PLY, OFF, 3MF readers and writers |
| `mm-remesh` | Isotropic remeshing (feature/boundary preserving, region), quadric-error reduction |
| `mm-repair` | Hole filling (minimal/flat/smooth, Liepa), Laplacian/Taubin smoothing, thin-plate fairing, Inspector auto-repair |
| `mm-edit` | Plane cut (incl. local vessel cuts), caps, flow extensions, extrude, offset, bridge, mirror, transforms |
| `mm-solid` | Fast winding numbers, sparse narrow-band SDF, manifold surface extraction (Make Solid), offsets, SDF booleans |
| `polymix` | The application (egui + wgpu), centerline extraction, CFD tools |

Polymix's algorithms are independent implementations of published methods (references in the source).

## License

MIT OR Apache-2.0.
