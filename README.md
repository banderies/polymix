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

## Workspaces

Polymix has three workspaces, shown as tabs at the top of the window (⌘1 / ⌘2 / ⌘3):

- **Volume**: find, organise and preview medical images (DICOM, NIfTI, NRRD, …) in a folder.
- **Segmentation**: segment the image selected in Volume, in three planes and in 3D, with
  3D Slicer's Segment Editor tools; segments go to Mesh as surfaces.
- **Mesh**: the mesh editor described below.

**Volume** (*Add folder…*): Polymix looks through the folder and everything under it,
recognising files by their contents rather than their names (DICOM files often have none), and
skips tool folders (virtual environments, version control). It lists:

- **DICOM** organised like 3D Slicer's DICOM browser: Patient → Study (date, description,
  accession) → Series, with a series split into volumes where it mixes orientations, echoes or
  sizes; localizers, derived images (reformats, projections, maps) and non-image objects
  (screenshots, reports) are marked. Enhanced multi-frame files work, and every common pixel
  compression (JPEG baseline and lossless, JPEG-LS, JPEG 2000, RLE, deflate), all in Rust.
- **NIfTI** (`.nii`, `.nii.gz`, NIfTI-2), **Analyze** (`.hdr` + `.img`), **NRRD** (including
  Slicer `.seg.nrrd` segmentations), **MetaImage** (`.mha`, `.mhd`), **FreeSurfer MGH / MGZ**,
  multi-page **TIFF** and folders of 2D images (JPEG, PNG, BMP, TIFF slices), by folder,
  marked as image, label map, segmentation, vector field or time series.
- What it recognised but cannot read yet (MINC, Imaris, MRC, …), what failed, and what it
  skipped.

Select a series or file for its header (grouped: patient, study, series, equipment,
acquisition, geometry, pixel) and a slice in the radiological convention. Warnings flag
missing or uneven slices, gantry tilt, missing orientation and pixel formats not viewable yet.
**Hide identifiers** masks names, IDs, birth dates and accession numbers, including inside
file names. Folders are remembered and rescans only read changed files. Images and folders
can also be opened from the Finder (*Open With → Polymix*, or dropped on the Dock icon).

*Prepare image* (under the slice) works on the selected image in the background: N4 bias
correction, denoising, resampling, crop / pad, reorientation and header fixes, registration to
another scan (rigid / affine, saved with a `.tfm` for 3D Slicer), subtraction, intensity
normalisation, vessel enhancement (Frangi / Sato / Jerman), MIP / MinIP slabs, 4D time points,
phase-contrast velocities, and de-identified copies for sharing. Results are saved as NIfTI
images under *Derived images* and can be segmented like any other image. *Viewer* shows the
selected image in three planes under the browser; *Compare with* puts another image over it
or beside it (blend, a wipe with a divider, or toggle with T), e.g. before and after N4.

**Segmentation** shows the image selected in Volume: axial, sagittal and coronal planes with
a shared crosshair, and a 3D view (maximum intensity projection, volume rendering, or the
surface above a threshold). Segments are drawn over the image and as surfaces in 3D. The
effects follow 3D Slicer's Segment Editor (keys too: 1–0 / ⇧1–⇧0 effects, Space the one
before, i intensity masking, q / w segments): Threshold (eleven automatic methods, or the
range from a box, circle or line dragged over the structure), Paint and Erase, Draw, Level
tracing, Scissors, Islands, Margin, Hollow, Smoothing (median, opening, closing, Gaussian,
joint), Logical operators, Grow from seeds, Watershed, Fill between slices, Flood fill, Fast
marching, Draw tube, Surface cut, Mask volume and Split volume, with Slicer's masking rules
(editable area, intensity range, which segments may be overwritten) and undo. For vessels:
*Level set* grows a rough segment to the vessel wall (on the threshold range, or following a
vesselness overlay), *Guided vessel* segments a vessel through points clicked along it, and
*Centerline* gives the segment's centerline tree with the diameter along each branch, its
narrowest section (a click there turns the planes across and along the vessel) and a CSV of
the cross-sections. In the 3D view a click paints, picks an island or a flood-fill seed, or
places a point; ⌘ / Ctrl + drag paints.
*Statistics* measures every segment (volume, surface, intensities, largest diameter,
roundness, elongation; copy as CSV). *Make mesh* sends segments to Mesh as closed surfaces;
*Save* / *Load segmentation* read and write 3D Slicer `.seg.nrrd` and NIfTI label maps; and a
mesh comes back as a segment from Mesh (right-click an object → *Send to Segmentation*).

The image tabs share their display: window presets (CT and automatic), colour maps, thick
slabs (MIP, MinIP, mean), layouts (four-up, 3D over the planes, planes in a row), *Overlay*
(another image over this one), rulers, angles and regions (area and statistics), and oblique
planes (**⌥ + drag** in a view turns the other two; *Straighten* puts them back). In the Mesh
tab, View → *Image slices* shows the Segmentation tab's image as three planes among the
meshes, to check a surface against the image.

Every image view uses the same mouse: **left drag** scrolls through slices, **right drag**
pans, **middle drag** zooms, **left + right drag** sets window / level, the **wheel** scrolls
slices (⌘ + wheel zooms) and a **click** moves the crosshair. While the window / level is
dragged, a histogram in the bottom left of the view shows the image's values with the window
over them. In the Segmentation tab a **double-click** makes a view fill the view area, and
another brings the four views back (**Fit** resets zoom and pan, **Auto** the window); in the
Volume tab's thumbnail a double-click resets. In the 3D view, left drag rotates (left + right
drag sets the window there too).

Each workspace is independent: if one runs into an internal error it stops on its own (its
tab shows what happened, with a Restart button and a log in the data folder) and the
others keep working. Dropped files go to the workspace that opens them (meshes and
`.polymix` projects to Mesh, images and folders to Volume).

## Agent access (MCP)

AI agents such as Claude Code can work with the open session over the [Model Context
Protocol](https://modelcontextprotocol.io). They can read the mesh scene with per-object
statistics and the selection, the image catalog and headers, the segmentation, the log, a
screenshot and the panels on screen, and every tool's state (a stent's picked ends, a
measurement, a preview's checks) as data. If you allow changes, they can also select, edit,
run tools (typed settings such as Remesh's edge length, a tool's actions such as placing a
stent's ends, clicks on the model, or any panel's buttons and sliders), segment images
(threshold, paint, islands, margin, … and make meshes), prepare images, undo, and open and
export files. Use it to debug Polymix or to have an
agent help with the data on screen. To turn it on, click **Agents** at the right of the tab
bar, tick *Let AI agents read this session* (and *Let agents make changes* if you want them
to), then press **Add to Claude Code**.

The server listens on this computer only and needs a token. Every change an agent makes is an
undo step labelled "Agent: …". Patient identifiers are always masked for agents and dates are
cut to the year. Tools, privacy rules and design: [docs/agent_access.md](docs/agent_access.md).

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

1. **Import** the segmentation (STL/OBJ/PLY/OFF/3MF; drag & drop works). As in Meshmixer,
   *Preferences → File → Flip Z-Y axis on import/export* (on by default) turns Z-up files
   (scanner / RAS, most CAD) upright in the Y-up view and turns them back on export, so the
   exported file keeps its original coordinates. With the flip off, view Z-up data with
   *View → Up axis → Z up*. Optionally *Edit → Duplicate object* first to keep the original
   for a later deviation check.
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

Treatment variants: **Vessel → Virtual stent** places a stent between two clicked
cross-sections (or two *Measure → Vessel diameter* rings) and reshapes the vessel around it:
diameter and length (centred on the two ends along the stent's own axis), *Straightness*
(follow the centerline ↔ the smoothest G0 / G1 / G2 path between the ends), *Stiffness* (how
far narrower wall opens to the stent; wider wall is left alone), *End blend*, and *Covered*
(the lumen becomes the stent's tube: dilated stretches narrowed, side branches sealed off).
Previewed live; kept in place or as a copy. Details and limits:
[docs/virtual_stenting.md](docs/virtual_stenting.md).

Printed flow models: cut the inlets and outlets square (**Vessel → Trim branch**, or Plane
Cut), then **Vessel → Thicken into a shell** (also **Edit → Thicken**): a wall of the given
thickness (e.g. 2 mm) grown outward (the lumen keeps its shape), inward or centred, the cut
ends kept open with flat rims. The new side comes from a distance field, so it never folds
into itself at a bifurcation or a tight bend as moving the surface along its normals would,
and the result is a closed, manifold wall ready to print.

Analysis extras: **Centerline** (vessel tree, per-branch length and diameters, CSV / VTK export
for ParaView/VMTK), **Measure** (cross-section equivalent diameter, distances), **Deviation**
(signed distance to a reference with colour map, mean/RMS/95th percentile/Hausdorff — handy for
reporting how much a cleanup changed the geometry), **Units/Dimensions**.

Work queue: [docs/ROADMAP.md](docs/ROADMAP.md) · Meshmixer parity status: [docs/feature_parity.md](docs/feature_parity.md).

## Navigation & hotkeys

| Action | Mouse / key |
|---|---|
| Tumble (free trackball in every direction; *View → Free rotation* off = turntable) | Shift + middle-drag (Fusion 360 mode), right-drag, or Alt + left-drag |
| Pan | Middle-drag, Shift + right-drag, or Alt + Shift + left-drag; the point you grab follows the cursor exactly (off the model, the object's centre of mass does) |
| Zoom | Scroll / pinch, Alt + right-drag (Meshmixer mode), or Alt + Cmd + left-drag. The wheel heads for the point under the cursor (or the centre of the view), each step a fraction of the distance to it, so it slows near the surface and never zooms through the model; speed and both choices in *Preferences → General → Scroll zoom* |
| Navigation scheme | *View → Navigation*: Fusion 360 (default) or Meshmixer, as in Meshmixer's preferences, or Custom: *Preferences → General → Mouse buttons* sets Orbit / Pan / Zoom for each middle and right drag (plain, Shift, Alt) |
| Brush size / strength (Select, Sculpt) | Shift + scroll (or Control / Option / Command: *Preferences → General → Brush size scroll*), Cmd + middle-drag or `[` `]` / Cmd + right-drag (a plain right-drag always navigates) |
| `S` | Opens Select with nothing selected (accepting the open tool, e.g. Erase & Fill) |
| 3D mouse (3Dconnexion SpaceMouse, macOS) | Move the cap the way you want the model to move; Menu button = Hotbox, Fit = focus. Settings and a live readout under *View → 3D mouse…*; needs the 3DxWare driver |
| Fit / wireframe / hotbox | `F` or the *Fit* button beside the view cube (frames the visible objects) / `W` / `Space` |
| Select / Inspector / Transform | `S` / `I` / `T` |
| Accept / Cancel / Undo / Redo | `Enter` / `Esc` / `Cmd+Z` / `Shift+Cmd+Z` |
| Switching tools | Opening another tool (hotkey, palette, menu) accepts the current one, as in Meshmixer; only `Esc` / Cancel discards |
| With a selection | `T` transform the selected faces, `Shift+T` soft transform, `X` discard, `F` erase & fill, `D` extrude, `Cmd+D` offset, `R` remesh, `Shift+R` reduce, `Cmd+F` smooth, `B` smooth boundary, `Y` separate, `Cmd+B` bridge, `E` connected, `I` invert, `G` groups, `O` optimise boundary, `<` `>` shrink/grow, `[` `]` brush size, `S` clear selection |
| Select tool | Drag on the model: paint · Drag from the background: lasso (closed loop selects inside) or laser cut (open curve selects the smaller side, through the model) · Double-click a triangle at an open edge: select the ring around that hole (then `F` to erase & fill it cleanly; with the rings of two holes selected, `F` lofts them together with a tube: Minimal / Straight / Smooth, G0 / G1 / G2 at each end, Round, Fair, Along the axis (the tube follows a curve joining the vessels' axes, its cross-section morphing without pinching), Context per end (how far back along each vessel the tube takes its direction from), Follow the vessel centerlines, Blend (tidy ragged openings and smooth the join into the vessels), Remesh the join; Vessel → Bridge boundaries previews the tube live before Join) · Double-click elsewhere: connected part · Shift: deselect |

Shading: *View → Sharp edges* (on by default) keeps hard corners on CAD parts crisp: edges
that bend more than the crease angle (default 40°) are shaded sharp, so flat faces render flat.
Smooth organic surfaces such as segmentations are unaffected; turn it off to smooth everything.

## Tools

| Category | Tools |
|---|---|
| Select | Sphere brush (fixed size on the model), lasso and laser lasso (through or visible only), hole-border ring, grow/shrink, connected, invert, groups, optimise boundary · Discard, Erase & Fill, Extrude / Offset, Separate, Duplicate, Flip normals, face groups · Smooth, Remesh, Reduce on the selection |
| Sculpt | Robust smooth, Smooth, Inflate, Flatten, Drag, the draw and pinch brushes and the rest of Meshmixer's set; spacing, brush textures turned with the stroke, a screen stencil; symmetry across any plane; Stamp (a library of shapes, icons and text for labelling prints, plus your own) |
| Edit | Transform (gizmo + numeric, local / world frame, sizes, snapping, centre, drop to ground, principal axes; Face Transform on a selection), Align (picked face to ground / origin / another surface), Plane Cut, Mirror, Make Solid (colours from vertices or the texture), Remesh, Reduce, Smooth, Offset / Hollow (with escape holes), Thicken (a shell with open ends), Boolean (union/difference/intersection; exact or volumetric), Add Tube, Fit Primitive (`P` in Select; an adjust gizmo), Create Pivot (incl. from the last tool, symmetric, linked to the target, pivot-to-pivot Align), Align / Attract To Target, Bridge, Separate Shells, Combine, Duplicate, Flip normals |
| Vessel | Trim branch, Flow extensions, Cap open ends, Thicken into a shell, Measure diameter, Centerline, Virtual stent, Boundary patches / CFD export, Check for CFD |
| Analysis | Inspector, Measure (distance, thickness, gaps, cylinder fit, vessel diameter, snapping), Mesh Query (normals, quality, curvature), Units / Dimensions (incl. scale from measurements), Clearance, Check for CFD, Deviation; centre of mass in the statistics |
| View & files | Textures (OBJ `vt` and `map_Kd`, or right-click → Texture…), UV checker, X-ray, grid size, image slices from the Segmentation tab; Import reference (a see-through guide), Import parts folder, export one file per region or per object |

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
