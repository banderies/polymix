# Segmentation workflow: 3D Slicer → Polymix tools

Every tool the image → segmentation → mesh workflow needs, with its status and the tab it
belongs in: 3D Slicer's Segment Editor (core effects, the SegmentEditorExtraEffects
extension, the vessel tools of SlicerVMTK), the image preparation done before segmenting (N4
bias correction, denoising, registration, …), the viewer features they share, and what
happens to segments afterwards. Goal: at least everything Slicer does. Sources: the
[Segment Editor guide](https://slicer.readthedocs.io/en/latest/user_guide/modules/segmenteditor.html),
[SegmentEditorExtraEffects](https://github.com/lassoan/SlicerSegmentEditorExtraEffects) and
SlicerVMTK.

**Which tab:** a tool lives in the tab of what it produces: images in **Volume**, segments
(label maps) in **Segmentation**, surfaces in **Mesh**. A derived image (bias-corrected,
denoised, resampled, vesselness) joins the Volume catalog as a Derived volume and can then be
chosen as the Segmentation tab's source volume. Hand-offs are the exception: Make mesh and
Mask volume sit where the work finishes (Segmentation) as buttons that send the result to the
tab that owns it.

| Tab | Sections |
|---|---|
| Volume | [7. Image preparation](#7-image-preparation--volume-tab) |
| Segmentation | [1](#1-foundations-needed-by-most-tools)–[6](#6-ai-assisted-segmentation-slicer-extensions) |
| Volume and Segmentation | [8. Viewer features](#8-viewer-features--volume-and-segmentation-tabs-polymix-imaging) |
| Mesh | [9. After segmentation](#9-after-segmentation--mesh-tab) |

**Status:** ✅ done · 🟡 partial (see notes) · ❌ missing · ⚪ low priority.
**Engine** is the GUI-free algorithm (`crates/mm-segment`, unit-tested); **UI** is the tool in
the Segmentation tab (`crates/polymix-segmentation`). A tool is finished when both are ✅.

**Mouse:** with no tool active, Polymix's slice views keep the user's mapping (left drag =
slices, right = pan, middle = zoom, left + right = window / level, wheel = slices). A tool that
needs the left button takes it (`SliceOptions::left_drag_scrolls = false` →
`SliceResponse::left_drag_voxel`); right drag stays pan, so where Slicer uses a right click
(Draw: fill) Polymix uses Enter or a double-click.

---

## 1. Foundations (needed by most tools)

| Slicer | Engine | UI | Notes |
|---|---|---|---|
| Segments in one segmentation: binary label maps, shared layers when they do not overlap | 🟡 | 🟡 | One `u8` label map (up to 255 segments); no overlapping segments yet |
| Segment list: Add, Remove, rename, colour + terminology, visibility (eye), state flag (Not started / In progress / Completed / Flagged), layer column, filter bar, q / w = previous / next segment | — | ✅ | Add, Remove, rename (double-click), colour, visibility, volume in mL, status dot (click cycles; saved in `.seg.nrrd`), filter by name and status, q / w; terminology ⚪ |
| Segments in the slice views: fill and outline, opacity per segment | — | ✅ | Fill (opacity slider) and outline; one opacity for all segments |
| Show 3D: segment surfaces kept up to date while editing, smoothing factor (0 = none, 0.1 weak, ≥ 0.5 strong) | — | ✅ | Surfaces update after each edit / brush stroke; smoothing 0–8 (blur passes over the segments' occupancy) |
| Undo / redo (z / y), last 10 states | ✅ | ✅ | ⌘Z / ⇧⌘Z and buttons; 10 run-length-encoded states; a brush stroke is one step |
| Masking — Editable area: Everywhere, Inside / Outside all segments, Inside / Outside all visible segments, Inside / Outside one segment | ✅ | ✅ | |
| Masking — Editable intensity range (i toggles; preview with Threshold) | ✅ | ✅ | Threshold's Use for masking sets it; i toggles it |
| Masking — Modify other segments: Overwrite all (default), Overwrite visible, Allow overlap | 🟡 | 🟡 | Overwrite all / visible, and Keep other segments in place of Allow overlap (needs layers) |
| Source volume and segmentation geometry: Specify geometry (oversampling, ROI extent, Pad output) | ❌ | 🟡 | Labels follow the volume's grid; selecting a volume on the same grid (e.g. its N4-corrected copy) keeps the segments, another grid asks before discarding them |
| Effects in the 3D view (Paint, Scissors, Islands clicks) | — | ✅ | `VolumeView3d::pick` (first shown segment, else the surface / window centre / brightest voxel as rendered): a click paints a ball, picks an island or a flood-fill seed, or places a Draw tube / Surface cut / Guided vessel point; ⌘ / Ctrl + drag paints or draws Scissors (plain drags turn the camera) |
| Shortcuts: ← / → slice, Shift + move = scroll to the pointer, Ctrl + wheel zoom, Esc, Space = last two effects, 1–0 / Shift + 1–0 = effects, i = intensity masking | — | ✅ | All but Shift + move (⚪); − / + resize the brush |

## 2. Segment Editor effects (Slicer core)

| Slicer effect and options | Engine | UI | Notes |
|---|---|---|---|
| **Threshold** — range slider over the histogram; live preview; Automatic threshold (Otsu, Huang, IsoData, Kittler-Illingworth, Li, Maximum entropy, Moments, Renyi entropy, Shanbhag, Triangle, Yen) and how it is used (lower, upper, or around the automatic value); range from a local histogram (box, circle or line drawn on a slice); Use for masking | ✅ | ✅ | All eleven automatic methods (`mm_segment::autothreshold`), from the threshold up or up to it; local histogram: a box, circle or line dragged on a slice, its histogram shown, the range from its minimum / 5th percentile / average to its average / 95th percentile / maximum |
| **Paint** — brush radius in mm or % of the view; click = one dab, drag = stroke (applied on release); Sphere brush (3D); Pixel mode; Edit in 3D views; Shift + wheel and − / + resize | ✅ | ✅ | Radius in mm or % of the view, disc or sphere, Pixel mode, brush outline in the views, Shift + wheel and − / +; applied live; in the 3D view ⌘ / Ctrl + drag |
| **Draw** — click points or drag a line on a slice; double-click / right click fills; x deletes the last point, a / Enter applies | ✅ | ✅ | Right click → Enter / double-click (right drag pans); `mm_segment::shapes::polygon_prism` |
| **Erase** — Paint that removes; with Editable area = a segment, erased voxels move into that segment (split a segment) | ✅ | ✅ | Shares Paint's brush; the split works |
| **Level tracing** — moving the pointer outlines the same-intensity region under it on the slice; click applies | ✅ | ✅ | Brighter or darker than the pointer's value (`shapes::trace_level`) |
| **Grow from seeds** — fast GrowCut from ≥ 2 visible seed segments; Initialize / Update / Auto-update / Cancel / Apply; seed locality; preview | ✅ | ✅ | `mm_segment::grow_from_seeds` (competing shortest paths, as fast GrowCut); margin around the seeds |
| **Fill between slices** — morphological contour interpolation across any number of skipped slices; Initialize / Update / Auto-update / Cancel / Apply; shape only | ✅ | ✅ | `mm_segment::fill_between_slices` (signed-distance interpolation), any plane |
| **Margin** — grow or shrink by mm; Apply to visible segments | ✅ | ✅ | Apply to visible segments is one undo step |
| **Hollow** — shell of a given thickness; current segment as the inside, medial or outside surface; Apply to visible segments | ✅ | ✅ | `mm_segment::hollow`: cut from a lightly smoothed sub-voxel distance (shells within ~7%, where plain grow / shrink come out ~16% thin), always closed. For a printed wall, Thicken in the Mesh tab is exact |
| **Smoothing** — Median, Opening, Closing (kernel in mm), Gaussian (σ), Joint smoothing (all visible segments together, watertight boundaries); Apply to all or brush a region | ✅ | 🟡 | Median, Opening, Closing, Fill holes, Gaussian, Joint (`mm_segment::joint_smooth`: no gaps or overlaps between the visible segments, hidden ones untouched), Apply to visible segments; brushing a region ❌ |
| **Scissors** — Erase inside / Erase outside / Fill inside / Fill outside; free-form, circle or rectangle (lines can snap horizontal / vertical); slice cut: Unlimited, Positive, Negative, Symmetric with thickness; slice or 3D views; Apply to visible segments | ✅ | ✅ | In the slice views (with slice ranges), and in the 3D view: ⌘ / Ctrl + drag, cutting through the whole depth along the view. Line snapping ⚪ |
| **Islands** — Keep largest; Remove small (minimum size); Split islands to segments; Keep selected; Remove selected; Add selected (click an empty region) | ✅ | ✅ | All six; island clicks in the slice views |
| **Logical operators** — Copy, Add, Subtract, Intersect, Invert, Clear, Fill; Bypass masking | ✅ | ✅ | All seven; Bypass masking |
| **Mask volume** — Fill inside, Fill outside, or Fill inside and outside (binary) with fill values; Soft edge (Gaussian σ); output to a new or the same volume | ✅ | ✅ | Always a new image among the derived images (Volume tab) |

## 3. SegmentEditorExtraEffects

| Slicer effect and options | Engine | UI | Notes |
|---|---|---|---|
| **Local threshold** — ⌘ / Ctrl-click adds the structure under the pointer within a threshold range (set visually, from a local histogram, or a slider); Minimum diameter stops leaks through narrow necks; region of interest; algorithm GrowCut, Watershed or Masking, with feature size | ✅ | 🟡 | As Flood fill with the threshold range, a minimum diameter (opening) and edges refined by Grow from seeds or Watershed within a feature size; ROI ❌ |
| **Flood filling** — similar-intensity voxels connected to the click (intensity tolerance, neighbourhood size) | ✅ | ✅ | Tolerance around the clicked value or the threshold range; minimum diameter stops leaks |
| **Draw tube** — tube through clicked points with a radius (catheters, nerves, vessels) | ✅ | ✅ | `shapes::tube` (centripetal Catmull–Rom); points in slice views or the 3D view |
| **Fast marching** — grows the current segment, target size (% of volume) limits leaks | ✅ | ✅ | `mm_segment::fast_march` |
| **Watershed** — complete segmentation from seeds, object scale smooths the result | ✅ | ✅ | `mm_segment::watershed` (seeded, on the smoothed gradient) |
| **Surface cut** — smooth closed blob through points placed on its surface | ✅ | ✅ | `shapes::hull` (the points' convex hull) |
| **Split volume** — crop the volume around each segment into separate volumes | ✅ | ✅ | One image per visible segment, a margin in mm, optionally a value outside the segment; saved among the derived images |
| **Engrave** — emboss or engrave text into a segment | ✅ | ✅ | Click in a slice view: the text (A–Z, 0–9, - . + / :, Polymix's own letters) reads left to right there, cut in or raised, a letter height and a depth through the slices |

## 4. Vessels (SlicerVMTK) — the main Polymix workflow

| Slicer tool | Engine | UI | Notes |
|---|---|---|---|
| **Vesselness filtering** — Frangi filter: minimum / maximum vessel diameter, contrast, suppression of plates and blobs; result as a volume to threshold | ✅ | ✅ | In the Volume tab (§7): `mm_image::vessel`, Frangi / Sato / Jerman |
| **Level set segmentation** — geodesic active contours from seeds, on the image or its vesselness | ✅ | ✅ | `mm_segment::level_set` (Caselles 1997, narrow band): Level set effect from the segment, on the threshold range or following the overlay shown (e.g. vesselness) |
| **Guided arterial segmentation** — vessels grown along seed points / a rough centerline | ✅ | ✅ | `mm_segment::guided_vessel`: Guided vessel effect — a minimal path through the clicked points, a thin tube along it grown to the wall by a level set |
| **Extract centerline** — centerline network / tree with radii, from a segment | ✅ | ✅ | `mm_segment::centerlines` (TEASAR, Sato 2000): Centerline section, ends placed by clicking or found; drawn in the slice and 3D views |
| **Cross-section analysis** — diameter and area along a centerline | ✅ | ✅ | `mm_segment::{cross_sections, stenosis}`: area, equivalent / min / max diameter every 0.5 mm; the narrowest section and its narrowing; a diameter plot (a click goes there, the planes turned with the vessel); CSV |

## 5. Segmentations module, statistics, import and export

| Slicer | Engine | UI | Notes |
|---|---|---|---|
| Export segments as a label map: NIfTI, `.seg.nrrd` with segment names and colours | ✅ | ✅ | Save segmentation…; Slicer and the Volume tab read them back |
| Import a label map / `.seg.nrrd` / NIfTI labels as segments | ✅ | ✅ | Load segmentation…: `.seg.nrrd` (names, colours, status) or NIfTI (names from Polymix's description); another grid is resampled by position (nearest) |
| Export segments as models (STL / OBJ / PLY), closed-surface smoothing | ✅ | 🟡 | Make mesh → Mesh tab (closed, manifold, LPS like Slicer), exported from there |
| **Send to the Mesh tab** (Polymix) | ✅ | ✅ | Selected or every visible segment; arrives through Mesh's import (Append / Replace) |
| Import a model as a segment (voxelise) | ✅ | ✅ | Mesh tab: right-click an object → Send to Segmentation; LPS or RAS chosen by overlap, inside by winding number |
| Copy / move segments between segmentations; convert representations | ❌ | ❌ | ⚪ |
| **Segment statistics** — voxel count, volume (mm³, cm³), surface area, mean / SD / min / max / median intensity, centroid, bounding box, Feret diameter, roundness, elongation | ✅ | ✅ | `mm_segment::segment_stats` (surface within 1% on balls); Statistics section with a CSV copy; agents: `segmentation_statistics` |
| Standard terminology (anatomical names and colours) | ❌ | ❌ | ⚪ |

## 6. AI-assisted segmentation (Slicer extensions)

| Slicer extension | Engine | UI | Notes |
|---|---|---|---|
| TotalSegmentator (automatic whole-body, includes vessels on CT) | ❌ | ❌ | ⚪ external model |
| MONAI Auto3DSeg / MONAI Label | ❌ | ❌ | ⚪ external model |
| nnInteractive (segment from clicks, boxes, scribbles) | ❌ | ❌ | ⚪ external model |
| Brain mask / skull stripping (SynthStrip, HD-BET) | ❌ | ❌ | A mask for N4, cropping and editable area; a classic (non-AI) fallback is possible |

## 7. Image preparation — Volume tab

Image → image. Each result is a Derived volume (with where it came from, e.g. "N4 of Series 8")
that the Segmentation tab can segment. **Engine** is GUI-free code (`mm-volume` or a new
crate), **UI** the Volume tab. Parked ideas (slice geometry regularization, CT padding, CTA bone
removal, TOF slab correction, skull stripping, …): [future_work.md](future_work.md).

| Tool | Engine | UI | Notes |
|---|---|---|---|
| **N4 bias field correction** — removes MR coil shading; mask (automatic Otsu, or a segment), shrink factor, iterations per level, B-spline control-point spacing, convergence; corrected volume and the bias field | ✅ | ✅ | `mm_image::n4` (Tustison 2010: Wiener histogram sharpening + B-spline fit, levels); automatic head mask; segment as mask ❌ |
| **Denoising** — Gaussian, median, curvature / gradient anisotropic diffusion (edge-preserving), non-local means | ✅ | ✅ | `mm_image::{filter, denoise}`; noise σ estimate shown before / after |
| **Resample** — isotropic or chosen spacing; linear, cubic or windowed-sinc interpolation | ✅ | ✅ | `mm_image::resample` (Catmull-Rom cubic, Lanczos-3 sinc; anti-aliased when shrinking) |
| **Crop** to a box drawn in the views; pad | ✅ | 🟡 | Box by voxel ranges (outlined on the thumbnail) or *Fit to the head*; drawing it in the views ❌ |
| **Reorient** — to standard axes; flip / permute axes when a header is wrong | ✅ | ✅ | `mm_image::grid`: any axis code (RAS, LPS, …), mirror across a patient axis, swap header axes |
| **Registration** — rigid / affine (deformable later) between series of a patient; resample one onto another's grid | ✅ | ✅ | `mm_image::register` (mutual information or correlation, pyramid); ITK `.tfm` for Slicer; deformable ❌ |
| **Subtraction** — post − pre contrast after registration | ✅ | ✅ | The other image is registered (optional) and resampled onto this one's grid |
| **Intensity normalisation** — rescale, percentile clip, z-score, histogram matching | ✅ | ✅ | `mm_image::intensity`; statistics inside the head (optional) |
| **Vessel enhancement** — Frangi / Sato / Jerman vesselness, multi-scale (min / max diameter) | ✅ | ✅ | `mm_image::vessel`; bright or dark vessels |
| **MIP / MinIP slab volumes** | ✅ | ✅ | `mm_image::slab` (sliding, any plane, max / min / mean); as a viewing option (§8) ❌ |
| 4D: choose a time point or echo, average, split | ✅ | ✅ | Every time point of DICOM 4D series is kept; pick / mean / max / min / save each |
| 4D flow MRI velocity fields (phase contrast) | ✅ | ✅ | `mm_image::pc`: velocities, mean / peak speed, magnitude-weighted angiogram (VENC, phase range guessed, per-axis flip); flow through a plane ❌ |
| **Save derived volumes** (NIfTI / NRRD) | ✅ | ✅ | `mm_volume::write::nifti` (u8 / i16 / f32); saved to `<data dir>/volume/derived`, listed as *Derived images* |
| De-identified export (identifiers stripped, optional defacing) | ✅ | ✅ | `mm_volume::deid`: DICOM PS3.15 basic profile subset (UIDs replaced, dates shiftable, private tags dropped, screenshots skipped), NIfTI / Analyze / NRRD / MetaImage free text cleared, files named after the label; defacing ❌ |

## 8. Viewer features — Volume and Segmentation tabs (`polymix-imaging`)

| Feature | Status | Notes |
|---|---|---|
| Before / after comparison (toggle or split view) | ✅ | Compare with (Volume viewer) / Overlay (Segmentation): wipe with a divider, or toggle (T) |
| Two volumes overlaid: foreground over background, opacity, colour map | ✅ | `layers::Fusion`: resampled by position onto the shown grid; window, colour map, opacity |
| Window presets (CT brain / bone / angio; MR automatic) and colour maps | ✅ | Window ▾ (automatic, file, full range, six CT windows); grey, hot, rainbow, viridis and single hues |
| Histogram while the window / level is dragged | ✅ | `polymix_imaging::histogram`: bottom left of the view, log counts, the window's ends and ramp; volume in slice views, the slice in the thumbnail |
| Thick-slab MIP in the slice views | ✅ | MIP / MinIP / mean over a thickness in mm, on grid and oblique planes |
| Oblique reslicing: rotate a plane, plane along a vessel | ✅ | ⌥ + drag turns the other two planes about a view; Straighten; the planes turned with a vessel at a centerline section. Drawing tools (Draw, Scissors, Level tracing, local histogram, measurements) stay on grid slices |
| Layouts: one view, 1 × 3, 2 × 2; maximise a view | ✅ | Four-up, 3D over the three planes, three planes in a row; double-click fills the view area with a view and back |
| A viewer in the Volume tab (for preparing images) | ✅ | *Viewer*: three planes under the browser with the same controls and Compare with |
| Measurements in the views: ruler, angle, region statistics | ✅ | Ruler, Angle, Region (circle: area, mean ± SD, min, max) in the toolbar; listed, copied as text |
| Screenshots | ❌ | ⚪ |

## 9. After segmentation — Mesh tab

| Feature | Status | Notes |
|---|---|---|
| Receive segment meshes (`MeshWorkspace::add_meshes`) | ✅ | Through the import prompts |
| Image slices in the Mesh 3D view, to check the surface against the image | ✅ | View → Image slices: the Segmentation tab's image through its crosshair (or placed by sliders), LPS or RAS |
| Mesh → segment (voxelise; round trip for fixes) | ✅ | Right-click an object → Send to Segmentation |
| Aneurysm morphology: neck width, dome height, aspect and size ratios | ❌ | ⚪ on surface + centerline |
| Smoothing, remeshing, trim branch, flow extensions, centerline, CFD check | ✅ | See [feature_parity.md](feature_parity.md) |

---

## Suggested order (vessel segmentation → CFD mesh)

1. Foundations: segment list, labels drawn in the slice and 3D views, undo / redo, masking
   (editable area, intensity range, overwrite modes); source volume picker.
2. Threshold (preview, automatic), Islands, Paint / Erase (sphere brush), Local threshold and
   Flood filling. In Volume: N4, crop, resample, saving derived volumes, and a viewer to
   compare before and after.
3. Make mesh → Mesh tab; save `.seg.nrrd` / NIfTI.
4. Margin, Smoothing, Logical operators, Scissors, Hollow; thick-slab MIP in the views.
5. Grow from seeds, Fill between slices, Level tracing, Draw.
6. Vessel tools: vesselness, level set, draw tube, centerline; registration and subtraction.
7. Statistics, Mask volume, import of label maps and models, the remaining extra effects,
   denoising, AI.
