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
| Segment list: Add, Remove, rename, colour + terminology, visibility (eye), state flag (Not started / In progress / Completed / Flagged), layer column, filter bar, q / w = previous / next segment | — | 🟡 | Add, Remove, rename (double-click), colour, visibility, volume in mL; terminology, state flag, filter, q / w ❌ |
| Segments in the slice views: fill and outline, opacity per segment | — | ✅ | Fill (opacity slider) and outline; one opacity for all segments |
| Show 3D: segment surfaces kept up to date while editing, smoothing factor (0 = none, 0.1 weak, ≥ 0.5 strong) | — | 🟡 | Surfaces update after each edit / brush stroke; smoothing factor ❌ |
| Undo / redo (z / y), last 10 states | ✅ | ✅ | ⌘Z / ⇧⌘Z and buttons; 10 run-length-encoded states; a brush stroke is one step |
| Masking — Editable area: Everywhere, Inside / Outside all segments, Inside / Outside all visible segments, Inside / Outside one segment | ✅ | ✅ | |
| Masking — Editable intensity range (i toggles; preview with Threshold) | ✅ | 🟡 | Threshold's Use for masking sets it; the i key ❌ |
| Masking — Modify other segments: Overwrite all (default), Overwrite visible, Allow overlap | 🟡 | 🟡 | Overwrite all / visible, and Keep other segments in place of Allow overlap (needs layers) |
| Source volume and segmentation geometry: Specify geometry (oversampling, ROI extent, Pad output) | ❌ | 🟡 | Labels follow the volume's grid; selecting a volume on the same grid (e.g. its N4-corrected copy) keeps the segments, another grid asks before discarding them |
| Effects in the 3D view (Paint, Scissors, Islands clicks) | — | ❌ | needs picking in `VolumeView3d` |
| Shortcuts: ← / → slice, Shift + move = scroll to the pointer, Ctrl + wheel zoom, Esc, Space = last two effects, 1–0 / Shift + 1–0 = effects, i = intensity masking | — | 🟡 | Wheel, ⌘ / Ctrl + wheel zoom, Esc, Shift + wheel brush size; the rest ❌ |

## 2. Segment Editor effects (Slicer core)

| Slicer effect and options | Engine | UI | Notes |
|---|---|---|---|
| **Threshold** — range slider over the histogram; live preview; Automatic threshold (Otsu, Huang, IsoData, Kittler-Illingworth, Li, Maximum entropy, Moments, Renyi entropy, Shanbhag, Triangle, Yen) and how it is used (lower, upper, or around the automatic value); range from a local histogram (box, circle or line drawn on a slice); Use for masking | 🟡 | 🟡 | Histogram with draggable range, live preview of the result, Auto (Otsu), Use for masking; other automatic methods and local histogram ❌ |
| **Paint** — brush radius in mm or % of the view; click = one dab, drag = stroke (applied on release); Sphere brush (3D); Pixel mode; Edit in 3D views; Shift + wheel and − / + resize | ✅ | 🟡 | Radius in mm, disc or sphere, brush outline in the views, Shift + wheel; applied live; % of view, Pixel mode, 3D views, − / + ❌ |
| **Draw** — click points or drag a line on a slice; double-click / right click fills; x deletes the last point, a / Enter applies | ❌ | ❌ | Right click → Enter / double-click (right drag pans) |
| **Erase** — Paint that removes; with Editable area = a segment, erased voxels move into that segment (split a segment) | ✅ | ✅ | Shares Paint's brush; the split works |
| **Level tracing** — moving the pointer outlines the same-intensity region under it on the slice; click applies | ❌ | ❌ | 2D flood fill on the slice |
| **Grow from seeds** — fast GrowCut from ≥ 2 visible seed segments; Initialize / Update / Auto-update / Cancel / Apply; seed locality; preview | ❌ | ❌ | `region_grow` is a different (threshold flood) algorithm |
| **Fill between slices** — morphological contour interpolation across any number of skipped slices; Initialize / Update / Auto-update / Cancel / Apply; shape only | ❌ | ❌ | |
| **Margin** — grow or shrink by mm; Apply to visible segments | ✅ | 🟡 | Selected segment; Apply to visible segments ❌ |
| **Hollow** — shell of a given thickness; current segment as the inside, medial or outside surface; Apply to visible segments | 🟡 | ❌ | Composes from `grow` / `shrink` / `subtract`; vessel walls |
| **Smoothing** — Median, Opening, Closing (kernel in mm), Gaussian (σ), Joint smoothing (all visible segments together, watertight boundaries); Apply to all or brush a region | 🟡 | 🟡 | Median, Opening, Closing, Fill holes; Gaussian, Joint, brush region, Apply to visible segments ❌ |
| **Scissors** — Erase inside / Erase outside / Fill inside / Fill outside; free-form, circle or rectangle (lines can snap horizontal / vertical); slice cut: Unlimited, Positive, Negative, Symmetric with thickness; slice or 3D views; Apply to visible segments | ❌ | ❌ | Lasso projected through the volume |
| **Islands** — Keep largest; Remove small (minimum size); Split islands to segments; Keep selected; Remove selected; Add selected (click an empty region) | ✅ | ✅ | All six; island clicks in the slice views |
| **Logical operators** — Copy, Add, Subtract, Intersect, Invert, Clear, Fill; Bypass masking | ✅ | 🟡 | All seven; Bypass masking ❌ (masking applies) |
| **Mask volume** — Fill inside, Fill outside, or Fill inside and outside (binary) with fill values; Soft edge (Gaussian σ); output to a new or the same volume | ❌ | ❌ | Needs a scalar-volume writer |

## 3. SegmentEditorExtraEffects

| Slicer effect and options | Engine | UI | Notes |
|---|---|---|---|
| **Local threshold** — ⌘ / Ctrl-click adds the structure under the pointer within a threshold range (set visually, from a local histogram, or a slider); Minimum diameter stops leaks through narrow necks; region of interest; algorithm GrowCut, Watershed or Masking, with feature size | 🟡 | 🟡 | As Flood fill with the threshold range and a minimum diameter (opening); ROI, GrowCut / Watershed variants ❌ |
| **Flood filling** — similar-intensity voxels connected to the click (intensity tolerance, neighbourhood size) | ✅ | ✅ | Tolerance around the clicked value or the threshold range; minimum diameter stops leaks |
| **Draw tube** — tube through clicked points with a radius (catheters, nerves, vessels) | ❌ | ❌ | Mesh tab has Add Tube; here as voxels |
| **Fast marching** — grows the current segment, target size (% of volume) limits leaks | ❌ | ❌ | |
| **Watershed** — complete segmentation from seeds, object scale smooths the result | ❌ | ❌ | |
| **Surface cut** — smooth closed blob through points placed on its surface | ❌ | ❌ | |
| **Split volume** — crop the volume around each segment into separate volumes | ❌ | ❌ | ⚪ |
| **Engrave** — emboss or engrave text into a segment | ❌ | ❌ | ⚪ |

## 4. Vessels (SlicerVMTK) — the main Polymix workflow

| Slicer tool | Engine | UI | Notes |
|---|---|---|---|
| **Vesselness filtering** — Frangi filter: minimum / maximum vessel diameter, contrast, suppression of plates and blobs; result as a volume to threshold | ❌ | ❌ | Multi-scale Hessian |
| **Level set segmentation** — geodesic active contours from seeds, on the image or its vesselness | ❌ | ❌ | |
| **Guided arterial segmentation** — vessels grown along seed points / a rough centerline | ❌ | ❌ | |
| **Extract centerline** — centerline network / tree with radii, from a segment | ❌ | ❌ | Mesh tab has centerlines of meshes |
| **Cross-section analysis** — diameter and area along a centerline | ❌ | ❌ | |

## 5. Segmentations module, statistics, import and export

| Slicer | Engine | UI | Notes |
|---|---|---|---|
| Export segments as a label map: NIfTI, `.seg.nrrd` with segment names and colours | ✅ | ✅ | Save segmentation…; Slicer and the Volume tab read them back |
| Import a label map / `.seg.nrrd` / NIfTI labels as segments | ❌ | ❌ | The Volume tab already finds them (kinds Label map, Segmentation) |
| Export segments as models (STL / OBJ / PLY), closed-surface smoothing | ✅ | 🟡 | Make mesh → Mesh tab (closed, manifold, LPS like Slicer), exported from there |
| **Send to the Mesh tab** (Polymix) | ✅ | ✅ | Selected or every visible segment; arrives through Mesh's import (Append / Replace) |
| Import a model as a segment (voxelise) | ❌ | ❌ | `mm-solid` voxelises meshes |
| Copy / move segments between segmentations; convert representations | ❌ | ❌ | ⚪ |
| **Segment statistics** — voxel count, volume (mm³, cm³), surface area, mean / SD / min / max / median intensity, centroid, bounding box, Feret diameter, roundness, elongation | 🟡 | 🟡 | Volume (mL) in the segment list; the rest ❌ |
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
crate), **UI** the Volume tab.

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
| Before / after comparison (toggle or split view) | ❌ | Judging N4, denoising, registration |
| Two volumes overlaid: foreground over background, opacity, colour map | ❌ | Registration check, vesselness over MRA |
| Window presets (CT brain / bone / angio; MR automatic) and colour maps | 🟡 | Auto and File windows only |
| Histogram while the window / level is dragged | ✅ | `polymix_imaging::histogram`: bottom left of the view, log counts, the window's ends and ramp; volume in slice views, the slice in the thumbnail |
| Thick-slab MIP in the slice views | ❌ | Following vessels in TOF / CTA |
| Oblique reslicing: rotate a plane, plane along a vessel | ❌ | Views follow the voxel grid now |
| Layouts: one view, 1 × 3, 2 × 2; maximise a view | 🟡 | 2 × 2; double-click fills the view area with a view and back (not while a tool has the left button); 1 × 3 ❌ |
| A viewer in the Volume tab (for preparing images) | 🟡 | Thumbnail only; reuse `SliceView` |
| Measurements in the views: ruler, angle, region statistics | ❌ | |
| Screenshots | ❌ | ⚪ |

## 9. After segmentation — Mesh tab

| Feature | Status | Notes |
|---|---|---|
| Receive segment meshes (`MeshWorkspace::add_meshes`) | ✅ | Through the import prompts |
| Image slices in the Mesh 3D view, to check the surface against the image | ❌ | |
| Mesh → segment (voxelise; round trip for fixes) | ❌ | |
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
