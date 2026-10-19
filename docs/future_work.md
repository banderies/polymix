# Future work: image preparation

Ideas for the Volume tab's *Prepare image* that are parked until the vessel workflow needs
them (discussed 2026-10-19). Each entry says what goes wrong without it, how to build it and
where it plugs in. What exists today: [segmentation_tools.md §7](segmentation_tools.md#7-image-preparation--volume-tab).
Ordered by value for brain vessels → CFD.

## 1. Regular slice geometry on load (CT and MR)

**Problem.** `mm_volume::organize` warns about gantry tilt, missing slices and uneven
spacing (the "missing slice(s)", "Uneven slice spacing" and "sheared (gantry tilt)"
warnings) but builds the grid from the image normal and the median slice spacing
(`Geometry::from_axes(…, [row, col, normal], …)`). So a tilted head CT is stored sheared: at
20° every vertical vessel leans 20° and the top of a 10 cm stack sits ~3.6 cm sideways of
where it really is. A missing slice squeezes the anatomy across the gap; uneven spacing
stretches it. Segments look fine, but lengths, angles and aneurysm ratios come out wrong.

**Fix.** Resample from each slice's real position onto a regular, orthogonal grid (3D
Slicer's *acquisition geometry regularization*), automatically on load or as a one-click fix
next to the warning. Tilt: the slice step vector (`span / (n − 1)`) as the third axis gives a
sheared affine that is correct; resample it onto an orthogonal grid with `mm_image::resample`.
Gaps and uneven spacing: interpolate along the slice axis between the neighbouring slices.

## 2. CT padding outside the field of view

**Problem.** The DICOM Pixel Padding Value (0028,0120) is not read, so voxels outside the
reconstruction circle stay at −2000 / −3024 HU. They skew percentile clipping, z-scores,
histogram matching and Otsu, and ring at the circle's edge when resampling with cubic or sinc.

**Fix.** Read the padding value (and Pixel Padding Range Limit, 0028,0121); set padded and
out-of-circle voxels to −1000 HU and leave them out of statistics (like *Statistics inside the
head only*).

## 3. CTA bone removal

**Problem.** Contrast-filled arteries and thin skull-base bone overlap in HU, so Threshold and
Flood fill leak into the skull around the carotid siphon.

**Fix.** Matched-mask bone elimination when a non-contrast CT of the same session exists:
register it to the CTA (`mm_image::register`, rigid), threshold bone on the plain scan, dilate
1–2 voxels for misregistration and partial volume, and overwrite those voxels in the CTA with a
soft-tissue value. Cleaner than *Subtract*, which leaves noise and misregistration rims.
Fallback without a plain scan: high threshold + largest connected component + dilation as a
bone mask to edit by hand. Needs an *apply a mask to an image* step (Slicer's *Mask volume*,
§2 of the tracking doc), which would also let N4 use a segment as its mask.

## 4. TOF slab boundaries

**Problem.** Multi-slab 3D TOF (MOTSA) has intensity steps where the slabs join ("venetian
blind" artifact); distal vessels fade at each junction. N4's smooth B-spline field cannot model
a step.

**Fix.** A per-slice intensity scale from stationary tissue (robust mean or median of each
slice inside the head mask), smoothed within slabs with the steps kept, divided out. One
threshold then works across all slabs.

## 5. Rician-aware non-local means (MR)

**Problem.** MR magnitude noise is Rician, not Gaussian, so the current non-local means leaves
dark regions biased bright, lowering the contrast of small distal vessels against background.

**Fix.** Manjón's unbiased non-local means: filter I², subtract 2σ², take the square root
(clamped at 0). σ from the background (Rayleigh mode) or the existing robust estimate. A small
change to `mm_image::denoise`.

## 6. 4D flow cleanup (for CFD inflow conditions)

Only if 4D flow will supply boundary conditions. Standard steps (4D flow consensus statement,
Dyverfeldt et al. 2015) missing from `mm_image::pc`:

- Background phase (eddy-current) correction: fit a low-order polynomial to the velocities in
  static tissue, per direction, and subtract it.
- Velocity unwrapping where |v| exceeds the VENC (temporal or Laplacian 4D unwrapping).
- Noise masking of velocities outside vessels (magnitude or angiogram threshold).

Then *flow through a plane* (§7 notes) gives inlet flow waveforms.

## 7. Preparation recipes

A saved chain of Prepare steps, run per case in one click, intermediates not kept, every
step's parameters recorded (methods sections, reproducibility). Examples — TOF: N4 → slab
correction → non-local means → resample to 0.3 mm. CTA: geometry fix → padding → crop → bone
removal → denoise → resample. The script steps (`vol.prepare:op=…`) already drive each
operation.

## 8. Skull stripping

**Which mask.** For vessels the mask is intracranial (brain + CSF), not a tight brain mask:
the circle of Willis lies in the CSF cisterns. Any brain mask also cuts the petrous and
cavernous ICA and the vertebral arteries below the foramen magnum, where CFD inlets often sit.
So it limits editing and statistics rather than cropping: a mask for N4 and normalisation,
hiding scalp fat and marrow (bright on TOF MIPs, thresholds and the 3D view), removing the
skull vault on CTA. Grow it by a few mm and let Paint add the inlet vessels back.

**AI tools.**

| Tool | Contrasts | Licence | Notes |
|---|---|---|---|
| [SynthStrip](https://surfer.nmr.mgh.harvard.edu/docs/synthstrip) (FreeSurfer, 2022) | Any MR contrast (trained on synthetic images); test set includes MRA and CT | Weights MIT or CC BY 4.0 (can be bundled) | CSF included by default (a no-CSF model exists); 29 MB PyTorch; < 1 min on CPU at ~1 mm; predicts a distance to the brain boundary, so the margin is a setting in mm |
| [HD-BET](https://github.com/MIC-DKFZ/HD-BET) (DKFZ, 2019) | T1, T1 + contrast, T2, FLAIR (tumour patients) | Apache-2.0 | `pip install hd-bet`, Apple MPS; not trained on TOF or CT |
| [CTA-BET](https://pubs.rsna.org/doi/full/10.1148/ryai.240847) (Radiology: AI, 2025) | CTA, plain CT (Dice 0.99 / 0.98) | — | No public code found (2026-10); [HD-CTBET](https://github.com/CAAI/HD-CTBET) is a CT fork of HD-BET, unverified |

**Classic methods** (no model, plain Rust):
- **CT** (easiest): the [validated recipe](https://pmc.ncbi.nlm.nih.gov/articles/PMC4446187)
  of Muschelli et al. 2015 keeps 0–100 HU, smooths, then fits a BET surface with a low
  fractional intensity (0.01). Simpler: fill the cavity inside the skull (bone threshold,
  closing, fill, largest component). On CTA, contrast vessels fall outside 0–100 HU; hole
  filling brings the inner ones back.
- **MR, morphological** (BrainSuite BSE style): smooth (`mm_image::denoise`), threshold
  (Otsu), shrink to break brain–scalp bridges, keep the largest island, grow back, fill holes
  (`mm_segment`). Good on T1 / T2, fragile on TOF (dim brain, bright fat, vessels bridging).
- **MR, BET** (Smith 2002): a deformable sphere expanding to the brain edge; more robust, more
  work. Write it from the paper (FSL's licence is non-commercial).

**Running SynthStrip:**
1. Call an installed copy (`mri_synthstrip` from FreeSurfer ≥ 7.3, or the Docker image):
   write a NIfTI, run it, read the mask back. Quick, but every user installs FreeSurfer or
   Docker.
2. Inside Polymix: export the network to ONNX and run it with the `ort` crate (Core ML), or in
   pure Rust with `burn` on the wgpu backend Polymix already uses; weights bundled.
   Self-contained, but SynthStrip's preprocessing (1 mm grid, intensity scaling, padding to a
   multiple of 64) must match exactly.

**Recommendation.** TOF / contrast MRA: SynthStrip inside Polymix (one model for TOF,
contrast MRA and CT, keeps the CSF, licence allows bundling). Mostly CTA: the classic CT method
first, no model needed. The result is a label map usable as N4's mask, an editable area, or
(with *Mask volume*) to blank the image outside.

## Also

- **A viewer in the Volume tab, before / after comparison** (§8 of the tracking doc): tuning
  any of the above on a thumbnail is hard.

## Deferred further

- **Metal artifact reduction** (coils, clips): real methods need the scanner's raw projection
  data; image-domain methods are weak.
- **Deformable registration**: affine is usually enough in the head.
- **Gibbs unringing** (Kellner et al. 2016).
- **Dual-energy CT** material decomposition (bone removal without a second scan).
