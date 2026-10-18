# Volumetric image formats

What the Volume workspace reads today, and the formats other fields use that it could read
later. "Effort" is for a reader of headers and voxels in Rust: **small** (a fixed or text
header, a day or less), **medium** (a structured container), **large** (a complex codec or
sparse structure). Researched October 2026; crate versions as of then.

## Supported now

| Format | Files | Notes |
|---|---|---|
| DICOM | `.dcm`, `.ima`, no extension, DICOMDIR | dicom-rs 0.10. Classic and Enhanced multi-frame; pixels in uncompressed, deflate, RLE and JPEG (baseline, extended, lossless) syntaxes, all pure Rust. Organised Patient → Study → Series like 3D Slicer |
| NIfTI-1 / NIfTI-2 | `.nii`, `.nii.gz`, `.hdr` + `.img` | Own reader (both header sizes, both byte orders); sform → qform → voxel sizes |
| Analyze 7.5 | `.hdr` + `.img` | No orientation in the format (flagged) |
| NRRD | `.nrrd`, `.nhdr` + data file | Raw and gzip data; Slicer segmentations (`.seg.nrrd`) with segment names |

Recognised but not read yet (listed as such by the scanner): MetaImage, MGH/MGZ, MINC, Imaris,
DREAM3D, MRC/CCP4, FITS, VTK legacy, Interfile, ECAT 7, and folders of 2D images (image stacks).

## Next steps for the formats above

| Gap | Approach | Effort |
|---|---|---|
| JPEG-LS and JPEG 2000 / HTJ2K DICOM pixels | Pure Rust first: `j2k-native` (8/12/16-bit, HTJ2K) or the codecs of `dicom-toolkit-rs`; otherwise `charls-sys` (C++) and `openjpeg-sys` (C) through dicom-rs features | medium |
| Siemens mosaics (fMRI, DTI) | Unpack the tiles; slice count from the CSA header | medium |
| DICOM SEG, RT Structure Set | Segmentations and contours from DICOM (useful for the Segmentation workspace) | medium |
| NRRD bzip2, multi-file NRRD | `bzip2` crate; `data file: LIST` | small |

## Medical and research imaging

| Field | Formats | Rust support | Effort |
|---|---|---|---|
| Clinical radiology (CT, MR, PET, SPECT, ultrasound) | DICOM (also RT Dose grids, SEG, Parametric Map; Enhanced US Volume) | dicom-rs | done / medium for the extras |
| Research exports | NIfTI, NRRD, MetaImage (`.mha`, `.mhd`) | MetaImage: write our own (text `key = value` header) | small |
| Neuroimaging | NIfTI-1/2, Analyze, MGH/MGZ (FreeSurfer), MINC 1/2, CIFTI (NIfTI-2), BIDS folders (NIfTI + JSON sidecars), Bruker ParaVision `2dseq`, Varian/Agilent `.fdf` | MGH: `neuroformats` crate or our own (284-byte header); MINC 2 needs HDF5 | small (MGH, BIDS) / medium (MINC) |
| PET legacy | ECAT 7 (`.v`), Interfile (`.hv`, `.h33`) | write our own | small |
| Preclinical | DICOM, Analyze, Interfile, Siemens Inveon (`.hdr` text + `.img`), vendor raw + header | write our own | small |
| Ultrasound 3D/4D | DICOM multi-frame; vendor files (GE Kretz `.vol`, Philips private data) | vendor formats are undocumented | large |

## Industrial CT and non-destructive testing

| Formats | Rust support | Effort |
|---|---|---|
| Raw voxels + a header: Volume Graphics `.vgi` / `.vgl` + `.vol`, Nikon `.xtekct` + `.vol`, Waygate / GE phoenix `.pcr` + `.vol`, North Star `.nsihdr` | text headers, write our own | small each |
| Zeiss Xradia `.txm`, `.txrm` | OLE2 compound files (`cfb` crate) | medium |
| Bruker SkyScan reconstructions | BMP / PNG / TIFF stack + `.log` | small (with image stacks) |
| TIFF / BigTIFF stacks | `tiff` 0.11 (pure Rust) | small |
| DICONDE (DICOM for industrial inspection) | dicom-rs | small |

## Microscopy and biology

| Formats | Rust support | Effort |
|---|---|---|
| OME-TIFF, ImageJ TIFF | `tiff` + our own OME-XML / ImageJ metadata parsing | small |
| OME-Zarr (NGFF), N5 | `zarrs` 0.23 + `ome_zarr_metadata` (pure Rust, mature) | small–medium |
| Imaris `.ims`, BigDataViewer HDF5 | HDF5: `hdf5-metno` (C libhdf5) or the young pure-Rust readers (`hdf5-pure`, `hdf5-reader`) | medium |
| Zeiss `.czi`, Nikon `.nd2`, Leica `.lif` / `.lof`, Olympus `.oir` / `.vsi` | `czi-rs`, `nd2-rs` (young); Bio-Formats is Java and GPL (not usable) | medium–large |
| Cryo-EM: MRC / CCP4 (`.mrc`, `.map`, `.rec`) | `mrc` crate, or our own (1024-byte header) | small |

## Materials science

| Formats | Rust support | Effort |
|---|---|---|
| DREAM3D `.dream3d`, synchrotron DXchange / NeXus | HDF5 | medium |
| Avizo / Amira `.am` | text header + raw / RLE | small |
| TIFF stacks | `tiff` | small |

## Geoscience, oil and gas

| Formats | Rust support | Effort |
|---|---|---|
| SEG-Y seismic cubes | `giga-segy-in` 0.5 | medium (EBCDIC header, IBM floats, trace headers) |
| GOCAD Voxet (`.vo` + property files) | text header + raw floats | small |
| OpenVDS / VDS, Petrel ZGY, RESQML | vendor libraries / HDF5 | large |

## Weather, climate and oceanography

| Formats | Rust support | Effort |
|---|---|---|
| NetCDF 3 / 4 | `netcdf` (C library) or `netcdf-reader` (pure Rust, young); NetCDF-4 is HDF5 | small (v3) / medium (v4) |
| GRIB 1 / 2 | `grib` crate (pure Rust) | medium |
| Zarr (cloud datasets) | `zarrs` | small–medium |

## Astronomy

| Formats | Rust support | Effort |
|---|---|---|
| FITS data cubes (radio spectral cubes, integral field units) | `fitsrs` (pure Rust) or `fitsio` (cfitsio, C) | small |
| CASA image tables, HDF5 | — / HDF5 | large / medium |

## Simulation, visual effects

| Formats | Rust support | Effort |
|---|---|---|
| VTK image data (`.vti`, legacy `.vtk` structured points) | `vtkio` 0.6 / 0.7-rc | small |
| OpenVDB, NanoVDB | `vdb-rs` (partial), `nanovdb-rs` | large (sparse tree, Blosc) |
| Raw volumes (`.raw`, `.dat`) with the size given by the user | our own (ask for dims, type, spacing) | small |

## Suggested order

1. **MetaImage** and **MGH/MGZ**: common in research pipelines, small readers.
2. **Image stacks** (TIFF, PNG, JPEG, BMP) and **raw + header** industrial CT formats: one
   reader covers many vendors; also reads exported slice folders like the reference data's
   INV2 series.
3. **JPEG 2000 / JPEG-LS** DICOM pixels, so every clinical DICOM previews.
4. **OME-Zarr** and **MRC** for microscopy and cryo-EM.
5. HDF5-based formats (MINC 2, Imaris, DREAM3D) once a pure-Rust HDF5 reader matures.

## Sources

dicom-rs (github.com/Enet4/dicom-rs), nifti-rs (github.com/Enet4/nifti-rs), the NIfTI-1 and
NRRD specifications, 3D Slicer (`DICOMScalarVolumePlugin.py`, `DICOMUtils.py`), CTK
(`ctkDICOMDatabase.cpp`), dcm2niix (`isSameSet`), ITK/GDCM (`GDCMSeriesFileNames`,
`IPPSorter`), Bio-Formats' supported-formats list, and crates.io / docs.rs for the crates named.
